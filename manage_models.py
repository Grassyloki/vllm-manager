#!/usr/bin/env python3
"""
manage_models.py — list, delete, download, profile local LLM models.

Commands:
    manage_models.py list                              show installed
    manage_models.py delete <name> [--yes]             remove a model
    manage_models.py download <repo_or_url>            GGUF picker + fetch
    manage_models.py profile list <model>              list launch profiles
    manage_models.py profile show <model> [name]       print profile(s)
    manage_models.py profile add  <model> <name>       new profile, seeded from default
    manage_models.py profile copy <model> <src> <dst>  duplicate profile
    manage_models.py profile delete <model> <name>     remove profile
    manage_models.py profile edit <model>              open profiles.toml in $EDITOR
    manage_models.py profile path <model>              print path to profiles.toml
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, field, dataclass

import model_lib
from model_lib import (
    DEFAULT_MAX_MODEL_LEN,
    LocalModel,
    Profile,
    copy_profile,
    detect_model_max_len,
    dump_profiles_toml,
    ensure_profiles_exist,
    extract_quant_tag,
    human_gb,
    load_profiles,
    make_default_profile,
    profiles_path,
    save_profiles,
    scan_all,
    scan_local_model,
)

# --- config ------------------------------------------------------------------
DEFAULT_MODEL_DIR = os.environ.get(
    "VLLM_MGR_MODEL_DIR", "/mnt/stor1/vllm/models")
LMSTUDIO_MODEL_DIR = os.environ.get(
    "LMSTUDIO_MODEL_DIR", "/mnt/stor1/LMStudio/models")
DEFAULT_VRAM_GB = 64
KV_HEADROOM = 0.30
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_TREE_API = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"

# Multi-root layout. "vllm" = flat: <root>/<repo>. "lmstudio" = two-level:
# <root>/<publisher>/<repo>. Each (root, layout) pair is scanned for `list`
# and is a valid `download --target` destination.
ROOTS: list[tuple[str, str]] = [
    (DEFAULT_MODEL_DIR,  "vllm"),
    (LMSTUDIO_MODEL_DIR, "lmstudio"),
]

# Where vllm_manager's setup caches the vLLM parser registry. Used to gate
# parser flags in auto-created profiles (same rule as vllm_manager itself).
PID_DIR = os.environ.get("VLLM_MGR_PID_DIR", "/root/.vllm-pids")
PARSER_CACHE = os.path.join(PID_DIR, "vllm-version.txt.json")

QUANT_PREFERENCE = [
    "UD-Q4_K_XL",
    "Q4_K_M",
    "UD-Q5_K_XL",
    "Q5_K_M",
    "UD-Q3_K_XL",
    "Q3_K_M",
    "UD-Q6_K_XL",
    "Q6_K",
    "Q8_0",
    "UD-IQ3_XXS",
    "IQ4_XS",
    "IQ3_M",
    "IQ2_M",
]

SHARD_RE = model_lib.SHARD_RE


# == Host probing (GPUs, V100 detection, parser registry, RAM) ===============
# Auto-created profiles must match the host they'll run on: SM70 hosts need
# the V100 arg set (and must NOT get --enable-expert-parallel), and parser
# flags must be gated by what the installed vLLM build actually ships.

_HOST_GPUS: list[dict] | None = None


def _host_gpus() -> list[dict]:
    """[{index, name, mem_total_mb, compute_cap}] via nvidia-smi; cached.

    [] on GPU-less hosts (e.g. the dev laptop) — callers fall back to
    generic defaults then.
    """
    global _HOST_GPUS
    if _HOST_GPUS is not None:
        return _HOST_GPUS
    out: list[dict] = []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    out.append({
                        "index": int(parts[0]), "name": parts[1],
                        "mem_total_mb": int(float(parts[2])),
                        "compute_cap": parts[3] if len(parts) > 3 else "",
                    })
                except ValueError:
                    continue
    except (OSError, subprocess.SubprocessError):
        pass
    _HOST_GPUS = out
    return out


def _host_is_v100() -> bool:
    return any((g.get("compute_cap") or "").startswith("7.0")
               or "V100" in (g.get("name") or "")
               for g in _host_gpus())


def _default_vram_gb() -> float:
    """Total VRAM across host GPUs, or DEFAULT_VRAM_GB if none detected."""
    gpus = _host_gpus()
    if gpus:
        return sum(g["mem_total_mb"] for g in gpus) / 1024
    return float(DEFAULT_VRAM_GB)


def _cached_parsers() -> tuple[set[str], set[str]]:
    return model_lib.read_cached_parsers(PARSER_CACHE)


def _host_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


# == Multi-root scanning =====================================================
# vLLM root is flat (one level); LMStudio root is publisher/repo (two levels).
# Both are surfaced via the same LocalModel shape; only the .name format
# differs ("repo" vs "publisher/repo").

def _scan_lmstudio(root: str) -> list[LocalModel]:
    if not os.path.isdir(root):
        return []
    out: list[LocalModel] = []
    for pub in sorted(os.listdir(root)):
        pub_path = os.path.join(root, pub)
        if not os.path.isdir(pub_path):
            continue
        for repo in sorted(os.listdir(pub_path)):
            full = os.path.join(pub_path, repo)
            if not os.path.isdir(full):
                continue
            # scan_local_model takes (root, name); pass the publisher dir
            # as root so name == repo, then prefix the surfaced name.
            m = scan_local_model(pub_path, repo)
            m.name = f"{pub}/{repo}"
            out.append(m)
    return out


def scan_root(root: str, layout: str) -> list[LocalModel]:
    if layout == "vllm":
        return scan_all(root)
    if layout == "lmstudio":
        return _scan_lmstudio(root)
    raise ValueError(f"unknown layout: {layout}")


def scan_all_roots() -> list[tuple[str, str, LocalModel]]:
    """Return [(root, layout, LocalModel), ...] across every configured root."""
    out: list[tuple[str, str, LocalModel]] = []
    for root, layout in ROOTS:
        for m in scan_root(root, layout):
            out.append((root, layout, m))
    return out


def resolve_model(name: str) -> tuple[str, str, LocalModel] | None:
    """Find a (root, layout, LocalModel) for `name`, searching every root.

    For LMStudio, `name` may be either 'publisher/repo' (preferred) or just
    'repo' (matched if unambiguous).
    """
    matches: list[tuple[str, str, LocalModel]] = []
    for root, layout, m in scan_all_roots():
        if m.name == name:
            return (root, layout, m)
        if layout == "lmstudio" and m.name.endswith("/" + name):
            matches.append((root, layout, m))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"  '{name}' is ambiguous — use the full publisher/repo name:")
        for _, _, m in matches:
            print(f"    {m.name}")
    return None


def model_path_in_root(root: str, layout: str, name: str) -> str:
    """Filesystem path for a model named `name` under (root, layout)."""
    return os.path.join(root, name)  # name already includes 'publisher/' for lmstudio


# == Shared TUI helpers come from tui_lib ====================================

import tui_lib as tui

_tui_addstr   = tui.addstr
_tui_select   = tui.select
_tui_pause    = tui.pause
_tui_run_cmd  = tui.run_cmd
_tui_text     = tui.text
_C_TITLE   = tui.C_TITLE
_C_GREEN   = tui.C_GREEN
_C_YELLOW  = tui.C_YELLOW
_C_DIM     = tui.C_DIM
_C_CYAN    = tui.C_CYAN
_C_RED     = tui.C_RED


# == TUI actions ==


def _create_planned_profile(root: str, short: str) -> None:
    """Create a [default] profile sized by compute_plan for THIS host's GPUs.

    Replaces the old naive default (tp_size=1, native context) that OOM'd
    for anything bigger than one GPU. Picks tp_size / max_model_len from the
    sizing planner, applies the V100 arg set on SM70 hosts, and gates parser
    flags by the installed build's registry. GPU-less hosts (dev laptop)
    fall back to the generic auto-template.
    """
    if load_profiles(root, short):
        print(f"    Kept existing {profiles_path(root, short)}")
        return
    v100 = _host_is_v100()
    tools, reasonings = _cached_parsers()
    gpus = _host_gpus()

    if not gpus:
        ensure_profiles_exist(
            root, short, interactive=False, v100=v100,
            available_tool_parsers=tools or None,
            available_reasoning_parsers=reasonings or None,
        )
        print(f"    Wrote {profiles_path(root, short)} "
              f"(no GPU detected — generic default; run `plan` on the host)")
        return

    m = scan_local_model(root, short)
    info = model_lib.detect_family(root, short)
    vram = max(1, min(g["mem_total_mb"] for g in gpus) // 1024)
    mem_util = 0.88 if v100 else 0.90
    plan = model_lib.compute_plan(
        m, info, vram_per_gpu_gb=vram, num_gpus=len(gpus),
        gpu_mem_util=mem_util, concurrency=2,
    )
    if plan.fits and plan.recommended_max_len:
        max_len = plan.recommended_max_len
    else:
        max_len = detect_model_max_len(root, short) or DEFAULT_MAX_MODEL_LEN
    extra = model_lib.suggest_extra_args(
        info, tp_size=plan.tp_size, v100=v100,
        available_tool_parsers=tools or None,
        available_reasoning_parsers=reasonings or None,
    )
    prof = make_default_profile(
        description=(f"Auto-planned at download for "
                     f"{plan.tp_size}x{vram}GB, concurrency=2."),
        tp_size=plan.tp_size,
        gpu=",".join(str(i) for i in range(plan.tp_size)),
        gpu_mem_util=mem_util,
        max_model_len=max_len,
        extra_args=extra or None,
    )
    save_profiles(root, short, {"default": prof})
    print(f"    Wrote {profiles_path(root, short)}")
    print(f"    Planned: tp_size={plan.tp_size}  max_model_len={max_len}  "
          f"gpu_mem_util={mem_util}" + ("  (V100 arg set)" if v100 else ""))
    if not plan.fits:
        print(f"    WARNING: the planner says this model does NOT fit "
              f"({plan.tp_size}x{vram}GB) — inspect `plan {short}` before starting.")
    for n in plan.notes:
        print(f"    note: {n}")


def _post_download_profile(dest: str, target: str, kind: str):
    """Auto-create a default profile for a freshly-downloaded model.

    Skipped for lmstudio targets and any GGUF model — those have no profile
    concept in this manager.
    """
    if target != "vllm" or kind != "hf":
        print("\n  (no profile created — lmstudio target or GGUF model)")
        return
    print("\n  Creating launch profile ...")
    _create_planned_profile(DEFAULT_MODEL_DIR, os.path.basename(dest))


def _tui_act_download(stdscr):
    import curses

    # Step 1: pick target before leaving curses (uses the menu widget).
    target_items = [
        (f"vllm      ({DEFAULT_MODEL_DIR})", 0),
        (f"lmstudio  ({LMSTUDIO_MODEL_DIR})", 0),
        ("Back", curses.A_DIM),
    ]
    t_idx = _tui_select(
        stdscr, "Download — Select Destination",
        target_items,
        header=[
            ("vllm:     flat layout, profile auto-created for HF models",
             curses.A_DIM),
            ("lmstudio: publisher/repo layout, no profile (GGUF-friendly)",
             curses.A_DIM),
        ],
    )
    if t_idx not in (0, 1):
        return
    target = "vllm" if t_idx == 0 else "lmstudio"

    curses.endwin()
    print("\n  == Download Model ==")
    print(f"  Target: {target}")
    repo = input("  Enter HF repo (org/name or URL): ").strip()
    if not repo:
        print("\n  Aborted.")
        input("\n  Press Enter to continue ...")
        return
    dest_name = input("  Destination folder name (leave blank for default): ").strip()
    try:
        repo_id = parse_repo_id(repo)
    except SystemExit as e:
        print(f"\n  Error: {e}")
        input("\n  Press Enter to continue ...")
        return
    _, _, dest = _resolve_download_dest(target, repo_id, dest_name or None)
    print(f"\n  Fetching file tree for {repo_id} ...")
    try:
        tree = fetch_tree(repo_id)
    except SystemExit as e:
        print(f"\n  Error: {e}")
        input("\n  Press Enter to continue ...")
        return
    kind = detect_repo_kind(tree)
    if kind == "empty":
        print("\n  Repo has neither .gguf nor HF-format weight files.")
        input("\n  Press Enter to continue ...")
        return

    if kind == "hf":
        weight_bytes, n_files, quant = summarize_hf_bundle(tree, repo_id)
        size_gb = weight_bytes / (1024 ** 3)
        vram = _default_vram_gb()
        budget, _ = weights_budget(vram)
        fit = "ok" if size_gb <= budget else ("tight" if size_gb <= vram else "no")
        print(f"\n  HF-format repo (safetensors). Quant hint: {quant or '-'}")
        print(f"  Weight files: {n_files} file(s), {size_gb:.1f} GB  —  fit: {fit}")
        problem = _v100_compat_problem(_fetch_repo_config(repo_id, tree)) \
            if _host_is_v100() else None
        if problem:
            print(f"\n  WARNING: this checkpoint will NOT run on this host's "
                  f"V100s (SM70):")
            print(f"    {problem}")
            print(f"    Look for a plain 'awq' (GEMM) or 'gptq' quant instead.")
            resp = input("  Download anyway? [y/N]: ").strip().lower()
            if resp not in ("y", "yes"):
                print("\n  Aborted.")
                input("\n  Press Enter to continue ...")
                return
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"\n  Destination already exists and is non-empty: {dest}")
            print(f"  Delete it first or pick a different folder.")
            input("\n  Press Enter to continue ...")
            return
        confirm = input(f"\n  Download to {dest}? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            print("\n  Aborted.")
            input("\n  Press Enter to continue ...")
            return
        print(f"\n  Downloading HF bundle ({size_gb:.1f} GB) ...")
        rc = _hf_download_bundle(repo_id, dest)
        if rc != 0:
            print(f"\n  hf download failed (exit {rc}).")
            input("\n  Press Enter to continue ...")
            return
        print(f"\n  Done. Model dir: {dest}")
        _post_download_profile(dest, target, "hf")
        input("\n  Press Enter to continue ...")
        return

    vram = _default_vram_gb()
    budget, budget_desc = weights_budget(vram)
    variants = group_variants(tree)
    rec = recommend(variants, budget)
    header = [
        (f"Budget: {budget_desc}.", curses.A_DIM),
        ("", 0),
        (
            f"  {'':<2}{'#':>3}  {'Quant':<14} {'Size':>9}  {'Files':>5}  "
            f"{'Fit':<5}  Pattern",
            curses.A_BOLD,
        ),
    ]
    items = []
    indices = []
    for i, v in enumerate(variants):
        fit = (
            "ok"
            if v.size_gb <= budget
            else ("tight" if v.size_gb <= vram else "no")
        )
        star = "*" if v is rec else " "
        label = (
            f"  {star} {i + 1:>2}  "
            f"{v.quant:<14} {v.size_gb:>7.1f} GB "
            f"{len(v.files):>5}  {fit:<5}  {v.include_pattern}"
        )
        items.append((label, 0))
        indices.append(i)
    selected = _tui_select(
        stdscr, "Download \u2014 Select Variant", items, header=header
    )
    # Leave curses BEFORE any input()/print/subprocess below \u2014 the download
    # progress and prompts must run on a normal terminal.
    curses.endwin()
    if selected < 0 or selected >= len(indices):
        print("\n  Aborted.")
        return
    chosen = variants[indices[selected]]
    if os.path.isdir(dest) and os.listdir(dest):
        print(
            f"\n  Destination already exists and is non-empty: {dest}\n"
            f"  Use --force with CLI to overwrite, or delete it first."
        )
        input("\n  Press Enter to continue ...")
        return
    confirm = (
        input(
            f"\n  Download {chosen.quant} ({chosen.size_gb:.1f} GB, "
            f"{len(chosen.files)} file(s)) \u2192 {dest}? [Y/n]: "
        )
        .strip()
        .lower()
    )
    if confirm in ("n", "no"):
        print("\n  Aborted.")
        return
    main_file = chosen.files[0]["path"]
    print(f"\n  Downloading {chosen.quant} ...")
    rc = _hf_download(repo_id, chosen, dest)
    if rc != 0:
        print(f"\n  hf download failed (exit {rc}).")
        input("\n  Press Enter to continue ...")
        return
    if chosen.is_sharded:
        print(
            f"\n  Done: {len(chosen.files)} shards, "
            f"first file: {os.path.join(dest, main_file)}"
        )
    else:
        print(f"\n  Done. Primary file: {os.path.join(dest, main_file)}")
    _post_download_profile(dest, target, "gguf")
    input("\n  Press Enter to continue ...")


def _tui_act_delete(stdscr, triples):
    """triples: list[(root, layout, LocalModel)] from scan_all_roots()."""
    import curses

    if not triples:
        curses.endwin()
        print("\n  No models installed.")
        input("\n  Press Enter to continue ...")
        return
    items = [(f"[{layout:<8}] {m.name}", 0) for _, layout, m in triples]
    items.append(("Back", curses.A_DIM))
    idx = _tui_select(stdscr, "Delete Model \u2014 Select Model", items)
    if idx < 0 or idx >= len(triples):
        return
    root, layout, info = triples[idx]
    curses.endwin()
    print(f"\n  About to delete:")
    print(f"    Where: {layout}  ({root})")
    print(f"    Name:  {info.name}")
    print(f"    Path:  {info.path}")
    print(f"    Size:  {info.size_bytes / (1024**3):.1f} GB")
    print(
        f"    Kind:  {info.kind}  Quant: {info.quant or '-'}  Variants: {info.variants}"
    )
    print()
    resp = input(f"  Type '{info.name}' to confirm: ").strip()
    if resp != info.name:
        print("\n  Name did not match. Aborted.")
        input("\n  Press Enter to continue ...")
        return
    shutil.rmtree(info.path)
    print(f"\n  Deleted {info.path}")
    if layout == "lmstudio":
        parent = os.path.dirname(info.path)
        try:
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
                print(f"  Removed empty publisher dir {parent}")
        except OSError:
            pass
    input("\n  Press Enter to continue ...")


def _tui_act_profiles(stdscr, model_name):
    import curses

    profs = load_profiles(DEFAULT_MODEL_DIR, model_name)
    actions = [
        ("List Profiles", lambda: _tui_pf_list(stdscr, model_name)),
        ("Show Profiles", lambda: _tui_pf_show(stdscr, model_name, profs)),
        ("Add Profile", lambda: _tui_pf_add(stdscr, model_name)),
        ("Copy Profile", lambda: _tui_pf_copy(stdscr, model_name)),
        ("Delete Profile", lambda: _tui_pf_delete(stdscr, model_name, profs)),
        ("Edit profiles.toml", lambda: _tui_pf_edit(stdscr, model_name)),
        ("Show profiles.toml Path", lambda: _tui_pf_path(stdscr, model_name)),
        ("Back", None),
    ]
    items = [(name, curses.A_DIM if func is None else 0) for name, func in actions]
    idx = _tui_select(stdscr, f"Profile Management \u2014 {model_name}", items)
    if idx >= 0 and actions[idx][1] is not None:
        actions[idx][1]()


def _tui_pf_list(stdscr, model_name):
    # Reuse the CLI printer via run_cmd, which handles the leave-curses /
    # print / pause / restore dance. Printing directly here (as this used to)
    # runs input() in curses raw mode: staircase output and an Enter key
    # (\r, never \n) that input() waits on forever.
    tui.run_cmd(stdscr, cmd_profile_list,
                argparse.Namespace(dir=DEFAULT_MODEL_DIR, model=model_name))


def _tui_pf_show(stdscr, model_name, profs):
    import curses

    if not profs:
        curses.endwin()
        print("\n  No profiles.")
        input("\n  Press Enter to continue ...")
        return
    names = sorted(profs.keys(), key=lambda n: (n != "default", n.lower()))
    items = [
        (
            f"{n:<14} engine={profs[n].engine:<6} "
            f"tp={profs[n].tp_size} gpu={profs[n].gpu}"
            if n != "default"
            else f"{n:<14} engine={profs[n].engine:<6} "
            f"tp={profs[n].tp_size} gpu={profs[n].gpu} **(default)**",
            0,
        )
        for n in names
    ]
    items.append(("Show All", 0))
    items.append(("Back", curses.A_DIM))
    idx = _tui_select(stdscr, f"Show Profile \u2014 {model_name}", items)
    if idx < 0 or idx >= len(profs) + 1:
        return
    if idx == len(profs):
        # Show all
        curses.endwin()
        for name in names:
            print(_format_profile(profs[name]))
            print()
        print()
        input("\n  Press Enter to continue ...")
        return
    p = profs[names[idx]]
    curses.endwin()
    print()
    print(_format_profile(p))
    print()
    input("\n  Press Enter to continue ...")


def _tui_pf_add(stdscr, model_name):
    """Walk-through profile creation with inline explanations."""
    import curses

    profs = load_profiles(DEFAULT_MODEL_DIR, model_name)

    # -- Step 1: name ---------------------------------------------------------
    name = _tui_text(stdscr, "  Profile name: ")
    if not name:
        return
    if name in profs:
        curses.endwin()
        print(f"\n  Profile '{name}' already exists.")
        input("\n  Press Enter to continue ...")
        return

    # -- Step 2: detect family for sensible defaults --------------------------
    try:
        info = model_lib.detect_family(DEFAULT_MODEL_DIR, model_name)
    except Exception:
        info = None

    fam_label = info.family if info else "unknown"
    detected = [
        f"Detected family: {fam_label}",
        f"Architecture:    {(info.arch if info else '?')}",
        f"Quantization:    {(info.quant_method or 'none') if info else '?'}",
    ]

    # -- Step 3: tool-use Y/N -------------------------------------------------
    tool_use = tui.confirm(
        stdscr, "Enable tool-call parser?",
        default=True,
        explain=detected + [
            "",
            "Tool-call parsing lets the model emit structured tool calls",
            "(function calls) that clients like OpenCode/Claude Code can",
            "execute. Picks the right parser per family automatically.",
        ],
    )

    # -- Step 4: reasoning Y/N ------------------------------------------------
    reasoning = tui.confirm(
        stdscr, "Enable reasoning parser?",
        default=True,
        explain=[
            "Surfaces the model's <think>...</think> as a separate field",
            "instead of inline text. Useful for chain-of-thought models",
            "(GLM-4.5/4.7, Qwen3 family). Adds tokens \u2014 turn off",
            "for fastest replies on simple chats.",
        ],
    )

    # -- Step 5: max_model_len ------------------------------------------------
    suggested_ctx = detect_model_max_len(DEFAULT_MODEL_DIR, model_name) \
                    or DEFAULT_MAX_MODEL_LEN
    ctx_str = _tui_text(
        stdscr,
        f"  Max context length [{suggested_ctx}]: ",
        default=str(suggested_ctx),
    )
    try:
        max_model_len = int(ctx_str)
    except ValueError:
        max_model_len = suggested_ctx

    # -- Step 6: gpu_mem_util -------------------------------------------------
    util_str = _tui_text(
        stdscr,
        "  gpu_memory_utilization [0.90]: ",
        default="0.90",
    )
    try:
        gpu_mem_util = float(util_str)
    except ValueError:
        gpu_mem_util = 0.90

    # -- Step 7: tp_size ------------------------------------------------------
    tp_str = _tui_text(stdscr, "  tensor_parallel_size [2]: ", default="2")
    try:
        tp_size = int(tp_str)
    except ValueError:
        tp_size = 2

    gpu_field = ",".join(str(i) for i in range(tp_size))

    # -- Build the profile ----------------------------------------------------
    p = make_default_profile(max_model_len=max_model_len)
    p.name = name
    p.tp_size = tp_size
    p.gpu = gpu_field
    p.gpu_mem_util = gpu_mem_util

    extra: list[str] = []
    if info is not None:
        try:
            tools, reasonings = _cached_parsers()
            extra = model_lib.suggest_extra_args(
                info, tp_size=tp_size,
                tool_use=tool_use, reasoning=reasoning,
                v100=_host_is_v100(),
                available_tool_parsers=tools or None,
                available_reasoning_parsers=reasonings or None,
            )
        except Exception:
            extra = []
    p.extra_args = extra

    profs[name] = p
    path = save_profiles(DEFAULT_MODEL_DIR, model_name, profs)
    curses.endwin()
    print(f"\n  Added [{name}] to {path}")
    print(f"    tp_size={tp_size}  ctx={max_model_len}  "
          f"gpu_mem_util={gpu_mem_util}")
    if extra:
        print(f"    extra_args: {' '.join(extra)}")
    input("\n  Press Enter to continue ...")


def _tui_pf_copy(stdscr, model_name):
    import curses

    profs = load_profiles(DEFAULT_MODEL_DIR, model_name)
    if not profs:
        curses.endwin()
        print("\n  No profiles to copy from.")
        input("\n  Press Enter to continue ...")
        return
    names = sorted(profs.keys(), key=lambda n: (n != "default", n.lower()))
    items = [(n, 0) for n in names]
    items.append(("Back", curses.A_DIM))
    idx = _tui_select(stdscr, f"Copy from profile \u2014 {model_name}", items)
    if idx < 0 or idx >= len(names):
        return
    src = names[idx]
    dst = _tui_text(stdscr, f"  New profile name (copy of '{src}'): ")
    if not dst:
        return
    if dst in profs:
        curses.endwin()
        print(f"\n  Profile '{dst}' already exists.")
        input("\n  Press Enter to continue ...")
        return
    profs[dst] = copy_profile(profs[src], dst)
    path = save_profiles(DEFAULT_MODEL_DIR, model_name, profs)
    curses.endwin()
    print(f"  Copied [{src}] \u2192 [{dst}] in {path}")
    input("\n  Press Enter to continue ...")


def _tui_pf_delete(stdscr, model_name, profs):
    import curses

    if not profs:
        curses.endwin()
        print("\n  No profiles to delete.")
        input("\n  Press Enter to continue ...")
        return
    names = sorted(profs.keys(), key=lambda n: (n != "default", n.lower()))
    items = []
    for n in names:
        tag = " (default - type name to delete)" if n == "default" else ""
        items.append((f"{n:<14}{tag}", 0))
    items.append(("Back", curses.A_DIM))
    idx = _tui_select(stdscr, f"Delete Profile \u2014 {model_name}", items)
    if idx < 0 or idx >= len(names):
        return
    name = names[idx]
    if name not in profs:
        return
    if name == "default":
        curses.endwin()
        resp = input(f"  Type '{name}' to delete default profile: ").strip()
        if resp != name:
            print("\n  Aborted.")
            input("\n  Press Enter to continue ...")
            return
    else:
        curses.endwin()
        resp = input(f"  Delete profile '{name}'? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("\n  Aborted.")
            input("\n  Press Enter to continue ...")
            return
    del profs[name]
    if profs:
        path = save_profiles(DEFAULT_MODEL_DIR, model_name, profs)
        curses.endwin()
        print(f"  Removed [{name}] from {path}")
    else:
        path = profiles_path(DEFAULT_MODEL_DIR, model_name)
        if os.path.isfile(path):
            os.remove(path)
        curses.endwin()
        print(f"  Removed [{name}] (profiles.toml deleted \u2014 no profiles left)")
    input("\n  Press Enter to continue ...")


def _tui_pf_edit(stdscr, model_name):
    import curses

    path = profiles_path(DEFAULT_MODEL_DIR, model_name)
    if not os.path.isfile(path):
        ensure_profiles_exist(DEFAULT_MODEL_DIR, model_name, interactive=False)
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    curses.endwin()
    print(f"  Opening {path} in {editor} ...")
    subprocess.run([editor, path])
    input("\n  Press Enter to continue ...")


def _tui_pf_path(stdscr, model_name):
    import curses

    path = profiles_path(DEFAULT_MODEL_DIR, model_name)
    curses.endwin()
    print(f"  {path}")
    input("\n  Press Enter to continue ...")


# == TUI main loop ==


def _tui_main(stdscr):
    import curses

    curses.curs_set(0)
    stdscr.keypad(True)
    tui.init_colors()
    h, w = stdscr.getmaxyx()
    if h < 10 or w < 50:
        curses.endwin()
        print("Terminal too small (need at least 50x10).")
        print("Use CLI commands instead: python manage_models.py --help")
        return

    MENU_ACTIONS = [
        ("Download Model", 0),
        ("Delete Model", 0),
        ("Profile Manager", 0),
        ("Seed Curated Profiles", 0),
        ("Help", 0),
        ("Quit", None),
    ]

    def _build_header():
        triples = scan_all_roots()
        total = sum(m.size_bytes for _, _, m in triples) if triples else 0
        header = [
            (
                f"Models: {len(triples)}    Disk: {(total / (1024**3)):.1f} GB",
                curses.color_pair(_C_YELLOW),
            )
        ]
        if not triples:
            header.append(("", 0))
            return header
        hdr = [
            (
                f"  {'Where':<9}  {'Name':<40}  {'Kind':<5}  {'Quant':<10}  "
                f"{'Size':>9}",
                curses.A_BOLD,
            ),
            ("  " + "\u2500" * 80, curses.A_DIM),
        ]
        for root, layout, m in triples[:6]:
            name = m.name if len(m.name) <= 40 else m.name[:39] + "\u2026"
            line = (
                f"  {layout:<9}  {name:<40}  {m.kind:<5}  "
                f"{(m.quant or '-'):<10}  {human_gb(m.size_bytes):>9}"
            )
            hdr.append((line, curses.A_DIM))
        if len(triples) > 6:
            hdr.append((f"  ... and {len(triples) - 6} more", curses.A_DIM))
        return header + hdr

    while True:
        header = _build_header()
        menu_items = [
            (label, curses.A_DIM if func is None else 0) for label, func in MENU_ACTIONS
        ]

        def _refresh():
            return _build_header(), menu_items

        idx = _tui_select(
            stdscr,
            "vLLM Models Manager \u2014 Local LLM Management",
            menu_items,
            header=header,
            refresh_cb=_refresh,
            refresh_ms=8000,
        )

        if idx < 0 or idx == len(MENU_ACTIONS) - 1:
            break

        if idx == 0:
            # Download Model
            _tui_act_download(stdscr)
        elif idx == 1:
            # Delete Model — multi-root
            triples = scan_all_roots()
            if not triples:
                curses.endwin()
                print("\n  No models installed.")
                input("\n  Press Enter to continue ...")
            else:
                _tui_act_delete(stdscr, sorted(
                    triples, key=lambda t: (t[1], t[2].name)))
        elif idx == 2:
            # Profile Manager — vllm root only (lmstudio has no profiles)
            models = scan_all(DEFAULT_MODEL_DIR)
            if not models:
                curses.endwin()
                print("\n  No vLLM models installed.")
                input("\n  Press Enter to continue ...")
                continue
            model_items = [(m.name, 0) for m in sorted(models, key=lambda x: x.name)]
            model_items.append(("Back", curses.A_DIM))
            m_idx = _tui_select(
                stdscr, "Profile Manager \u2014 Select Model", model_items
            )
            if 0 <= m_idx < len(model_items) - 1:
                sel = sorted(models, key=lambda x: x.name)[m_idx]
                curses.endwin()
                print(f"\n  Model: {sel.name}")
                print(
                    f"  Kind: {sel.kind}  Quant: {sel.quant or '-'}  "
                    f"Size: {human_gb(sel.size_bytes)}"
                )
                profs = load_profiles(DEFAULT_MODEL_DIR, sel.name)
                print(f"  Profiles: {', '.join(profs.keys()) if profs else '(none)'}")
                print(f"  Directory: {sel.path}")
                print("\n  [D] Delete this model")
                print(
                    f"  [P] Manage profiles ({', '.join(profs.keys()) if profs else '(none)'})"
                )
                print("  [Q] Back to models")
                ch = input("\n  Choice: ").strip().lower()
                if ch in ("d", "del"):
                    _tui_act_delete(
                        stdscr, [(DEFAULT_MODEL_DIR, "vllm", sel)])
                elif ch in ("p", "profile"):
                    _tui_act_profiles(stdscr, sel.name)
                stdscr.touchwin()
        elif idx == 3:
            # Seed Curated Profiles \u2014 interactive confirms run inside run_cmd
            # (already out of curses there).
            tui.run_cmd(stdscr, cmd_seed_profiles,
                        argparse.Namespace(models=[], yes=False, list=False))
        elif idx == 4:
            # Help
            curses.endwin()
            print("""
  vLLM Models Manager \u2014 Help
  ===
  Navigation:
    [Up]/[Down]    Navigate main menu
    [Enter]        Execute selected action
  Actions:
    Download Model         Fetch a GGUF model from HuggingFace (GGUF picker)
    Delete Model           Remove an installed model folder
    Profile Manager        Manage launch profiles for a model
    Seed Curated Profiles  Write the documented profile sets for the
                           standard models (Qwen3.6 35B/27B, GLM-4.7-Flash)
    Help                   Show this screen
    Quit                   Exit the TUI
  Download:
    Interactive GGUF variant picker with VRAM budget, size, shard count,
    and automatic profile creation after download.
  Profile Manager:
    List                  Show all profiles in a table
    Show                  View a profile's settings
    Add                   Create a new profile (seed from existing or default)
    Copy                  Duplicate an existing profile
    Delete                Remove a profile
    Edit profiles.toml     Open in $EDITOR (or nano)
    Show profiles.toml Path   Print the path to profiles.toml
""")
            input("\n  Press Enter to continue ...")


def _tui_launch():
    """Entry point for the interactive TUI."""
    tui.launch(_tui_main)


# =============================================================================
# Remote (HF) side — GGUF picker
# =============================================================================


@dataclass
class RemoteVariant:
    quant: str
    files: list[dict] = field(default_factory=list)
    total_size: int = 0

    @property
    def size_gb(self) -> float:
        return self.total_size / (1024**3)

    @property
    def is_sharded(self) -> bool:
        return len(self.files) > 1

    @property
    def include_pattern(self) -> str:
        path = self.files[0]["path"]
        if self.is_sharded:
            return SHARD_RE.sub("-*-of-*.gguf", path)
        return path


def parse_repo_id(raw: str) -> str:
    s = raw.strip().rstrip("/")
    if s.startswith("http"):
        m = re.match(r"https?://huggingface\.co/([^/]+/[^/?#]+)", s)
        if not m:
            sys.exit(f"Could not parse repo id from URL: {raw}")
        return m.group(1)
    if s.count("/") != 1:
        sys.exit(f"Expected 'org/repo' or an HF URL, got: {raw}")
    return s


def fetch_tree(repo_id: str) -> list[dict]:
    # The tree endpoint paginates at 1000 entries (RFC5988 Link header);
    # follow rel="next" so huge multi-quant repos aren't silently truncated.
    url: str | None = HF_TREE_API.format(repo=repo_id)
    out: list[dict] = []
    while url:
        req = urllib.request.Request(url)
        if HF_TOKEN:
            req.add_header("Authorization", f"Bearer {HF_TOKEN}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                out.extend(json.loads(resp.read()))
                link = resp.headers.get("Link") or ""
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                sys.exit(
                    f"Repo requires auth. Accept the license at "
                    f"https://huggingface.co/{repo_id} and set HF_TOKEN."
                )
            if e.code == 404:
                sys.exit(f"Repo not found: {repo_id}")
            sys.exit(f"HF API error {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            sys.exit(f"Network error: {e.reason}")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return out


def group_variants(tree: list[dict]) -> list[RemoteVariant]:
    by_quant: dict[str, RemoteVariant] = {}
    for f in tree:
        if f.get("type") != "file":
            continue
        path = f.get("path", "")
        if not path.lower().endswith(".gguf"):
            continue
        quant = extract_quant_tag(path) or "UNKNOWN"
        v = by_quant.setdefault(quant, RemoteVariant(quant=quant))
        size = int(f.get("size") or 0)
        v.files.append({"path": path, "size": size})
        v.total_size += size

    def shard_idx(p: str) -> int:
        m = SHARD_RE.search(p)
        return int(m.group(1)) if m else 0

    for v in by_quant.values():
        v.files.sort(key=lambda f: shard_idx(f["path"]))

    return sorted(by_quant.values(), key=lambda v: v.total_size)


def weights_budget(vram_gb: float, *, offload: bool = False,
                   ram_gb: float | None = None) -> tuple[float, str]:
    """(budget_gb, human description) of how many GB of weights can be held.

    Default: VRAM minus the KV-cache reservation. With offload=True (CPU+GPU
    MoE offload, llama.cpp-style), 85% of system RAM joins the budget — the
    other 15% stays for OS/page-cache so decode never faults to disk.
    """
    gpu_budget = vram_gb * (1.0 - KV_HEADROOM)
    if not offload:
        return gpu_budget, (
            f"{vram_gb:.0f} GB VRAM, ~{gpu_budget:.0f} GB usable for weights "
            f"(reserving {int(KV_HEADROOM * 100)}% for KV cache)")
    ram = ram_gb if ram_gb is not None else _host_ram_gb()
    total = gpu_budget + ram * 0.85
    return total, (
        f"offload mode: ~{gpu_budget:.0f} GB GPU + ~{ram * 0.85:.0f} GB of "
        f"{ram:.0f} GB RAM = ~{total:.0f} GB for weights")


def recommend(variants: list[RemoteVariant], budget_gb: float):
    """Best variant that fits budget_gb of weights.

    If NOTHING fits, returns the smallest variant (the least-impossible
    choice) — callers must check `rec.size_gb > budget_gb` and say so
    instead of presenting it as a fit.
    """
    if not variants:
        return None
    fits = [v for v in variants if v.size_gb <= budget_gb]
    if not fits:
        return min(variants, key=lambda v: v.total_size)
    for pref in QUANT_PREFERENCE:
        for v in fits:
            if pref.upper() in v.quant.upper():
                return v
    return max(fits, key=lambda v: v.total_size)


# -- HF-format (safetensors / AWQ / GPTQ) -----------------------------------

HF_INCLUDE_PATTERNS = [
    "*.safetensors", "*.safetensors.index.json",
    "*.bin", "*.bin.index.json",
    "*.json", "tokenizer*", "*.model", "*.txt",
    "*.py",            # trust_remote_code model/modeling files
    "chat_template*",  # some repos ship separate chat template files
    "*.md", "LICENSE*",
]


def detect_repo_kind(tree: list[dict]) -> str:
    """Return 'gguf', 'hf', 'hf+gguf', or 'empty'."""
    has_gguf = False
    has_hf = False
    for f in tree:
        if f.get("type") != "file":
            continue
        path = f.get("path", "").lower()
        if path.endswith(".gguf"):
            has_gguf = True
        elif os.path.basename(path) == "config.json":
            has_hf = True
        elif path.endswith((".safetensors", ".safetensors.index.json")):
            has_hf = True
    if has_hf and has_gguf:
        return "hf+gguf"
    if has_hf:
        return "hf"
    if has_gguf:
        return "gguf"
    return "empty"


def summarize_hf_bundle(tree: list[dict],
                        repo_id: str = "") -> tuple[int, int, str | None]:
    """Return (weight_bytes, weight_file_count, quant_method_hint).

    weight_bytes sums safetensors + bin shards (not configs/tokenizers, since
    those are negligible). quant_method_hint is sniffed from path names and
    the repo id.
    """
    weight_bytes = 0
    weight_files = 0
    for f in tree:
        if f.get("type") != "file":
            continue
        plow = f.get("path", "").lower()
        if plow.endswith((".safetensors", ".bin")):
            weight_bytes += int(f.get("size") or 0)
            weight_files += 1
    # Quant hint from paths + repo id. Cosmetic; real detection happens when
    # the file is on disk via _detect_quant_from_config in model_lib.
    hay = repo_id.lower() + " " + " ".join(
        (f.get("path") or "").lower() for f in tree
    )
    quant: str | None = None
    for tag in ("AWQ", "GPTQ", "FP8", "NVFP4", "INT4", "INT8"):
        if tag.lower() in hay:
            quant = tag
            break
    return weight_bytes, weight_files, quant


# Quant methods whose vLLM kernels require newer-than-Volta GPUs. Keyed by
# quantization_config.quant_method as it appears in repo config.json.
# The trap this catches: repos NAMED "AWQ" that are actually one of these
# (e.g. cyankiwi/GLM-4.7-Flash-AWQ-4bit is compressed-tensors → Marlin).
_SM80_QUANT_METHODS = {
    "compressed-tensors": "routes to Marlin kernels (needs SM80+/Ampere)",
    "compressed_tensors": "routes to Marlin kernels (needs SM80+/Ampere)",
    "fp8":                "FP8 needs SM89+ (Ada/Hopper)",
    "modelopt":           "ModelOpt FP8/NVFP4 needs SM89+",
    "modelopt_fp4":       "NVFP4 needs SM100 (Blackwell)",
    "nvfp4":              "NVFP4 needs SM100 (Blackwell)",
    "mxfp4":              "MXFP4 needs SM90+",
    "w4afp8":             "W4AFP8 needs SM90 (Hopper)",
    "awq_marlin":         "Marlin kernels need SM80+ (use plain 'awq')",
    "gptq_marlin":        "Marlin kernels need SM80+ (use plain 'gptq')",
}


def _fetch_repo_config(repo_id: str, tree: list[dict]) -> dict | None:
    """Fetch and parse the repo's (shallowest) config.json; None on failure."""
    paths = [f.get("path", "") for f in tree if f.get("type") == "file"
             and os.path.basename(f.get("path", "")) == "config.json"]
    if not paths:
        return None
    paths.sort(key=lambda p: p.count("/"))
    url = f"https://huggingface.co/{repo_id}/resolve/main/{paths[0]}"
    req = urllib.request.Request(url)
    if HF_TOKEN:
        req.add_header("Authorization", f"Bearer {HF_TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _v100_compat_problem(cfg: dict | None) -> str | None:
    """Why this checkpoint can't run on SM70, or None if it looks fine.

    Only judges the quantization method; unquantized and awq/gptq/bnb
    checkpoints pass (bf16 configs are served as fp16 on Volta).
    """
    if not cfg:
        return None
    qc = cfg.get("quantization_config") or {}
    if not qc and isinstance(cfg.get("text_config"), dict):
        qc = cfg["text_config"].get("quantization_config") or {}
    method = str(qc.get("quant_method")
                 or qc.get("quantization_method") or "").lower()
    if method in _SM80_QUANT_METHODS:
        return f"quant_method '{method}': {_SM80_QUANT_METHODS[method]}"
    return None


def _hf_download_bundle(repo_id: str, dest: str,
                        extra_excludes: list[str] | None = None) -> int:
    os.makedirs(dest, exist_ok=True)
    env = os.environ.copy()
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
    cmd = ["hf", "download", repo_id, "--local-dir", dest]
    for pat in HF_INCLUDE_PATTERNS:
        cmd += ["--include", pat]
    for pat in (extra_excludes or ["*.gguf"]):
        cmd += ["--exclude", pat]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, env=env).returncode


# =============================================================================
# list / delete
# =============================================================================


def cmd_list(args):
    triples = scan_all_roots()
    print()
    for root, layout in ROOTS:
        present = "(present)" if os.path.isdir(root) else "(missing)"
        print(f"  {layout:<9}  {root}  {present}")
    if not triples:
        print("\n  (no models installed)\n")
        return
    total = sum(m.size_bytes for _, _, m in triples)
    print(f"\n  {len(triples)} model(s), {total / (1024**3):.1f} GB total\n")
    print(
        f"  {'Where':<9}  {'Name':<48}  {'Kind':<5}  {'Quant':<10}  "
        f"{'Size':>9}  Profiles"
    )
    print("  " + "-" * 100)
    for root, layout, m in triples:
        if layout == "vllm" and m.kind == "hf":
            profs = load_profiles(root, m.name)
            prof_str = ", ".join(sorted(profs.keys())) or "(none)"
        else:
            prof_str = "-"
        name = m.name if len(m.name) <= 48 else m.name[:47] + "…"
        print(
            f"  {layout:<9}  {name:<48}  {m.kind:<5}  "
            f"{(m.quant or '-'):<10}  {human_gb(m.size_bytes)}  {prof_str}"
        )
    print()


def cmd_delete(args):
    found = resolve_model(args.name)
    if not found:
        sys.exit(f"No model named '{args.name}' in any configured root.")
    root, layout, info = found
    path = info.path

    print(f"\n  About to delete:")
    print(f"    Where: {layout}  ({root})")
    print(f"    Name:  {info.name}")
    print(f"    Path:  {path}")
    print(f"    Size:  {info.size_bytes / (1024**3):.1f} GB")
    print(
        f"    Kind:  {info.kind}  Quant: {info.quant or '-'}  "
        f"Variants: {info.variants}"
    )
    print()

    if not args.yes:
        try:
            resp = input("  Type the model name to confirm: ").strip()
        except EOFError:
            resp = ""
        if resp != info.name:
            print("  Name did not match. Aborted.")
            return

    shutil.rmtree(path)
    print(f"  Deleted {path}")

    # For lmstudio (publisher/repo), prune the publisher dir if it's now empty.
    if layout == "lmstudio":
        parent = os.path.dirname(path)
        try:
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
                print(f"  Removed empty publisher dir {parent}")
        except OSError:
            pass


# =============================================================================
# download (GGUF picker + post-download profile creation)
# =============================================================================


def _print_remote_table(variants, rec, budget: float, budget_desc: str):
    # "tight" = over the weights budget but within the raw capacity (i.e.
    # would only fit with no KV/OS headroom — possible but unwise).
    raw_capacity = budget / (1.0 - KV_HEADROOM)
    print()
    print(f"  Budget: {budget_desc}.")
    print()
    print(
        f"  {'':<2}{'#':>3}  {'Quant':<14} {'Size':>9}  "
        f"{'Files':>5}  {'Fit':<5}  Pattern"
    )
    print("  " + "-" * 74)
    for i, v in enumerate(variants, 1):
        fit = ("ok" if v.size_gb <= budget
               else "tight" if v.size_gb <= raw_capacity else "no")
        star = "*" if v is rec else " "
        print(
            f"  {star} {i:>2}  {v.quant:<14} {v.size_gb:>7.1f} GB  "
            f"{len(v.files):>5}  {fit:<5}  {v.include_pattern}"
        )
    print()
    if rec and rec.size_gb > budget:
        print(
            f"  NOTHING here fits the budget. Smallest is {rec.quant} "
            f"({rec.size_gb:.1f} GB) — needs ~{rec.size_gb - budget:.0f} GB "
            f"more. Consider --offload (CPU+GPU) or a smaller model."
        )
    elif rec:
        print(
            f"  Recommended: {rec.quant} ({rec.size_gb:.1f} GB) — "
            f"best quality from preference list that fits the budget."
        )
    print()


def _pick_variant(variants, rec):
    prompt = f"  Select [1-{len(variants)}], Enter for recommendation, 'q' to quit: "
    while True:
        try:
            raw = input(prompt).strip().lower()
        except EOFError:
            return None
        if raw in ("q", "quit"):
            return None
        if raw == "":
            return rec
        try:
            i = int(raw)
        except ValueError:
            print("  Enter a number, Enter, or 'q'.")
            continue
        if 1 <= i <= len(variants):
            return variants[i - 1]
        print("  Out of range.")


def _hf_download(repo_id: str, variant, dest: str) -> int:
    os.makedirs(dest, exist_ok=True)
    env = os.environ.copy()
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN

    includes = [
        variant.include_pattern,
        "*.json",
        "*.md",
        "tokenizer*",
        "*.gguf.md5",
    ]
    cmd = ["hf", "download", repo_id, "--local-dir", dest]
    for pat in includes:
        cmd += ["--include", pat]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, env=env).returncode


def _prompt_target(default: str = "vllm") -> str:
    """Interactive vllm-vs-lmstudio chooser used by CLI when --target is unset."""
    print()
    print("  Where should this model live?")
    print(f"    [1] vllm      ({DEFAULT_MODEL_DIR})    — flat layout, profiles auto-created for HF")
    print(f"    [2] lmstudio  ({LMSTUDIO_MODEL_DIR})   — publisher/repo layout, no profiles")
    while True:
        try:
            raw = input(f"  Choose [1/2, Enter for {default}]: ").strip().lower()
        except EOFError:
            return default
        if raw == "":
            return default
        if raw in ("1", "vllm", "v"):
            return "vllm"
        if raw in ("2", "lmstudio", "lms", "l"):
            return "lmstudio"
        print("  Enter 1, 2, or Enter.")


def _resolve_download_dest(target: str, repo_id: str,
                           name_override: str | None) -> tuple[str, str, str]:
    """Return (root, short_name, dest_path) for a download.

    short_name is used for downstream profile/scan calls — it matches the
    on-disk folder relative to root (e.g. 'Qwen3.6-35B-A3B-AWQ' or
    'lmstudio-community/Qwen3.6-35B-A3B-GGUF').
    """
    org, _, repo = repo_id.partition("/")
    if target == "vllm":
        short = name_override or repo
        return DEFAULT_MODEL_DIR, short, os.path.join(DEFAULT_MODEL_DIR, short)
    if target == "lmstudio":
        # Preserve HF org/ structure. --name overrides only the repo half.
        repo_short = name_override or repo
        short = f"{org}/{repo_short}"
        return LMSTUDIO_MODEL_DIR, short, os.path.join(
            LMSTUDIO_MODEL_DIR, org, repo_short)
    raise ValueError(f"unknown target: {target}")


def cmd_download(args):
    # --dry-run never downloads, so it shouldn't require the hf CLI.
    if not args.dry_run and subprocess.run(
            ["which", "hf"], capture_output=True).returncode != 0:
        sys.exit(
            "'hf' CLI not found. Install with: pip install -U 'huggingface_hub[cli]'"
        )

    repo_id = parse_repo_id(args.repo)

    # --yes means non-interactive: don't stop to ask where it goes.
    target = args.target or ("vllm" if args.yes else _prompt_target())
    root, short, dest = _resolve_download_dest(target, repo_id, args.name)
    vram = args.vram if args.vram else _default_vram_gb()
    budget, budget_desc = weights_budget(
        vram, offload=args.offload, ram_gb=args.ram)

    if os.path.isdir(dest) and os.listdir(dest) and not args.force:
        sys.exit(
            f"Destination already exists and is non-empty: {dest}\n"
            f"  Use --name to pick a different folder, --force to reuse, "
            f"or `delete {short}` first."
        )

    print(f"Fetching file tree for {repo_id} ...")
    tree = fetch_tree(repo_id)
    kind = detect_repo_kind(tree)

    if kind == "empty":
        sys.exit("Repo has neither .gguf nor HF-format weight files.")

    if kind == "hf":
        weight_bytes, n_files, quant = summarize_hf_bundle(tree, repo_id)
        size_gb = weight_bytes / (1024 ** 3)
        fit = "ok" if size_gb <= budget else ("tight" if size_gb <= vram else "no")
        print()
        print(f"  HF-format repo (safetensors). Quant hint: {quant or '-'}")
        print(f"  Weight files: {n_files} file(s), {size_gb:.1f} GB total — fit: {fit}")
        print()
        if _host_is_v100():
            problem = _v100_compat_problem(_fetch_repo_config(repo_id, tree))
            if problem:
                print(f"  WARNING: this checkpoint will NOT run on this "
                      f"host's V100s (SM70):")
                print(f"    {problem}")
                print(f"    Look for a plain 'awq' (GEMM) or 'gptq' quant "
                      f"of the same model instead.")
                print()
                if args.yes:
                    sys.exit("  Refusing under --yes. Re-run interactively "
                             "to override.")
                if not args.dry_run:
                    resp = input("  Download anyway? [y/N]: ").strip().lower()
                    if resp not in ("y", "yes"):
                        print("  Aborted.")
                        return
        if args.dry_run:
            return
        if not args.yes:
            resp = input(f"  Download to {dest}? [Y/n]: ").strip().lower()
            if resp in ("n", "no"):
                print("  Aborted.")
                return
        print(f"\nDownloading HF bundle ({size_gb:.1f} GB) → {dest}")
        rc = _hf_download_bundle(repo_id, dest)
        if rc != 0:
            sys.exit(f"\nhf download failed (exit {rc}).")
        print(f"\nDone. Model dir: {dest}")

        if target == "vllm":
            print()
            print("Creating launch profile ...")
            _create_planned_profile(root, short)
        else:
            print("  (lmstudio target — no profile created.)")
        return

    else:  # "gguf" or "hf+gguf" — pick a GGUF variant
        variants = group_variants(tree)
        if kind == "hf+gguf":
            print("  (Repo has both HF and GGUF files. Picking a GGUF quant "
                  "below; use --hf-bundle to grab the safetensors instead.)")
            if getattr(args, "hf_bundle", False):
                rc = _hf_download_bundle(repo_id, dest)
                if rc != 0:
                    sys.exit(f"\nhf download failed (exit {rc}).")
                print(f"\nDone. Model dir: {dest}")
                return
        rec = recommend(variants, budget)
        _print_remote_table(variants, rec, budget, budget_desc)

        if args.dry_run:
            return

        if args.yes and rec and rec.size_gb > budget:
            sys.exit(
                f"  --yes given but nothing fits the budget "
                f"(smallest: {rec.quant}, {rec.size_gb:.1f} GB). "
                f"Try --offload or pick explicitly without --yes.")
        chosen = rec if args.yes else _pick_variant(variants, rec)
        if not chosen:
            print("  Aborted.")
            return
        if args.yes:
            print(f"  --yes: picking {chosen.quant}")

        print(
            f"\nDownloading {chosen.quant} "
            f"({chosen.size_gb:.1f} GB, {len(chosen.files)} file(s)) → {dest}"
        )
        rc = _hf_download(repo_id, chosen, dest)
        if rc != 0:
            sys.exit(f"\nhf download failed (exit {rc}).")

        main_file = chosen.files[0]["path"]
        print(f"\nDone. Primary file: {os.path.join(dest, main_file)}")
        if chosen.is_sharded:
            print(f"  ({len(chosen.files)} shards in the set.)")

    # GGUF (and lmstudio in general) get no profile.
    print("\n  (GGUF model — no profile created.)")


# =============================================================================
# seed-profiles — curated, benchmarked profile sets for this box's standard
# models. Single source of truth for what the docs describe:
#   docs/qwen3.6-profiles.md   (Qwen3.6 35B / 27B on 2x V100)
#   docs/glm-4.7-flash.md      (GLM-4.7-Flash on 2x V100)
# Parser flags here are intentionally NOT gated by the registry probe — these
# sets are the documented, validated configs (measured on the box for Qwen;
# GLM pending on-box verification, see its guide).
# =============================================================================


def _qwen36_profiles(port: int) -> dict[str, Profile]:
    """The 2x2 {default, MTP} x {128k, 256k} set from docs/qwen3.6-profiles.md."""
    base = [
        "--attention-backend", "FLASH_ATTN_V100",
        "--quantization", "awq",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
        "--skip-mm-profiling",
    ]
    mtp = ["--speculative-config",
           '{"method": "mtp", "num_speculative_tokens": 2}']

    def prof(name: str, desc: str, ctx: int, seqs: int,
             tail: list[str]) -> Profile:
        return Profile(
            name=name, description=desc, engine="vllm", port=port,
            tp_size=2, gpu="0,1", dtype="half", gpu_mem_util=0.88,
            max_model_len=ctx,
            extra_args=base + ["--max-num-seqs", str(seqs),
                               "--max-num-batched-tokens", "8192"] + tail,
        )

    return {
        "default": prof(
            "default", "Baseline @128k. TP=2 across both GPUs.",
            131072, 16, []),
        "default-256k": prof(
            "default-256k", "Baseline @ full 256k context.",
            262144, 8, []),
        "default-mtp": prof(
            "default-mtp",
            "Balanced @128k + MTP speculative decoding (2). Recommended.",
            131072, 16, mtp),
        "default-mtp-256k": prof(
            "default-mtp-256k",
            "Full 256k context + MTP (2). Recommended for long context.",
            262144, 8, mtp),
    }


def _glm47_flash_profiles(port: int) -> dict[str, Profile]:
    """The set from docs/glm-4.7-flash.md. MLA model: no attention-backend
    pin (FLASH_ATTN_V100 is MHA-only), and MTP starts at 1 draft token
    (the model ships one MTP layer)."""
    base = [
        "--quantization", "awq",
        "--reasoning-parser", "glm45",
        "--enable-auto-tool-choice", "--tool-call-parser", "glm47",
    ]
    mtp = ["--speculative-config",
           '{"method": "mtp", "num_speculative_tokens": 1}']

    def prof(name: str, desc: str, ctx: int, seqs: int, tail: list[str],
             tp: int = 2, gpu: str = "0,1") -> Profile:
        return Profile(
            name=name, description=desc, engine="vllm", port=port,
            tp_size=tp, gpu=gpu, dtype="half", gpu_mem_util=0.88,
            max_model_len=ctx,
            extra_args=base + ["--max-num-seqs", str(seqs),
                               "--max-num-batched-tokens", "8192"] + tail,
        )

    return {
        "default": prof(
            "default", "Baseline @128k. TP=2 across both GPUs.",
            131072, 16, []),
        "default-mtp": prof(
            "default-mtp",
            "Balanced @128k + MTP (1 draft token). Recommended once verified.",
            131072, 16, mtp),
        "default-200k": prof(
            "default-200k", "Full native 202k context + MTP (1).",
            202752, 8, mtp),
        "solo-gpu": prof(
            "solo-gpu", "Single-GPU variant; leaves GPU 0 free for co-running.",
            131072, 16, [], tp=1, gpu="1"),
    }


# Canonical folder name -> (profile builder, note shown in listings).
CURATED_MODELS: dict = {
    "Qwen3.6-35B-A3B-AWQ": (
        lambda: _qwen36_profiles(7001),
        "MoE 35B; measured on-box, MTP +17%"),
    "Qwen3.6-27B-AWQ-MTP": (
        lambda: _qwen36_profiles(7002),
        "dense 27B; measured on-box"),
    "GLM-4.7-Flash-AWQ": (
        lambda: _glm47_flash_profiles(7003),
        "MLA MoE 30B; UNVERIFIED on the fork — see docs/glm-4.7-flash.md"),
}


def _find_model_folder(canonical: str) -> str | None:
    """On-disk folder for a curated model, matched case-insensitively."""
    if not os.path.isdir(DEFAULT_MODEL_DIR):
        return None
    for entry in os.listdir(DEFAULT_MODEL_DIR):
        if (entry.lower() == canonical.lower()
                and os.path.isdir(os.path.join(DEFAULT_MODEL_DIR, entry))):
            return entry
    return None


def cmd_seed_profiles(args):
    """Write the curated profile sets for this box's standard models."""
    if args.list:
        print(f"\n  Curated models (model dir: {DEFAULT_MODEL_DIR}):\n")
        for canon, (_builder, note) in CURATED_MODELS.items():
            folder = _find_model_folder(canon)
            state = f"present as '{folder}'" if folder else "NOT on disk"
            print(f"    {canon:<24} {state:<32} {note}")
        print()
        return

    if args.models:
        targets = []
        for name in args.models:
            canon = next((c for c in CURATED_MODELS
                          if c.lower() == name.lower()), None)
            if canon is None:
                sys.exit(f"'{name}' is not a curated model. "
                         f"Have: {', '.join(CURATED_MODELS)}")
            folder = _find_model_folder(canon)
            if folder is None:
                sys.exit(f"'{canon}' is not on disk under {DEFAULT_MODEL_DIR}. "
                         f"Download it first (see the docs/ guide).")
            targets.append((canon, folder))
    else:
        targets = [(c, f) for c in CURATED_MODELS
                   if (f := _find_model_folder(c))]
        missing = [c for c in CURATED_MODELS if not _find_model_folder(c)]
        if missing:
            print(f"\n  Skipping (not on disk): {', '.join(missing)}")
        if not targets:
            sys.exit("  No curated models found on disk. "
                     "Run with --list to see the catalog.")

    for canon, folder in targets:
        builder, note = CURATED_MODELS[canon]
        curated = builder()
        existing = load_profiles(DEFAULT_MODEL_DIR, folder)

        added, updated, unchanged = [], [], []
        for pname, prof in curated.items():
            if pname not in existing:
                added.append(pname)
            elif asdict(existing[pname]) != asdict(prof):
                updated.append(pname)
            else:
                unchanged.append(pname)
        custom = [p for p in existing if p not in curated]

        print(f"\n  {folder}  ({note})")
        for label, names in (("add", added), ("update", updated),
                             ("keep (already match)", unchanged),
                             ("keep (custom, untouched)", custom)):
            if names:
                print(f"    {label}: {', '.join(sorted(names))}")

        if not added and not updated:
            print("    Nothing to do.")
            continue

        if not args.yes:
            try:
                resp = input("    Write these profiles? [Y/n]: ").strip().lower()
            except EOFError:
                resp = ""
            if resp in ("n", "no"):
                print("    Skipped.")
                continue

        path = profiles_path(DEFAULT_MODEL_DIR, folder)
        if os.path.isfile(path):
            shutil.copy2(path, path + ".bak")
            print(f"    Backed up to {path}.bak")
        merged = dict(existing)
        merged.update(curated)
        save_profiles(DEFAULT_MODEL_DIR, folder, merged)
        print(f"    Wrote {path}")


# =============================================================================
# profile subcommands
# =============================================================================


def _require_model_dir(args):
    path = os.path.join(args.dir, args.model)
    if not os.path.isdir(path):
        sys.exit(f"No model directory at {path}")


def _format_profile(p: Profile) -> str:
    lines = [f"  [{p.name}]"]
    d = asdict(p)
    d.pop("name", None)
    for k in (
        "description",
        "engine",
        "port",
        "tp_size",
        "gpu",
        "dtype",
        "gpu_mem_util",
        "max_model_len",
        "extra_args",
        "env",
        "launch_prefix",
    ):
        if k in d:
            lines.append(f"    {k:<14} = {d[k]!r}")
    return "\n".join(lines)


def cmd_profile_list(args):
    _require_model_dir(args)
    profs = load_profiles(args.dir, args.model)
    path = profiles_path(args.dir, args.model)
    print(f"\n  {args.model}  ({path})")
    if not profs:
        print("    (no profiles; run `profile add <name>` to create one)\n")
        return
    print(
        f"\n  {'Name':<14}  {'Engine':<6}  {'TP':>2}  {'GPU':<8}  "
        f"{'Ctx':>7}  Description"
    )
    print("  " + "-" * 72)
    names = sorted(profs.keys(), key=lambda n: (n != "default", n.lower()))
    for name in names:
        p = profs[name]
        desc = p.description[:40] + ("…" if len(p.description) > 40 else "")
        print(
            f"  {name:<14}  {p.engine:<6}  {p.tp_size:>2}  "
            f"{p.gpu:<8}  {p.max_model_len:>7}  {desc}"
        )
    print()


def cmd_profile_show(args):
    _require_model_dir(args)
    profs = load_profiles(args.dir, args.model)
    if not profs:
        sys.exit(f"No profiles for {args.model}.")
    if args.name:
        if args.name not in profs:
            sys.exit(f"No profile '{args.name}'. Have: {sorted(profs)}")
        print()
        print(_format_profile(profs[args.name]))
        print()
        return
    print()
    for name in sorted(profs.keys(), key=lambda n: (n != "default", n.lower())):
        print(_format_profile(profs[name]))
        print()


def cmd_profile_add(args):
    _require_model_dir(args)
    profs = load_profiles(args.dir, args.model)
    if args.name in profs:
        sys.exit(f"Profile '{args.name}' already exists.")

    if args.copy_from:
        if args.copy_from not in profs:
            sys.exit(f"No source profile '{args.copy_from}'. Have: {sorted(profs)}")
        new_p = copy_profile(profs[args.copy_from], args.name)
    elif "default" in profs:
        new_p = copy_profile(profs["default"], args.name)
    else:
        # No default yet — seed a fresh one, with the same auto-templated
        # extra_args (quantization flag, parsers, V100 arg set) the
        # auto-created default would get.
        suggested = detect_model_max_len(args.dir, args.model) or DEFAULT_MAX_MODEL_LEN
        new_p = make_default_profile(max_model_len=suggested)
        new_p.name = args.name
        new_p.description = "New profile."
        try:
            info = model_lib.detect_family(args.dir, args.model)
            tools, reasonings = _cached_parsers()
            new_p.extra_args = model_lib.suggest_extra_args(
                info, tp_size=new_p.tp_size, v100=_host_is_v100(),
                available_tool_parsers=tools or None,
                available_reasoning_parsers=reasonings or None,
            )
        except Exception:
            pass

    profs[args.name] = new_p
    path = save_profiles(args.dir, args.model, profs)
    print(f"  Added [{args.name}] to {path}")
    print(f"  Edit it: $EDITOR {path}")


def cmd_profile_copy(args):
    _require_model_dir(args)
    profs = load_profiles(args.dir, args.model)
    if args.src not in profs:
        sys.exit(f"No source profile '{args.src}'. Have: {sorted(profs)}")
    if args.dst in profs:
        sys.exit(f"Profile '{args.dst}' already exists.")
    profs[args.dst] = copy_profile(profs[args.src], args.dst)
    path = save_profiles(args.dir, args.model, profs)
    print(f"  Copied [{args.src}] → [{args.dst}] in {path}")


def cmd_profile_delete(args):
    _require_model_dir(args)
    profs = load_profiles(args.dir, args.model)
    if args.name not in profs:
        sys.exit(f"No profile '{args.name}'. Have: {sorted(profs)}")
    if args.name == "default" and not args.yes:
        sys.exit("Refusing to delete the 'default' profile without --yes.")
    del profs[args.name]
    if profs:
        path = save_profiles(args.dir, args.model, profs)
        print(f"  Removed [{args.name}] from {path}")
    else:
        # Deleting the last profile wipes the file rather than leaving a stub.
        path = profiles_path(args.dir, args.model)
        if os.path.isfile(path):
            os.remove(path)
        print(f"  Removed [{args.name}] (profiles.toml deleted — no profiles left)")


def cmd_profile_edit(args):
    _require_model_dir(args)
    path = profiles_path(args.dir, args.model)
    if not os.path.isfile(path):
        # Create a default-only file so $EDITOR has something to edit.
        ensure_profiles_exist(args.dir, args.model, interactive=False)
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    subprocess.run([editor, path])


def cmd_profile_path(args):
    _require_model_dir(args)
    print(profiles_path(args.dir, args.model))


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Manage local LLM models: list, delete, download, profiles."
    )
    ap.add_argument(
        "--dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Model directory (default: {DEFAULT_MODEL_DIR})",
    )
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List installed models")

    sp = sub.add_parser("delete", help="Delete an installed model")
    sp.add_argument("name", help="Model folder name")
    sp.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")

    sp = sub.add_parser("download", help="Download a model from HF (GGUF or HF-format)")
    sp.add_argument("repo", help="HF repo id (org/name) or full URL")
    sp.add_argument("--name", help="Override destination folder name")
    sp.add_argument(
        "--target",
        choices=["vllm", "lmstudio"],
        default=None,
        help="Where to install. Default: prompt interactively. "
             "'vllm' = flat layout + auto profile (HF only). "
             "'lmstudio' = publisher/repo layout, no profile.",
    )
    sp.add_argument(
        "--vram",
        type=float,
        default=None,
        help="Available VRAM in GB (default: sum of detected GPUs, "
             f"else {DEFAULT_VRAM_GB})",
    )
    sp.add_argument(
        "--offload",
        action="store_true",
        help="Budget for CPU+GPU MoE offload (llama.cpp-style): adds 85%% "
             "of system RAM to the weight budget when recommending quants",
    )
    sp.add_argument(
        "--ram",
        type=float,
        default=None,
        help="System RAM in GB for --offload (default: read /proc/meminfo)",
    )
    sp.add_argument(
        "--yes",
        action="store_true",
        help="Take recommendation and default profile without prompts",
    )
    sp.add_argument(
        "--dry-run", action="store_true", help="List variants and exit; don't download"
    )
    sp.add_argument(
        "--force",
        action="store_true",
        help="Allow downloading into a non-empty directory",
    )
    sp.add_argument(
        "--hf-bundle",
        action="store_true",
        help="For repos with both HF and GGUF files, grab the safetensors bundle "
             "instead of picking a GGUF quant",
    )

    sp = sub.add_parser(
        "seed-profiles",
        help="Write the curated (documented, benchmarked) profile sets for "
             "this box's standard models",
    )
    sp.add_argument(
        "models", nargs="*", metavar="model",
        help="Curated model(s) to seed (default: all found on disk)",
    )
    sp.add_argument("--yes", action="store_true",
                    help="Write without per-model confirmation")
    sp.add_argument("--list", action="store_true",
                    help="Show the catalog and each model's on-disk state")

    pp = sub.add_parser("profile", help="Manage per-model launch profiles")
    psub = pp.add_subparsers(dest="psub", required=True)

    sp = psub.add_parser("list", help="List profiles for a model")
    sp.add_argument("model")

    sp = psub.add_parser("show", help="Print one or all profiles")
    sp.add_argument("model")
    sp.add_argument("name", nargs="?", default=None)

    sp = psub.add_parser("add", help="Add a new profile seeded from default")
    sp.add_argument("model")
    sp.add_argument("name")
    sp.add_argument(
        "--copy-from",
        default=None,
        help="Source profile to copy from (default: 'default')",
    )

    sp = psub.add_parser("copy", help="Duplicate an existing profile")
    sp.add_argument("model")
    sp.add_argument("src")
    sp.add_argument("dst")

    sp = psub.add_parser("delete", help="Remove a profile")
    sp.add_argument("model")
    sp.add_argument("name")
    sp.add_argument(
        "--yes", action="store_true", help="Required to delete the 'default' profile"
    )

    sp = psub.add_parser("edit", help="Open profiles.toml in $EDITOR")
    sp.add_argument("model")

    sp = psub.add_parser("path", help="Print path to profiles.toml")
    sp.add_argument("model")

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    # No subcommand on CLI — launch interactive TUI
    if not args.cmd:
        _tui_launch()
        return

    dispatch = {
        "list": cmd_list,
        "delete": cmd_delete,
        "download": cmd_download,
        "seed-profiles": cmd_seed_profiles,
    }
    if args.cmd == "profile":
        sub_dispatch = {
            "list": cmd_profile_list,
            "show": cmd_profile_show,
            "add": cmd_profile_add,
            "copy": cmd_profile_copy,
            "delete": cmd_profile_delete,
            "edit": cmd_profile_edit,
            "path": cmd_profile_path,
        }
        sub_dispatch[args.psub](args)
        return
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
