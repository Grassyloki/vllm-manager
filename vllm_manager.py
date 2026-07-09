#!/usr/bin/env python3
"""
vLLM Service Manager
====================
Set up a vLLM environment and manage multiple vLLM inference servers on
separate ports, each launched from a named profile stored alongside its
model weights. Model downloads and profile authoring live in
manage_models.py; this script is about *running* what's already on disk.

Designed for: 2x NVIDIA V100 32 GB (64 GB total), Arch Linux, Python 3.14.
vLLM runs inside a Python 3.12 venv; this manager runs on system Python.

Usage (interactive):
    python vllm_manager.py              # launches interactive TUI

Usage (CLI):
    python vllm_manager.py setup [--v100]
    python vllm_manager.py list
    python vllm_manager.py start   <model> [--profile NAME] [--port N]
    python vllm_manager.py stop    <model>
    python vllm_manager.py stop-all
    python vllm_manager.py restart <model> [--profile NAME]
    python vllm_manager.py status
    python vllm_manager.py test    <model>
    python vllm_manager.py logs    <model>

Launch profiles:
    Profiles are TOML files at <MODEL_DIR>/<model>/profiles.toml.
    Create / edit them with manage_models.py:
        python manage_models.py profile list   <model>
        python manage_models.py profile edit   <model>
        python manage_models.py profile add    <model> <name>
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from datetime import datetime

import model_lib
import tui_lib as tui
from model_lib import (
    Profile,
    human_gb,
    load_profiles,
    profiles_path,
    save_profiles,
    scan_all,
    scan_local_model,
)

# =============================================================================
# CONFIGURATION — env vars override these. Edit defaults to match your host.
# =============================================================================

# Filesystem layout
MODEL_DIR  = os.environ.get("VLLM_MGR_MODEL_DIR",  "/mnt/stor1/vllm/models")
VENV_DIR   = os.environ.get("VLLM_MGR_VENV_DIR",   "/mnt/stor1/vllm/vllm-env")
PID_DIR    = os.environ.get("VLLM_MGR_PID_DIR",    "/root/.vllm-pids")
LOG_DIR    = os.environ.get("VLLM_MGR_LOG_DIR",    "/root/.vllm-logs")
STATE_FILE = os.path.join(PID_DIR, "state.json")
VLLM_VERSION_FILE = os.path.join(PID_DIR, "vllm-version.txt")

# Networking
BASE_PORT   = int(os.environ.get("VLLM_MGR_BASE_PORT", "1001"))
VLLM_HOST   = os.environ.get("VLLM_MGR_BIND_HOST",  "0.0.0.0")
# Public hostname/URL printed by `endpoints` for client config snippets.
PUBLIC_HOST = os.environ.get("VLLM_MGR_PUBLIC_HOST", "aibox.lok.tech")
PUBLIC_SCHEME = os.environ.get("VLLM_MGR_PUBLIC_SCHEME", "http")

# Hardware. Per-GPU VRAM is auto-detected from nvidia-smi at startup unless
# VLLM_MGR_PER_GPU_VRAM_GB is set. This provisional value is refined below,
# once the GPU-query helpers are defined.
_PER_GPU_VRAM_ENV = os.environ.get("VLLM_MGR_PER_GPU_VRAM_GB")
PER_GPU_VRAM = int(_PER_GPU_VRAM_ENV) if _PER_GPU_VRAM_ENV else 32

# vLLM build
PYTHON_VERSION = "3.12"  # vLLM wheels are built for 3.12
# Pin the V100 fork to a known-good commit. Bump when you've tested a newer ref.
# `main` is acceptable but means `setup --v100` rebuilds 30-90 min if upstream moves.
VLLM_FORK_REPO = "https://github.com/1CatAI/1Cat-vLLM.git"
VLLM_FORK_REF  = os.environ.get("VLLM_MGR_FORK_REF", "main")
# Distribution (PyPI metadata) name the fork declares. It is NOT "vllm": the
# fork's pyproject names the package "1cat-vllm" even though the import module
# stays `vllm`. uv refuses a `name @ git+...` spec whose name does not match the
# project metadata, so this must track whatever the fork declares.
VLLM_FORK_PKG  = "1cat-vllm"

# Source-build parallelism for `setup --v100` (compiling the fork from source).
# cicc / cudafe++ / ptxas / cc1plus are each SINGLE-THREADED — one invocation
# uses at most one core — so the only way to light up more cores is to run more
# compile jobs at once (ninja -j = MAX_JOBS). Each CUDA translation unit also
# needs RAM, so jobs are capped by RAM to avoid the OOM killer (exit 9). Knobs:
#   VLLM_MGR_BUILD_JOBS         force an exact job count (overrides the RAM cap)
#   VLLM_MGR_BUILD_MEM_PER_JOB  GiB of RAM budgeted per job      (default 1.5)
#   VLLM_MGR_BUILD_RESERVE_GB   GiB held back for the OS/others   (default 2)
#   VLLM_MGR_BUILD_NVCC_THREADS threads per nvcc (1 is fine for single-arch 7.0)
# NOTE: this build runs via uv/pip -> cmake/ninja, NOT makepkg, so makepkg.conf
# (MAKEFLAGS / NINJAFLAGS) has no effect here — these vars are the real lever.
BUILD_JOBS         = int(os.environ.get("VLLM_MGR_BUILD_JOBS", "0"))          # 0 = auto
BUILD_MEM_PER_JOB  = float(os.environ.get("VLLM_MGR_BUILD_MEM_PER_JOB", "1.5"))
BUILD_RESERVE_GB   = float(os.environ.get("VLLM_MGR_BUILD_RESERVE_GB", "2"))
BUILD_NVCC_THREADS = int(os.environ.get("VLLM_MGR_BUILD_NVCC_THREADS", "1"))

# Scratch dir for the build (wheel extraction + nvcc temps). MUST be on a roomy
# disk: torch's CUDA wheel alone needs several GiB to unpack, and the default
# /tmp is often a small RAM-backed tmpfs — extraction dies with ENOSPC and it
# steals RAM the compilers need. Defaults next to the venv on its big volume.
BUILD_TMP_DIR = (os.environ.get("VLLM_MGR_BUILD_TMP")
                 or os.path.join(os.path.dirname(VENV_DIR), "build-tmp"))

# Process lifecycle
STOP_TIMEOUT_SEC = int(os.environ.get("VLLM_MGR_STOP_TIMEOUT", "30"))

# HuggingFace token — set HF_TOKEN env var or run `hf auth login`
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Test prompt used by the `test` command
TEST_PROMPT = "Briefly explain what a buffer overflow is."

# Benchmark defaults (overridable via `benchmark` CLI flags / the wizard).
BENCH_PROMPT = "Briefly explain what a buffer overflow is"
BENCH_MAX_TOKENS    = 4096      # generate enough to get a steady-state window
BENCH_WARMUP        = 1        # discarded warm-up generations
BENCH_RUNS          = 3        # measured generations
BENCH_WINDOW_SEC    = 5.0      # sliding window for steady-state tok/s
BENCH_READY_TIMEOUT = 600      # max secs to wait for model load/readiness
BENCH_FREEZE_TIMEOUT = 90      # no-token gap that counts as a freeze (abort run)
BENCH_STALL_GAP     = 2.0      # inter-token gap (s) counted as a "stall"


# =============================================================================
# HELPERS
# =============================================================================

def _venv_bin(name: str) -> str:
    """Path to an executable inside the venv."""
    return os.path.join(VENV_DIR, "bin", name)


def _venv_python() -> str:
    return _venv_bin("python")


def _ensure_dirs():
    for d in (MODEL_DIR, PID_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def _save_state(state: dict):
    _ensure_dirs()
    # Atomic write: a crash or full disk mid-write must not corrupt state.json
    # (we'd lose track of running servers). Same temp+rename as save_profiles.
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_FILE)


def _port_free(port: int) -> bool:
    """True if nothing on this host is listening on `port` (bind probe).

    Catches ports held by processes the state file can't see (Docker tenants,
    a manually launched vLLM, sshd port-forwards, ...).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def _next_port(state: dict) -> int:
    """Lowest port from BASE_PORT not used by us AND actually bindable."""
    used = {v["port"] for v in state.values()}
    port = BASE_PORT
    while port in used or not _port_free(port):
        port += 1
        if port > BASE_PORT + 1000:  # something is very wrong; fail loudly
            raise RuntimeError(
                f"No free port found in {BASE_PORT}-{BASE_PORT + 1000}.")
    return port


def _pid_alive(pid: int) -> bool:
    """True if PID is alive AND looks like a vLLM server (or we can't tell)."""
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    # Cross-check /proc/<pid>/cmdline so a recycled PID belonging to some
    # unrelated process after a reboot doesn't masquerade as ours.
    cmdline_path = f"/proc/{pid}/cmdline"
    if not os.path.exists(cmdline_path):
        return True  # not Linux-with-procfs; trust os.kill
    try:
        with open(cmdline_path, "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return True
    return ("vllm" in cmdline) or ("api_server" in cmdline)


def _model_dir_path(name: str) -> str:
    return os.path.join(MODEL_DIR, name)


def _model_exists(name: str) -> bool:
    p = _model_dir_path(name)
    if not os.path.isdir(p):
        return False
    # Any of: a config.json at root, a .safetensors/.bin, or a .gguf file.
    for _, _, fns in os.walk(p):
        if "config.json" in fns:
            return True
        for f in fns:
            if f.lower().endswith((".safetensors", ".bin", ".gguf")):
                return True
    return False


def _resolve_serve_path(name: str) -> str:
    """What to pass as `--model` to vLLM.

    - HF-format models: the directory containing config.json.
    - GGUF-only repos (no config.json): the path of the first .gguf shard.
      vLLM auto-discovers the other shards and pulls the tokenizer from the
      GGUF file itself. If your repo lacks tokenizer metadata, add
      `--tokenizer <hf-base-repo>` via the profile's extra_args.
    """
    root = _model_dir_path(name)
    gguf_files: list[str] = []
    for dp, _, fns in os.walk(root):
        if "config.json" in fns:
            return dp
        for f in fns:
            if f.lower().endswith(".gguf"):
                gguf_files.append(os.path.join(dp, f))
    if gguf_files:
        gguf_files.sort()  # shard-01-of-NN sorts before shard-02-of-NN
        return gguf_files[0]
    return root


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, printing it first."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def _has_command(name: str) -> bool:
    return subprocess.run(
        ["which", name], capture_output=True
    ).returncode == 0


def _prune_dead(state: dict) -> tuple[dict, list[str]]:
    """Remove entries whose processes are no longer alive."""
    alive = {}
    dead = []
    for name, entry in state.items():
        if _pid_alive(entry["pid"]):
            alive[name] = entry
        else:
            dead.append(name)
    if dead:
        _save_state(alive)
    return alive, dead


def _gpu_overlap(gpu_a: str, gpu_b: str) -> set[str]:
    """Return the set of GPU IDs shared between two CUDA_VISIBLE_DEVICES strings."""
    return set(gpu_a.split(",")) & set(gpu_b.split(","))


def _get_running() -> dict:
    """Load state and return only entries whose process is alive."""
    state = _load_state()
    alive, _ = _prune_dead(state)
    return alive


# Readiness probe cache: {port: (unix_ts, bool_ready)}.
# TTL keeps the TUI from hammering the server on every redraw.
_READY_CACHE: dict[int, tuple[float, bool]] = {}
_READY_TTL = 4.0
_READY_TIMEOUT = 0.4


def _probe_ready(port: int) -> bool:
    """Cheap TCP + HTTP check: server is accepting requests on /v1/models."""
    now = time.monotonic()
    cached = _READY_CACHE.get(port)
    if cached and (now - cached[0]) < _READY_TTL:
        return cached[1]

    ready = False
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models",
            timeout=_READY_TIMEOUT,
        ) as resp:
            ready = resp.status == 200
    except Exception:
        ready = False

    _READY_CACHE[port] = (now, ready)
    return ready


def _gpu_vram_used(state: dict) -> dict[str, int]:
    """Return {gpu_id: total_weight_gb} for running models.

    Reads the weight size that was stashed into the state entry at start
    time (see cmd_start). Falls back to scanning the model folder.
    """
    usage: dict[str, int] = {}
    for name, entry in state.items():
        vram_gb = int(entry.get("weight_gb") or 0)
        if vram_gb == 0 and os.path.isdir(_model_dir_path(name)):
            try:
                vram_gb = int(round(
                    scan_local_model(MODEL_DIR, name).weight_bytes / (1024 ** 3)
                ))
            except Exception:
                vram_gb = 0
        gpus = entry.get("gpu", "").split(",") if entry.get("gpu") else []
        # Tensor parallel divides weights across GPUs.
        per_gpu = vram_gb // max(len(gpus), 1)
        for gid in gpus:
            usage[gid] = usage.get(gid, 0) + per_gpu
    return usage


def _query_gpus(timeout: float = 6.0) -> list[dict]:
    """Best-effort per-GPU info via nvidia-smi. [] if unavailable.

    Each entry: {index, name, mem_total_mb, mem_free_mb, compute_cap|None, uuid}.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.free,name,compute_cap,uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if r.returncode != 0:
        return []
    out: list[dict] = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            tot = int(float(parts[1]))
            free = int(float(parts[2]))
        except ValueError:
            continue
        cc = parts[4] if len(parts) > 4 else ""
        out.append({
            "index": idx, "name": parts[3],
            "mem_total_mb": tot, "mem_free_mb": free,
            "compute_cap": cc if cc and not cc.startswith("[") else None,
            "uuid": parts[5] if len(parts) > 5 else None,
        })
    return out


_GPU_CACHE: list[dict] | None = None


def _gpus(refresh: bool = False) -> list[dict]:
    """Cached _query_gpus — one nvidia-smi call per process unless refreshed."""
    global _GPU_CACHE
    if _GPU_CACHE is None or refresh:
        _GPU_CACHE = _query_gpus()
    return _GPU_CACHE


def _is_vllm_pid(pid: int) -> bool:
    """True if a GPU-using pid is one of ours (runs from the vLLM venv) or
    otherwise looks like a vLLM server/worker. Lets us split GPU memory into
    'ours' vs 'other' (e.g. a Docker tenant on a different venv)."""
    try:
        if os.readlink(f"/proc/{pid}/exe").startswith(VENV_DIR):
            return True
    except OSError:
        pass
    low = _proc_cmdline(pid).lower()
    return ("vllm.entrypoints" in low or "-m vllm" in low
            or "enginecore" in low or "vllm::" in low)


def _gpu_usage_split(timeout: float = 6.0) -> list[dict]:
    """Live per-GPU memory split into vLLM vs other processes. [] if no smi.

    Each entry: {index, name, total_mb, used_mb, vllm_mb, other_mb}. Uses
    nvidia-smi compute-apps (per-process MiB + gpu uuid) and attributes each
    pid to vLLM via _is_vllm_pid; 'other' is whatever's left of used memory.
    """
    gpus = _gpus(refresh=True)
    if not gpus:
        return []
    uuid_to_idx = {g["uuid"]: g["index"] for g in gpus if g.get("uuid")}
    vllm_by_idx: dict[int, int] = {g["index"]: 0 for g in gpus}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                idx = uuid_to_idx.get(parts[0])
                try:
                    pid, mem = int(parts[1]), int(float(parts[2]))
                except ValueError:
                    continue
                if idx is not None and _is_vllm_pid(pid):
                    vllm_by_idx[idx] += mem
    except (OSError, subprocess.SubprocessError):
        pass
    out: list[dict] = []
    for g in gpus:
        used = g["mem_total_mb"] - g["mem_free_mb"]
        vllm = min(vllm_by_idx.get(g["index"], 0), used)
        out.append({
            "index": g["index"], "name": g["name"],
            "total_mb": g["mem_total_mb"], "used_mb": used,
            "vllm_mb": vllm, "other_mb": max(0, used - vllm),
        })
    return out


# -- TUI usage-bar rendering (segment lists for tui.select headers) -----------

def _stacked_bar(segvals, total, width=20):
    """Bar body: filled blocks per (value, color_attr) segment, then dim free."""
    import curses
    segs = []
    filled = 0
    for val, attr in segvals:
        n = round(val / total * width) if total > 0 else 0
        n = max(0, min(width - filled, n))
        if n:
            segs.append(("█" * n, attr))
        filled += n
    segs.append(("░" * (width - filled), curses.color_pair(tui.C_DIM)))
    return segs


def _gpu_bar_line(g, width=20):
    """Segmented header line for one GPU: vLLM (green) + other (yellow) + free."""
    import curses
    label = f"GPU{g['index']} {g.get('name') or 'GPU'}"[:26].ljust(26)
    bar = _stacked_bar(
        [(g["vllm_mb"], curses.color_pair(tui.C_GREEN)),
         (g["other_mb"], curses.color_pair(tui.C_YELLOW))],
        g["total_mb"], width)
    return ([(f"  {label} [", curses.color_pair(tui.C_DIM))] + bar + [
        (f"] {g['used_mb'] / 1024:5.1f}/{g['total_mb'] / 1024:.0f}G", 0),
        (f"  █{g['vllm_mb'] / 1024:.1f}", curses.color_pair(tui.C_GREEN)),
        (f" █{g['other_mb'] / 1024:.1f}", curses.color_pair(tui.C_YELLOW)),
    ])


def _ram_bar_line(used_gb, total_gb, width=20):
    """Segmented header line for system RAM (cyan used + dim free)."""
    import curses
    label = "System RAM"[:26].ljust(26)
    bar = _stacked_bar([(used_gb, curses.color_pair(tui.C_CYAN))],
                       total_gb, width)
    return ([(f"  {label} [", curses.color_pair(tui.C_DIM))] + bar +
            [(f"] {used_gb:5.1f}/{total_gb:.0f}G", 0)])


# Refine the per-GPU VRAM default from real hardware (unless pinned via env).
# Uses the smallest GPU so sizing math never over-promises on a mixed rig.
if _PER_GPU_VRAM_ENV is None:
    _detected_gpus = _gpus()
    if _detected_gpus:
        PER_GPU_VRAM = max(
            1, round(min(g["mem_total_mb"] for g in _detected_gpus) / 1024))


def _print_log_tail(path: str, n: int = 25):
    """Print the last n lines of a log file, indented (for crash post-mortems)."""
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        print("    (no log captured)")
        return
    for line in lines[-n:]:
        print("    | " + line.rstrip())


def _preflight_gpu_check(prof, weight_gb: int,
                         activation_reserve_gb: float = 2.5
                         ) -> tuple[list[str], bool]:
    """Validate a profile against live GPU state. Returns (problems, fatal).

    Catches what vLLM would hit at load: a target GPU id that doesn't exist,
    tp_size > GPUs present, and a GPU whose free VRAM is below what vLLM will
    try to reserve (gpu_mem_util * total) — which includes memory held by
    *external* processes the state file can't see (e.g. a Docker tenant).
    """
    gpus = _gpus()
    if not gpus:
        return (["nvidia-smi unavailable — skipping GPU pre-flight check."], False)
    by_idx = {g["index"]: g for g in gpus}
    targets = [int(x) for x in prof.gpu.split(",") if x.strip().isdigit()]
    problems: list[str] = []
    fatal = False

    if prof.tp_size > len(gpus):
        problems.append(
            f"tp_size={prof.tp_size} but only {len(gpus)} GPU(s) present.")
        fatal = True

    per_gpu_need_mb = (weight_gb / max(1, prof.tp_size)
                       + activation_reserve_gb) * 1024
    for tid in targets:
        g = by_idx.get(tid)
        if g is None:
            problems.append(
                f"profile targets GPU {tid}, not reported by nvidia-smi "
                f"(have {sorted(by_idx)}).")
            fatal = True
            continue
        free_mb, total_mb = g["mem_free_mb"], g["mem_total_mb"]
        want_mb = prof.gpu_mem_util * total_mb
        used_others = total_mb - free_mb
        if want_mb > free_mb + 64:  # vLLM errors when util*total can't be met
            problems.append(
                f"GPU {tid}: profile targets {want_mb / 1024:.1f} GB "
                f"(util {prof.gpu_mem_util:.2f} x {total_mb / 1024:.0f} GB) but only "
                f"{free_mb / 1024:.1f} GB free "
                f"({used_others / 1024:.1f} GB held by other processes). "
                f"Lower gpu_mem_util or free that GPU.")
            fatal = True
        elif free_mb < per_gpu_need_mb:
            problems.append(
                f"GPU {tid}: {free_mb / 1024:.1f} GB free, but weights+overhead "
                f"need ~{per_gpu_need_mb / 1024:.1f} GB — likely OOM.")
            fatal = True
    return problems, fatal


def _read_mem_gb() -> tuple[int, int]:
    """Return (MemTotal, MemAvailable) in whole GiB; 0 for anything unreadable."""
    total = avail = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // (1024 * 1024)
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // (1024 * 1024)
    except OSError:
        pass
    return total, avail


def _physical_cores() -> int:
    """Physical core count (SMT siblings collapsed); cpu_count() fallback.

    Compile jobs are compute-bound, so running one per SMT *thread* just
    doubles RAM pressure for a few percent of throughput — on a 2x Xeon 6230
    (40C/80T) that's 79 jobs instead of 39 for no real gain.
    """
    try:
        siblings: set[str] = set()
        base = "/sys/devices/system/cpu"
        for entry in os.listdir(base):
            if not re.fullmatch(r"cpu\d+", entry):
                continue
            topo = os.path.join(base, entry, "topology")
            for fname in ("core_cpus_list", "thread_siblings_list"):
                p = os.path.join(topo, fname)
                if os.path.isfile(p):
                    with open(p) as f:
                        siblings.add(f.read().strip())
                    break
        if siblings:
            return len(siblings)
    except OSError:
        pass
    return os.cpu_count() or 4


def _compute_build_jobs() -> tuple[int, str]:
    """Choose ninja MAX_JOBS for the source build (== cores busy at peak).

    Aims for almost all PHYSICAL cores, but caps by RAM (BUILD_MEM_PER_JOB
    per job, holding BUILD_RESERVE_GB back) so the OOM killer doesn't SIGKILL
    the single-threaded CUDA compilers. VLLM_MGR_BUILD_JOBS overrides the cap.
    Returns (jobs, human_readable_reason).
    """
    logical = os.cpu_count() or 4
    if BUILD_JOBS > 0:
        return (max(1, min(BUILD_JOBS, logical * 2)),
                f"VLLM_MGR_BUILD_JOBS={BUILD_JOBS} (manual override)")
    cpu = _physical_cores()
    total_gb, avail_gb = _read_mem_gb()
    basis_gb = max(1.0, (total_gb or 8) - BUILD_RESERVE_GB)
    ram_jobs = max(1, int(basis_gb / max(0.25, BUILD_MEM_PER_JOB)))
    cpu_jobs = max(1, cpu - 1)               # "almost all physical cores"
    jobs = max(1, min(cpu_jobs, ram_jobs))
    why = (f"min(physical-cores-1={cpu_jobs}, "
           f"{basis_gb:.0f}GiB/{BUILD_MEM_PER_JOB:.1f}GiB={ram_jobs}); "
           f"host {cpu}C/{logical}T, {total_gb}GiB RAM, {avail_gb}GiB free now")
    return jobs, why


# =============================================================================
# COMMANDS (CLI)
# =============================================================================

def cmd_setup(args):
    """Create a Python 3.12 venv and install vLLM + dependencies."""
    _ensure_dirs()

    # Route build scratch (wheel extraction + nvcc temps) onto a roomy disk.
    # The default /tmp is often a small RAM-backed tmpfs, and torch's CUDA wheel
    # needs several GiB to unpack -> ENOSPC; nvcc temps pile up there too. All
    # child processes inherit TMPDIR. Override with VLLM_MGR_BUILD_TMP.
    try:
        os.makedirs(BUILD_TMP_DIR, exist_ok=True)
        os.environ["TMPDIR"] = BUILD_TMP_DIR
        import shutil as _shutil
        _free_gb = _shutil.disk_usage(BUILD_TMP_DIR).free / (1024 ** 3)
        _scratch_note = f"{BUILD_TMP_DIR}  ({_free_gb:.0f} GiB free)"
        if _free_gb < 20:
            _scratch_note += "  [WARNING: <20 GiB free — build may run out of space]"
    except OSError as e:
        _scratch_note = f"(could not prepare {BUILD_TMP_DIR}: {e})"

    print("=" * 60)
    print("  vLLM Environment Setup")
    print("=" * 60)
    print(f"  Build scratch (TMPDIR): {_scratch_note}")

    # -- Step 1: Ensure uv is available ---------------------------------------
    print("\n[1/3] Checking for uv package manager ...")
    if not _has_command("uv"):
        print("  uv not found. Installing ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"],
                       capture_output=True)

        if not _has_command("uv"):
            print("  Trying official installer ...")
            subprocess.run(
                ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
                capture_output=True,
            )
            for p in ["~/.local/bin", "~/.cargo/bin"]:
                expanded = os.path.expanduser(p)
                if os.path.isdir(expanded):
                    os.environ["PATH"] = expanded + ":" + os.environ.get("PATH", "")

        if not _has_command("uv"):
            print("\n  ERROR: Could not install uv.")
            print("  Install manually: https://docs.astral.sh/uv/getting-started/installation/")
            sys.exit(1)

    print("  uv is available.")

    # -- Step 2: Create venv with Python 3.12 ---------------------------------
    print(f"\n[2/3] Setting up venv at {VENV_DIR} ...")
    if os.path.isfile(_venv_python()):
        print(f"  venv already exists.")
        result = subprocess.run(
            [_venv_python(), "--version"], capture_output=True, text=True
        )
        print(f"  {result.stdout.strip()}")
    else:
        print(f"  Creating venv with Python {PYTHON_VERSION} ...")
        print(f"  (uv will download Python {PYTHON_VERSION} if not installed)")
        result = _run([
            "uv", "venv", VENV_DIR,
            "--python", PYTHON_VERSION,
            "--seed",
        ])
        if result.returncode != 0:
            print(f"\n  ERROR: Failed to create venv.")
            print(f"  Try installing python{PYTHON_VERSION} manually:")
            print(f"    sudo pacman -S python312  # or from AUR")
            sys.exit(1)

    # -- Step 3: Install vLLM -------------------------------------------------
    print(f"\n[3/3] Installing vLLM ...")

    if args.v100:
        print()
        print("  V100 (SM70) mode selected.")
        print("  Standard vLLM AWQ kernels require SM75+ GPUs.")
        print("  Installing 1Cat-vLLM fork with SM70 AWQ support.")
        print("  Source: https://github.com/1CatAI/1Cat-vLLM")
        print()

        # cicc / cudafe++ / ptxas / cc1plus are each SINGLE-THREADED, so cores
        # busy at peak == number of parallel compile jobs (ninja -j = MAX_JOBS).
        # _compute_build_jobs aims for almost all cores but caps by RAM so the
        # OOM killer doesn't nuke the compilers (SIGKILL / exit 9). Tune via the
        # BUILD_* vars at the top of this file (or VLLM_MGR_BUILD_* env vars).
        jobs, why = _compute_build_jobs()

        build_env = os.environ.copy()
        build_env["MAX_JOBS"] = str(jobs)
        build_env["NVCC_THREADS"] = str(BUILD_NVCC_THREADS)
        build_env["TORCH_CUDA_ARCH_LIST"] = "7.0"
        build_env["VLLM_TARGET_DEVICE"] = "cuda"
        build_env.pop("CMAKE_BUILD_PARALLEL_LEVEL", None)

        print(f"  Build parallelism: MAX_JOBS={jobs}, "
              f"NVCC_THREADS={BUILD_NVCC_THREADS}")
        print(f"    ({why})")
        print(f"    Override: VLLM_MGR_BUILD_JOBS=N  |  "
              f"VLLM_MGR_BUILD_MEM_PER_JOB=GiB  |  VLLM_MGR_BUILD_RESERVE_GB=GiB")
        print(f"  TORCH_CUDA_ARCH_LIST=7.0 (V100).")
        print("  Building vLLM from source — this will take 30-90 minutes.")
        print()

        # Wipe stale CMake/ninja caches from prior aborted builds. The cached
        # CMakeCache.txt pins absolute paths into uv's per-run build-isolation
        # venv (e.g. /root/.cache/uv/builds-v0/.tmpXXX/bin/ninja); those dirs
        # are deleted when the build exits, so a re-run finds "no such file".
        import shutil, glob
        for checkout in glob.glob(
            os.path.expanduser("~/.cache/uv/git-v0/checkouts/*/*")
        ):
            for sub in ("build", ".deps"):
                p = os.path.join(checkout, sub)
                if os.path.isdir(p):
                    print(f"  Clearing stale build cache: {p}")
                    shutil.rmtree(p, ignore_errors=True)

        fork_spec = f"{VLLM_FORK_PKG} @ git+{VLLM_FORK_REPO}@{VLLM_FORK_REF}"
        print(f"  Installing pinned ref: {VLLM_FORK_REF}")
        print(f"  (override with VLLM_MGR_FORK_REF=<sha|tag|branch>)")
        print()
        # Drop any prior upstream `vllm` distribution first: the fork installs
        # into the same `vllm` import dir but under a different dist name
        # ("1cat-vllm"), so leaving the old dist installed leaves orphaned files
        # and a confusing double-listing in `pip list`.
        _run([
            "uv", "pip", "uninstall",
            "--python", _venv_python(),
            "vllm",
        ], env=build_env)
        result = _run([
            "uv", "pip", "install",
            "--python", _venv_python(),
            "--reinstall-package", VLLM_FORK_PKG,
            "--no-cache",
            fork_spec,
        ], env=build_env)

        if result.returncode != 0:
            print()
            print("  Auto-install from source failed.")
            print("  This likely means a build dependency is missing.")
            print()
            print("  Manual installation steps:")
            print("    1. Visit https://github.com/1CatAI/1Cat-vLLM/releases")
            print("    2. Download the .whl matching your Python/CUDA version")
            print(f"    3. Run: uv pip install --python {_venv_python()} <path-to-wheel>")
            print()
            print("  Required system packages (Arch):")
            print("    sudo pacman -S gcc cuda cudnn")
            print()
            print("  Alternatively, use their Docker image:")
            print("    See the README at https://github.com/1CatAI/1Cat-vLLM")
            sys.exit(1)
    else:
        result = _run([
            "uv", "pip", "install",
            "--python", _venv_python(),
            "vllm",
            "--torch-backend=auto",
        ])
        if result.returncode != 0:
            print("\n  ERROR: vLLM installation failed.")
            print("  Check CUDA drivers are installed: nvidia-smi")
            sys.exit(1)

    _run([
        "uv", "pip", "install",
        "--python", _venv_python(),
        "huggingface_hub[cli]",
    ])

    # -- Verify ---------------------------------------------------------------
    print("\nVerifying installation ...")
    result = subprocess.run(
        [_venv_python(), "-c", "import vllm; print(vllm.__version__)"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        version = result.stdout.strip()
        # Cache the version + parser names so plan/start can adapt without
        # paying for a venv import on every invocation.
        try:
            _ensure_dirs()
            with open(VLLM_VERSION_FILE, "w") as f:
                f.write(version + "\n")
        except OSError:
            pass

        print(f"  vLLM {version}")
        print()
        print("  Setup complete.")
        print(f"    venv:   {VENV_DIR}")
        print(f"    models: {MODEL_DIR}")
        print(f"    logs:   {LOG_DIR}")
        print()
        print("  Next steps:")
        print(f"    python manage_models.py download <hf-repo>   # grab a model")
        print(f"    python {sys.argv[0]} list                    # see what's on disk")
        print(f"    python {sys.argv[0]} plan <model>            # check VRAM fit")
        print(f"    python {sys.argv[0]} start <model>           # launch default profile")
    else:
        print(f"\n  WARNING: vLLM import failed:")
        print(f"  {result.stderr.strip()}")
        if not args.v100:
            print("\n  If you have V100 GPUs, re-run with: python vllm_manager.py setup --v100")



def _resolve_profile(name: str, profile_name: str) -> Profile:
    """Load a profile by name, or auto-create a default if it doesn't exist."""
    profs = load_profiles(MODEL_DIR, name)
    if profile_name in profs:
        return profs[profile_name]

    if profile_name != "default":
        print(f"No profile '{profile_name}' for '{name}'.")
        if profs:
            print(f"  Available: {', '.join(sorted(profs))}")
        else:
            print("  No profiles exist yet. "
                  f"Create one: manage_models.py profile add {name} {profile_name}")
        sys.exit(1)

    # Silently auto-create a default profile so `start <model>` Just Works
    # the first time after download without an extra step.
    from model_lib import ensure_profiles_exist
    profs, created = ensure_profiles_exist(MODEL_DIR, name, interactive=False)
    if created:
        print(f"  Auto-created default profile at {profiles_path(MODEL_DIR, name)}.")
    return profs["default"]


def _strip_reasoning_args(extra: list[str]) -> list[str]:
    """Drop --reasoning-parser <value> (and --reasoning-parser=<value>) pairs."""
    out: list[str] = []
    skip_next = False
    for tok in extra:
        if skip_next:
            skip_next = False
            continue
        if tok == "--reasoning-parser":
            skip_next = True
            continue
        if tok.startswith("--reasoning-parser="):
            continue
        out.append(tok)
    return out


def _build_vllm_cmd(model_path: str, served_name: str, port: int,
                    prof: Profile) -> list[str]:
    # Optional NUMA/etc. wrapper (e.g. ["numactl", "--cpunodebind=0",
    # "--membind=0"]). Dropped with a warning if the binary isn't installed,
    # so a profile written on the aibox still starts elsewhere.
    prefix: list[str] = []
    if prof.launch_prefix:
        import shutil
        if shutil.which(prof.launch_prefix[0]):
            prefix = list(prof.launch_prefix)
        else:
            print(f"  WARNING: launch_prefix binary "
                  f"'{prof.launch_prefix[0]}' not found — ignoring prefix.")
    cmd = prefix + [
        _venv_python(), "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model_path,
        "--served-model-name",      served_name,
        "--host",                   VLLM_HOST,
        "--port",                   str(port),
        "--dtype",                  prof.dtype,
        "--tensor-parallel-size",   str(prof.tp_size),
        "--gpu-memory-utilization", str(prof.gpu_mem_util),
        "--max-model-len",          str(prof.max_model_len),
        "--trust-remote-code",
    ]
    cmd.extend(prof.extra_args)
    return cmd


def cmd_start(args):
    """Start serving a model via vLLM on a dedicated port."""
    _ensure_dirs()
    name = args.model
    profile_name = args.profile or "default"

    if not _model_exists(name):
        print(f"No model directory at {_model_dir_path(name)}.")
        print(f"  Available: {', '.join(m.name for m in scan_all(MODEL_DIR)) or '(none)'}")
        print(f"  Download: python manage_models.py download <hf-repo>")
        sys.exit(1)

    if not os.path.isfile(_venv_python()):
        print("venv not found. Run 'setup' first.")
        sys.exit(1)

    prof = _resolve_profile(name, profile_name)
    if prof.engine != "vllm":
        print(f"Profile '{prof.name}' requests engine '{prof.engine}', "
              f"but this manager only launches vllm today.")
        sys.exit(1)

    # Sanity: gpu ids vs tp_size.
    gpus = [g.strip() for g in prof.gpu.split(",") if g.strip()]
    if len(gpus) != prof.tp_size:
        print(f"WARNING: profile [{prof.name}] has gpu='{prof.gpu}' "
              f"({len(gpus)} ids) but tp_size={prof.tp_size}. "
              f"These must match.")
        print(f"  Edit: manage_models.py profile edit {name}")
        print()

    state = _load_state()
    if name in state and _pid_alive(state[name]["pid"]):
        entry = state[name]
        print(f"'{name}' is already running.")
        print(f"  Port: {entry['port']}  PID: {entry['pid']}  "
              f"Profile: {entry.get('profile', '?')}")
        return
    state.pop(name, None)

    # Port priority: --port flag > profile's pinned port > auto from BASE_PORT.
    if args.port:
        port = args.port
    elif getattr(prof, "port", 0):
        port = prof.port
    else:
        port = _next_port(state)

    # For an explicitly chosen port (flag or pinned), verify it's actually
    # available before paying a multi-minute model load that dies at bind.
    if (args.port or getattr(prof, "port", 0)) and not _port_free(port):
        in_use = {e["port"]: n for n, e in state.items()
                  if _pid_alive(e["pid"])}
        holder = (f"'{in_use[port]}'" if port in in_use
                  else "another process (not managed by this tool)")
        print(f"ERROR: port {port} is already in use by {holder}.")
        print(f"  Stop it first, change the profile's port, or use --port.")
        if not getattr(args, "force", False):
            sys.exit(1)
        print("  --force given — continuing anyway (vLLM will likely "
              "fail at bind).")
    model_info = scan_local_model(MODEL_DIR, name)
    weight_gb = int(round(model_info.weight_bytes / (1024 ** 3)))
    serve_path = _resolve_serve_path(name)

    # Co-tenancy warning on shared GPUs.
    for other_name, other_entry in state.items():
        if not _pid_alive(other_entry["pid"]):
            continue
        overlap = _gpu_overlap(other_entry["gpu"], prof.gpu)
        if overlap:
            other_vram = int(other_entry.get("weight_gb") or 0)
            total = other_vram + weight_gb
            gpu_ids = ",".join(sorted(overlap))
            print(f"NOTE: GPU {gpu_ids} is shared with '{other_name}' "
                  f"(~{other_vram} GB weights)")
            print(f"  Combined weight estimate: ~{total} GB / {PER_GPU_VRAM} GB per GPU")
            if total > PER_GPU_VRAM * prof.gpu_mem_util:
                print(f"  WARNING: this likely exceeds GPU memory.")
            print()

    # Pre-flight: validate against live GPU state (free VRAM incl. external
    # tenants, target GPUs exist, tp_size fits) so we fail fast with a clear
    # message instead of a cryptic vLLM OOM at load. Bypass with --force.
    problems, fatal = _preflight_gpu_check(prof, weight_gb)
    for _msg in problems:
        print(f"  GPU check: {_msg}")
    if fatal and not getattr(args, "force", False):
        print()
        print("  Refusing to start — this would likely fail at load. "
              "Override with --force.")
        sys.exit(1)
    if problems:
        print()

    is_gguf = serve_path.lower().endswith(".gguf")

    if getattr(args, "no_reasoning", False):
        prof.extra_args = _strip_reasoning_args(prof.extra_args)

    print(f"Starting {name}")
    print(f"  Profile:  {prof.name}")
    print(f"  Model:    {serve_path}")
    if is_gguf:
        print(f"  (GGUF mode — tokenizer read from file metadata. If loading")
        print(f"   fails on tokenizer, add '--tokenizer <hf-base-repo>' to")
        print(f"   extra_args in profiles.toml.)")
    print(f"  Port:     {port}")
    print(f"  GPU(s):   {prof.gpu}  (tensor parallel = {prof.tp_size})")
    print(f"  Dtype:    {prof.dtype}")
    print(f"  Max len:  {prof.max_model_len}")
    if getattr(args, "no_reasoning", False):
        print(f"  Reasoning: OFF (stripped --reasoning-parser for this launch)")
    if prof.extra_args:
        print(f"  Extra:    {' '.join(prof.extra_args)}")
    if prof.launch_prefix:
        print(f"  Prefix:   {' '.join(prof.launch_prefix)}")

    log_file = os.path.join(LOG_DIR, f"{name}.log")
    # Keep the previous run's log (so a crash's output survives the next start)
    # instead of truncating it — one generation at <model>.log.1.
    if os.path.isfile(log_file) and os.path.getsize(log_file) > 0:
        try:
            os.replace(log_file, log_file + ".1")
        except OSError:
            pass

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = prof.gpu
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
    env.update(prof.env)

    cmd = _build_vllm_cmd(serve_path, name, port, prof)

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    state[name] = {
        "pid":       proc.pid,
        "port":      port,
        "gpu":       prof.gpu,
        "tp":        prof.tp_size,
        "profile":   prof.name,
        "weight_gb": weight_gb,
        "started":   datetime.now().isoformat(timespec="seconds"),
    }
    _save_state(state)

    print(f"  PID:      {proc.pid}")
    print(f"  Log:      {log_file}")
    print()

    # Watch for an immediate crash (bad flag, missing dep) so we don't report a
    # false "loading...". Load-time OOM can take longer — use --wait to catch it.
    print("  Checking it stays up ...", end="", flush=True)
    crash_deadline = time.monotonic() + 8
    while time.monotonic() < crash_deadline:
        if proc.poll() is not None:
            print(" crashed.")
            state.pop(name, None)
            _save_state(state)
            print()
            print(f"  ERROR: {name} exited within seconds "
                  f"(exit code {proc.returncode}).")
            print(f"  Last lines of {log_file}:")
            _print_log_tail(log_file, 25)
            print()
            print(f"  Full log: python {sys.argv[0]} logs {name}")
            sys.exit(1)
        time.sleep(0.5)
    print(" still up.")

    # Optional: block until the server answers /v1/models (or --wait elapses).
    wait_secs = int(getattr(args, "wait", 0) or 0)
    if wait_secs > 0:
        print(f"  Waiting up to {wait_secs}s for readiness ...",
              end="", flush=True)
        ready_deadline = time.monotonic() + wait_secs
        ready = False
        while time.monotonic() < ready_deadline:
            if proc.poll() is not None:
                print(" process exited.")
                state.pop(name, None)
                _save_state(state)
                print(f"\n  ERROR: {name} exited (code {proc.returncode}) while "
                      f"starting. Last lines of {log_file}:")
                _print_log_tail(log_file, 25)
                sys.exit(1)
            if _probe_ready(port):
                ready = True
                break
            time.sleep(2)
        print(" ready." if ready else " still loading (timed out; process alive).")

    print()
    if wait_secs <= 0:
        print(f"  Model is loading (may take 1-5 min for large models).")
    print(f"  Watch progress:  python {sys.argv[0]} logs {name}")
    print(f"  Check status:    python {sys.argv[0]} status")
    print(f"  Test it:         python {sys.argv[0]} test {name}")
    print(f"  API endpoint:    http://localhost:{port}/v1/chat/completions")


def _signal_and_wait(name: str, pid: int, port: int, timeout: int) -> bool:
    """SIGTERM, poll until dead or timeout, then SIGKILL. Returns True if exited cleanly."""
    if not _pid_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.5)
    print(f"  {name}: didn't exit in {timeout}s, sending SIGKILL ...")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return False


def cmd_stop(args):
    """Stop a running model."""
    state = _load_state()
    name = args.model
    timeout = getattr(args, "timeout", None) or STOP_TIMEOUT_SEC

    if name not in state:
        print(f"'{name}' is not in the state file.")
        return

    entry = state[name]
    pid = entry["pid"]

    if _pid_alive(pid):
        print(f"Stopping {name} (PID {pid}, port {entry['port']}, "
              f"timeout {timeout}s) ...")
        clean = _signal_and_wait(name, pid, entry["port"], timeout)
        print("  Stopped." if clean else "  Killed.")
    else:
        print(f"  PID {pid} was not running (stale entry removed).")

    del state[name]
    _save_state(state)


def cmd_stop_all(args):
    """Stop all running models in parallel (SIGTERM all, then wait once)."""
    state = _load_state()
    if not state:
        print("No models are running.")
        return

    timeout = getattr(args, "timeout", None) or STOP_TIMEOUT_SEC
    names = list(state.keys())
    print(f"Stopping {len(names)} model(s) in parallel: {', '.join(names)}")
    print(f"  Per-process timeout: {timeout}s")

    # Phase 1: send SIGTERM to all
    pending: list[tuple[str, int, int]] = []  # (name, pid, port)
    for name in names:
        entry = state[name]
        pid = entry["pid"]
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                pending.append((name, pid, entry["port"]))
                print(f"  SIGTERM -> {name} (PID {pid})")
            except ProcessLookupError:
                pass

    # Phase 2: wait for all (single deadline, not per-model serial)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        pending = [p for p in pending if _pid_alive(p[1])]
        if pending:
            time.sleep(0.5)

    # Phase 3: SIGKILL any stragglers
    for name, pid, _port in pending:
        print(f"  {name}: didn't exit in {timeout}s, sending SIGKILL ...")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    _save_state({})
    print(f"  All stopped.")


def cmd_restart(args):
    """Restart a model using the profile it was started with (or --profile)."""
    state = _load_state()
    name = args.model

    old = state.get(name, {})
    profile = args.profile or old.get("profile") or "default"

    if name in state:
        cmd_stop(argparse.Namespace(model=name))

    print()
    cmd_start(argparse.Namespace(model=name, profile=profile, port=None,
                                 no_reasoning=False))


def cmd_status(args):
    """Show all running models, their ports, and GPU assignments."""
    state = _load_state()

    if not state:
        print("No models are currently running.")
        print(f"  Start one: python {sys.argv[0]} start <model>")
        return

    alive, dead = _prune_dead(state)

    if dead:
        print(f"Cleaned up {len(dead)} stale entry(s): {', '.join(dead)}\n")

    if not alive:
        print("No models are currently running.")
        return

    hdr = (f"{'Model':<24} {'Profile':<12} {'Port':<6} {'PID':<7} "
           f"{'GPU':<6} {'TP':<4} {'Started'}")
    print(hdr)
    print("-" * len(hdr))

    total_vram = 0
    gpu_usage: dict[str, list] = {}

    for name, entry in sorted(alive.items(), key=lambda x: x[1]["port"]):
        vram = int(entry.get("weight_gb") or 0)
        print(
            f"{name:<24} {entry.get('profile','-'):<12} "
            f"{entry['port']:<6} {entry['pid']:<7} "
            f"{entry['gpu']:<6} {entry['tp']:<4} {entry['started']}"
        )
        total_vram += vram
        gpus = entry["gpu"].split(",")
        per_gpu = vram // max(len(gpus), 1)
        for gid in gpus:
            gpu_usage.setdefault(gid, []).append((name, per_gpu))

    print()
    print("GPU memory estimates (weight-only; actual usage is higher):")
    for gid in sorted(gpu_usage):
        models_on_gpu = gpu_usage[gid]
        gpu_total = sum(v for _, v in models_on_gpu)
        model_list = ", ".join(f"{n}(~{v}GB)" for n, v in models_on_gpu)
        bar_len = int(gpu_total / PER_GPU_VRAM * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"  GPU {gid}: [{bar}] ~{gpu_total}/{PER_GPU_VRAM} GB  ({model_list})")

    print(f"\n  Total weight VRAM: ~{total_vram} GB / {PER_GPU_VRAM * 2} GB")


def cmd_test(args):
    """Send a test prompt to a running model and display the response."""
    state = _load_state()
    name = args.model

    if name not in state:
        print(f"'{name}' is not running.")
        if _model_exists(name):
            print(f"  Start it: python {sys.argv[0]} start {name}")
        sys.exit(1)

    if not _pid_alive(state[name]["pid"]):
        print(f"'{name}' process (PID {state[name]['pid']}) is no longer running.")
        print(f"  Check logs: python {sys.argv[0]} logs {name}")
        del state[name]
        _save_state(state)
        sys.exit(1)

    port = state[name]["port"]
    url = f"http://localhost:{port}/v1/chat/completions"

    print(f"Testing {name} on port {port}")
    print(f"  Prompt: \"{TEST_PROMPT}\"")
    print()

    payload = json.dumps({
        "model":       name,
        "messages":    [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens":  256,
        "temperature": 0.7,
        "stream":      True,
        "stream_options": {"include_usage": True},
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept":       "text/event-stream",
        },
    )

    print("  Response (streaming):")
    print()
    print("    ", end="", flush=True)

    request_start = time.monotonic()
    first_token_time = None
    last_token_time = None
    completion_tokens = 0
    prompt_tokens = None
    usage_total = None
    pieces: list[str] = []
    col = 4  # indent width

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\n").rstrip("\r")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Usage-only final chunk (when stream_options.include_usage).
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    usage_total = usage.get("total_tokens", usage_total)
                    if usage.get("completion_tokens") is not None:
                        completion_tokens = max(
                            completion_tokens, usage["completion_tokens"]
                        )

                for choice in chunk.get("choices", []) or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    if not piece:
                        continue
                    now = time.monotonic()
                    if first_token_time is None:
                        first_token_time = now
                    last_token_time = now
                    completion_tokens += 1  # chunk-level count (fallback)
                    pieces.append(piece)
                    for ch in piece:
                        if ch == "\n":
                            print()
                            print("    ", end="", flush=True)
                            col = 4
                        else:
                            if col >= 76:
                                print()
                                print("    ", end="", flush=True)
                                col = 4
                            print(ch, end="", flush=True)
                            col += 1

            print()

        # Prefer the server-reported completion count (usage) for TPS.
        if usage_total is not None and prompt_tokens is not None:
            srv_completion = usage_total - prompt_tokens
            if srv_completion > 0:
                completion_tokens = srv_completion

        total_elapsed = (last_token_time or time.monotonic()) - request_start
        ttft = (first_token_time - request_start) if first_token_time else None
        gen_elapsed = (
            (last_token_time - first_token_time)
            if first_token_time and last_token_time and last_token_time > first_token_time
            else 0.0
        )
        tps = (completion_tokens / gen_elapsed) if gen_elapsed > 0 else 0.0
        tps_wall = (
            completion_tokens / (total_elapsed - (ttft or 0.0))
            if total_elapsed > (ttft or 0.0) and completion_tokens
            else tps
        )

        print()
        print("  Performance:")
        if ttft is not None:
            print(f"    Time to first token: {ttft * 1000:.0f} ms")
        print(f"    Generation time:     {gen_elapsed:.2f} s")
        print(f"    Total wall time:     {total_elapsed:.2f} s")
        print(
            f"    Tokens: prompt={prompt_tokens if prompt_tokens is not None else '?'}, "
            f"completion={completion_tokens}, "
            f"total={usage_total if usage_total is not None else '?'}"
        )
        if completion_tokens and gen_elapsed > 0:
            print(f"    Throughput:          {tps:.2f} tok/s (post-first-token)")
        elif completion_tokens and total_elapsed > 0:
            print(f"    Throughput:          {tps_wall:.2f} tok/s (wall)")
        else:
            print("    Throughput:          n/a (no tokens generated)")

        if completion_tokens:
            print(f"\n  [PASS] {name} is responding.")
        else:
            print(f"\n  [FAIL] {name} returned no content.")

    except urllib.error.URLError as e:
        print()
        print(f"  [FAIL] Could not reach {url}")
        print(f"         {e}")
        print()
        print("  The model may still be loading. Check logs:")
        print(f"    python {sys.argv[0]} logs {name}")
    except Exception as e:
        print()
        print(f"  [FAIL] {type(e).__name__}: {e}")


def cmd_list(args):
    """List models on disk with kind/quant/size and running status."""
    state = _get_running()
    models = scan_all(MODEL_DIR)

    print(f"\n  Model dir: {MODEL_DIR}")
    if not models:
        print("  (no models installed)")
        print(f"  Download: python manage_models.py download <hf-repo>\n")
        return

    total = sum(m.size_bytes for m in models)
    print(f"  {len(models)} model(s), {total / (1024**3):.1f} GB total\n")
    print(f"  {'Name':<26}  {'Kind':<6}  {'Quant':<10}  {'Size':>9}  "
          f"{'Profiles':<18}  Status")
    print("  " + "-" * 88)

    for m in models:
        profs = load_profiles(MODEL_DIR, m.name)
        prof_str = ", ".join(sorted(profs.keys())) or "(none)"
        if len(prof_str) > 18:
            prof_str = prof_str[:17] + "…"

        if m.name in state:
            entry = state[m.name]
            status = (f"running :{entry['port']} "
                      f"[{entry.get('profile','-')}]")
        else:
            status = "-"

        print(f"  {m.name:<26}  {m.kind:<6}  {(m.quant or '-'):<10}  "
              f"{human_gb(m.size_bytes)}  {prof_str:<18}  {status}")
    print()
    print(f"  Start:   python {sys.argv[0]} start <model> [--profile NAME]")
    print(f"  Profiles: python manage_models.py profile list <model>")


# =============================================================================
# ENDPOINTS — copy-paste config snippets for IDE/CLI clients
# =============================================================================
#
# Each running model becomes a usable OpenAI-compatible endpoint at
# http://PUBLIC_HOST:<port>/v1. This command prints config blocks for the
# clients the user actually uses.

def _endpoint_url(port: int) -> str:
    return f"{PUBLIC_SCHEME}://{PUBLIC_HOST}:{port}/v1"


def _print_endpoint_block(name: str, port: int):
    base = _endpoint_url(port)
    print()
    print("─" * 72)
    print(f"  {name}  →  {base}")
    print("─" * 72)

    # OpenAI-compatible cURL smoke test
    print()
    print("  Smoke test:")
    print(f"    curl {base}/chat/completions \\")
    print( "      -H 'Content-Type: application/json' \\")
    print(f"      -d '{{\"model\":\"{name}\",\"messages\":["
          f"{{\"role\":\"user\",\"content\":\"hi\"}}]}}'")

    # OpenCode (~/.config/opencode/opencode.json)
    print()
    print("  OpenCode  (~/.config/opencode/opencode.json):")
    print( "    {")
    print( "      \"provider\": {")
    print(f"        \"vllm-{name}\": {{")
    print( "          \"npm\": \"@ai-sdk/openai-compatible\",")
    print(f"          \"name\": \"vLLM ({name})\",")
    print( "          \"options\": {")
    print(f"            \"baseURL\": \"{base}\"")
    print( "          },")
    print( "          \"models\": {")
    print(f"            \"{name}\": {{ \"name\": \"{name}\" }}")
    print( "          }")
    print( "        }")
    print( "      }")
    print( "    }")

    # Claude Code (env-var or settings.json — supports OpenAI-compatible endpoints
    # via ANTHROPIC_BASE_URL when paired with a Claude-format adapter; for
    # native OpenAI usage, use a litellm proxy or the openai-compatible setting).
    print()
    print("  Claude Code  (one-shot env vars):")
    print(f"    ANTHROPIC_BASE_URL={base.rsplit('/v1',1)[0]} \\")
    print(f"    ANTHROPIC_AUTH_TOKEN=dummy \\")
    print( "    claude")
    print("    # NOTE: requires a Claude-format proxy (e.g., LiteLLM) in front")
    print("    # of vLLM. For raw OpenAI usage, prefer OpenCode/Zed below.")

    # Zed (~/.config/zed/settings.json)
    print()
    print("  Zed  (settings.json):")
    print( "    {")
    print( "      \"language_models\": {")
    print( "        \"openai_compatible\": {")
    print(f"          \"vllm-{name}\": {{")
    print(f"            \"api_url\": \"{base}\",")
    print( "            \"api_key\": \"dummy\",")
    print( "            \"available_models\": [")
    print(f"              {{ \"name\": \"{name}\", \"max_tokens\": 32000 }}")
    print( "            ]")
    print( "          }")
    print( "        }")
    print( "      }")
    print( "    }")

    # Continue (~/.continue/config.yaml)
    print()
    print("  Continue  (~/.continue/config.yaml):")
    print( "    models:")
    print(f"      - name: vllm-{name}")
    print(f"        provider: openai")
    print(f"        model: {name}")
    print(f"        apiBase: {base}")
    print( "        apiKey: dummy")


def cmd_endpoints(args):
    """Print client-config snippets for running (or named) models."""
    state = _get_running()
    if args.model:
        if args.model not in state:
            # Try to locate it as a stopped model and synthesize a port note.
            if _model_exists(args.model):
                print(f"'{args.model}' is not running — start it first to get a port.")
                print(f"  python {sys.argv[0]} start {args.model}")
                sys.exit(1)
            print(f"No model named '{args.model}'.")
            sys.exit(1)
        targets = [(args.model, state[args.model])]
    else:
        if not state:
            print("No models are running. Start one to get an endpoint:")
            print(f"  python {sys.argv[0]} start <model>")
            return
        targets = sorted(state.items(), key=lambda x: x[1]["port"])

    print(f"  Public host: {PUBLIC_HOST}  ({len(targets)} endpoint(s))")
    for name, entry in targets:
        _print_endpoint_block(name, entry["port"])


def cmd_logs(args):
    """Tail the log file for a model."""
    name = args.model
    log_file = os.path.join(LOG_DIR, f"{name}.log")

    if not os.path.isfile(log_file):
        print(f"No log file found for '{name}' at {log_file}")
        sys.exit(1)

    print(f"Tailing {log_file}  (Ctrl+C to stop)\n")
    try:
        subprocess.run(["tail", "-f", "-n", "50", log_file])
    except KeyboardInterrupt:
        print()


# =============================================================================
# PLAN — sizing assistant
# =============================================================================

def _format_plan(name: str, m, info, p) -> str:
    """Render a Plan as a human-readable block (no ANSI colour, plain text)."""
    out: list[str] = []
    out.append("=" * 64)
    out.append(f"  Sizing plan for {name}")
    out.append("=" * 64)
    out.append(f"  Family:        {info.family}  ({info.arch or '?'})")
    out.append(f"  Quant:         {info.quant_method or 'fp16'}"
               + (f"-int{info.quant_bits}" if info.quant_bits else ""))
    out.append(f"  Attention:     {info.attn_kind}"
               + (f", hybrid ({info.full_attn_layers}/{info.num_hidden_layers} "
                  f"layers carry KV)" if info.is_hybrid_attn
                  else f", {info.num_hidden_layers} layers"))
    if info.attn_kind == "mla":
        out.append(f"  MLA:           kv_lora={info.kv_lora_rank} "
                   f"qk_rope={info.qk_rope_head_dim}")
    else:
        out.append(f"  KV heads/dim:  {info.num_kv_heads} x {info.head_dim}")
    out.append(f"  Model max ctx: {info.max_position_embeddings or '?'}")
    out.append("")
    out.append(f"  GPU pool:      {p.tp_size} x {PER_GPU_VRAM} GB"
               f"  (mem util {p.gpu_mem_util:.0%})")
    out.append(f"  Weights:       {p.weight_gb_total:.1f} GB total -> "
               f"{p.weight_gb_per_gpu:.1f} GB / GPU")
    out.append(f"  Activations:   ~{p.activation_reserve_gb:.0f} GB / GPU reserve")
    out.append(f"  KV budget:     {p.kv_budget_per_gpu_gb:.1f} GB / GPU "
               f"(= {p.kv_budget_per_gpu_gb * p.tp_size:.1f} GB total)")
    if p.kv_bytes_per_token:
        out.append(f"  KV per token:  {p.kv_bytes_per_token / 1024:.1f} KB "
                   f"(across all full-attention layers)")
    out.append("")
    if not p.fits:
        out.append(f"  >>> DOES NOT FIT on {p.tp_size}x{PER_GPU_VRAM} GB.")
    else:
        out.append(f"  Recommended for concurrency={p.concurrency}:")
        out.append(f"    tp_size       = {p.tp_size}")
        out.append(f"    max_model_len = {p.recommended_max_len}")
        if p.kv_bytes_per_token:
            kv_used_gb = (p.recommended_max_len * p.concurrency
                          * p.kv_bytes_per_token / (1024 ** 3))
            out.append(f"    -> KV used   ~{kv_used_gb:.1f} GB total")
    if p.notes:
        out.append("")
        out.append("  Notes:")
        for n in p.notes:
            out.append(f"    - {n}")
    return "\n".join(out)


def _plan_to_profile_overrides(p, info, *, base: Profile | None = None) -> dict:
    """Map a Plan onto Profile field overrides for the auto-template."""
    gpus = ",".join(str(i) for i in range(p.tp_size))
    overrides = {
        "tp_size":       p.tp_size,
        "gpu":           gpus,
        "max_model_len": p.recommended_max_len,
        "gpu_mem_util":  p.gpu_mem_util,
    }
    return overrides


def cmd_plan(args):
    """Recommend tp_size + max_model_len for a model on this host."""
    name = args.model
    if not _model_exists(name):
        print(f"No model directory at {_model_dir_path(name)}.")
        sys.exit(1)

    m = scan_local_model(MODEL_DIR, name)
    info = model_lib.detect_family(MODEL_DIR, name)

    # Detect available GPU count from CUDA_VISIBLE_DEVICES or default to 2.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_gpus = (len([x for x in visible.split(",") if x.strip()])
                if visible else 2)

    p = model_lib.compute_plan(
        m, info,
        vram_per_gpu_gb=PER_GPU_VRAM,
        num_gpus=num_gpus,
        tp_size=args.tp,
        gpu_mem_util=args.gpu_mem_util,
        concurrency=args.concurrency,
    )

    print(_format_plan(name, m, info, p))

    if not p.fits:
        sys.exit(2)

    # Build the recommended profile (auto-template + sizing overrides).
    tool_set, reasoning_set = model_lib.read_cached_parsers(VLLM_VERSION_FILE + ".json")
    # SM70 (V100) needs the fork's attention backend and omits expert-parallel;
    # detect it so `plan --apply` writes a profile that actually runs here.
    is_v100 = any(
        (g.get("compute_cap") or "").startswith("7.0") or "V100" in g.get("name", "")
        for g in _gpus())
    extra_args = model_lib.suggest_extra_args(
        info, tp_size=p.tp_size,
        tool_use=not args.no_tool_use,
        reasoning=not args.no_reasoning,
        v100=is_v100,
        available_tool_parsers=tool_set or None,
        available_reasoning_parsers=reasoning_set or None,
    )
    overrides = _plan_to_profile_overrides(p, info)
    overrides["extra_args"] = extra_args
    overrides["description"] = (
        f"Auto-planned for {p.tp_size}x{PER_GPU_VRAM}GB, "
        f"concurrency={p.concurrency}."
    )
    new_default = model_lib.make_default_profile(**overrides)

    # Diff against existing [default] if any.
    existing = load_profiles(MODEL_DIR, name)
    print()
    print("  Recommended profile:")
    for line in model_lib.dump_profiles_toml({"default": new_default}).splitlines():
        if line.startswith("#") or not line.strip():
            continue
        print(f"    {line}")

    if "default" in existing:
        print()
        print("  Differences from current [default]:")
        any_diff = False
        cur = existing["default"]
        for field_name in ("tp_size", "gpu", "max_model_len",
                           "gpu_mem_util", "extra_args"):
            cur_val = getattr(cur, field_name)
            new_val = getattr(new_default, field_name)
            if cur_val != new_val:
                any_diff = True
                print(f"    {field_name}:  {cur_val!r}  ->  {new_val!r}")
        if not any_diff:
            print("    (none — your default already matches the recommendation)")

    print()
    if args.apply:
        # Back up then overwrite [default].
        path = profiles_path(MODEL_DIR, name)
        if os.path.isfile(path):
            import shutil
            shutil.copy2(path, path + ".bak")
            print(f"  Backed up existing profiles to {path}.bak")
        new_profs = dict(existing)
        new_profs["default"] = new_default
        save_profiles(MODEL_DIR, name, new_profs)
        print(f"  Wrote {path}  (updated [default])")
    elif args.write_as:
        new_profs = dict(existing)
        copy = model_lib.copy_profile(new_default, args.write_as,
                                      description=new_default.description)
        new_profs[args.write_as] = copy
        path = save_profiles(MODEL_DIR, name, new_profs)
        print(f"  Wrote {path}  (added [{args.write_as}])")
    else:
        cmd = (f"  python {sys.argv[0]} plan {name} "
               f"--concurrency {p.concurrency} --apply")
        print(f"  To apply this to [default] (with backup):")
        print(cmd)
        cmd = (f"  python {sys.argv[0]} plan {name} "
               f"--concurrency {p.concurrency} --write-as planned")
        print(f"  Or save as a new variant:")
        print(cmd)


# =============================================================================
# BENCHMARK — load model+profile, measure steady-state tok/s, then hard-stop
# =============================================================================
# Wizard-driven throughput probe. Per (model, profile): launch the server, wait
# for readiness (surviving crashes/freezes), run a warm-up + N measured
# streaming generations, then HARD-stop everything (kill -9 of the process group
# + port holder, escalating to leftover vllm procs if VRAM stays stuck). The
# headline tok/s is the best sustained `window`-second slice, so the 10-30s
# mid-stream stalls vLLM sometimes shows don't poison the number.


class _BenchCfg:
    """Knobs for a benchmark run (CLI flags / the wizard fill these in)."""

    def __init__(self, prompt=BENCH_PROMPT, max_tokens=BENCH_MAX_TOKENS,
                 warmup=BENCH_WARMUP, runs=BENCH_RUNS, window=BENCH_WINDOW_SEC,
                 ready_timeout=BENCH_READY_TIMEOUT,
                 freeze_timeout=BENCH_FREEZE_TIMEOUT, stall_gap=BENCH_STALL_GAP,
                 escalate=True, save_json=True, concurrency=1):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.warmup = warmup
        self.runs = runs
        self.window = window
        self.ready_timeout = ready_timeout
        self.freeze_timeout = freeze_timeout
        self.stall_gap = stall_gap
        self.escalate = escalate
        self.save_json = save_json
        self.concurrency = max(1, int(concurrency))


# -- process / kill helpers ---------------------------------------------------

def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _hard_kill_pid(pid: int, sig=signal.SIGKILL):
    """Send `sig` to a pid AND its process group (reaps TP worker children)."""
    try:
        pgid = os.getpgid(pid)
    except (OSError, ProcessLookupError):
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
        except (OSError, ProcessLookupError):
            pass
    try:
        os.kill(pid, sig)
    except (OSError, ProcessLookupError):
        pass


def _port_pids(port: int) -> set[int]:
    """Pids holding a listening socket on `port` (via `ss`); empty if ss missing."""
    pids: set[int] = set()
    try:
        r = subprocess.run(["ss", "-tlnpH"], capture_output=True,
                           text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return pids
    addr_tok = re.compile(rf"(?:^|\s)\S*:{port}(?:\s|$)")
    for line in r.stdout.splitlines():
        if addr_tok.search(line):
            for m in re.finditer(r"pid=(\d+)", line):
                pids.add(int(m.group(1)))
    return pids


def _vllm_pids() -> list[int]:
    """Live pids that are clearly a vLLM server/worker.

    Deliberately precise: requires a python process AND a real vLLM invocation
    marker, so we never match an editor or `tail` that merely has 'vllm' in a
    path/arg (which kill -9 escalation would otherwise nuke). Excludes us.
    """
    out: list[int] = []
    me = os.getpid()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for e in entries:
        if not e.isdigit():
            continue
        pid = int(e)
        if pid == me:
            continue
        low = _proc_cmdline(pid).lower()
        if "python" not in low:
            continue
        if ("vllm.entrypoints" in low or "-m vllm" in low
                or "enginecore" in low or "vllm::" in low):
            out.append(pid)
    return out


def _target_gpu_free_mb(prof) -> int | None:
    """Min free VRAM (MB) across the profile's target GPUs, live; None if unknown."""
    g = {x["index"]: x for x in _gpus(refresh=True)}
    ids = [int(x) for x in prof.gpu.split(",") if x.strip().isdigit()]
    vals = [g[i]["mem_free_mb"] for i in ids if i in g]
    return min(vals) if vals else None


def _http_ok(port: int, path: str = "/v1/models", timeout: float = 2.0) -> bool:
    """Uncached readiness probe (avoids _probe_ready's TTL bleeding across restarts)."""
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _tail_lines(path: str, n: int = 12) -> list[str]:
    try:
        with open(path, errors="replace") as f:
            return [ln.rstrip() for ln in f.readlines()[-n:]]
    except OSError:
        return []


# -- measurement --------------------------------------------------------------

def _bench_steady_tps(token_times: list[float], window: float) -> float:
    """Best sustained tokens/sec over any `window`-second slice (stall-proof).

    A long mid-stream stall just shrinks the qualifying window to exclude the
    gap, so the reported rate reflects real decode speed, not the pause.
    """
    n = len(token_times)
    if n < 2:
        return 0.0
    best = 0.0
    j = 0
    for i in range(1, n):
        while token_times[i] - token_times[j] > window:
            j += 1
        span = token_times[i] - token_times[j]
        if span > 0:
            best = max(best, (i - j) / span)
    return best


def _bench_stream(port, model_name, prompt, max_tokens, freeze_timeout) -> dict:
    """One streaming generation. Never raises. Returns metrics + a per-token
    arrival timeline; on freeze/crash sets ['error'] but keeps the partial
    timeline. The socket read timeout (= freeze_timeout) is what turns a true
    hang into an abort while still tolerating the normal 10-30s stalls."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json", "Accept": "text/event-stream"})

    t0 = time.monotonic()
    token_times: list[float] = []
    ttft = None
    completion_tokens = prompt_tokens = None
    error = None
    resp = None
    try:
        resp = urllib.request.urlopen(req, timeout=freeze_timeout)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                if usage.get("completion_tokens") is not None:
                    completion_tokens = usage["completion_tokens"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("content")
                         or delta.get("reasoning_content")
                         or delta.get("reasoning") or "")
                if not piece:
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t0
                token_times.append(now)
    except (socket.timeout, TimeoutError):
        error = f"freeze (no data for {freeze_timeout}s)"
    except (urllib.error.URLError, ConnectionError) as e:
        error = f"connection: {getattr(e, 'reason', e)}"
    except Exception as e:  # never let a benchmark run take down the wizard
        error = f"{type(e).__name__}: {e}"
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    n = len(token_times)
    comp = completion_tokens if completion_tokens else n
    gen_span = (token_times[-1] - token_times[0]) if n >= 2 else 0.0
    gaps = [token_times[k + 1] - token_times[k] for k in range(n - 1)]
    return {
        "error": error, "ttft": ttft, "token_times": token_times,
        "chunk_tokens": n, "completion_tokens": comp,
        "prompt_tokens": prompt_tokens, "gen_span": gen_span, "gaps": gaps,
    }


def _bench_streams_concurrent(port, model_name, cfg, sample_cb=None) -> dict:
    """Run cfg.concurrency parallel streaming generations; merge the metrics.

    concurrency=1 degrades to a single _bench_stream. For N>1 the streams'
    token timelines are merged, so steady/mean tok/s measure AGGREGATE decode
    throughput across the batch (the number that tells you whether spec decode
    beats plain batching at this load). Each stream gets a slightly different
    prompt so prefix caching can't share the prefill across the batch.
    `sample_cb` is polled while streams run (GPU peak-VRAM sampling).
    """
    n = cfg.concurrency
    if n == 1:
        s = _bench_stream(port, model_name, cfg.prompt, cfg.max_tokens,
                          cfg.freeze_timeout)
        if sample_cb:
            sample_cb()
        return s

    from concurrent.futures import ThreadPoolExecutor
    prompts = [f"{cfg.prompt} [stream {k + 1}]" for k in range(n)]
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(_bench_stream, port, model_name, pr,
                          cfg.max_tokens, cfg.freeze_timeout)
                for pr in prompts]
        while any(not f.done() for f in futs):
            if sample_cb:
                sample_cb()
            time.sleep(1.0)
    streams = [f.result() for f in futs]

    merged = sorted(t for s in streams for t in s["token_times"])
    gaps = [merged[i + 1] - merged[i] for i in range(len(merged) - 1)]
    ttfts = [s["ttft"] for s in streams if s["ttft"] is not None]
    errors = sorted({s["error"] for s in streams if s["error"]})
    prompt_toks = [s["prompt_tokens"] for s in streams
                   if s["prompt_tokens"] is not None]
    return {
        "error": "; ".join(errors) or None,
        "ttft": (sum(ttfts) / len(ttfts)) if ttfts else None,
        "token_times": merged,
        "chunk_tokens": sum(s["chunk_tokens"] for s in streams),
        "completion_tokens": sum(s["completion_tokens"] or 0 for s in streams),
        "prompt_tokens": sum(prompt_toks) if prompt_toks else None,
        "gen_span": (merged[-1] - merged[0]) if len(merged) >= 2 else 0.0,
        "gaps": gaps,
    }


def _bench_wait_ready(proc, port, timeout, log) -> str:
    """Poll until ready, the process dies, or timeout. Returns one of
    'ready' | 'crashed' | 'timeout'."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return "crashed"
        if _http_ok(port):
            return "ready"
        time.sleep(2)
    return "timeout"


def _bench_stop(pid, port, prof, baseline_free_mb, *, escalate, log):
    """Hard-stop a benchmarked server: SIGTERM->SIGKILL the group, kill the port
    holder, and (if escalate and VRAM is still stuck) kill -9 leftover vllm."""
    _hard_kill_pid(pid, signal.SIGTERM)
    t = time.monotonic() + 5
    while time.monotonic() < t and _pid_alive(pid):
        time.sleep(0.3)
    if _pid_alive(pid):
        _hard_kill_pid(pid, signal.SIGKILL)

    for pp in _port_pids(port) - {pid}:
        log(f"    kill -9 port-{port} holder pid {pp}")
        _hard_kill_pid(pp, signal.SIGKILL)

    t = time.monotonic() + 12  # let the port actually free
    while time.monotonic() < t and _port_pids(port):
        time.sleep(0.5)

    if not escalate:
        return
    # Escalate only if the target GPUs' VRAM hasn't come back (orphaned workers
    # still resident). >2 GB short of baseline => kill leftover vllm procs.
    time.sleep(3)
    cur = _target_gpu_free_mb(prof)
    if (baseline_free_mb is not None and cur is not None
            and cur < baseline_free_mb - 2048):
        leftover = _vllm_pids()
        if leftover:
            log(f"    VRAM not released ({cur} MB free vs {baseline_free_mb} "
                f"baseline); kill -9 leftover vllm {leftover}")
            for lp in leftover:
                _hard_kill_pid(lp, signal.SIGKILL)
            time.sleep(2)


def _bench_one(name: str, profile_name: str, cfg: "_BenchCfg", log) -> dict:
    """Full start -> measure -> hard-stop cycle for one (model, profile).
    Never raises; records failures (crash/timeout/freeze) in the result."""
    res = {
        "model": name, "profile": profile_name, "status": "error", "detail": "",
        "concurrency": cfg.concurrency,
        "ttft_ms": None, "steady_tps": None, "mean_tps": None,
        "completion_tokens": None, "stalls": None, "max_gap_s": None,
        "load_s": None, "gpu_peak_mb": {}, "runs": [],
    }
    log(f"\n=== {name} / {profile_name}"
        + (f"  (concurrency {cfg.concurrency})" if cfg.concurrency > 1 else "")
        + " ===")

    try:
        prof = _resolve_profile(name, profile_name)
    except SystemExit:
        res["detail"] = "could not resolve profile"
        return res
    if prof.engine != "vllm":
        res["status"], res["detail"] = "skip", f"engine={prof.engine}"
        return res

    state = _load_state()
    if name in state and _pid_alive(state[name]["pid"]):
        log(f"  {name} already running — stopping it first ...")
        _bench_stop(state[name]["pid"], state[name]["port"], prof,
                    _target_gpu_free_mb(prof), escalate=cfg.escalate, log=log)
        state.pop(name, None)
        _save_state(state)

    serve_path = _resolve_serve_path(name)
    weight_gb = int(round(
        scan_local_model(MODEL_DIR, name).weight_bytes / (1024 ** 3)))
    port = _next_port(state)
    baseline_free = _target_gpu_free_mb(prof)

    # Track peak VRAM used on the profile's target GPUs (up to 4) across the run.
    target_ids = [int(x) for x in prof.gpu.split(",") if x.strip().isdigit()][:4]
    gpu_peak = {i: 0 for i in target_ids}
    res["gpu_peak_mb"] = gpu_peak  # same dict object; mutated in place below

    def _sample_gpu():
        info = {g["index"]: g for g in _gpus(refresh=True)}
        for i in target_ids:
            if i in info:
                used = info[i]["mem_total_mb"] - info[i]["mem_free_mb"]
                gpu_peak[i] = max(gpu_peak[i], used)

    for msg in _preflight_gpu_check(prof, weight_gb)[0]:
        log(f"  GPU check: {msg}")

    log_file = os.path.join(LOG_DIR, f"{name}.log")
    if os.path.isfile(log_file) and os.path.getsize(log_file) > 0:
        try:
            os.replace(log_file, log_file + ".1")
        except OSError:
            pass

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = prof.gpu
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
    env.update(prof.env)
    cmd = _build_vllm_cmd(serve_path, name, port, prof)

    log(f"  launching on port {port} (gpu {prof.gpu}, ctx {prof.max_model_len}) ...")
    t_launch = time.monotonic()
    try:
        lf = open(log_file, "w")
    except OSError as e:
        res["detail"] = f"cannot open log: {e}"
        return res
    try:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    except Exception as e:
        lf.close()
        res["detail"] = f"spawn failed: {e}"
        return res

    state[name] = {
        "pid": proc.pid, "port": port, "gpu": prof.gpu, "tp": prof.tp_size,
        "profile": prof.name, "weight_gb": weight_gb,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    _save_state(state)

    try:
        log(f"  waiting up to {cfg.ready_timeout}s for readiness ...")
        status = _bench_wait_ready(proc, port, cfg.ready_timeout, log)
        res["load_s"] = round(time.monotonic() - t_launch, 1)
        if status != "ready":
            res["status"] = "crash" if status == "crashed" else "timeout"
            res["detail"] = (f"exit code {proc.returncode}" if status == "crashed"
                             else f"not ready within {cfg.ready_timeout}s")
            log(f"  FAILED: {res['status']} ({res['detail']}). Last log lines:")
            for ln in _tail_lines(log_file, 12):
                log(f"    | {ln}")
            return res
        log(f"  ready in {res['load_s']}s — warm-up x{cfg.warmup}, "
            f"measured x{cfg.runs}")
        _sample_gpu()  # weights + KV pool now resident

        for _ in range(cfg.warmup):
            if proc.poll() is not None:
                break
            # Warm up at the measured concurrency so CUDA graphs / batch
            # paths for that batch size are captured before timing starts.
            _bench_streams_concurrent(port, name, cfg)

        measured = []
        for i in range(cfg.runs):
            if proc.poll() is not None:
                res["status"] = "crash"
                res["detail"] = f"server exited during run {i + 1}"
                log(f"  FAILED: server exited during run {i + 1}.")
                for ln in _tail_lines(log_file, 12):
                    log(f"    | {ln}")
                return res
            s = _bench_streams_concurrent(port, name, cfg,
                                          sample_cb=_sample_gpu)
            steady = _bench_steady_tps(s["token_times"], cfg.window)
            mean = (s["completion_tokens"] / s["gen_span"]
                    if s["gen_span"] > 0 else 0.0)
            stalls = sum(1 for g in s["gaps"] if g > cfg.stall_gap)
            max_gap = max(s["gaps"]) if s["gaps"] else 0.0
            rm = {
                "ttft_ms": round(s["ttft"] * 1000, 1) if s["ttft"] else None,
                "steady_tps": round(steady, 1), "mean_tps": round(mean, 1),
                "completion_tokens": s["completion_tokens"], "stalls": stalls,
                "max_gap_s": round(max_gap, 1), "error": s["error"],
            }
            res["runs"].append(rm)
            measured.append(rm)
            tag = f"  [{s['error']}]" if s["error"] else ""
            log(f"    run {i + 1}: steady {rm['steady_tps']} tok/s, mean "
                f"{rm['mean_tps']} tok/s, ttft {rm['ttft_ms']} ms, "
                f"{s['completion_tokens']} tok, {stalls} stall(s){tag}")
            _sample_gpu()  # capture activation high-water mark

        if gpu_peak:
            log("    GPU mem peak: " + ", ".join(
                f"GPU{i} {gpu_peak[i] / 1024:.1f} GiB" for i in target_ids))

        ok = [r for r in measured if r["error"] is None and r["steady_tps"]]
        pick = ok or measured
        if pick:
            res["steady_tps"] = max(r["steady_tps"] or 0 for r in pick)
            res["mean_tps"] = max(r["mean_tps"] or 0 for r in pick)
            ttfts = [r["ttft_ms"] for r in pick if r["ttft_ms"]]
            res["ttft_ms"] = round(sum(ttfts) / len(ttfts), 1) if ttfts else None
            res["completion_tokens"] = max((r["completion_tokens"] or 0)
                                           for r in pick)
            res["stalls"] = max((r["stalls"] or 0) for r in pick)
            res["max_gap_s"] = max((r["max_gap_s"] or 0) for r in pick)
        res["status"] = "ok" if ok else "degraded"
        if not ok:
            res["detail"] = "; ".join(sorted(
                {r["error"] for r in measured if r["error"]})) or "no tokens"
    finally:
        log(f"  stopping {name} (pid {proc.pid}) ...")
        _bench_stop(proc.pid, port, prof, baseline_free,
                    escalate=cfg.escalate, log=log)
        try:
            lf.close()
        except Exception:
            pass
        st = _load_state()
        st.pop(name, None)
        _save_state(st)
    return res


def _print_bench_table(results: list[dict], log):
    # Per-GPU peak-VRAM columns: union of target GPU indices across results,
    # capped at 4. Values are peak USED GiB on that physical GPU during the run.
    def _peak(r):
        return {int(k): v for k, v in (r.get("gpu_peak_mb") or {}).items()}
    gpu_ids = sorted({i for r in results for i in _peak(r)})[:4]
    gpu_hdr = "".join(f" {('G' + str(i) + ' GiB'):>9}" for i in gpu_ids)
    width = 99 + len(gpu_hdr)
    log("")
    log("=" * width)
    log(f"  {'Model':<22} {'Profile':<14} {'Status':<8} {'Conc':>4} {'Steady':>8} "
        f"{'Mean':>7} {'TTFT':>8} {'Tokens':>7} {'Stall':>6} {'Load':>6}{gpu_hdr}")
    log("  " + "-" * (width - 2))
    for r in results:
        steady = f"{r['steady_tps']:.1f}" if r['steady_tps'] else "-"
        mean = f"{r['mean_tps']:.1f}" if r['mean_tps'] else "-"
        ttft = f"{r['ttft_ms']:.0f}ms" if r['ttft_ms'] else "-"
        toks = r['completion_tokens'] if r['completion_tokens'] else "-"
        stall = r['stalls'] if r['stalls'] is not None else "-"
        load = f"{r['load_s']:.0f}s" if r['load_s'] is not None else "-"
        peak = _peak(r)
        gpu_cells = "".join(
            f" {(f'{peak[i] / 1024:.1f}' if peak.get(i) else '-'):>9}"
            for i in gpu_ids)
        conc = r.get("concurrency") or 1
        log(f"  {r['model'][:22]:<22} {r['profile'][:14]:<14} {r['status']:<8} "
            f"{conc:>4} {steady:>8} {mean:>7} {ttft:>8} {str(toks):>7} "
            f"{str(stall):>6} {load:>6}{gpu_cells}")
        if r['status'] != "ok" and r['detail']:
            log(f"      reason: {r['detail']}")
    log("  " + "-" * (width - 2))
    log("  Steady = best sustained tok/s over the sliding window (stall-proof);"
        " Mean = overall incl. stalls.")


def _run_benchmarks(targets: list[tuple], cfg: "_BenchCfg", log=print) -> list[dict]:
    """targets: [(model_name, [profile_names]), ...]. Runs sequentially."""
    total = sum(len(p) for _, p in targets)
    log(f"Benchmarking {total} profile-run(s) across {len(targets)} model(s).")
    log(f"  ~{cfg.max_tokens} tok/run, warm-up {cfg.warmup}, measured {cfg.runs}, "
        f"concurrency {cfg.concurrency}, window {cfg.window}s, "
        f"freeze>{cfg.freeze_timeout}s, "
        f"escalate={'on' if cfg.escalate else 'off'}")
    if cfg.concurrency > 1:
        log(f"  (tok/s figures are AGGREGATE across {cfg.concurrency} parallel "
            f"streams; each stream gets a distinct prompt)")
    results: list[dict] = []
    i = 0
    for name, profiles in targets:
        for pn in profiles:
            i += 1
            log(f"\n[{i}/{total}]")
            try:
                results.append(_bench_one(name, pn, cfg, log))
            except Exception as e:  # belt-and-suspenders; one run can't abort all
                log(f"  UNEXPECTED ERROR: {type(e).__name__}: {e}")
                results.append({
                    "model": name, "profile": pn, "status": "error",
                    "detail": f"{type(e).__name__}: {e}", "steady_tps": None,
                    "mean_tps": None, "ttft_ms": None, "completion_tokens": None,
                    "stalls": None, "max_gap_s": None, "load_s": None, "runs": [],
                })
    _print_bench_table(results, log)
    if cfg.save_json and results:
        _ensure_dirs()
        path = os.path.join(LOG_DIR, f"benchmark-{datetime.now():%Y%m%d-%H%M%S}.json")
        try:
            with open(path, "w") as f:
                json.dump({"config": vars(cfg), "results": results}, f, indent=2)
            log(f"\n  Results saved: {path}")
        except OSError as e:
            log(f"\n  (could not save results: {e})")
    return results


def cmd_benchmark(args):
    """CLI: benchmark one model's profiles (TUI wizard covers multi-model)."""
    _ensure_dirs()
    if not os.path.isfile(_venv_python()):
        print("venv not found. Run 'setup' first.")
        sys.exit(1)
    name = args.model
    if not _model_exists(name):
        print(f"No model directory at {_model_dir_path(name)}.")
        sys.exit(1)
    profs = load_profiles(MODEL_DIR, name)
    if not profs:
        print(f"No profiles for '{name}'. Create one with manage_models.py.")
        sys.exit(1)
    if args.profile:
        missing = [p for p in args.profile if p not in profs]
        if missing:
            print(f"Unknown profile(s): {missing}. Have: {sorted(profs)}")
            sys.exit(1)
        chosen = list(args.profile)
    else:
        chosen = sorted(profs.keys(), key=lambda n: (n != "default", n.lower()))

    cfg = _BenchCfg(
        max_tokens=args.max_tokens, warmup=args.warmup, runs=args.runs,
        window=args.window, ready_timeout=args.ready_timeout,
        freeze_timeout=args.freeze_timeout, escalate=not args.no_escalate,
        save_json=not args.no_save, concurrency=args.concurrency,
    )
    if args.prompt:
        cfg.prompt = args.prompt
    _run_benchmarks([(name, chosen)], cfg)


# =============================================================================
# INTERACTIVE TUI (curses)
# =============================================================================
# Launched when no CLI subcommand is given. Curses widgets/colours live in
# tui_lib.py — this section is just the menu wiring and per-action submenus.
# =============================================================================

def _tui_launch():
    """Entry point for the interactive menu."""
    tui.launch(_tui_main)


# Backwards-compat shims for the action functions that still reference the
# old `_tui_*` names directly. Cheaper than renaming every callsite below.
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


# -- TUI actions (each is a submenu) ------------------------------------------

def _tui_act_start(stdscr):
    """Start Model: pick model -> pick profile -> start."""
    import curses
    running = _get_running()
    models = scan_all(MODEL_DIR)

    if not models:
        curses.endwin()
        print("\nNo models installed.")
        print("  Download one: python manage_models.py download <hf-repo>")
        input("\nPress Enter to continue ...")
        stdscr.touchwin()
        stdscr.refresh()
        return

    # -- Step 1: choose model -------------------------------------------------
    items = []
    names = []
    for m in models:
        is_running = m.name in running
        tag = f"  (running :{running[m.name]['port']})" if is_running else ""
        weight_gb = m.weight_bytes / (1024 ** 3)
        attr = curses.color_pair(_C_GREEN) if is_running else 0
        label = (f"{m.name:<24} {m.kind:<5} "
                 f"~{weight_gb:>4.0f} GB weights{tag}")
        items.append((label, attr))
        names.append(m.name)

    items.append(("Back", curses.A_DIM))

    idx = _tui_select(stdscr, "Start Model \u2014 Select Model", items)
    if idx < 0 or idx >= len(names):
        return

    model_name = names[idx]

    if model_name in running:
        curses.endwin()
        print(f"\n'{model_name}' is already running on port "
              f"{running[model_name]['port']}.")
        input("\nPress Enter to continue ...")
        stdscr.touchwin()
        stdscr.refresh()
        return

    # -- Step 2: choose profile ----------------------------------------------
    profs = load_profiles(MODEL_DIR, model_name)
    if not profs:
        # Auto-seed a default profile and proceed.
        curses.endwin()
        print(f"\nNo profiles yet for '{model_name}'. Creating default ...")
        from model_lib import ensure_profiles_exist
        profs, _ = ensure_profiles_exist(MODEL_DIR, model_name, interactive=False)
        print(f"  Wrote {profiles_path(MODEL_DIR, model_name)}")
        input("\nPress Enter to continue ...")
        stdscr.touchwin()
        stdscr.refresh()

    prof_items = []
    prof_names = sorted(profs.keys(), key=lambda n: (n != "default", n.lower()))
    for pn in prof_names:
        p = profs[pn]
        prof_items.append(
            (f"{pn:<14} engine={p.engine}  tp={p.tp_size}  "
             f"gpu={p.gpu}  ctx={p.max_model_len}",
             0)
        )
    prof_items.append(("Back", curses.A_DIM))

    hdr = [(f"Model: {model_name}", curses.A_BOLD)]
    p_idx = _tui_select(stdscr, "Start Model \u2014 Select Profile",
                        prof_items, header=hdr)
    if p_idx < 0 or p_idx >= len(prof_names):
        return

    profile_name = prof_names[p_idx]
    chosen = profs[profile_name]

    # -- Step 3: optional per-launch toggles ----------------------------------
    no_reasoning = False
    has_reasoning = any(
        a == "--reasoning-parser" or a.startswith("--reasoning-parser=")
        for a in chosen.extra_args
    )
    if has_reasoning:
        explain = [
            f"Profile [{profile_name}] sets a reasoning parser.",
            "Reasoning surfaces a model's thinking as a separate field",
            "(e.g. <think>...</think>) instead of inline text.",
            "Turn OFF for faster, lower-token replies on simple chats.",
        ]
        keep = tui.confirm(stdscr, "Keep reasoning parser ON for this launch?",
                           default=True, explain=explain)
        no_reasoning = not keep

    # -- Step 4: run start ----------------------------------------------------
    _tui_run_cmd(stdscr, cmd_start,
                 argparse.Namespace(model=model_name, profile=profile_name,
                                    port=None, no_reasoning=no_reasoning,
                                    force=False, wait=0))


def _tui_act_stop(stdscr):
    """Stop Model: checkbox list of running models."""
    import curses
    running = _get_running()

    if not running:
        import curses as _c
        _c.endwin()
        print("\nNo models are currently running.")
        input("\nPress Enter to continue ...")
        stdscr.touchwin()
        stdscr.refresh()
        return

    items = []
    names = []
    for name, entry in sorted(running.items(), key=lambda x: x[1]["port"]):
        vram = int(entry.get("weight_gb") or 0)
        prof = entry.get("profile", "-")
        label = (f"{name:<24} [{prof}] :{entry['port']}  "
                 f"GPU {entry['gpu']}  ~{vram} GB")
        items.append((label, 0))
        names.append(name)

    selected = _tui_select(stdscr, "Stop Model \u2014 Toggle to Select",
                           items, multi=True)
    if not selected:
        return

    def _do_stop():
        for idx in selected:
            cmd_stop(argparse.Namespace(model=names[idx]))

    _tui_run_cmd(stdscr, _do_stop)


def _tui_act_test(stdscr):
    """Test Model: pick a running model."""
    import curses
    running = _get_running()

    if not running:
        import curses as _c
        _c.endwin()
        print("\nNo models are currently running.")
        input("\nPress Enter to continue ...")
        stdscr.touchwin()
        stdscr.refresh()
        return

    items = []
    names = []
    for name, entry in sorted(running.items(), key=lambda x: x[1]["port"]):
        label = f"{name:<26} :{entry['port']}  GPU {entry['gpu']}"
        items.append((label, 0))
        names.append(name)

    items.append(("Back", curses.A_DIM))

    idx = _tui_select(stdscr, "Test Model \u2014 Select Model", items)
    if idx < 0 or idx >= len(names):
        return

    _tui_run_cmd(stdscr, cmd_test, argparse.Namespace(model=names[idx]))


def _tui_act_status(stdscr):
    """Show full status (exits curses to print)."""
    _tui_run_cmd(stdscr, cmd_status, argparse.Namespace())


def _tui_act_logs(stdscr):
    """View Logs: pick a model, then tail -f (no pause after — Ctrl+C exits)."""
    import curses

    # Show all models that have log files (matches any *.log in LOG_DIR).
    items = []
    names = []
    running = _get_running()

    log_names: list[str] = []
    if os.path.isdir(LOG_DIR):
        for f in sorted(os.listdir(LOG_DIR)):
            if f.endswith(".log"):
                log_names.append(f[:-4])

    for name in log_names:
        is_running = name in running
        tag = f"  (running :{running[name]['port']})" if is_running else "  (stopped)"
        attr = curses.color_pair(_C_GREEN) if is_running else curses.A_DIM
        items.append((f"{name:<26}{tag}", attr))
        names.append(name)

    if not items:
        curses.endwin()
        print("\nNo log files found.")
        input("\nPress Enter to continue ...")
        stdscr.touchwin()
        stdscr.refresh()
        return

    items.append(("Back", curses.A_DIM))

    idx = _tui_select(stdscr, "View Logs \u2014 Select Model", items)
    if idx < 0 or idx >= len(names):
        return

    # tail -f doesn't need a pause — Ctrl+C exits back to curses
    _tui_run_cmd(stdscr, cmd_logs, argparse.Namespace(model=names[idx]),
                 pause_after=False)


def _tui_act_plan(stdscr):
    """Plan Sizing: pick a model, run cmd_plan to print recommendations."""
    import curses
    models = scan_all(MODEL_DIR)
    if not models:
        curses.endwin()
        print("\nNo models installed.")
        input("\nPress Enter to continue ...")
        stdscr.touchwin()
        stdscr.refresh()
        return

    items = []
    names = []
    for m in models:
        weight_gb = m.weight_bytes / (1024 ** 3)
        items.append((f"{m.name:<26} {m.kind:<5} ~{weight_gb:>4.0f} GB", 0))
        names.append(m.name)
    items.append(("Back", curses.A_DIM))

    idx = _tui_select(stdscr, "Plan Sizing — Select Model", items)
    if idx < 0 or idx >= len(names):
        return

    _tui_run_cmd(stdscr, cmd_plan, argparse.Namespace(
        model=names[idx], tp=None, concurrency=2, gpu_mem_util=0.90,
        no_tool_use=False, no_reasoning=False, apply=False, write_as=None,
    ))


def _tui_act_endpoints(stdscr):
    """Endpoints: show client config snippets for running models."""
    _tui_run_cmd(stdscr, cmd_endpoints, argparse.Namespace(model=None))


def _tui_act_edit_profile(stdscr):
    """Edit Profile: pick a model, shell out to manage_models.py profile edit."""
    import curses
    models = scan_all(MODEL_DIR)
    if not models:
        curses.endwin()
        print("\nNo models installed.")
        input("\nPress Enter to continue ...")
        stdscr.touchwin()
        stdscr.refresh()
        return

    items = []
    names = []
    for m in models:
        items.append((f"{m.name:<26} {m.kind:<5}", 0))
        names.append(m.name)
    items.append(("Back", curses.A_DIM))

    idx = _tui_select(stdscr, "Edit Profile — Select Model", items)
    if idx < 0 or idx >= len(names):
        return

    mgr = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "manage_models.py")
    tui.shell_out(stdscr,
                  [sys.executable, mgr, "profile", "edit", names[idx]])


def _tui_act_benchmark(stdscr):
    """Benchmark wizard: walk each model, space-select profiles, then run all."""
    import curses
    models = scan_all(MODEL_DIR)
    if not models:
        curses.endwin()
        print("\nNo models installed.")
        input("\nPress Enter to continue ...")
        stdscr.touchwin()
        stdscr.refresh()
        return

    # Walk each model; space-select profiles to benchmark (Esc/none = skip).
    targets: list[tuple] = []
    for m in models:
        profs = load_profiles(MODEL_DIR, m.name)
        if not profs:
            continue
        names = sorted(profs.keys(), key=lambda n: (n != "default", n.lower()))
        items = []
        for pn in names:
            p = profs[pn]
            items.append((f"{pn:<16} tp={p.tp_size}  gpu={p.gpu}  "
                          f"ctx={p.max_model_len}", 0))
        header = [
            (f"Model: {m.name}", curses.A_BOLD),
            ("Space toggles, Enter confirms, Esc skips this model.",
             curses.A_DIM),
        ]
        picks = _tui_select(stdscr, f"Benchmark — Select Profiles ({m.name})",
                            items, header=header, multi=True)
        if picks:
            targets.append((m.name, [names[i] for i in picks]))

    if not targets:
        curses.endwin()
        print("\nNo profiles selected — nothing to benchmark.")
        input("\nPress Enter to continue ...")
        stdscr.touchwin()
        stdscr.refresh()
        return

    total = sum(len(p) for _, p in targets)
    explain = ["Will start -> measure -> HARD-stop each of these:", ""]
    explain += [f"  {n}: {', '.join(ps)}" for n, ps in targets]
    explain += [
        "",
        f"{total} profile-run(s). Each: {BENCH_WARMUP} warm-up + {BENCH_RUNS} "
        f"runs, ~{BENCH_MAX_TOKENS} tok.",
        "Headline tok/s = best sustained 5s window (stall-proof).",
        "Cleanup: kill -9 process group + port holder, escalate if VRAM stuck.",
    ]
    if not tui.confirm(stdscr, "Start benchmark run?", default=True,
                       explain=explain):
        return

    _tui_run_cmd(stdscr, lambda: _run_benchmarks(targets, _BenchCfg()))


def _tui_act_setup(stdscr):
    """Run Setup: ask about V100, then run setup."""
    import curses

    items = [
        ("Standard (SM75+ GPUs: A100, RTX 3090, etc.)", 0),
        ("V100 mode (SM70 — installs 1Cat-vLLM fork for AWQ)", 0),
        ("Back", curses.A_DIM),
    ]

    idx = _tui_select(stdscr, "Setup \u2014 Select GPU Type", items)
    if idx < 0 or idx == 2:
        return

    v100 = (idx == 1)
    _tui_run_cmd(stdscr, cmd_setup, argparse.Namespace(v100=v100))


# -- TUI main loop -----------------------------------------------------------

def _tui_main(stdscr):
    """Main interactive menu loop."""
    import curses

    curses.curs_set(0)       # hide cursor
    stdscr.keypad(True)      # enable arrow keys
    tui.init_colors()

    if tui.too_small(stdscr, min_h=10, min_w=50):
        curses.endwin()
        print("Terminal too small (need at least 50x10).")
        print("Use CLI commands instead: python vllm_manager.py --help")
        return

    MENU_ACTIONS = [
        ("Start Model",        _tui_act_start),
        ("Stop Model",         _tui_act_stop),
        ("Status",             _tui_act_status),
        ("Test Model",         _tui_act_test),
        ("Benchmark",          _tui_act_benchmark),
        ("View Logs",          _tui_act_logs),
        ("Plan Sizing",        _tui_act_plan),
        ("Endpoints",          _tui_act_endpoints),
        ("Edit Profile",       _tui_act_edit_profile),
        ("Setup Environment",  _tui_act_setup),
        ("Quit",               None),
    ]

    refresh_ms = 5000

    def _build_header():
        running = _get_running()
        header = []
        ts = time.strftime("%H:%M:%S")
        every = f"{refresh_ms // 1000}s"

        # -- status line: running count + live update interval ------------
        if running:
            n_ready = sum(1 for e in running.values()
                          if _probe_ready(e["port"]))
            header.append((
                f"Running: {len(running)} model(s), {n_ready} ready"
                f"     \u23f1 {ts} \u00b7 refresh {every}",
                curses.A_BOLD | curses.color_pair(_C_GREEN)))
            for name, entry in sorted(running.items(),
                                      key=lambda x: x[1]["port"]):
                prof = entry.get("profile", "-")
                ready = _probe_ready(entry["port"])
                url = _endpoint_url(entry["port"])
                dattr = (curses.color_pair(_C_GREEN if ready else _C_CYAN)
                         | curses.A_BOLD)
                header.append([
                    ("  ", 0),
                    ("\u25cf", dattr),
                    (f" {name[:22]:<22} ", 0),
                    (f"[{prof[:10]:<10}] ", curses.color_pair(_C_DIM)),
                    (url, curses.color_pair(_C_CYAN)),
                    (f"  {'ready' if ready else 'loading'}", dattr),
                ])
        else:
            header.append((
                f"Running: none     \u23f1 {ts} \u00b7 refresh {every}",
                curses.A_DIM))

        # -- live GPU memory: vLLM (green) vs other (yellow) vs free ------
        header.append(("", 0))
        gsplit = _gpu_usage_split()
        if gsplit:
            header.append(("GPU memory (live, actual):", curses.A_BOLD))
            for g in gsplit:
                header.append(_gpu_bar_line(g))
            header.append([
                ("    ", 0),
                ("\u2588", curses.color_pair(_C_GREEN)),
                (" vLLM   ", curses.color_pair(_C_DIM)),
                ("\u2588", curses.color_pair(_C_YELLOW)),
                (" other   ", curses.color_pair(_C_DIM)),
                ("\u2591", curses.color_pair(_C_DIM)),
                (" free", curses.color_pair(_C_DIM)),
            ])
        else:
            header.append(("GPU memory: nvidia-smi unavailable",
                           curses.A_DIM))

        # -- system RAM ---------------------------------------------------
        tot, avail = _read_mem_gb()
        if tot:
            header.append(_ram_bar_line(tot - avail, tot))
        return header

    while True:
        header = _build_header()

        # -- build menu items ------------------------------------------------
        menu_items = [(label, 0) for label, _ in MENU_ACTIONS]
        # dim the Quit option
        menu_items[-1] = ("Quit", curses.A_DIM)

        def _refresh():
            return _build_header(), menu_items

        # -- show menu and get selection -------------------------------------
        idx = _tui_select(stdscr, "vLLM Service Manager", menu_items,
                          header=header,
                          refresh_cb=_refresh,
                          refresh_ms=refresh_ms)

        if idx == -1 or idx == len(MENU_ACTIONS) - 1:
            # ESC / q / "Quit"
            break

        action = MENU_ACTIONS[idx][1]
        if action:
            action(stdscr)


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="vLLM Service Manager \u2014 setup, serve, manage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            When run without a subcommand, an interactive TUI menu is shown.

            Examples:
              %(prog)s setup --v100                Setup venv with V100 AWQ support
              %(prog)s list                        Show installed models + profiles
              %(prog)s start qwen3.5-27b           Launch using the default profile
              %(prog)s start qwen3.6 --profile tool_use
              %(prog)s status                      Show running models
              %(prog)s stop qwen3.5-27b
              %(prog)s stop-all

            Downloads and profile authoring live in manage_models.py:
              python manage_models.py download unsloth/Qwen3.6-35B-A3B-GGUF
              python manage_models.py profile list qwen3.6
              python manage_models.py profile edit qwen3.6
        """),
    )
    sub = p.add_subparsers(dest="command")

    # setup
    sp = sub.add_parser("setup", help="Create venv and install vLLM")
    sp.add_argument(
        "--v100", action="store_true",
        help="Install 1Cat-vLLM fork for V100/SM70 AWQ support "
             "(standard vLLM AWQ requires SM75+)",
    )

    # start
    sp = sub.add_parser("start", help="Start serving a model using a profile")
    sp.add_argument("model", metavar="model",
                    help="Model folder name under MODEL_DIR")
    sp.add_argument("--profile", default="default",
                    help="Named profile from profiles.toml (default: 'default')")
    sp.add_argument("--port", type=int, default=None,
                    help="Override the auto-assigned port")
    sp.add_argument("--no-reasoning", action="store_true",
                    help="Strip --reasoning-parser from this launch only "
                         "(profile is not modified)")
    sp.add_argument("--force", action="store_true",
                    help="Skip the pre-flight GPU check and start anyway")
    sp.add_argument("--wait", type=int, default=0, metavar="SECS",
                    help="Block until /v1/models responds, up to SECS "
                         "(0 = only watch for an immediate crash)")

    # stop
    sp = sub.add_parser("stop", help="Stop a running model")
    sp.add_argument("model", metavar="model")
    sp.add_argument("--timeout", type=int, default=None,
                    help=f"Seconds to wait for SIGTERM before SIGKILL "
                         f"(default: {STOP_TIMEOUT_SEC})")

    # stop-all
    sp = sub.add_parser("stop-all", help="Stop all running models in parallel")
    sp.add_argument("--timeout", type=int, default=None,
                    help=f"Seconds to wait for SIGTERM before SIGKILL "
                         f"(default: {STOP_TIMEOUT_SEC})")

    # restart
    sp = sub.add_parser("restart",
                        help="Restart a model using its previous profile "
                             "(or --profile to change it)")
    sp.add_argument("model", metavar="model")
    sp.add_argument("--profile", default=None,
                    help="Override the profile used on restart")

    # status
    sub.add_parser("status", help="Show running models, ports, and GPU usage")

    # test
    sp = sub.add_parser("test", help="Test a running model with a prompt")
    sp.add_argument("model", metavar="model")

    # benchmark
    sp = sub.add_parser(
        "benchmark",
        help="Load profile(s), measure steady-state tok/s, then hard-stop")
    sp.add_argument("model", metavar="model")
    sp.add_argument("--profile", action="append", default=None, metavar="NAME",
                    help="Profile to benchmark (repeatable; default: all)")
    sp.add_argument("--prompt", default=None, help="Override the benchmark prompt")
    sp.add_argument("--max-tokens", type=int, default=BENCH_MAX_TOKENS)
    sp.add_argument("--runs", type=int, default=BENCH_RUNS,
                    help=f"Measured runs per profile (default: {BENCH_RUNS})")
    sp.add_argument("--warmup", type=int, default=BENCH_WARMUP,
                    help=f"Discarded warm-up runs (default: {BENCH_WARMUP})")
    sp.add_argument("--concurrency", type=int, default=1, metavar="N",
                    help="Parallel streams per measured run; tok/s is the "
                         "aggregate across streams (default: 1)")
    sp.add_argument("--window", type=float, default=BENCH_WINDOW_SEC,
                    help="Sliding window (s) for steady-state tok/s")
    sp.add_argument("--ready-timeout", type=int, default=BENCH_READY_TIMEOUT,
                    help="Max secs to wait for model readiness")
    sp.add_argument("--freeze-timeout", type=int, default=BENCH_FREEZE_TIMEOUT,
                    help="Abort a run if no token arrives for this many seconds")
    sp.add_argument("--no-escalate", action="store_true",
                    help="Don't escalate to kill -9 of leftover vllm if VRAM stuck")
    sp.add_argument("--no-save", action="store_true",
                    help="Don't write the JSON results file")

    # list
    sub.add_parser("list",
                   help="List models installed on disk (kind, quant, profiles)")

    # logs
    sp = sub.add_parser("logs", help="Tail log file for a model")
    sp.add_argument("model", metavar="model")

    # plan — sizing assistant
    sp = sub.add_parser(
        "plan",
        help="Recommend tp_size + max_model_len for a model on this host",
    )
    sp.add_argument("model", metavar="model")
    sp.add_argument("--tp", type=int, default=None,
                    help="Pin tp_size (default: smallest that fits)")
    sp.add_argument("--concurrency", type=int, default=2,
                    help="Concurrent sequences to size for (default: 2)")
    sp.add_argument("--gpu-mem-util", type=float, default=0.90,
                    help="GPU memory utilisation fraction (default: 0.90)")
    sp.add_argument("--no-tool-use", action="store_true",
                    help="Omit tool-call parser flags from extra_args")
    sp.add_argument("--no-reasoning", action="store_true",
                    help="Omit reasoning-parser flag from extra_args")
    grp = sp.add_mutually_exclusive_group()
    grp.add_argument("--apply", action="store_true",
                     help="Overwrite [default] in profiles.toml (backs up first)")
    grp.add_argument("--write-as", metavar="NAME", default=None,
                     help="Save recommendation as a new named profile")

    # endpoints — copy-paste client config snippets
    sp = sub.add_parser(
        "endpoints",
        help="Print copy-paste client config (OpenCode/Claude Code/Zed/Continue)",
    )
    sp.add_argument("model", nargs="?", default=None,
                    help="Limit to one model (default: all running)")

    args = p.parse_args()

    # No subcommand -> launch interactive TUI
    if not args.command:
        _tui_launch()
        return

    commands = {
        "setup":    cmd_setup,
        "start":    cmd_start,
        "stop":     cmd_stop,
        "stop-all": cmd_stop_all,
        "restart":  cmd_restart,
        "plan":     cmd_plan,
        "status":   cmd_status,
        "test":     cmd_test,
        "benchmark": cmd_benchmark,
        "list":     cmd_list,
        "logs":     cmd_logs,
        "endpoints": cmd_endpoints,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
