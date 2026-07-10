"""
model_lib.py — shared types/helpers for manage_models.py and vllm_manager.py.

Two concerns:
  1. Filesystem scanning of the models directory (LocalModel, scan_all).
  2. Per-model launch profiles in <model_dir>/<model>/profiles.toml.

Profile schema (TOML). Each [section] is a named profile; "default" is used
when no profile is specified at start time.

Stored fields:
    description    free-form note
    engine         "vllm" (only engine supported right now)
    tp_size        tensor-parallel size; must equal number of gpu ids
    gpu            comma-separated CUDA device ids (e.g. "0" or "0,1")
    dtype          "half" (fp16), "bfloat16", or "auto"
    gpu_mem_util   fraction of each GPU's VRAM vLLM may use (0.0-1.0)
    max_model_len  max context length in tokens
    extra_args     raw flag list appended to `vllm serve`
                   e.g. ["--reasoning-parser", "qwen3"]
    env            extra env vars passed to the server process
    launch_prefix  argv prepended to the server command, e.g.
                   ["numactl", "--cpunodebind=0", "--membind=0"] to pin the
                   whole server to one NUMA node on a multi-socket host

Intentionally NOT stored (derived at launch, would only cause drift):
    - model_path   the folder containing profiles.toml identifies the model
    - port         auto-assigned from BASE_PORT at start
    - host         global config in vllm_manager.py
"""

from __future__ import annotations
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field


# =============================================================================
# Filesystem scanning
# =============================================================================

@dataclass
class LocalModel:
    name: str
    path: str
    size_bytes: int       # full folder size (weights + configs + readme)
    weight_bytes: int     # weight files only (.safetensors / .bin / .gguf)
    kind: str             # "hf" | "gguf" | "mixed" | "empty"
    quant: str | None
    variants: int


SHARD_RE = re.compile(r"-0*(\d+)-of-0*(\d+)\.gguf$", re.IGNORECASE)
QUANT_RE = re.compile(
    r"(UD-(?:IQ|Q)\d+(?:_[A-Z0-9]+)*"
    r"|IQ\d+(?:_[A-Z0-9]+)*"
    r"|Q\d+(?:_[A-Z0-9]+)*"
    r"|BF16|F16|F32)",
    re.IGNORECASE,
)


def extract_quant_tag(path: str) -> str | None:
    stem = SHARD_RE.sub(".gguf", path)
    matches = QUANT_RE.findall(stem)
    if not matches:
        return None
    ud = [m for m in matches if m.upper().startswith("UD-")]
    if ud:
        return max(ud, key=len).upper()
    return max(matches, key=len).upper()


def _detect_quant_from_config(cfg_path: str) -> str | None:
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return None
    qc = cfg.get("quantization_config") or {}
    return (qc.get("quant_method")
            or qc.get("quantization_method")
            or cfg.get("quantization"))


def _detect_max_len_from_config(cfg_path: str) -> int | None:
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return None
    for key in ("max_position_embeddings", "model_max_length",
                "max_sequence_length"):
        val = cfg.get(key)
        if isinstance(val, int) and val > 0:
            return val
    return None


def _walk_model(root: str) -> tuple[int, int, list[str], list[str]]:
    """Return (total_bytes, weight_bytes, hf_dirs_with_config, gguf_paths)."""
    total = weight = 0
    hf_dirs: list[str] = []
    gguf_paths: list[str] = []
    for dp, _, fns in os.walk(root):
        if "config.json" in fns:
            hf_dirs.append(dp)
        for f in fns:
            fp = os.path.join(dp, f)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            total += sz
            lower = f.lower()
            if lower.endswith((".safetensors", ".bin", ".gguf")):
                weight += sz
            if lower.endswith(".gguf"):
                gguf_paths.append(fp)
    return total, weight, hf_dirs, gguf_paths


def scan_local_model(root: str, name: str) -> LocalModel:
    path = os.path.join(root, name)
    total, weight, hf_dirs, gguf_paths = _walk_model(path)

    if hf_dirs and gguf_paths:
        kind = "mixed"
    elif hf_dirs:
        kind = "hf"
    elif gguf_paths:
        kind = "gguf"
    else:
        kind = "empty"

    quant = None
    if hf_dirs:
        quant = _detect_quant_from_config(os.path.join(hf_dirs[0], "config.json"))
    if not quant and gguf_paths:
        quant = extract_quant_tag(gguf_paths[0])

    return LocalModel(
        name=name, path=path, size_bytes=total, weight_bytes=weight,
        kind=kind, quant=quant, variants=len(hf_dirs) + len(gguf_paths),
    )


def scan_all(root: str) -> list[LocalModel]:
    if not os.path.isdir(root):
        return []
    out = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            out.append(scan_local_model(root, entry))
    return out


def human_gb(n: int) -> str:
    gb = n / (1024 ** 3)
    if gb >= 10:
        return f"{gb:>6.0f} GB"
    return f"{gb:>6.1f} GB"


def detect_model_max_len(root: str, name: str) -> int | None:
    """Best-effort: look for config.json and read max_position_embeddings."""
    model_dir = os.path.join(root, name)
    if not os.path.isdir(model_dir):
        return None
    for dp, _, fns in os.walk(model_dir):
        if "config.json" in fns:
            v = _detect_max_len_from_config(os.path.join(dp, "config.json"))
            if v:
                return v
    return None


# =============================================================================
# Model family detection — drives auto-templating + sizing math
# =============================================================================
#
# Goal: from a model's config.json, classify the family and pull every field
# the launcher / sizing assistant needs. Must handle:
#
#   - Plain HF causal LMs:      keys at top level
#   - Vision-language models:   sizing keys live under text_config.*
#   - Hybrid attn (Qwen3.6):    layer_types + full_attention_interval
#                               (linear/Mamba layers carry no per-token KV)
#   - GGUF folders:             no config.json — return a stub family
#
# Returns FamilyInfo. None of these fields should ever be a guess; if we
# couldn't read it, leave it None and let callers handle it explicitly.

@dataclass
class FamilyInfo:
    family: str             # qwen3_moe | qwen3_dense | glm4_moe | glm4_moe_lite
                            # | llama | gguf | unknown
    arch: str | None        # first entry of architectures[]
    quant_method: str | None   # awq | gptq | bitsandbytes | None
    quant_bits: int | None
    is_moe: bool
    is_hybrid_attn: bool    # True if some layers don't keep per-token KV
    full_attn_layers: int   # count of layers whose KV cache scales with tokens
    num_hidden_layers: int | None
    # KV cache geometry — varies by attention style:
    #  - Standard MHA/GQA: per-layer KV bytes/token = 2 * num_kv_heads * head_dim * dtype_bytes
    #  - MLA (DeepSeek-style):     = (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
    attn_kind: str          # "mha" | "mla"
    num_kv_heads: int | None
    head_dim: int | None
    kv_lora_rank: int | None
    qk_rope_head_dim: int | None
    hidden_size: int | None
    max_position_embeddings: int | None
    dtype: str | None       # torch_dtype
    has_vision: bool
    raw: dict               # pruned copy of the relevant config block (for diagnostics)


_KNOWN_FAMILIES = {
    # arch substring -> family tag
    "GlmMoeDsaForCausalLM":               "glm5_moe",   # GLM-5.x (744B; MLA + DSA)
    "Glm4MoeLiteForCausalLM":             "glm4_moe_lite",
    "Glm4MoeForCausalLM":                 "glm4_moe",
    "Qwen3_5MoeForConditionalGeneration": "qwen3_moe",
    "Qwen3_5ForConditionalGeneration":    "qwen3_dense",
    "Qwen3MoeForCausalLM":                "qwen3_moe",
    "Qwen3NextForCausalLM":               "qwen3_moe",
    "Qwen3ForCausalLM":                   "qwen3_dense",
    "LlamaForCausalLM":                   "llama",
    "MistralForCausalLM":                 "mistral",
    "Glm4ForCausalLM":                    "glm4_dense",
}


def _full_attn_layer_count(text_cfg: dict) -> tuple[bool, int, int]:
    """
    Inspect layer_types / full_attention_interval to count layers whose KV
    cache scales with token count.

    Returns (is_hybrid, full_attn_layers, total_layers).
    """
    total = int(text_cfg.get("num_hidden_layers") or 0)
    layer_types = text_cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        full = sum(1 for t in layer_types if isinstance(t, str)
                   and "full" in t.lower() and "attention" in t.lower())
        hybrid = any("linear" in str(t).lower() or "mamba" in str(t).lower()
                     for t in layer_types)
        if hybrid and full > 0:
            return True, full, total
        if not hybrid and full > 0:
            return False, full, total
    interval = text_cfg.get("full_attention_interval")
    if isinstance(interval, int) and interval > 1 and total:
        # 1-in-N layers are full attention; rest are linear/Mamba.
        return True, max(1, total // interval), total
    return False, total, total


def _read_config(model_dir: str) -> dict | None:
    """Find the first config.json under model_dir and return it, or None."""
    if not os.path.isdir(model_dir):
        return None
    for dp, _, fns in os.walk(model_dir):
        if "config.json" in fns:
            try:
                with open(os.path.join(dp, "config.json")) as f:
                    return json.load(f)
            except Exception:
                return None
    return None


def detect_family(root: str, name: str) -> FamilyInfo:
    cfg = _read_config(os.path.join(root, name))
    if cfg is None:
        return FamilyInfo(
            family="gguf", arch=None, quant_method=None, quant_bits=None,
            is_moe=False, is_hybrid_attn=False, full_attn_layers=0,
            num_hidden_layers=None, attn_kind="mha", num_kv_heads=None,
            head_dim=None, kv_lora_rank=None, qk_rope_head_dim=None,
            hidden_size=None, max_position_embeddings=None, dtype=None,
            has_vision=False, raw={},
        )

    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else None

    family = "unknown"
    if arch:
        for needle, tag in _KNOWN_FAMILIES.items():
            if needle == arch:
                family = tag
                break
        else:
            for needle, tag in _KNOWN_FAMILIES.items():
                if needle.lower() in arch.lower():
                    family = tag
                    break

    # Vision/conditional-generation models stash text-side keys in text_config.
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    has_vision = "vision_config" in cfg

    qc = cfg.get("quantization_config") or text_cfg.get("quantization_config") or {}
    quant_method = qc.get("quant_method") or qc.get("quantization_method")
    quant_bits = qc.get("bits") if isinstance(qc.get("bits"), int) else None

    # Detect MLA (DeepSeek-style multi-head latent attention). Models with
    # kv_lora_rank store one compressed KV vector per token instead of K+V
    # per head — completely different memory math from MHA/GQA.
    kv_lora_rank = text_cfg.get("kv_lora_rank")
    qk_rope_head_dim = text_cfg.get("qk_rope_head_dim")
    if isinstance(kv_lora_rank, int) and kv_lora_rank > 0:
        attn_kind = "mla"
        # MLA still has num_attention_heads but only the latent vector caches.
        head_dim = text_cfg.get("v_head_dim") or text_cfg.get("qk_nope_head_dim")
    else:
        attn_kind = "mha"
        head_dim = text_cfg.get("head_dim")
        if head_dim is None:
            h = text_cfg.get("hidden_size")
            a = text_cfg.get("num_attention_heads")
            if isinstance(h, int) and isinstance(a, int) and a > 0:
                head_dim = h // a

    is_moe = ("num_experts" in text_cfg
              or "n_routed_experts" in text_cfg
              or family.endswith("_moe")
              or family.endswith("_moe_lite"))

    is_hybrid, full_layers, total = _full_attn_layer_count(text_cfg)

    raw = {k: text_cfg.get(k) for k in (
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "num_experts",
        "num_experts_per_tok", "max_position_embeddings",
        "full_attention_interval", "kv_lora_rank", "qk_rope_head_dim",
        "v_head_dim", "qk_nope_head_dim",
    ) if k in text_cfg}

    return FamilyInfo(
        family=family,
        arch=arch,
        quant_method=quant_method,
        quant_bits=quant_bits,
        is_moe=is_moe,
        is_hybrid_attn=is_hybrid,
        full_attn_layers=full_layers,
        num_hidden_layers=total or None,
        attn_kind=attn_kind,
        num_kv_heads=text_cfg.get("num_key_value_heads"),
        head_dim=head_dim,
        kv_lora_rank=kv_lora_rank if isinstance(kv_lora_rank, int) else None,
        qk_rope_head_dim=qk_rope_head_dim if isinstance(qk_rope_head_dim, int) else None,
        hidden_size=text_cfg.get("hidden_size"),
        max_position_embeddings=text_cfg.get("max_position_embeddings"),
        dtype=text_cfg.get("dtype") or text_cfg.get("torch_dtype")
              or cfg.get("torch_dtype"),
        has_vision=has_vision,
        raw=raw,
    )


# =============================================================================
# Auto-templates: map (family, quant, vllm_version) -> extra_args list
# =============================================================================
#
# Single source of truth for "what flags does this model want by default".
# Centralised so manage_models.py (profile creation) and vllm_manager.py
# (sizing assistant, per-launch overrides) stay in sync.

# Family -> (tool_parser, reasoning_parser). None means the family doesn't
# support that capability in any currently shipping vLLM.
_FAMILY_PARSERS: dict[str, tuple[str | None, str | None]] = {
    "glm5_moe":      ("glm47", "glm45"),
    "glm4_moe":      ("glm47", "glm45"),
    "glm4_moe_lite": ("glm47", "glm45"),
    "glm4_dense":    ("glm47", "glm45"),
    "qwen3_moe":     ("hermes", "qwen3"),
    "qwen3_dense":   ("hermes", "qwen3"),
    "llama":         ("llama3_json", None),
    "mistral":       ("mistral", "mistral"),
    "unknown":       (None, None),
    "gguf":          (None, None),
}


def family_parsers(family: str) -> tuple[str | None, str | None]:
    """Return (tool_parser, reasoning_parser) names for a family."""
    return _FAMILY_PARSERS.get(family, (None, None))


def suggest_extra_args(
    info: FamilyInfo,
    *,
    tp_size: int = 1,
    tool_use: bool = True,
    reasoning: bool = True,
    v100: bool = False,
    available_tool_parsers: set[str] | None = None,
    available_reasoning_parsers: set[str] | None = None,
) -> list[str]:
    """
    Build the recommended `extra_args` list for a profile.

    Honours the user's tool_use / reasoning toggles and gates each parser by
    the registry probed at setup time (so we never write a flag the running
    vLLM build can't load). Pass v100=True on SM70 hosts so the result is
    actually runnable on the 1Cat fork (dedicated attention backend, no
    expert-parallel — which is experimental there).
    """
    args: list[str] = []

    # V100 (SM70): the fork serves attention through a dedicated backend; pin it
    # so vLLM doesn't try to select an SM80+ one (FlashInfer/awq_marlin/etc.).
    # MLA models (kv_lora_rank set, e.g. GLM-4.7-Flash / DeepSeek) use a
    # different attention path — the fork's FLASH_ATTN_V100 kernel is
    # standard-MHA only, so leave the backend unpinned and let vLLM pick
    # (typically Triton MLA); pinning it would fail at load.
    if v100 and info.attn_kind != "mla":
        args += ["--attention-backend", "FLASH_ATTN_V100"]

    # Quantisation: always be explicit for AWQ — the only kernel V100 (SM70)
    # supports is the legacy GEMM AWQ; awq_marlin needs SM 8.0+.
    if info.quant_method == "awq":
        args += ["--quantization", "awq"]
    elif info.quant_method == "gptq":
        args += ["--quantization", "gptq"]

    tool_parser, reasoning_parser = family_parsers(info.family)

    if tool_use and tool_parser:
        if available_tool_parsers is None or tool_parser in available_tool_parsers:
            args += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]

    if reasoning and reasoning_parser:
        if (available_reasoning_parsers is None
                or reasoning_parser in available_reasoning_parsers):
            args += ["--reasoning-parser", reasoning_parser]

    # MoE + tensor-parallel: experts are partitioned across GPUs differently
    # from pure TP; this is faster for most layouts. On V100 the fork's
    # validated MoE command omits it (experimental there), so skip it then.
    if info.is_moe and tp_size > 1 and not v100:
        args += ["--enable-expert-parallel"]

    return args


# =============================================================================
# vLLM-version + parser-registry probe
# =============================================================================
#
# Reads the snapshot vllm_manager.py writes at end of `setup`. Used by the
# auto-templater so we never write a parser name the installed build can't
# resolve. Cached on disk so we don't pay venv-import latency per CLI call.

def read_vllm_version(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


def probe_parsers(venv_python: str) -> tuple[set[str], set[str]]:
    """
    Best-effort: spawn the venv interpreter to dump tool/reasoning parser
    names. Returns ({tool_parsers}, {reasoning_parsers}) or (set(), set())
    if anything fails. Caller decides what to do with empty sets.
    """
    import subprocess
    code = (
        "import json, sys\n"
        "out = {'tool': [], 'reasoning': []}\n"
        "try:\n"
        "    from vllm.tool_parsers import ToolParserManager as _T\n"
        "    out['tool'] = sorted(set(_T.tool_parsers) | set(_T.lazy_parsers))\n"
        "except Exception: pass\n"
        "try:\n"
        "    from vllm.reasoning import ReasoningParserManager as _R\n"
        "    eager = getattr(_R, 'reasoning_parsers', {})\n"
        "    lazy  = getattr(_R, 'lazy_parsers', {})\n"
        "    out['reasoning'] = sorted(set(eager) | set(lazy))\n"
        "except Exception: pass\n"
        "json.dump(out, sys.stdout)\n"
    )
    try:
        r = subprocess.run([venv_python, "-c", code],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return set(), set()
        data = json.loads(r.stdout)
        return set(data.get("tool", [])), set(data.get("reasoning", []))
    except Exception:
        return set(), set()


def cache_parsers(path: str, tool: set[str], reasoning: set[str]) -> None:
    """Stash the registry alongside the version file (as JSON)."""
    try:
        with open(path, "w") as f:
            json.dump({"tool": sorted(tool), "reasoning": sorted(reasoning)}, f)
    except OSError:
        pass


def read_cached_parsers(path: str) -> tuple[set[str], set[str]]:
    try:
        with open(path) as f:
            data = json.load(f)
        return set(data.get("tool", [])), set(data.get("reasoning", []))
    except (OSError, ValueError):
        return set(), set()


# =============================================================================
# Sizing assistant
# =============================================================================
#
# Single-user / low-concurrency sizing for V100-class hardware. The math:
#
#   weight_gb_per_gpu  = weight_bytes / (1024^3 * tp_size)
#   per_gpu_budget_gb  = vram_per_gpu_gb * gpu_mem_util
#   activation_overhead = ~2 GB per GPU (worker/activations/cudagraphs)
#   kv_budget_per_gpu  = per_gpu_budget_gb - weight_gb_per_gpu - activation_overhead
#   kv_per_token_bytes = (see below; depends on attn_kind, hybrid, dtype)
#   max_tokens_total   = kv_budget_per_gpu * tp_size * 1024^3 / kv_per_token_bytes
#                        (MLA: no tp_size factor — the latent KV is replicated
#                         on every TP rank, so one GPU's budget is the cap)
#   recommended_max_len = max_tokens_total / concurrency
#
# KV cache per token (across the model, ALL gpus combined; tp splits it):
#   MHA/GQA:   2 * full_attn_layers * num_kv_heads * head_dim * dtype_bytes
#   MLA:           full_attn_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
#
# For hybrid models (Qwen3.6), only `full_attn_layers` (counted by
# detect_family) contribute. Linear-attention layers carry constant SSM
# state, not per-token cache.

# Common context lengths to round recommendations to.
_NICE_CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072, 200000, 262144]

# Bytes per element for KV cache. vLLM defaults match weight dtype on V100
# (no fp8 KV — that's SM89+).
_DTYPE_BYTES = {
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
    "float32": 4, "fp32": 4,
}


@dataclass
class Plan:
    fits: bool
    tp_size: int
    weight_gb_total: float
    weight_gb_per_gpu: float
    per_gpu_budget_gb: float
    activation_reserve_gb: float
    kv_budget_per_gpu_gb: float
    kv_bytes_per_token: int     # combined across all full-attention layers
    max_tokens_at_concurrency: int
    recommended_max_len: int
    requested_max_len: int      # what we'd cap to (model's max_position_embeddings)
    concurrency: int
    gpu_mem_util: float
    notes: list[str]            # human-readable observations / warnings


def _kv_bytes_per_token(info: FamilyInfo) -> int:
    """Total per-token KV cache bytes across all full-attention layers."""
    layers = info.full_attn_layers or info.num_hidden_layers or 0
    bytes_per_elem = _DTYPE_BYTES.get((info.dtype or "float16").lower(), 2)
    if info.attn_kind == "mla":
        per_layer_elems = (info.kv_lora_rank or 0) + (info.qk_rope_head_dim or 0)
        return layers * per_layer_elems * bytes_per_elem
    # MHA/GQA: K and V each have num_kv_heads * head_dim elements per layer
    kv_heads = info.num_kv_heads or 0
    head_dim = info.head_dim or 0
    return 2 * layers * kv_heads * head_dim * bytes_per_elem


def _round_down_to_nice(n: int) -> int:
    """Round n down to a nice context length (4k/8k/16k/32k/...)."""
    if n <= 0:
        return 0
    candidates = [c for c in _NICE_CONTEXTS if c <= n]
    if candidates:
        return max(candidates)
    # Below 4k — round down to nearest 1k.
    return max(1024, (n // 1024) * 1024)


def compute_plan(
    model: LocalModel,
    info: FamilyInfo,
    *,
    vram_per_gpu_gb: int = 32,
    num_gpus: int = 2,
    tp_size: int | None = None,
    gpu_mem_util: float = 0.90,
    concurrency: int = 2,
    activation_reserve_gb: float = 2.0,
    cap_to_model_max: bool = True,
) -> Plan:
    """
    Recommend tp_size + max_model_len for `model` on the given GPU pool.

    If tp_size is None, picks the smallest tp_size in {1, 2, ..., num_gpus}
    that fits the weights + activation reserve + at least 4k tokens of KV
    headroom; falls back to num_gpus with a warning if even that doesn't fit.
    """
    notes: list[str] = []
    weight_gb_total = model.weight_bytes / (1024 ** 3)

    # Pick tp_size if not given.
    if tp_size is None:
        tp_size = 1
        for tp in range(1, num_gpus + 1):
            per_gpu = (weight_gb_total / tp) + activation_reserve_gb
            budget = vram_per_gpu_gb * gpu_mem_util
            if per_gpu + 4.0 <= budget:  # 4 GB minimum KV headroom
                tp_size = tp
                break
        else:
            tp_size = num_gpus
            notes.append(
                f"Even with tp_size={num_gpus}, weight+overhead exceeds "
                f"{gpu_mem_util:.0%} of {vram_per_gpu_gb} GB per GPU."
            )

    weight_gb_per_gpu = weight_gb_total / tp_size
    per_gpu_budget_gb = vram_per_gpu_gb * gpu_mem_util
    kv_budget_per_gpu_gb = per_gpu_budget_gb - weight_gb_per_gpu - activation_reserve_gb

    if kv_budget_per_gpu_gb <= 0:
        notes.append("Negative KV budget — model won't fit at this gpu_mem_util.")
        return Plan(
            fits=False, tp_size=tp_size,
            weight_gb_total=weight_gb_total, weight_gb_per_gpu=weight_gb_per_gpu,
            per_gpu_budget_gb=per_gpu_budget_gb,
            activation_reserve_gb=activation_reserve_gb,
            kv_budget_per_gpu_gb=kv_budget_per_gpu_gb,
            kv_bytes_per_token=0, max_tokens_at_concurrency=0,
            recommended_max_len=0,
            requested_max_len=info.max_position_embeddings or 0,
            concurrency=concurrency, gpu_mem_util=gpu_mem_util, notes=notes,
        )

    kv_bytes_per_token = _kv_bytes_per_token(info)
    if kv_bytes_per_token == 0:
        notes.append("Unknown attention geometry — can't size KV cache. "
                     "Defaulting to model's max_position_embeddings.")
        recommended = info.max_position_embeddings or DEFAULT_MAX_MODEL_LEN
        return Plan(
            fits=True, tp_size=tp_size,
            weight_gb_total=weight_gb_total, weight_gb_per_gpu=weight_gb_per_gpu,
            per_gpu_budget_gb=per_gpu_budget_gb,
            activation_reserve_gb=activation_reserve_gb,
            kv_budget_per_gpu_gb=kv_budget_per_gpu_gb,
            kv_bytes_per_token=0,
            max_tokens_at_concurrency=0,
            recommended_max_len=recommended,
            requested_max_len=info.max_position_embeddings or 0,
            concurrency=concurrency, gpu_mem_util=gpu_mem_util, notes=notes,
        )

    # MHA/GQA: KV cache is sharded across tp_size GPUs (vLLM splits by
    # num_kv_heads), so the pool is per-gpu-budget * tp_size.
    # MLA: there is ONE compressed latent per token and vLLM replicates it on
    # every TP rank — capacity is bounded by a single GPU's budget, and TP>1
    # buys zero extra KV room (only weight/compute sharding).
    if info.attn_kind == "mla":
        kv_total_budget_bytes = int(kv_budget_per_gpu_gb * (1024 ** 3))
        if tp_size > 1:
            notes.append(
                "MLA KV cache is replicated across TP ranks — capacity is one "
                "GPU's KV budget, not the sum. TP>1 still shards the weights.")
    else:
        kv_total_budget_bytes = int(kv_budget_per_gpu_gb * tp_size * (1024 ** 3))
    max_tokens_total = kv_total_budget_bytes // kv_bytes_per_token
    max_tokens_at_concurrency = max_tokens_total // max(1, concurrency)

    if cap_to_model_max and info.max_position_embeddings:
        max_tokens_at_concurrency = min(
            max_tokens_at_concurrency, info.max_position_embeddings
        )

    recommended = _round_down_to_nice(max_tokens_at_concurrency)
    if recommended < 4096:
        notes.append(f"Tight: only ~{max_tokens_at_concurrency} tokens fit at "
                     f"concurrency={concurrency}; consider --concurrency 1.")

    if (info.max_position_embeddings
            and recommended >= info.max_position_embeddings):
        notes.append(f"Capped to model's max_position_embeddings "
                     f"({info.max_position_embeddings}); KV budget allows more.")

    return Plan(
        fits=True, tp_size=tp_size,
        weight_gb_total=weight_gb_total, weight_gb_per_gpu=weight_gb_per_gpu,
        per_gpu_budget_gb=per_gpu_budget_gb,
        activation_reserve_gb=activation_reserve_gb,
        kv_budget_per_gpu_gb=kv_budget_per_gpu_gb,
        kv_bytes_per_token=kv_bytes_per_token,
        max_tokens_at_concurrency=max_tokens_at_concurrency,
        recommended_max_len=recommended,
        requested_max_len=info.max_position_embeddings or 0,
        concurrency=concurrency, gpu_mem_util=gpu_mem_util, notes=notes,
    )


# =============================================================================
# Profiles
# =============================================================================

PROFILES_FILE = "profiles.toml"
SUPPORTED_ENGINES = ("vllm",)
DEFAULT_MAX_MODEL_LEN = 32768


@dataclass
class Profile:
    name: str
    description: str = "Baseline launch config. Copy this block to add a variant."
    engine: str = "vllm"
    port: int = 0            # fixed listen port; 0 = auto-assign from BASE_PORT
    tp_size: int = 1
    gpu: str = "0"
    dtype: str = "half"
    gpu_mem_util: float = 0.90
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    launch_prefix: list[str] = field(default_factory=list)


def profiles_path(model_dir: str, model_name: str) -> str:
    return os.path.join(model_dir, model_name, PROFILES_FILE)


def load_profiles(model_dir: str, model_name: str) -> dict[str, Profile]:
    path = profiles_path(model_dir, model_name)
    if not os.path.isfile(path):
        return {}
    with open(path, "rb") as f:
        data = tomllib.load(f)
    out: dict[str, Profile] = {}
    for pname, d in data.items():
        if not isinstance(d, dict):
            continue
        out[pname] = _profile_from_dict(pname, d)
    return out


def _profile_from_dict(name: str, d: dict) -> Profile:
    return Profile(
        name=name,
        description=str(d.get("description", "")),
        engine=str(d.get("engine", "vllm")),
        port=int(d.get("port", 0) or 0),
        tp_size=int(d.get("tp_size", 1)),
        gpu=str(d.get("gpu", "0")),
        dtype=str(d.get("dtype", "half")),
        gpu_mem_util=float(d.get("gpu_mem_util", 0.90)),
        max_model_len=int(d.get("max_model_len", DEFAULT_MAX_MODEL_LEN)),
        extra_args=[str(x) for x in (d.get("extra_args") or [])],
        env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
        launch_prefix=[str(x) for x in (d.get("launch_prefix") or [])],
    )


def save_profiles(model_dir: str, model_name: str,
                  profiles: dict[str, Profile]) -> str:
    path = profiles_path(model_dir, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = dump_profiles_toml(profiles)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def make_default_profile(**overrides) -> Profile:
    kwargs = {"name": "default"}
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return Profile(**kwargs)


def copy_profile(src: Profile, new_name: str,
                 description: str | None = None) -> Profile:
    d = asdict(src)
    d["name"] = new_name
    d["description"] = description if description is not None \
        else f"Copied from '{src.name}'."
    # Deep-ish copy for mutable fields.
    d["extra_args"] = list(src.extra_args)
    d["env"] = dict(src.env)
    d["launch_prefix"] = list(src.launch_prefix)
    return Profile(**d)


def ensure_profiles_exist(
    model_dir: str, model_name: str,
    *, interactive: bool = False,
    suggested_max_len: int | None = None,
    available_tool_parsers: set[str] | None = None,
    available_reasoning_parsers: set[str] | None = None,
    tool_use: bool = True,
    reasoning: bool = True,
    v100: bool = False,
) -> tuple[dict[str, Profile], bool]:
    """Create profiles.toml with a [default] profile if none exists yet.

    The default's extra_args are auto-populated from family detection (AWQ
    flag, tool/reasoning parsers, expert-parallel for MoE+TP). Pass
    `tool_use` / `reasoning` to opt out per profile.

    Returns (profiles_dict, created) where `created` is True iff the file
    was newly written.
    """
    existing = load_profiles(model_dir, model_name)
    if existing:
        return existing, False

    if suggested_max_len is None:
        suggested_max_len = (detect_model_max_len(model_dir, model_name)
                             or DEFAULT_MAX_MODEL_LEN)

    overrides: dict = {"max_model_len": suggested_max_len}
    if interactive:
        overrides.update(_prompt_profile_defaults(suggested_max_len))

    info = detect_family(model_dir, model_name)
    tp_size = int(overrides.get("tp_size", 1))
    extra = suggest_extra_args(
        info,
        tp_size=tp_size,
        tool_use=tool_use,
        reasoning=reasoning,
        v100=v100,
        available_tool_parsers=available_tool_parsers,
        available_reasoning_parsers=available_reasoning_parsers,
    )
    if extra:
        overrides["extra_args"] = extra

    prof = make_default_profile(**overrides)
    profiles = {"default": prof}
    save_profiles(model_dir, model_name, profiles)
    return profiles, True


def _prompt_profile_defaults(suggested_max_len: int) -> dict:
    print()
    print("  Setting up default launch profile (profiles.toml).")
    print("  Press Enter to accept each default; edit the file later for fine-tuning.")
    print()

    out: dict = {}

    raw = _prompt(f"  Max context length in tokens [{suggested_max_len}]: ")
    if raw:
        try:
            out["max_model_len"] = int(raw)
        except ValueError:
            print(f"    (not an integer, keeping {suggested_max_len})")

    raw = _prompt("  Tensor parallel size (number of GPUs to split across) [1]: ")
    tp = 1
    if raw:
        try:
            tp = int(raw)
            out["tp_size"] = tp
        except ValueError:
            pass

    default_gpu = "0" if tp == 1 else ",".join(str(i) for i in range(tp))
    raw = _prompt(f"  GPU ids (comma-separated) [{default_gpu}]: ")
    out["gpu"] = raw or default_gpu

    raw = _prompt("  GPU memory utilization fraction [0.95]: ")
    if raw:
        try:
            out["gpu_mem_util"] = float(raw)
        except ValueError:
            pass

    raw = _prompt("  Fixed listen port (blank/0 = auto-assign): ")
    if raw:
        try:
            out["port"] = int(raw)
        except ValueError:
            pass

    return out


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except EOFError:
        return ""


# =============================================================================
# TOML writer (minimal, tailored to our schema)
# =============================================================================

_IDENT_RE = re.compile(r"[A-Za-z0-9_-]+")


def _toml_str(s: str) -> str:
    # JSON strings are a TOML-compatible subset for the chars we handle.
    return json.dumps(s, ensure_ascii=False)


def _toml_key(k: str) -> str:
    if _IDENT_RE.fullmatch(k):
        return k
    return _toml_str(k)


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return _toml_str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        items = ", ".join(f"{_toml_key(k)} = {_toml_value(val)}"
                          for k, val in v.items())
        return "{" + items + "}"
    raise TypeError(f"Cannot TOML-encode {type(v).__name__}")


_FIELD_ORDER = [
    "description", "engine", "port", "tp_size", "gpu", "dtype",
    "gpu_mem_util", "max_model_len", "extra_args", "env", "launch_prefix",
]

_FIELD_NOTE = {
    "description":   "free-form note",
    "engine":        "'vllm' (only engine supported today)",
    "port":          "fixed listen port; 0 = auto-assign from BASE_PORT",
    "tp_size":       "tensor-parallel size; must equal number of gpu ids",
    "gpu":           "comma-separated CUDA device ids",
    "dtype":         "'half' (fp16), 'bfloat16', or 'auto'",
    "gpu_mem_util":  "fraction of each GPU's VRAM vLLM may use",
    "max_model_len": "max context length; trim to reduce KV-cache VRAM",
    "extra_args":    "raw flags appended to 'vllm serve'",
    "env":           "extra env vars for the server process",
    "launch_prefix": "argv prepended to the command, e.g. numactl pinning",
}

_FILE_HEADER = """\
# profiles.toml  -  launch profiles for this model.
#
# Each [section] is a named profile. "default" is used when --profile is
# omitted. Copy a section to a new name to make a variant, e.g. [tool_use].
#
# Derived at launch (not model_path/host):
#   model_path  - implied by this file's location
#   host        - global config in vllm_manager.py
#   port        - `port = 0` (or omitted) auto-assigns from BASE_PORT; set a
#                 non-zero `port` to pin this model to a fixed port.
#
# Rule: `gpu` and `tp_size` must agree. "0,1" + tp_size=2 is valid;
# "0" + tp_size=2 is not.

"""


def dump_profiles_toml(profiles: dict[str, Profile]) -> str:
    # "default" first, rest alphabetical.
    names = sorted(profiles.keys(), key=lambda n: (n != "default", n.lower()))
    parts = [_FILE_HEADER]
    for name in names:
        p = profiles[name]
        parts.append(f"[{_toml_key(name)}]")
        d = asdict(p)
        d.pop("name", None)
        for field_name in _FIELD_ORDER:
            if field_name not in d:
                continue
            val = d[field_name]
            note = _FIELD_NOTE.get(field_name, "")
            line = f"{field_name} = {_toml_value(val)}"
            if note:
                line += f"  # {note}"
            parts.append(line)
        parts.append("")
    return "\n".join(parts)
