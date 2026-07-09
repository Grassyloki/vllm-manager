# Guide: GLM-4.7-Flash on 2x V100 (32 GB)

The GLM model served on this box. Chosen as the stand-in for GLM-5.2, which
**cannot run here**: GLM-5.2 is a 744B-parameter MoE (~40B active) whose
smallest usable quant is ~223 GB (1-bit GGUF) — 4-bit is ~370+ GB — against
64 GB of total VRAM. No GLM-5.2 Air/Flash exists yet (as of 2026-07); when one
ships, revisit this guide (family detection for the GLM-5 architecture,
`GlmMoeDsaForCausalLM`, is already in `model_lib.py`).

## The model

| | GLM-4.7-Flash (AWQ) |
|---|---|
| HF repo | `QuantTrio/GLM-4.7-Flash-AWQ` |
| Type | MoE (64 routed experts, 4 active; ~3B active of 30B total) |
| Architecture | `Glm4MoeLiteForCausalLM` |
| Attention | **MLA** (kv_lora_rank 512, rope dim 64) — see caveats |
| Layers | 47 |
| Quantization | AWQ 4-bit, `quant_method: awq` (legacy GEMM — V100 OK) |
| Weights on disk | ~18.4 GiB |
| Config dtype | bfloat16 (forced to `half` on V100) |
| Native max context | 202752 |
| MTP speculative head | yes (`num_nextn_predict_layers: 1`) |
| Suggested fixed port | 7003 |

IMPORTANT repo choice: `cyankiwi/GLM-4.7-Flash-AWQ-4bit` (the most-downloaded
"AWQ" repo) is actually **compressed-tensors W4A16**, which routes to Marlin
kernels and does not run on Volta. Use `QuantTrio/GLM-4.7-Flash-AWQ`, whose
`quant_method` is genuinely `awq` (legacy GEMM kernel, SM70-capable).

```bash
python manage_models.py download QuantTrio/GLM-4.7-Flash-AWQ
```

## MLA changes the sizing math

Unlike the Qwen models, GLM-4.7-Flash caches one compressed latent per token
(~53 KB/token across all 47 layers) instead of per-head K/V. Two consequences:

1. **TP does not grow KV capacity.** The MLA latent is replicated on every
   tensor-parallel rank, so 2 GPUs hold the same KV pool as 1. `plan` accounts
   for this (fixed 2026-07). TP=2 still shards the weights and roughly doubles
   compute.
2. Capacity is generous anyway: at TP=2 / 0.88 util the full native 202k
   context fits (~340k tokens of KV per the planner); at TP=1 about 130k fits.

## The profiles

| Profile | TP | Context | MTP | When to use |
|---------|----|---------|-----|-------------|
| `default` | 2 | 131072 | no | Baseline / fallback. |
| `default-mtp` | 2 | 131072 | yes (1 draft token) | **Recommended** once verified. |
| `default-200k` | 2 | 202752 | yes (1) | Full native context. |
| `solo-gpu` | 1 | 131072 | no | Co-run next to another model on the other GPU. |

Shared base: `dtype = "half"`, `gpu_mem_util = 0.88`, `--quantization awq`,
GLM parsers, text-only. Note there is **no `--attention-backend` pin**: the
fork's `FLASH_ATTN_V100` kernel is standard-attention only and will not serve
MLA. Let vLLM pick (expect a Triton MLA path) — see caveats.

```toml
[default]
description   = "Baseline @128k. TP=2 across both GPUs."
engine        = "vllm"
port          = 7003
tp_size       = 2
gpu           = "0,1"
dtype         = "half"
gpu_mem_util  = 0.88
max_model_len = 131072
extra_args    = ["--quantization", "awq",
                 "--reasoning-parser", "glm45",
                 "--enable-auto-tool-choice", "--tool-call-parser", "glm47",
                 "--max-num-seqs", "16",
                 "--max-num-batched-tokens", "8192"]
env           = {}

[default-mtp]
description   = "Balanced @128k + MTP speculative decoding (1 draft token)."
# ... same as [default] plus:
# extra_args  += ["--speculative-config",
#                 "{\"method\": \"mtp\", \"num_speculative_tokens\": 1}"]

[solo-gpu]
description   = "Single-GPU variant; leaves the other V100 free."
tp_size       = 1
gpu           = "1"
max_model_len = 131072
# same extra_args as [default]
```

The model ships one MTP layer (`num_nextn_predict_layers: 1`), so start with
`num_speculative_tokens: 1`; 2-3 reuses the same head with lower acceptance —
benchmark before keeping anything above 1.

## MUST VERIFY ON THE BOX (in order)

This model is architecturally different from everything else served here, and
none of it has been proven on the 1Cat fork yet:

1. **MLA attention on the fork.** First `start` attempt tells you. If it fails
   at backend selection, try `--attention-backend TRITON_MLA` in extra_args;
   if no MLA backend loads on SM70, this model cannot run on this fork —
   document the failure in the profile description and move on.
2. **fp16 numerics.** The checkpoint is bf16, forced to `half` on Volta. Watch
   the first few generations for NaN/gibberish (MLA models are occasionally
   range-sensitive in fp16).
3. **MTP acceptance.** Benchmark `default` vs `default-mtp` single-stream and
   at `--concurrency 4` before recommending MTP.

```bash
python vllm_manager.py start GLM-4.7-Flash-AWQ --profile default --wait 300
python vllm_manager.py test  GLM-4.7-Flash-AWQ
python vllm_manager.py benchmark GLM-4.7-Flash-AWQ --profile default --profile default-mtp
python vllm_manager.py benchmark GLM-4.7-Flash-AWQ --profile default --concurrency 4
```

## V100 / SM70 caveats recap

- `dtype = "half"` only (no fast bf16 on Volta).
- AWQ legacy GEMM only (`--quantization awq`); Marlin / compressed-tensors
  W4A16 need SM80+. That's why the QuantTrio repo, not cyankiwi.
- Do NOT pin `FLASH_ATTN_V100` for this model (MLA).
- Do NOT use `--enable-expert-parallel` on this fork (crashes MoE decode
  under CUDA graphs).
- `gpu_mem_util = 0.88`; 0.95 OOMs during CUDA-graph capture.
