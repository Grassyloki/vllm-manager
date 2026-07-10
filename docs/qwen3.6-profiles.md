# Guide: Qwen3.6 35B and 27B on 2x V100 (32 GB)

This documents the two Qwen3.6 models served on this box and the launch profiles
authored for them. Both run as 4-bit AWQ on 2x Tesla V100-SXM2 32 GB (SM70, NVLink)
via the 1Cat-vLLM fork, tensor-parallel across both GPUs, text-only.

Recommended choice: use the **MTP** profiles. In observed single-stream testing
they delivered the best overall throughput (clearest on the 35B), thanks to the
multi-token-prediction speculative-decode head both models ship.

## The models

| | Qwen3.6-35B-A3B-AWQ | Qwen3.6-27B-AWQ-MTP |
|---|---|---|
| Type | MoE (256 experts, 8 active; ~3B active) | Dense |
| Architecture | hybrid linear + full attention, vision | hybrid linear + full attention, vision |
| Full-attention layers (carry KV) | 10 of 40 | 16 of 64 |
| Quantization | AWQ 4-bit, GEMM kernel | AWQ 4-bit, GEMM kernel |
| Weights on disk | ~23.7 GiB | ~18.2 GiB |
| Per-GPU weights at TP=2 | ~11.9 GiB | ~9.1 GiB |
| Config dtype | float16 | bfloat16 (forced to `half` on V100) |
| Native max context | 262144 | 262144 |
| MTP speculative head | yes | yes |
| Fixed port | 7001 | 7002 |

Both are vision-language models but the profiles here are text-only
(`--limit-mm-per-prompt image=0,video=0` plus `--skip-mm-profiling`) to keep
startup lean. Served model ids are the folder names: `Qwen3.6-35B-A3B-AWQ` and
`Qwen3.6-27B-AWQ-MTP`.

## The profiles

Don't hand-author these — they're in the curated catalog:

```bash
python manage_models.py seed-profiles          # writes all four per model
```

Four profiles per model, a 2x2 of {default, MTP} x {128k, 256k}:

| Profile | Context | MTP (spec decode) | max-num-seqs | When to use |
|---------|---------|-------------------|--------------|-------------|
| `default` | 131072 | no | 16 | Baseline / fallback. |
| `default-256k` | 262144 | no | 8 | Baseline at full context. |
| `default-mtp` | 131072 | yes (2 draft tokens) | 16 | **Recommended.** Best throughput at 128k. |
| `default-mtp-256k` | 262144 | yes (2 draft tokens) | 8 | **Recommended for long context.** |

All four share the same base: `tp_size = 2`, `gpu = "0,1"`, `dtype = "half"`
(fp16; never bf16 on Volta), `gpu_mem_util = 0.88`, and these flags:

```
--attention-backend FLASH_ATTN_V100      # V100/SM70 attention path
--quantization awq                       # legacy GEMM kernel (Marlin needs SM80+)
--reasoning-parser qwen3                 # split <think> into a reasoning field
--enable-auto-tool-choice --tool-call-parser hermes
--limit-mm-per-prompt {"image": 0, "video": 0}   # text-only
--skip-mm-profiling
--max-num-batched-tokens 8192
```

The MTP profiles add:

```
--speculative-config {"method": "mtp", "num_speculative_tokens": 2}
```

Example (`default-mtp` for the 35B; the others differ only in `max_model_len`,
`max-num-seqs`, the spec-config line, and `port`):

```toml
[default-mtp]
description   = "Balanced @128k + MTP speculative decoding (2)."
engine        = "vllm"
port          = 7001
tp_size       = 2
gpu           = "0,1"
dtype         = "half"
gpu_mem_util  = 0.88
max_model_len = 131072
extra_args    = ["--attention-backend", "FLASH_ATTN_V100", "--quantization", "awq",
                 "--reasoning-parser", "qwen3", "--enable-auto-tool-choice",
                 "--tool-call-parser", "hermes",
                 "--limit-mm-per-prompt", "{\"image\": 0, \"video\": 0}",
                 "--skip-mm-profiling", "--max-num-seqs", "16",
                 "--max-num-batched-tokens", "8192",
                 "--speculative-config", "{\"method\": \"mtp\", \"num_speculative_tokens\": 2}"]
env           = {}
```

## Why MTP

Both models ship a multi-token-prediction (MTP) head, so vLLM can draft a couple
of tokens per step and verify them in one pass. Observed single-stream results
(overall throughput, the meaningful number; warm server, ~512-token replies):

| Model | Profile | Overall tok/s | TTFT | Peak VRAM/GPU |
|-------|---------|--------------:|-----:|--------------:|
| 35B | `default` (no MTP) | ~98 | ~95 ms | ~30 GiB |
| 35B | `default-mtp` | ~115 | ~75 ms | ~31 GiB |
| 35B | `default-mtp-256k` | ~116 | ~85 ms | ~31 GiB |
| 27B | `default` (no MTP) | ~90 | ~108 ms | ~31 GiB |
| 27B | `default-mtp` | ~89 | ~108 ms | ~31 GiB |

MTP is the clear win on the 35B (about +17%) and roughly neutral on the dense
27B for this workload, so MTP is a safe default for both. These are single-stream
numbers (one request at a time); under heavy concurrency speculative decoding
gives back ground to plain batching, so for many simultaneous users prefer the
non-MTP `default` profiles or the `throughput` profile.

Note on the benchmark tool's two numbers: with MTP, tokens arrive in bursts, so
the per-window "steady" figure understates it; the "mean"/overall throughput
above is the accurate one for spec decode.

## Tuning for the primary workload: coding agents / long chats

These models mostly serve OpenCode/Zed/Continue-style agents: the whole
conversation (often 50k-150k tokens) is re-sent on every turn, with 1-3
streams live at a time. That workload shapes three decisions:

1. **Keep prefix caching ON** (it is vLLM's default; do not add
   `--no-enable-prefix-caching` to the agent-facing profiles). A re-sent
   prefix skips its prefill entirely — on a 100k-token conversation that is
   the difference between a sub-second and a multi-minute turn start. The
   upstream Qwen3.5/3.6 recipe's "+27% decode by disabling prefix caching"
   applies to single fresh prompts only; for agents the TTFT win dominates.
   Note the hybrid (linear-attention) layers make prefix caching partially
   experimental upstream ("align" mode) — if a profile misbehaves after a
   cache hit, that flag is the first suspect.
2. **Keep MTP** (`default-mtp`) — measured +17% on the 35B here, and agent
   traffic is low-concurrency where spec decode wins.
3. **Verify the crossover with the concurrency benchmark** instead of
   trusting folklore. As of 2026-07 the benchmark supports parallel streams;
   aggregate tok/s across N streams is the number that decides MTP-vs-plain:

   ```bash
   python vllm_manager.py benchmark Qwen3.6-35B-A3B-AWQ \
       --profile default --profile default-mtp --concurrency 4
   ```

Two optional variants worth authoring for non-agent use:

| Variant | Change vs `default-mtp` | Use case |
|---------|------------------------|----------|
| `latency` | add `--no-enable-prefix-caching` | Single-stream, fresh prompts each time (reported up to +27% decode with MTP upstream — verify here). |
| `throughput` | drop the `--speculative-config`, raise `--max-num-seqs` to 32 | Many simultaneous users; batching beats spec decode. |

Tool-parser note: profiles here use `hermes`. The upstream Qwen3.5/3.6 recipe
now suggests `qwen3_coder` for tool calling. Whether the 1Cat fork ships that
parser is machine-specific — check the registry snapshot at
`$VLLM_MGR_PID_DIR/vllm-version.txt.json` (written by `setup`); only switch if
it's listed there, and re-test tool calls in a real agent afterwards.

## Running them

Pinned ports mean each model always lands on the same address regardless of
profile, and both can run at once:

```bash
# 35B on 7001, 27B on 7002 (recommended MTP profiles)
python vllm_manager.py start Qwen3.6-35B-A3B-AWQ  --profile default-mtp
python vllm_manager.py start Qwen3.6-27B-AWQ-MTP  --profile default-mtp

# full 256k context
python vllm_manager.py start Qwen3.6-35B-A3B-AWQ  --profile default-mtp-256k

# baseline without speculative decoding
python vllm_manager.py start Qwen3.6-35B-A3B-AWQ  --profile default

python vllm_manager.py status
python vllm_manager.py benchmark Qwen3.6-35B-A3B-AWQ --profile default --profile default-mtp
```

Loading these takes roughly 90-150 s. `start` watches for an early crash and can
block until ready with `--wait <secs>`.

## Connecting

Base URL `http://<host>:<port>/v1`, model id = folder name. Reminder: `/v1` is a
base URL, not a GET endpoint (a bare GET returns 404); health is `/v1/models`,
chat is `POST /v1/chat/completions`, and the `model` field must match the id
exactly.

```bash
curl http://aibox.lok.tech:7001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B-AWQ","messages":[{"role":"user","content":"hi"}],"max_tokens":256}'
```

These are reasoning models: replies stream a hidden `<think>` section (separated
by the `qwen3` reasoning parser) before the visible answer. With a very small
`max_tokens` the visible `content` can come back empty because the budget went to
thinking; raise `max_tokens` or send `chat_template_kwargs: {"enable_thinking": false}`
for terse replies. Run `python vllm_manager.py endpoints` for OpenCode / Zed /
Continue config snippets.

## V100 / SM70 caveats that shaped these profiles

- `dtype = "half"` only. Volta has no fast bfloat16.
- AWQ GEMM kernel via `--quantization awq`; Marlin and compressed-tensors W4A16
  require SM80+ and will not run here.
- `gpu_mem_util = 0.88`. At 0.95 with a large batch, CUDA-graph capture or the
  first request OOMs (activation/graph overhead sits on top of the KV pool).
- The 35B is MoE, but do NOT use `--enable-expert-parallel` on this fork: it
  crashes the MoE decode worker under CUDA graphs. Plain tensor-parallel expert
  sharding is stable (and is what these profiles use).
- The 27B is dense, so expert-parallel does not apply to it at all.
