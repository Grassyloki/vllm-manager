# vLLM Service Manager

A single-host control plane for running multiple [vLLM](https://github.com/vllm-project/vllm)
inference servers at once: an interactive curses TUI plus a full CLI for
downloading models, authoring per-model launch profiles, starting/stopping
servers on dedicated ports, benchmarking throughput, and printing ready-to-paste
client configs.

It was built for and tested on a **2x NVIDIA Tesla V100-SXM2 32 GB** box (Volta,
SM70) running Arch Linux, using the [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)
fork that adds SM70 AWQ kernels. It works on other hardware too, but several
defaults and helpers assume that target (see "V100 / SM70 notes").

## Guides

- [Qwen3.6 35B and 27B profiles](docs/qwen3.6-profiles.md) - the two Qwen3.6
  models served here and their `default` / `default-mtp` profiles (128k and 256k
  variants), with the MTP-is-best observation, benchmark numbers, and
  agent-workload (prefix caching / concurrency) tuning guidance.
- [GLM-4.7-Flash profiles](docs/glm-4.7-flash.md) - the GLM model that actually
  fits this box (GLM-5.2 at 744B does not), which AWQ repo is V100-safe, and
  the MLA caveats that make its profiles different from the Qwen ones.

## Contents

| File | Purpose |
|------|---------|
| `vllm_manager.py` | Main entry point: TUI + CLI to set up the env, run servers, benchmark, and inspect status. |
| `manage_models.py` | Download models from Hugging Face and manage per-model launch profiles. |
| `model_lib.py` | Shared library: filesystem scanning, model-family detection, KV-cache sizing, and the `profiles.toml` schema/IO. |
| `tui_lib.py` | Shared curses widgets (selector, prompts, colored bars, screen plumbing). |
| `test_vllm.sh` | Smoke-test a running server (waits for readiness, then hits `/health`, `/v1/models`, `/v1/chat/completions`). |
| `api_tester.sh` | Interactive whiptail client for poking any OpenAI-compatible endpoint. |

The manager itself runs on the system Python; vLLM lives in a separate Python
3.12 virtualenv that `setup` creates.

## Features

- Interactive TUI home screen with live, per-GPU memory usage split into vLLM
  vs. other processes (two-color bars), GPU names, system RAM, running models
  with their API URLs, and an auto-refresh interval.
- Named launch profiles per model in `profiles.toml` (tensor-parallel size, GPU
  set, dtype, context length, memory utilization, extra flags, env, and an
  optional fixed port).
- Multiple servers at once, each on its own port, with a state file tracking
  PIDs/ports and stale-entry pruning.
- A sizing planner (`plan`) that reads a model's config and recommends
  `tensor-parallel-size` and `max-model-len` for the available VRAM.
- A benchmark wizard that loads each selected (model, profile), measures
  steady-state tokens/sec with a sliding-window metric (robust to mid-stream
  stalls), records per-GPU peak VRAM, then hard-stops the server. Survives
  crashes and freezes; cleans up rogue worker processes.
- Pre-flight GPU checks before launch (free VRAM, target-GPU existence, tp/GPU
  count), startup-failure detection, atomic state writes, and log rotation.
- `endpoints` prints copy-paste client configs (OpenCode, Zed, Continue, raw
  cURL) for whatever is running.

## Requirements

- Linux with NVIDIA drivers and `nvidia-smi` on `PATH`.
- Python 3.12 available to `uv` (the manager runs on system Python; vLLM runs in
  a 3.12 venv it builds).
- [uv](https://docs.astral.sh/uv/) (auto-installed by `setup` if missing).
- For the V100/SM70 path: a working CUDA toolchain to build the 1Cat-vLLM fork
  from source (`gcc`, `cuda`, `cudnn`).
- A Hugging Face token in `HF_TOKEN` (or `hf auth login`) for gated downloads.

## Setup

```bash
# Standard GPUs (Ampere and newer): installs upstream vLLM
python vllm_manager.py setup

# Volta / V100 (SM70): builds the 1Cat-vLLM fork from source (30-90 min)
python vllm_manager.py setup --v100
```

The source build is memory-hungry. `cicc`/`cudafe++`/`ptxas`/`cc1plus` are each
single-threaded, so core usage comes from running many compile jobs at once
(`MAX_JOBS`), capped by RAM to avoid the OOM killer. Tune with the `BUILD_*`
variables below. Build scratch is routed to a roomy disk because the default
`/tmp` is often a small RAM-backed tmpfs.

## Usage

Run with no arguments for the interactive TUI:

```bash
python vllm_manager.py
```

Or use the CLI:

```bash
python vllm_manager.py list                          # models on disk + profiles + status
python vllm_manager.py plan  <model>                 # recommend tp_size + max-model-len
python vllm_manager.py start <model> [--profile NAME] [--port N] [--wait SECS] [--force]
python vllm_manager.py status                         # running models, ports, GPU usage
python vllm_manager.py test  <model>                  # one streaming prompt + tok/s
python vllm_manager.py benchmark <model> [--profile NAME ...]   # measure steady-state tok/s
python vllm_manager.py logs  <model>                  # tail the server log
python vllm_manager.py endpoints [model]              # client config snippets
python vllm_manager.py restart <model> [--profile NAME]
python vllm_manager.py stop  <model>
python vllm_manager.py stop-all
```

Model downloads and profile authoring live in `manage_models.py`:

```bash
python manage_models.py download <org/repo>           # HF download (HF-format or GGUF picker)
python manage_models.py download <org/repo> --dry-run # inspect variants/fit, no download
python manage_models.py download <org/repo> --offload [--ram GB]  # budget VRAM + system RAM
python manage_models.py list
python manage_models.py delete <model>
python manage_models.py profile list  <model>
python manage_models.py profile show  <model> [name]
python manage_models.py profile add   <model> <name>
python manage_models.py profile edit  <model>         # open profiles.toml in $EDITOR
```

Downloads are host-aware:

- The VRAM budget defaults to the GPUs nvidia-smi actually reports (override
  with `--vram`); `--offload` adds 85% of system RAM for llama.cpp-style
  CPU+GPU MoE offload when picking a GGUF quant.
- On V100 (SM70) hosts, HF-format downloads are checked against the repo's
  `quantization_config` first: checkpoints whose kernels need SM80+
  (compressed-tensors/Marlin, FP8, NVFP4, ...) trigger a warning before any
  bytes move — many repos named "AWQ" are actually compressed-tensors.
- After an HF download, the default profile is sized by the same planner as
  `plan` (tp_size, max_model_len, V100 arg set, parser gating) instead of a
  naive tp=1 template.

## Launch profiles

Each model gets a `profiles.toml` next to its weights. A profile named
`default` is used when `--profile` is omitted; copy a section to make a variant.

```toml
[default]
description   = "Balanced. TP=2 across both GPUs."
engine        = "vllm"          # only engine supported today
port          = 7001            # fixed listen port; 0 or omit = auto from BASE_PORT
tp_size       = 2               # tensor-parallel size; must equal the number of gpu ids
gpu           = "0,1"           # comma-separated CUDA device ids
dtype         = "half"          # "half" (fp16), "bfloat16", or "auto"
gpu_mem_util  = 0.88            # fraction of each GPU's VRAM vLLM may use
max_model_len = 131072          # max context length
extra_args    = ["--quantization", "awq", "--reasoning-parser", "qwen3"]
env           = {}              # extra env vars for the server process
launch_prefix = []              # argv prepended to the server command,
                                # e.g. ["numactl", "--cpunodebind=0", "--membind=0"]
```

Port resolution at start is: `--port` flag, then the profile's `port`, then the
next free port from `BASE_PORT`.

## Benchmark

```bash
python vllm_manager.py benchmark <model> --profile latency --profile throughput
python vllm_manager.py benchmark <model> --concurrency 4   # parallel streams
```

For each profile it starts the server, waits for readiness, runs a warm-up plus
N measured generations, and reports:

- **Steady tok/s**: the best sustained tokens/sec over a sliding window
  (default 5s), so a long mid-stream stall does not skew the result.
- **Mean tok/s**: overall, including stalls (for contrast).
- TTFT, completion tokens, stall count, load time, and per-GPU peak VRAM.

With `--concurrency N`, each measured run drives N parallel streams (each with
a distinct prompt so prefix caching can't collapse the batch) and the tok/s
figures are the aggregate across streams — the number that settles whether
speculative decoding still wins at that load.

Results print as a table and are saved as timestamped JSON in the log dir. In
the TUI, the Benchmark action is a wizard: it walks each model and you
space-select which profiles to run.

## Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_MGR_MODEL_DIR` | `/mnt/stor1/vllm/models` | Where models live. |
| `VLLM_MGR_VENV_DIR` | `/mnt/stor1/vllm/vllm-env` | vLLM virtualenv location. |
| `VLLM_MGR_PID_DIR` | `/root/.vllm-pids` | State file directory. |
| `VLLM_MGR_LOG_DIR` | `/root/.vllm-logs` | Per-model server logs + benchmark JSON. |
| `VLLM_MGR_BASE_PORT` | `1001` | First auto-assigned port. |
| `VLLM_MGR_BIND_HOST` | `0.0.0.0` | Server bind address. |
| `VLLM_MGR_PUBLIC_HOST` | `aibox.lok.tech` | Hostname printed in client-config snippets. |
| `VLLM_MGR_PUBLIC_SCHEME` | `http` | Scheme for those snippets. |
| `VLLM_MGR_PER_GPU_VRAM_GB` | auto-detected | Per-GPU VRAM used for sizing (override to pin). |
| `VLLM_MGR_FORK_REF` | `main` | 1Cat-vLLM git ref to build for `setup --v100`. |
| `VLLM_MGR_STOP_TIMEOUT` | `30` | Seconds to wait for SIGTERM before SIGKILL. |
| `VLLM_MGR_BUILD_JOBS` | auto | Force ninja `MAX_JOBS` for the source build. |
| `VLLM_MGR_BUILD_MEM_PER_JOB` | `1.5` | GiB of RAM budgeted per compile job. |
| `VLLM_MGR_BUILD_RESERVE_GB` | `2` | GiB of RAM held back from the build. |
| `VLLM_MGR_BUILD_NVCC_THREADS` | `1` | Threads per nvcc invocation. |
| `VLLM_MGR_BUILD_TMP` | `<venv parent>/build-tmp` | Build scratch dir (kept off small `/tmp`). |
| `HF_TOKEN` | unset | Hugging Face token for downloads. |

`PUBLIC_HOST` is only a display default for the generated client snippets; set it
to your own host or `localhost`.

## Connecting clients

`python vllm_manager.py endpoints` prints ready-to-paste blocks. The base URL is
`http://<host>:<port>/v1` and the model id is the model's folder name. For an
OpenAI-compatible client, set `base_url` to that and the model to the folder
name. Note `/v1` is a base URL, not a GET-able path: a bare GET on it returns
404. Health checks should hit `/v1/models`; chat goes to `/v1/chat/completions`.

## V100 / SM70 notes

Hard-won specifics for serving AWQ models on Volta with the 1Cat-vLLM fork:

- Use `dtype = "half"` (fp16). Volta has no fast bfloat16; the fork's kernels
  assume fp16.
- AWQ must use the legacy GEMM kernel (`--quantization awq`); Marlin kernels
  require SM80+ (Ampere). compressed-tensors W4A16 also routes to Marlin and
  does not run on Volta.
- Pass `--attention-backend FLASH_ATTN_V100` for the fork's V100 attention path
  — but only for standard-attention (MHA/GQA) models. MLA models (GLM-4.7-Flash,
  DeepSeek-style; `kv_lora_rank` in config.json) use a different attention path
  the V100 kernel doesn't serve: leave the backend unpinned for those (the
  auto-templater does this) and see [the GLM guide](docs/glm-4.7-flash.md).
- Keep `gpu_memory_utilization` at about 0.88 for throughput-style profiles;
  0.95 plus a large `--max-num-batched-tokens` tends to OOM during CUDA-graph
  capture or on the first request, because activation/graph overhead sits on
  top of the KV-cache pool.
- On this fork, `--enable-expert-parallel` can crash MoE decode under CUDA
  graphs; plain tensor-parallel expert sharding is the stable path.
- Speculative decoding (MTP) helps latency at low/moderate concurrency but loses
  to plain batching at high concurrency. Verify the crossover on your workload
  with `benchmark --concurrency N` rather than assuming.

## Dual-socket hosts (NUMA)

On a multi-socket box (e.g. 2x Xeon Gold 6230), check where the GPUs attach
before tuning anything:

```bash
nvidia-smi topo -m     # look at the "CPU Affinity" / "NUMA Affinity" columns
```

- **Both GPUs on one socket** (common for SXM2 boards): pin the whole server to
  that node so the API server, tokenizer, and pinned staging buffers never
  cross the UPI link:

  ```toml
  launch_prefix = ["numactl", "--cpunodebind=0", "--membind=0"]
  ```

- **One GPU per socket**: don't bind to a single node (half the workers would
  pay remote-memory latency); interleave instead:

  ```toml
  launch_prefix = ["numactl", "--interleave=all"]
  ```

GPU-to-GPU traffic rides NVLink either way (it never touches UPI), so NCCL
needs no special settings on this topology. Benchmark with and without the
prefix — host-side effects are workload-dependent and usually worth a few
percent, not a rewrite.

## License

No license has been chosen yet. Add one before sharing if you intend others to
reuse it.
