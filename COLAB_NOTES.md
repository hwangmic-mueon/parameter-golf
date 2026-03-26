# Parameter Golf: Colab & Kaggle Notebook Reference

Practical troubleshooting and configuration guide for running the parameter golf
training pipeline on free-tier cloud GPUs. This document covers what works, what
crashes, and how to configure each GPU tier.

---

## GPU Tier Overview

| GPU | Platform | VRAM | Relative speed | Script to use |
|-----|----------|------|----------------|---------------|
| T4 | Colab Free, Kaggle Free | 16 GB | ~10x slower than H100 | `train_gpt.py` (root) |
| P100 | Kaggle Free | 16 GB | ~8x slower than H100 | `train_gpt.py` (root) |
| A100 40GB | Colab Pro | 40 GB | ~3x slower than H100 | `train_gpt.py` (root) |
| A10G | Colab Pro+ | 24 GB | ~4x slower than H100 | `train_gpt.py` (root) |
| H100 SXM | RunPod / cloud | 80 GB | baseline | records script (requires FA3) |

**Rule of thumb:** T4 and P100 are only useful for verifying the pipeline does
not crash. A100 can produce a meaningful but non-competitive BPB. H100 (ideally
8x) is required for a real leaderboard submission.

---

## Which Script to Run

There are two different training scripts in this repo:

| Script | Where | Use when |
|--------|-------|----------|
| `train_gpt.py` (repo root) | Root of repo | T4, P100, A100 — any non-H100 GPU |
| `records/track_10min_16mb/2026-03-26_ImprovedHyperparams/train_gpt.py` | Records folder | H100 only |

**Why they are separate:** The records/SOTA script has a top-level hard import:

```python
from flash_attn_interface import flash_attn_func as flash_attn_3_func
```

This line is at module load time (line 27), not behind a try/except. On any GPU
that does not have Flash Attention 3 installed, the script aborts immediately
with `ModuleNotFoundError: No module named 'flash_attn_interface'` before a
single training step runs. FA3 only works on Hopper architecture (H100). T4,
P100, and A100 will always crash with this script.

The root `train_gpt.py` uses `torch.nn.functional.scaled_dot_product_attention`
instead and runs on any CUDA GPU without additional packages.

---

## Known Compatibility Issues and Fixes

### Issue 1: Flash Attention 3 hard-import crash on T4/P100/A100

**Symptom:** Running the records script on a non-H100 GPU crashes immediately:
```
ModuleNotFoundError: No module named 'flash_attn_interface'
```

**Root cause:** `from flash_attn_interface import flash_attn_func` at line 27 of
the records script is a top-level import. FA3 requires Hopper SM90 instruction
set. T4 is Turing (SM75), P100 is Pascal (SM60), A100 is Ampere (SM80).

**Fix:** Use the root `train_gpt.py` for all non-H100 runs. The colab_run.ipynb
notebook already handles this via GPU detection:
```python
if is_h100:
    SCRIPT = f'{REPO}/records/track_10min_16mb/2026-03-26_ImprovedHyperparams/train_gpt.py'
else:
    SCRIPT = f'{REPO}/train_gpt.py'
```

---

### Issue 2: GQA (Grouped Query Attention) not supported on T4 SDPA flash backend

**Symptom:** On T4 with `NUM_KV_HEADS < NUM_HEADS`, training crashes with a
TorchDynamo error:
```
RuntimeError: CUDA error: invalid device function
```
or a torch.compile error mentioning "Invalid backend" or "enable_gqa not
supported".

**Root cause:** The base script calls:
```python
y = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    is_causal=True,
    enable_gqa=(self.num_kv_heads != self.num_heads),
)
```
The `enable_gqa=True` path in PyTorch SDPA routes to the flash backend. The
flash SDPA kernel on T4 (Turing) does not support GQA. This also applies to
P100 (Pascal). A100 should handle it, but T4 and P100 will crash.

**Fix:** Set `NUM_KV_HEADS` equal to `NUM_HEADS`. This causes
`enable_gqa=False`, falling back to standard multi-head attention (MHA), which
works on all GPU tiers.

```bash
# T4 / P100 safe config — no GQA
NUM_HEADS=6 NUM_KV_HEADS=6   # equal → enable_gqa=False → MHA mode
```

The colab_run.ipynb notebook already sets this for the T4/P100 branch:
```python
'NUM_HEADS': '6', 'NUM_KV_HEADS': '6',   # equal = no GQA
```

---

### Issue 3: TRAIN_BATCH_TOKENS too small for grad accumulation

**Symptom:** The script raises a `ValueError` at startup complaining that the
local batch size is smaller than `train_seq_len`.

**Root cause:** The script hardcodes `grad_accum_steps = 8 // world_size`. On a
single GPU (`world_size=1`), `grad_accum_steps=8`. The local batch per micro-step
is `TRAIN_BATCH_TOKENS // (world_size * grad_accum_steps)`. With `seq_len=1024`
and `grad_accum_steps=8`, the minimum `TRAIN_BATCH_TOKENS` is `1024 * 8 = 8192`.

Additionally, `WORLD_SIZE` must evenly divide 8 (valid values: 1, 2, 4, 8). Any
other value raises:
```
ValueError: WORLD_SIZE=3 must divide 8 so grad_accum_steps stays integral
```

**Fix:** Always set `TRAIN_BATCH_TOKENS >= train_seq_len * grad_accum_steps`.
With `train_seq_len=1024` and single-GPU: minimum is 8192. In practice, use at
least 65536 for meaningful throughput.

---

### Issue 4: VAL_BATCH_SIZE too small

**Symptom:**
```
ValueError: VAL_BATCH_SIZE must provide at least one sequence per rank
```

**Root cause:** Same math as above. `VAL_BATCH_SIZE // (world_size * grad_accum_steps)`
must be at least `train_seq_len`.

**Fix:** Set `VAL_BATCH_SIZE >= train_seq_len * grad_accum_steps`. On a single
GPU with `seq_len=1024`: minimum is 8192. Use 65536 as a safe floor.

---

### Issue 5: Environment path detection

**Symptom:** `git clone` or `cd` commands fail because the wrong base path was
hardcoded.

**Root cause:** Colab uses `/content/` as the working directory. Kaggle uses
`/kaggle/working/`. Hardcoding either path breaks the other.

**Fix:** Always auto-detect at runtime:
```python
if os.path.isdir('/content'):
    BASE = '/content'
elif os.path.isdir('/kaggle/working'):
    BASE = '/kaggle/working'
else:
    BASE = os.path.expanduser('~')
```
The colab_run.ipynb notebook does this in cells 4, 6, and 7.

---

### Issue 6: Kaggle Internet disabled by default

**Symptom:** `git clone` or `pip install` hang or fail with connection errors.

**Root cause:** Kaggle notebook kernels start with internet access **off** by
default for reproducibility reasons.

**Fix:** Before running any cell, go to the notebook sidebar:
**Settings → Internet → turn On**

This must be done before starting the kernel, not mid-session.

---

### Issue 7: Flash SDPA backend explicitly forced on — crashes T4/P100 regardless of GQA

In `train_gpt.py` (lines 957-960) the training setup does:
```python
enable_cudnn_sdp(False)
enable_flash_sdp(True)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)   # math (the only fallback) is disabled
```

Flash attention requires compute capability ≥ 8.0 (Ampere). T4=7.5 (Turing),
P100=6.0 (Pascal). With flash forced on and math disabled, there is **no valid
SDPA backend** on T4/P100 — the script crashes even with `enable_gqa=False` (MHA
mode). Setting `NUM_KV_HEADS == NUM_HEADS` is necessary but not sufficient.

**Root error:**
```
RuntimeError('Invalid backend')
```
even when `enable_gqa=False` is confirmed in the log.

**Fix:** After cloning the repo, patch `train_gpt.py` to swap flash→math on
non-Ampere GPUs. The `colab_run.ipynb` Cell 4 does this automatically:
```python
cc = torch.cuda.get_device_capability()
if cc[0] < 8:  # T4=7, P100=6
    txt = open(script).read()
    txt = txt.replace('enable_flash_sdp(True)',  'enable_flash_sdp(False)')
    txt = txt.replace('enable_math_sdp(False)',  'enable_math_sdp(True)')
    open(script, 'w').write(txt)
```
The math SDPA backend is slower than flash but works on all GPU generations.

---

### Issue 8: T4 bfloat16 torch.compile unsupported → 20+ minute validation stall

**Symptom:** Training completes all warmup steps then appears to freeze with no
output for 20+ minutes. Warnings appear:
```
Tesla T4 does not support bfloat16 compilation natively, skipping
Not enough SMs to use max_autotune_gemm mode
```

**Root cause:** `torch.compile` (TorchInductor) doesn't support bfloat16 on T4
(Turing, SM75). The compiled model silently falls back to float32. The first
validation pass evaluates all 62M val tokens in float32 on a single T4, which
takes 15-30 minutes — far longer than the 10-minute training budget.

**Fix:** Disable dynamo/compile on T4/P100 via `TORCHDYNAMO_DISABLE=1`. In
eager mode the script uses float16 which is fast enough on T4.

```python
# Set before launching torchrun
os.environ['TORCHDYNAMO_DISABLE'] = '1'
```

The `colab_run.ipynb` Cell 6 sets this automatically for compute_cap < 8.0.

---

## Environment Setup: Step by Step

### Google Colab (T4 Free or A100 Pro)

1. Open colab_run.ipynb
2. Runtime → Change runtime type → GPU (T4 for free, A100 for Pro)
3. Run Cell 2 (GPU check) to confirm GPU type
4. Run Cell 3 (pip install) — no internet toggle needed
5. Run Cell 4 (clone repo) — uses `/content/` automatically
6. Run Cell 5 (download data) — use `--train-shards 1` for testing
7. Run Cell 6 (train) — script auto-selected based on GPU

### Kaggle (T4 or P100 Free)

1. Create new notebook, attach GPU accelerator
2. **Settings → Internet → On** — do this first
3. Run all cells — uses `/kaggle/working/` automatically
4. Kaggle gives 30 GPU-hours/week free; sessions are limited to ~9 hours each

### Package installs

For T4/P100/A100 (root script), only these are needed:
```bash
pip install -q sentencepiece huggingface-hub datasets tqdm zstandard
```

FA3 is only needed for H100:
```bash
pip install flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
```

---

## Configuration Parameters: Safe vs Unsafe per GPU

### T4 and P100 (safe config)

```bash
# Model shape — keep small, no GQA
NUM_LAYERS=6
MODEL_DIM=384
NUM_HEADS=6
NUM_KV_HEADS=6          # MUST equal NUM_HEADS — GQA breaks T4/P100 SDPA
MLP_MULT=3

# Batch sizing — must satisfy: TRAIN_BATCH_TOKENS >= seq_len * 8
TRAIN_BATCH_TOKENS=131072   # 128 * 1024, safe for single GPU
VAL_BATCH_SIZE=65536

# Architecture features — all safe on T4/P100
MUON_BACKEND_STEPS=10   # more Newton-Schulz steps; zero throughput impact
XSA_LAST_N=99           # XSA on all layers; clamped to num_layers
BIGRAM_VOCAB_SIZE=2048  # BigramHash embedding
SMEAR_ENABLED=1
LN_SCALE=1
ROPE_DIMS=16

# Time limit
MAX_WALLCLOCK_SECONDS=600
```

### A100 (safe config)

Same as T4/P100 but can afford a larger model and GQA works:
```bash
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4          # GQA works on A100 — 2:1 ratio
MLP_MULT=3
TRAIN_BATCH_TOKENS=262144
VAL_BATCH_SIZE=131072
```
All architecture features (XSA, BigramHash, SmearGate, Partial RoPE, Muon NS=10)
are safe on A100.

### H100 (records script only)

```bash
NUM_LAYERS=11
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4          # GQA 2:1
MLP_MULT=3
TRAIN_BATCH_TOKENS=786432
VAL_BATCH_SIZE=524288

# H100-only features (not in root train_gpt.py)
EVAL_STRIDE=64
EVAL_SEQ_LEN=2048
VE_LAYERS=7,8,9,10
LATE_QAT_THRESHOLD=0.20
GPTQ_CALIB_BATCHES=512
```

### Parameters that are always unsafe (any GPU)

| Parameter | Unsafe when | Reason |
|-----------|-------------|--------|
| `NUM_KV_HEADS < NUM_HEADS` | T4, P100 | SDPA flash backend GQA not supported |
| `TRAIN_BATCH_TOKENS < seq_len * 8` | Any single GPU | `grad_accum_steps=8`, division produces 0 |
| `TRAIN_SEQ_LEN > 2048` | Any GPU (root script) | Partial RoPE trained at 2048 max; BPB degrades catastrophically beyond training length |
| `WORLD_SIZE` not in {1,2,4,8} | Any | Script requires `8 % world_size == 0` |

---

## Features Available per Script

| Feature | Root script (T4/P100/A100) | Records script (H100) |
|---------|---------------------------|----------------------|
| XSA-all (`XSA_LAST_N`) | Yes | Yes |
| BigramHash (`BIGRAM_VOCAB_SIZE`) | Yes | Yes |
| SmearGate (`SMEAR_ENABLED`) | Yes | Yes |
| Partial RoPE (`ROPE_DIMS`) | Yes | Yes |
| Muon NS steps (`MUON_BACKEND_STEPS`) | Yes | Yes |
| Layer-norm scaling (`LN_SCALE`) | Yes | Yes |
| GQA | A100 only (not T4/P100) | Yes |
| Value Embeddings (`VE_LAYERS`) | Yes (basic) | Yes (full) |
| GPTQ int6 + lzma | No (int8 + zlib only) | Yes |
| Sliding window eval (`EVAL_STRIDE`) | No | Yes |
| Late QAT (`LATE_QAT_THRESHOLD`) | No | Yes |
| Flash Attention 3 | No (uses PyTorch SDPA) | Required |

---

## Expected BPB by GPU (10-minute run)

These are rough ranges based on the model configs in colab_run.ipynb and the
training throughput of each GPU tier.

| GPU | Config | Tokens processed (10 min) | Expected BPB |
|-----|--------|--------------------------|--------------|
| T4 | 6L 384d, MHA | ~600K | 1.5 – 1.7 |
| P100 | 6L 384d, MHA | ~800K | 1.4 – 1.6 |
| A100 | 9L 512d, GQA | ~5–8M | 1.2 – 1.35 |
| H100 SXM (1x) | 11L 512d | ~10M | ~1.25 – 1.35 |
| H100 SXM (8x) | 11L 512d | ~78M | < 1.12 (SOTA target) |

T4/P100 numbers are far from competitive. The model barely converges in 10 minutes
at these sizes. Use T4/P100 exclusively to confirm the pipeline runs without
errors before committing GPU time.

**SOTA context:**
- Current record (2026-03-24): **1.1154 BPB** on 8xH100 SXM
- Target to beat: **< 1.1082 BPB** (must beat by >= 0.0072 BPB)
- Sliding window eval (stride=64) accounts for 0.0236 BPB of the gap between
  single-pass and sliding-window score — this is only used in the records script,
  not the root script

---

## Debugging Checklist

If a run crashes on Colab/Kaggle, check in this order:

1. **Wrong script** — Did you accidentally point to a records folder script on a
   non-H100? Check for `ModuleNotFoundError: flash_attn_interface`.

2. **GQA on T4/P100** — Is `NUM_KV_HEADS < NUM_HEADS`? Set them equal.

3. **Batch too small** — Is `TRAIN_BATCH_TOKENS` at least `train_seq_len * 8`?
   With `seq_len=1024` on a single GPU, minimum is 8192; use 65536+.

4. **Data not downloaded** — Did Cell 5 complete? Check that
   `./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin` exists. Use
   `--train-shards 1` for a quick smoke test.

5. **Kaggle internet off** — If `git clone` or `pip install` hangs, go to
   Settings → Internet → On and restart the kernel.

6. **OOM on T4** — Reduce `MODEL_DIM` (try 256) or `TRAIN_BATCH_TOKENS` (try
   32768). T4 has 16GB but bf16 activations + optimizer state add up fast.

7. **Seq len too long** — `TRAIN_SEQ_LEN > 2048` with Partial RoPE causes
   catastrophic BPB degradation (observed: 1.5695 at seq=4096). Keep at 1024 or
   2048 max.

---

## Notes on the Training Pipeline

- `torchrun --standalone --nproc_per_node=1` is the correct invocation for single
  GPU on Colab/Kaggle. The script handles `world_size=1` correctly.
- `MAX_WALLCLOCK_SECONDS=600` enforces the 10-minute competition limit. The script
  stops training when this is exceeded, then runs post-training quantization and
  serialization.
- Logs are written to `logs/{RUN_ID}.txt` under the repo directory. On Colab this
  is `/content/parameter-golf/logs/`. The colab_run.ipynb Cell 6 also writes a
  copy to `{BASE}/train_log.txt` for easy access.
- `torch.compile` is applied to the Newton-Schulz function
  (`zeropower_via_newtonschulz5`). First warmup steps trigger compilation; expect
  a 1–3 minute delay before training steps start logging.
- TF32 is enabled for both matmul and cuDNN. This is correct for Ampere+ (A100).
  On T4 (Turing), TF32 is not available on matmul but the flags are harmless —
  they simply have no effect.
- The `SDPA_BACKENDS` log line `sdp_backends:cudnn=False flash=True mem_efficient=False math=False`
  confirms the flash SDPA path is active. If this causes issues on T4, it can be
  changed in the training code around line 957, but with `NUM_KV_HEADS == NUM_HEADS`
  it should be stable.
