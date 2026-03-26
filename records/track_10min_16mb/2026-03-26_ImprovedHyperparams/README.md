# Improved Hyperparameters on XSA-all + FullGPTQ + Parallel Muon

**Author:** michaelhwang
**Base:** 2026-03-24 XSA-all FullGPTQ ParallelMuon (1.1154 BPB SOTA)
**val_bpb:** TBD (requires 8×H100 run with FA3)

## Summary

Identical architecture to the SOTA. Four untested hyperparameter improvements, all passed as env vars with no code changes beyond updating defaults:

| Param | SOTA default | This submission | Rationale |
|-------|-------------|-----------------|-----------|
| `MUON_BACKEND_STEPS` | 5 | **10** | The base `train_gpt.py` defaults to 10. Better Newton-Schulz orthogonalization at ~0% compute overhead (NS is <0.1% of step time) |
| `GPTQ_CALIB_BATCHES` | 256 | **512** | More calibration batches → better Hessian estimate → lower quantization loss. Current gap: 1.1356 BPB → 1.1389 BPB (+0.0033). Goal: shrink that gap. |
| `LATE_QAT_THRESHOLD` | 0.15 | **0.20** | QAT activates when LR < 20% of peak instead of 15%. More training steps with quantization noise simulation → more quantization-robust weights. |
| `VE_LAYERS` | "9,10" | **"7,8,9,10"** | Seed 7 used only 15.20 MB against the 15.9 MB target — 700 KB unused. Adding VE to 2 more layers uses ~200 KB of that headroom. More layers benefit from token identity injection. |

None of these appear in the SOTA's negative results table. All are non-destructive changes.

## Key observation

From the SOTA seed 7 log:
```
selective_prune: 4081896 ±1 candidates, unpruned=15.20MB target=15.9MB
selective_prune: already fits, no pruning needed
```

The model fits at 15.20 MB, leaving ~700 KB of unused capacity. The `VE_LAYERS` change specifically targets this headroom.

## Run command

```bash
# Requires Flash Attention 3 (Hopper only)
pip install --break-system-packages flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

SEED=7 TARGET_MB=15.9 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-26_ImprovedHyperparams/train_gpt.py
```

All four improved hyperparameters are now the defaults in this script. Individual knobs can still be overridden via env vars.

## N-gram interpolation (merged from pr638)

An n-gram cache with multi-order backoff and entropy-adaptive alpha is applied at final eval time (inside `eval_val_sliding`). This is a **runtime-only** cache: no serialized weights, zero artifact size impact.

| Env var | Default | Description |
|---------|---------|-------------|
| `NGRAM_CACHE` | `"1"` on multi-GPU, `"0"` on single-GPU | Enable n-gram interpolation |
| `NGRAM_ALPHA` | `0.40` | Base interpolation weight toward n-gram |
| `NGRAM_ORDER` | `7` | Maximum n-gram order |
| `NGRAM_MIN_ORDER` | `2` | Minimum n-gram order (backoff floor) |
| `NGRAM_ENTROPY` | `1` | Enable entropy-adaptive alpha |
| `NGRAM_ENT_BASE` | `0.05` | Minimum alpha (low-entropy / confident tokens) |
| `NGRAM_ENT_RANGE` | `0.55` | Alpha range across entropy values |
| `NGRAM_ENT_SCALE` | `2.0` | Sigmoid sharpness for entropy gating |
| `NGRAM_ENT_THRESH` | `4.0` | Entropy midpoint (nats) for sigmoid |

**Validated on 8×H100:** alpha=0.40, order=7 → **1.0336 BPB** improvement.

Multi-order backoff: highest order matched first; lower orders fill in unmatched positions. Score-first discipline: tables updated *after* scoring each segment window.

## Expected BPB

Each hyperparameter change is expected to give 0-0.005 BPB improvement. Combined effect: potentially 0.005-0.015 BPB.
N-gram interpolation adds a further ~0.003-0.005 BPB from the entropy-adaptive mixing.
Target: **< 1.1082 BPB** (requires beating SOTA by 0.0072 to qualify as new record).
