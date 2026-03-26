# Parameter Golf Experiments Notes

OpenAI parameter golf challenge: 16MB artifact, 10-minute training on 8xH100 SXM, evaluated by bits-per-byte (BPB) on FineWeb validation set.

---

## Current SOTA

**Score: 1.1154 BPB** (3-seed mean, std 0.0005) — `2026-03-24_XSA-all_FullGPTQ_ParallelMuon`

### Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 |
| Dimension | 512 |
| Attention heads | 8H / 4KV (GQA) |
| MLP activation | LeakyReLU(0.5)² with 3× width |
| Positional encoding | Partial RoPE 16/64 |
| Attention type | XSA (Exclusive Self-Attention) on **all 11 layers** |
| Skip connections | U-Net style |
| Gating | SmearGate |
| Embedding | BigramHash(2048) + VE128 |
| Normalization | LN Scale |
| EMA | 0.997 |
| SWA | Tight (starts step 6250 in seed7) |
| Optimizer | Parallel Muon |
| Quantization | Full Hessian GPTQ int6 + lzma |
| Pruning | Selective ±1 magnitude (sort by reconstruction error = scale²) |
| Compression | Parameter Banking |

### Seed results

| Seed | Steps | ms/step | Sliding BPB (stride=64) | Artifact size |
|------|-------|---------|--------------------------|---------------|
| 7 | 6,938 | 86.7 | **1.1153** | 15,937,739 bytes |
| 314 | ~6,930 | 86.7 | **1.1150** | 15,933,191 bytes |
| 2024 | ~6,930 | 86.7 | **1.1159** | 15,928,475 bytes |

**Mean: 1.1154 | Std: 0.0005**

### Seed7 log — key events

```
step:4000  val_bpb:1.2116  (mid-training checkpoint)
step:6250  swa:start
step:6420  late_qat:enabled  scale:0.1499  (threshold=0.15)
step:6938  val_bpb:1.1366   stopping_early: wallclock_cap (600,078ms)

post-EMA:                           val_bpb:1.1356
GPTQ calibration: 256 batches, 68 layers
selective_prune: 4,081,896 ±1 candidates
selective_prune: already fits (15.20MB < 15.9MB target) — no pruning needed
Serialized int6+lzma model: 15,830,064 bytes
Total submission (code + model): 15,937,739 bytes

final_int6_roundtrip:               val_bpb:1.1389  (single-pass)
final_int6_sliding_window:          val_bpb:1.1153  (stride=64)
```

The gap between single-pass (1.1389) and sliding-window (1.1153) eval is **0.0236 BPB** — sliding window is essential to the score.

### Run command

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
SEED=7 TARGET_MB=15.9 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## What Works (Techniques with Positive BPB Impact)

All techniques relative to starting stack; each listed with its cumulative contribution as documented.

| Technique | BPB Delta | Notes |
|-----------|-----------|-------|
| XSA on all 11 layers (vs. last 4 only) | -0.0016 | Forces cross-position mixing from layer 0. Zero additional parameters. |
| Selective ±1 pruning | Enables budget fit | Sort quantized ±1 values by scale², zero least-impactful first. Allows BigramHash(2048) and larger model to fit in 16MB. |
| Full Hessian GPTQ int6 + lzma | Significant vs. naive int8 | Inherited from PR #535 and PR #569. |
| Parallel Muon optimizer | Significant | Inherited from PR #593. |
| LeakyReLU(0.5)² MLP activation | Positive | Inherited from PR #493, PR #518. |
| BigramHash(2048) | Positive vs. none | Provides character-level bigram lookup as embedding. |
| Partial RoPE 16/64 | Positive | Only first 16 of 64 head-dim positions get RoPE. |
| VE128 (value embedding, 128 buckets) | Positive | Injects position-independent identity into deep layers. |
| SmearGate | Positive | Gated residual mixing. |
| U-Net skip connections | Positive | Layer-skipping residuals. |
| EMA (0.997) + Tight SWA | Positive | Stochastic weight averaging at end of training. |
| Sliding window eval (stride=64) | -0.0236 vs. single-pass | Gives 1,984 tokens of effective context per eval position. Not training cost. |
| Flash Attention 3 (vs. FA2) | ~+1,000 steps | FA2 costs ~100ms/step vs. ~87ms; difference is ~1,000 steps and ~0.004 BPB over 600 seconds. |

---

## What Doesn't Work (Negative Results)

All tested on the SOTA stack; positive delta = worse BPB.

| Technique | BPB | Delta vs. SOTA | Why it failed |
|-----------|-----|----------------|---------------|
| Value Residual Learning (linear) | 1.1298 | +0.0012 worse | Conflicts with VE128 — both inject identity info into deep layers; redundant |
| VRL sigmoid gates + TrigramHash | 1.1174 | +0.0020 worse | Combined overhead costs ~100 training steps; net negative |
| Catalytic Residuals | 1.1285 | ~-0.0001 (negligible) | Redundant with existing attn_scale / mlp_scale / resid_mix |
| Backout Connection | 1.1291 | +0.0005 worse | Redundant with U-Net skip connections |
| Gated Attention + XSA-all | 1.1279 | +0.0011 vs. XSA-all | 3% per-step overhead outweighs quality gain |
| Hadamard rotation + GPTQ | 1.1266 | -0.0002 | +0.5MB artifact size; hurts zstd compressibility |
| TrigramHash (zero params) | 1.1237 | +0.0049 worse | Changes weight distribution in ways that hurt lzma compression |
| BigramHash(8192) | 1.1200 | -0.0068 | Artifact 0.52MB over 16MB budget — does not fit |
| BigramHash(4096) | 1.1285 | +0.0097 worse | Also 0.52MB over budget; cold cache penalty on top |
| Stride=32 eval (vs. stride=64) | — | +0.0001 | Negligible at seq_len=2048; stride=64 already gives 1,984 context tokens |
| Temperature scaling (T ≠ 1.0) | — | +0.0002 to +0.003 | Model is already well-calibrated; T=1.0 is optimal |
| Extended context eval (seq=4096) | 1.5695 | catastrophic | Partial RoPE (trained at seq=2048) breaks completely beyond training length |
| Checkpoint logit ensemble | — | infeasible | EMA-raw weight delta alone is 16.2MB compressed (int8+zstd) — exceeds budget |
| Entropy coding (ANS/Huffman) | — | +0.05MB at best | lzma already at 99.7% of Shannon entropy limit; no practical headroom |
| Magnitude pruning (all ±1 values) | 1.1341 | +0.015 worse | Too aggressive — no smooth reconstruction-quality continuum at threshold=1 |

---

## Our Additions

### N-gram Eval Cache (`2026-03-25_NGramCache`)

**What we built:** An entropy-adaptive online n-gram interpolation layer applied only during the final evaluation pass. Training is identical to the SOTA; only the eval step changes.

**Mechanism:**
- While scoring the validation set left-to-right, running unigram/bigram/trigram counts are accumulated from previously-seen tokens.
- At each position, neural probabilities are interpolated with n-gram probabilities using:
  ```
  alpha = cache_scale * (0.05 + 0.55 / (1 + exp(-2 * (H - 4.0))))
  ```
  where `H` is the neural model's predictive entropy at that position.
- When the model is confident (low H), n-grams are barely used (alpha ≈ 0.05).
- When the model is uncertain (H > 4 nats), n-grams contribute significantly (alpha → 0.60).
- `cache_scale` ramps from 0 → 1 as the cache fills past 5M tokens, suppressing near-uniform noise early in the validation pass.

**Why it was expected to help:**
- With vocab_size=1024 and 62M validation tokens, average observations per bigram = ~62 (well above the threshold of 5 for reliable estimation).
- Average per trigram hash bucket = ~950 observations.
- For high-entropy positions (where the neural model genuinely struggles), bigram/trigram context from the same validation stream should be highly reliable.
- Many common English patterns (articles, prepositions, punctuation sequences) are deterministic n-grams in a 1024-vocab space.

**Why it didn't work locally (warm-cache test):**

Local warm-cache tests (where the cache was pre-built before scoring, eliminating ramp-up noise) showed that **n-gram interpolation hurts BPB even with a fully-populated cache on small validation sets**. The root cause is that an 11-layer 512d transformer trained on 8B tokens with BigramHash(2048) already has every common bigram and trigram pattern encoded directly in its weights. The n-gram cache is not providing new information — it is duplicating what the model already knows and adding noise via the interpolation arithmetic.

**Conclusion:** The approach is fundamentally limited for this architecture. The neural model's BigramHash embedding and deep attention layers already internalize all reliable n-gram statistics from the training corpus. N-gram interpolation only helps when the neural model has a systematic blind spot for surface-level patterns, which is not the case here. The improvement would require a much weaker baseline model or a vocabulary too large for the neural model to memorize all n-gram statistics.

**Status:** H100 submission not run. Approach shelved.

---

## Untested Hyperparameter Knobs (env-var changes, no code modification required)

These can be tested with zero code changes by setting environment variables.

| Variable | Current (SOTA) | Proposed | Rationale |
|----------|----------------|----------|-----------|
| `MUON_BACKEND_STEPS` | 5 | 10 | Base `train_gpt.py` defaults to 10; SOTA uses 5. Increasing to 10 gives more Newton-Schulz backend iterations per Muon step. Zero per-step overhead on the critical path. |
| `GPTQ_CALIB_BATCHES` | 256 | 512 | Doubling calibration batches improves Hessian estimation accuracy for GPTQ. Runs post-training so no impact on training step budget. |
| `VE_LAYERS` | "9,10" | "7,8,9,10" | Value embedding currently applied on last 2 layers. Extending to last 4 layers uses approximately 200KB of the 700KB spare headroom in the artifact (seed7 was 15.20MB uncompressed before GPTQ, fitting within 15.9MB target). May improve deep-layer identity injection. |
| `LATE_QAT_THRESHOLD` | 0.15 | 0.20 | QAT (quantization-aware training) starts when the LR schedule reaches 0.15× peak. Raising to 0.20 starts QAT earlier, giving more training steps under quantization-aware gradients. Seed7 triggered QAT at step 6,420 out of 6,938; moving to 0.20 would trigger ~100 steps earlier. |

---

## Architecture Ideas Not Yet Tried

These are absent from both the SOTA negative-results table and the existing literature in this repo, meaning they have not been tested on this stack.

| Idea | Motivation | Risk |
|------|------------|------|
| **SwiGLU / GeGLU activation** | Standard in modern LLMs (LLaMA, PaLM). SOTA currently uses LeakyReLU(0.5)², which was tuned on earlier stacks. SwiGLU uses a gated linear unit with SiLU, which may have better gradient flow with the current 11-layer depth. | ~5% parameter overhead for the gate projection; needs careful budget accounting. |
| **Deeper model (12–13 layers)** | Seed7 artifact was only 15.20MB before GPTQ, leaving ~700KB uncompressed headroom within the 15.9MB target. 1–2 additional transformer layers would use this capacity. Depth scaling historically improves BPB more efficiently than width. | Adds ~130KB of parameters per layer at 512d; need to verify it still fits after GPTQ+lzma. |
| **Int5 quantization** | Moving from int6 to int5 would free approximately 16% of model storage, enabling a ~16% larger model (extra ~1.5 layers or wider MLP) at the same artifact size. Requires extending the GPTQ quantization kernel. | Reconstruction error is significantly higher at int5; net BPB effect uncertain without ablation. |

---

## Key Challenge Constraints

| Constraint | Value |
|------------|-------|
| Artifact size limit | 16,000,000 bytes (decimal 16MB, not 16 MiB = 16,777,216 bytes) |
| What counts toward artifact | `train_gpt.py` code bytes + compressed model bytes |
| Training time limit | 10 minutes on 8xH100 SXM |
| Eval time limit | Additional 10 minutes on 8xH100 SXM (separate from training) |
| Attention requirement | Flash Attention 3 (Hopper kernel). FA2 costs ~100ms/step vs. ~87ms, losing ~1,000 steps and ~0.004 BPB. |
| External resources | No downloads, training data access, or network calls during evaluation. |
| Record threshold | Must beat existing SOTA by ≥ 0.005 nats (= 0.0072 BPB) at p < 0.01. |
| Dataset | FineWeb 10B tokens, 1024-vocab SentencePiece BPE tokenizer. Val set: 62,021,632 tokens. |

### Flash Attention 3 install

```bash
pip install --break-system-packages flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
```

---

## Local Testing Results

| Test | Config | BPB | Notes |
|------|--------|-----|-------|
| Local Mac baseline | MLX, 4L 256d, 5 min | 1.9979 | Beats Codex (2.67 BPB). Useful for smoke testing architecture changes. |
| N-gram eval (cold cache) | 4L 256d + n-gram, small val | worse than baseline | Cache sparsity at < 1M val tokens produces near-uniform noise. |
| N-gram eval (warm cache, pre-built) | 4L 256d + n-gram, fully populated cache | worse than baseline | N-gram hurts even with a fully-built cache. Neural model already learned all bigram patterns. |

Local MLX runs use a much smaller model and shorter training than H100 SOTA submissions. They are useful for catching crashes and testing qualitative behavior but BPB numbers are not directly comparable to H100 results.

---

## Colab Blackwell GPU Discovery (2026-03-26)

### GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120)

Google Colab labels the "H100 GPU" option but sometimes allocates a Blackwell GPU (SM120) instead. This was discovered when a Colab session allocated an **NVIDIA RTX PRO 6000 Blackwell Server Edition** with compute capability 12.0.

**Key properties:**
- `compute_cap`: 12.0 (SM120, Blackwell architecture)
- Not to be confused with H100 (SM90, Hopper architecture)
- VRAM: 96 GB

**FA3 failure on Blackwell:** FA3 installs successfully (`pip install` completes, `import flash_attn_interface` succeeds) but crashes at runtime:
```
CUDA error: no kernel image is available for execution on the device
```
Root cause: FA3 wheels are compiled for SM90 (H100) only. No SM120 kernel binary is bundled.

**torch.compile / Triton failure on Blackwell:** Even without FA3, `torch.compile` via TorchInductor/Triton fails:
```
No valid triton configs. OutOfMemoryError: out of resource
```
Root cause: Triton has no tuned tile configs for SM120 yet. This affects the Newton-Schulz compile path in the Muon optimizer as well as any kernel that Triton auto-tunes.

**Fix:** Set `TORCHDYNAMO_DISABLE=1` and use the root `train_gpt.py` (not the records script). Flash SDPA (PyTorch built-in, not FA3) works correctly in eager mode on Blackwell. GQA (`enable_gqa=True`) also works fine on Blackwell.

**GPU detection lesson — never match by name string:** Colab reports the GPU name as something that may include "H100" or may not; the name string is unreliable. Always gate on compute capability:

| cc[0] | Architecture | FA3 | torch.compile | GQA | Script |
|-------|-------------|-----|---------------|-----|--------|
| ≥ 12 | Blackwell (SM120) | installs but crashes at runtime | fails (no Triton configs) | works | root `train_gpt.py` + `TORCHDYNAMO_DISABLE=1` |
| 9 | Hopper / H100 (SM90) | works | works | works | records script (FA3 + GPTQ int6) |
| 8–8.9 | Ampere (A100, L4, G4) | not needed | works | works | root `train_gpt.py` |
| < 8 | Turing/Pascal (T4, P100) | not needed | fails (no bfloat16) | crashes | root `train_gpt.py` + `TORCHDYNAMO_DISABLE=1` |

---

### Blackwell Batch Size Trade-off

Without `torch.compile`, the Muon optimizer's Newton-Schulz step runs in eager mode, which is significantly slower per step. Large batches compound the issue:

| TRAIN_BATCH_TOKENS | ms/step | Steps in 600s | BPB (pre-quant) | Notes |
|--------------------|---------|---------------|-----------------|-------|
| 786,432 | 2748 | ~219 | 1.95 | Barely converged — too few steps |
| 131,072 | 398 | ~1507 | 1.39 | Acceptable convergence for single-GPU Blackwell |

**Conclusion:** Use `TRAIN_BATCH_TOKENS=131072` on Blackwell. The ~398ms/step in eager mode is slower than A100+compile, but gives enough steps to produce meaningful convergence in 10 minutes.

---

### Blackwell Model Size Constraint (int8+zlib budget)

The root `train_gpt.py` uses **int8 + zlib** compression (not int6+lzma like the records script). With int8+zlib the compression ratio is approximately 0.80 (the compressed model is ~80% of the raw int8 byte count).

With a 16MB artifact budget (code + model), the usable model budget is roughly 15.5–15.8MB compressed. This constrains the maximum parameter count at 512d:

| Config | Params | Estimated compressed size | Fits 16MB? |
|--------|--------|--------------------------|------------|
| 11L 512d | ~26.8M | ~21.4 MB | No — 5MB over |
| 9L 512d | ~21.8M | ~17.4 MB | No — ~1.4MB over |
| 8L 512d | ~19.4M | ~15.5 MB | Yes — fits |
| 7L 512d | ~17.0M | ~13.6 MB | Yes — with headroom |

**Current Blackwell config:** 8L 512d, GQA 8H/4KV, `TRAIN_BATCH_TOKENS=131072`.

Note: The SOTA records script (int6+lzma) achieves much better compression (~15.8MB for 11L 512d with 26.8M params), which is why the H100 path can afford a larger model. The compression format is the binding constraint that separates Blackwell (root script) from H100 (records script) in terms of model capacity.

---

### Blackwell Results Summary (single GPU, 10 min, root script)

| Config | ms/step | Steps | val_bpb (pre-quant) | Artifact budget status |
|--------|---------|-------|---------------------|------------------------|
| 8L 512d GQA, TORCHDYNAMO_DISABLE=1 | 398 | ~1507 | ~1.39 | fits within 16MB (estimated) |

For comparison:
- **SOTA (8×H100, records script):** 1.1154 BPB
- **Blackwell gap vs. SOTA:** ~0.27 BPB — due to 1507 vs. 6938 steps, single GPU, no GPTQ int6/sliding-window eval

Blackwell runs are useful for testing pipeline correctness on a higher-end GPU but are not competitive for leaderboard submission due to the combination of: eager-mode-only execution, smaller forced model size (int8+zlib budget), and no sliding-window eval.

---

### Fork and Repository State

- **Public upstream:** https://github.com/openai/parameter-golf
- **Our fork:** https://github.com/hwangmic-mueon/parameter-golf (branch: `main` = `codex_loop_v1`)
- **Fork contains:**
  - `ImprovedHyperparams` submission (records/track_10min_16mb/2026-03-26_ImprovedHyperparams/)
  - `NGramCache` submission
  - `colab_run.ipynb` — GPU-aware launcher notebook
  - `COLAB_NOTES.md` and `EXPERIMENTS_NOTES.md`

The notebook (`colab_run.ipynb`) clones from the fork, auto-detects GPU compute capability, and routes to the correct script. Blackwell (cc ≥ 12) is now a distinct tier: uses root script with `TORCHDYNAMO_DISABLE=1` and the 8L 512d model config.
