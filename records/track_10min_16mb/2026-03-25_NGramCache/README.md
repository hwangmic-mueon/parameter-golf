# N-gram Eval Cache on XSA-all + FullGPTQ + Parallel Muon

**Author:** michaelhwang
**Base:** 2026-03-24 XSA-all FullGPTQ ParallelMuon (1.1154 BPB SOTA)
**Approach:** Online n-gram interpolation during final evaluation

## Summary

This submission adds an entropy-adaptive online n-gram cache to the evaluation pass of the current SOTA model. Training is identical to the 2026-03-24 XSA submission; only the final evaluation step is changed.

### Key idea

While scoring the validation set left-to-right, we build running unigram/bigram/trigram statistics from previously-seen tokens. For each position, we interpolate neural model probabilities with n-gram probabilities using an entropy-adaptive alpha:

```
alpha = cache_scale * (0.05 + 0.55 / (1 + exp(-2 * (H - 4.0))))
```

where `H` is the neural model's predictive entropy at that position. This means:
- When the neural model is confident (low H), n-grams are barely used (alpha ≈ 0.05)
- When the neural model is uncertain (H > 4 nats), n-grams contribute significantly (alpha → 0.60)
- `cache_scale` ramps from 0 → 1 as the cache fills (>5M tokens), avoiding near-uniform noise early

### Why this helps

With vocab_size=1024 and 62M val tokens:
- Average 62 observations per bigram (well above the threshold of 5 for reliable estimates)
- Average 950 observations per trigram hash bucket
- For high-entropy positions where the neural model struggles, bigram/trigram context from the same validation stream is highly reliable

The improvement is specifically for the subset of tokens where the model is genuinely uncertain. With a 1024-vocab tokenizer, many common English n-gram patterns (articles, prepositions, punctuation sequences) are highly predictable.

### Architecture (unchanged from SOTA)

- 11 layers, 512 dim, 8H/4KV
- LeakyReLU(0.5)² MLP 3×
- BigramHash(2048), Partial RoPE 16/64
- XSA-all (Exclusive Self-Attention on all layers)
- U-Net skip connections, SmearGate, EMA(0.997), Tight SWA
- GPTQ int6 + lzma compression
- Parallel Muon optimizer
- Eval: seq_len=2048, stride=64 sliding window + n-gram enhanced pass

### Implementation notes

The `eval_val_ngram` function:
1. Processes sequences in batches of 64 for GPU efficiency (neural forward pass)
2. Updates the n-gram cache sequentially per-sequence (preserves causal order)
3. Runs on rank 0 only (sequential n-gram is inherently single-process)
4. Uses `NGramEvalCache` with int32 bigram counts (4 MB) and int16 trigram hash (128 MB)

### Estimated BPB improvement

With 62M val tokens and vocab=1024, bigrams are well-populated after the first ~5M tokens.
The expected improvement over pure neural eval is 0.01–0.05 BPB, depending on the distribution
of high-entropy positions in the val set.

## Submission details

- val_bpb: TBD (requires H100 run)
- bytes_total: TBD
- Training time: ≤10 min on 8×H100 (identical to base)
- Eval time: ≤10 min on 8×H100 (parallel sliding window + sequential n-gram)
