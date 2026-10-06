# M1 one-time test-split check

Run 2026-10-06 under `docs/contracts/c4-learning-curve.md` §7, which fixed the models, the target and the verdict before the validation curve was seen. This is the only time the test split was opened (access log: `reports/test_access.jsonl`, entries for eval, cand-stats and three rerank-v2 runs).

## Setup (fixed in advance)
- Models: reranker v2 trained on 3,000 labels, stratified policy, noise 0, seeds 0, 1, 2 (versions cadfa7929f8e, b43a45cb35c5, 6bdb7fec4b7b, from curve-v1). No other model, budget or seed was evaluated on test.
- Test split: 3,196 products, 6,709 queries, index a4a3ace2fd7a (frozen SigLIP2, product-level max cosine).
- Success: mean test R@5 over the three seeds ≥ test baseline R@5 + 0.03.

## Result: **not reached**

| | macro R@5 | Δ vs baseline | 95% interval (paired product bootstrap, B=1,000, seed 0) |
|---|---:|---:|---|
| Baseline | 0.8600 | | |
| Seed 0 | 0.8705 | +0.0105 | [+0.0059, +0.0149] |
| Seed 1 | 0.8736 | +0.0136 | [+0.0092, +0.0179] |
| Seed 2 | 0.8646 | +0.0046 | [+0.0011, +0.0080] |
| **Mean of 3 seeds** | **0.8696** | **+0.0096** | |

Required: 0.8600 + 0.03 = 0.8900. The mean falls short by 0.0204.

## Other K (per seed, Δ and interval)
| K | Seed 0 | Seed 1 | Seed 2 |
|---:|---|---|---|
| 1 | −0.0045 [−0.0138, +0.0047] | −0.0023 [−0.0104, +0.0057] | −0.0049 [−0.0144, +0.0046] |
| 10 | +0.0076 [+0.0043, +0.0105] | +0.0095 [+0.0057, +0.0129] | +0.0027 [−0.0002, +0.0058] |
| 20 | 0 | 0 | 0 |

R@20 does not move because reranking only reorders the top 20.

## Reading
- The test result matches the development-validation curve at 3,000 labels (mean Δ +0.0093 there, +0.0096 here). The small but positive R@5 gain carries over; the +0.03 target is not supported on either split.
- R@1 does not improve on test for any seed (all intervals include zero, point estimates below zero).
- Raw comparison output: `docs/results/m1-test-check/compare_seed{0,1,2}.json`.
