# DeepFurniture frozen baseline v0

Measured 2026-10-06 on the DeepFurniture **val** split. Frozen SigLIP2 base (`google/siglip2-base-patch16-224`), full-image query crop, exact FAISS search, product score = max cosine over gallery images. Same metric definition as M1: product-level macro Recall@K, product bootstrap 1,000 resamples, seed 0. Test split not opened.

## Data
- Manifest `deepfurniture-manifest-v1`, identity-level split (hash, seed 0), leakage gate ok (no identity in two splits, no shared image).
- val: 2,527 identities in the gallery (one rendered preview each), 312 of them have queries, 648 query crops. The other 2,215 identities are gallery-only distractors.
- Queries are crops of **rendered** scenes. The crop-to-identity link was checked on one scene archive (287/287 match); the other 24 archives were not downloaded.

## Result (index 72144653c570)

| K | macro R@K | 95% interval | micro R@K |
|---:|---:|---|---:|
| 1 | 0.5946 | [0.5433, 0.6416] | 0.5926 |
| 5 | 0.7991 | [0.7585, 0.8376] | 0.8009 |
| 10 | 0.8483 | [0.8093, 0.8841] | 0.8472 |
| 100 | 0.9722 | [0.9556, 0.9856] | 0.9722 |

R@100 − R@5 = 0.173: the right answer is usually among the first 100 candidates but not near the top, the same shape as the fashion baseline (0.981 − 0.857 = 0.124 on 3,243 products).

## Not claimed
- Real-photo performance: all queries are rendered. The IKEA real-photo check is not done yet.
- Comparison with fashion numbers: different galleries (2,527 vs 3,243 products) and one preview image per identity here.
