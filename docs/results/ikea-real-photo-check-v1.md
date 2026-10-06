# IKEA real-photo check v1

Measured 2026-10-06 under `docs/contracts/ikea-real-photo-check.md` v1.1. Frozen SigLIP2 base, whole room photo as the query (no crop), gallery = IKEA product images. Room-level bootstrap, 1,000 resamples, seed 0.

## Data
- 205 real room photos, 0 dropped. Gallery 2,173 products (2,175 found minus the 2 ambiguous images excluded by v1.1).
- Correct products per room: min 2, median 13, max 27.

## Result

| K | Hit@K | 95% interval | Recall@K | 95% interval |
|---:|---:|---|---:|---|
| 1 | 0.2244 | [0.1707, 0.2878] | 0.0312 | [0.0218, 0.0414] |
| 5 | 0.5707 | [0.5024, 0.6390] | 0.0823 | [0.0666, 0.0987] |
| 10 | 0.7415 | [0.6780, 0.7951] | 0.1183 | [0.1003, 0.1388] |
| 20 | 0.8341 | [0.7854, 0.8829] | 0.1430 | [0.1242, 0.1648] |
| 100 | 0.9463 | [0.9171, 0.9756] | 0.2413 | [0.2200, 0.2653] |

## Reading
- For 57% of rooms at least one of the listed products is in the top 5, but only 8% of all listed products are. A whole-room embedding finds one or two dominant items and misses the rest (median 13 products per room).
- This supports pointing at one item (a crop) before searching, which the product design already assumes. Cropping has not been measured yet.

## Not claimed
- Not comparable with single-answer product-level Recall@K (LRVS, DeepFurniture).
- Not a measure of pointing at one item in a real photo.
