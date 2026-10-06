# Crop before search v1 (exploratory)

Run `20261006T021624935656Z-f98ebcc8`, 2026-10-06, under `docs/contracts/crop-before-search.md` (sha256 10e64eee…). Same 205 real IKEA rooms and 2,173-product gallery as the IKEA check (population flag `matches_29_counts: true`, identical across arms). Frozen SigLIP2; detector OWLv2 at revision cfd3195b.

**Exploratory:** the settings were chosen after seeing the aggregate IKEA numbers, and this is the same set of rooms. The claim is about the detector → crop → merge pipeline against whole-photo search, not about cropping alone.

## Primary result
Mean Δ Recall@10 (crops − whole) = **+0.1295**, 95% cluster-bootstrap interval **[+0.1116, +0.1475]** (B = 10,000, 202 clusters). Rooms with Δ > 0 / = 0 / < 0: 157 / 39 / 9. Resample share with Δ ≤ 0: 0.0000 (not a p-value).

Verdict (wording fixed in the contract): **the crop pipeline raised Recall@10 on these rooms.**

## All K

| K | Hit whole | Hit crops | Δ | Recall whole | Recall crops | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2244 | 0.4732 | +0.2488 | 0.0312 | 0.0459 | +0.0147 |
| 5 | 0.5707 | 0.8683 | +0.2976 | 0.0823 | 0.1593 | +0.0770 |
| 10 | 0.7415 | 0.9463 | +0.2049 | 0.1183 | 0.2478 | +0.1295 |
| 20 | 0.8341 | 0.9756 | +0.1415 | 0.1430 | 0.3141 | +0.1711 |
| 100 | 0.9463 | 0.9902 | +0.0439 | 0.2413 | 0.4593 | +0.2179 |

The whole-photo column reproduces the IKEA check numbers exactly.

## Alongside
- Boxes per room: min 9, median 10, max 10; 7 rooms had fewer than 10; no room fell back to the whole photo.
- CPU time per room for the crop arm: detector about 1.08 s, crop embeddings about 0.33 s (mean). The whole-photo embeddings were cached from the IKEA check, so that arm shows 0 s.
- Visual check of one kitchen room: boxes land on a stool, a pendant lamp, the dining table and chairs, a cushion, the sofa, wall cabinets and a picture frame.

## Not claimed
- A user pointing at one item: unmeasured.
- Other detectors, vocabularies or M: a new contract version.
- Generalisation beyond staged catalogue rooms from one retailer. Confirmation needs a fresh real-photo set.
