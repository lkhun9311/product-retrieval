# Crop before search on real room photos v1

Fixed 2026-10-06 (main session), issue #32, before any crop result is computed.
Builds on `ikea-real-photo-check.md` v1.1 (same rooms, gallery, answers, metrics).
One adversarial review (Codex Astra, high) before running; 9 findings, all applied (marked `[R#]`).

## Question
Compared with searching the whole room photo, does a **detector → crop → merge** pipeline raise per-room Recall@K at the same K? The comparison is between two complete pipelines. It does not isolate the effect of cropping from the detector's knowledge or from searching up to 10 queries. `[R1]`

## Selection history (disclosed) `[R4]`
- The idea, vocabulary, M and merge rule were chosen after seeing the aggregate #29 numbers (Hit@5 0.571, Recall@5 0.082). No per-room result or ranking was inspected.
- These are the same 205 rooms. This run is therefore **exploratory**; no fresh real-photo set exists yet to confirm it. Every attempt is kept in `reports/crop/`.
- SigLIP2 and OWLv2 may have seen IKEA catalogue images during pre-training; this is not checked.

## Population `[R5]`
- Gallery, rooms and answer sets are exactly those of the #29 run (2,173 products, 205 rooms). The run writes their lists and sha256 and asserts: every room has ≥ 1 answer, and both arms see identical rooms, gallery and answers.
- The endpoint is recall **against the mapped labels**. Labels are not audited; visible but unlisted products earn no credit in either arm.

## Arms (same frozen SigLIP2 `google/siglip2-base-patch16-224`)
- **Whole**: the whole room photo is one query (recomputed in the same run).
- **Crops**: detector boxes → crops → per-crop ranking → merge.

## Detector, frozen `[R2]`
- `google/owlv2-base-patch16-ensemble` at revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`, `transformers` 5.18.0, CPU, float32. Images are EXIF-transposed and converted to RGB before anything else.
- Prompts, fixed: bed, sofa, armchair, chair, stool, table, desk, shelf, bookcase, cabinet, wardrobe, chest of drawers, lamp, ceiling lamp, mirror, rug, curtain, cushion, pillow, quilt, plant pot, clock, picture frame, storage box. Generic; no room's own product list is used.
- Box score = the maximum sigmoid score over the 24 prompts for that predicted box. Boxes come from `Owlv2Processor` post-processing with `target_sizes = (max(H, W), max(H, W))` (OWLv2 pads to a square), then clipped to the image.
- Keep boxes with score ≥ 0.10. Greedy NMS at IoU 0.5 over all kept boxes, highest score first; equal scores break by predicted-box index. Then keep the first **M = 10**.
- Crop = box expanded by 10% of its width/height on each side, clipped, rounded outward to integer pixels (floor for min, ceil for max). Crops narrower or shorter than 16 px are dropped.
- A room with zero surviving crops uses the whole photo as its only crop; such rooms are counted.

## Merge rule `[R3]`
- Each crop is ranked against the gallery by cosine (one image per product). Equal cosines break by product id (ascending).
- Traversal order is (rank depth, crop order): depth 1 of crop 1 … crop m, then depth 2 of crop 1 … crop m, where m is the number of surviving crops and crop order is detector score order.
- A product already in the merged list is skipped and its turn is consumed (traversal moves on). Continue until K unique products or until every crop list is exhausted.
- Both arms are cut at the same K, so each shows exactly K products.

## Metrics and comparison `[R6][R7][R8]`
- Per room Hit@K and Recall@K, K = 1, 5, 10, 20, 100 (definitions from the IKEA contract).
- **Primary: Δ Recall@10 = Crops − Whole**, paired by room. Report the observed mean Δ over the 205 rooms, and the counts of rooms with Δ > 0, = 0, < 0.
- Uncertainty: paired cluster bootstrap, B = 10,000, `numpy.random.default_rng(0)`, 95% percentile interval (`numpy.quantile`, default linear method). The interval describes how the mean Δ would vary over resampled room clusters like these; it is not a statement about other retailers or homes.
- Clusters: rooms whose whole-photo SigLIP2 embeddings have cosine ≥ 0.95 are linked; connected components are clusters (computed before any crop ranking). Report the cluster count. Name-based grouping found 205 distinct rooms.
- Verdict sentence, fixed now: lower bound > 0 → "the crop pipeline raised Recall@10 on these rooms"; upper bound < 0 → "the crop pipeline lowered Recall@10 on these rooms"; otherwise "the interval includes zero" (not evidence of equivalence).
- Secondary: Δ for the other K and for Hit@K, reported, not used for the verdict. "Δ ≤ 0 resample share" is reported and is not a p-value.

## Reported alongside `[R1]`
- Boxes per room (min / median / max), rooms with fewer than M boxes, rooms falling back to the whole photo.
- Detector, embedding and total time per room for each arm.

## Not claimed `[R9]`
- Not a measure of a user pointing at one item; that is unmeasured.
- One detector, one vocabulary, one M. Other settings are a new contract version, reported next to this one.
- Staged catalogue rooms from one retailer, not home snapshots.
