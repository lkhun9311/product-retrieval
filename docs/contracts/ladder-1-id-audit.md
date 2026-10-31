# Quality ladder step 1 · Product-ID audit v1

Drafted 2026-10-31 (main session), issue #43, milestone M5, before any audit judgement exists.
Order of work agreed with Codex Astra (high). One adversarial review (Codex Astra, high) on the draft: 19 findings (6 critical, 12 important, 1 minor), all applied (marked `[R#]`). §1 still needs the user's confirmation; nothing is reviewed before it.

## Question
How often are the LRVS-derived validation labels wrong in ways that change what "the same product" means for retrieval, and how often is the target of a query not identifiable? The answer decides whether the ladder continues on these labels, continues after fixes, needs a target-specification track, or stops.

## Why now
- Product IDs were grouped from URLs with regular expressions; CATEGORY is a synthetic label; BLIP2 captions were generated knowing the target. None of these is an independent truth.
- `check_manifest` checks missing images, byte-identical images shared across IDs (0 in the manifest) and one ID in two splits. It does not check whether two IDs are the same product. Passing it is not a semantic audit.
- "The bottleneck is the embedding" (D22) is a hypothesis. Label errors and target ambiguity have not been ruled out.

## 1. Same-product unit — needs the user's confirmation
- **Proposed:** one product = one model in one colour or pattern. Size and attributes a photo cannot show are "not visible", never a difference.
- Relations between two images: **same** · **variant** (same model, different colour or pattern) · **different** · **undecidable**.
- **Same needs positive evidence** `[R9]`: at least one distinctive feature visible in both (print or pattern, logo or label, a distinctive construction detail such as pocket, closure or seam layout) and no contradicting feature. "No visible difference" alone is **undecidable**, not same. A worked table of 12 example pairs (3 per relation) is fixed with the review tool before review starts.
- Only **same** is a valid truth relation. Variant is recorded as its own relation and **counts as a truth error** wherever it appears as the labelled truth `[R1]`.

## 2. Population and sample
- Population: the 3,243 validation products of `manifest_lrvs.jsonl`; the run records the manifest sha256 and refuses to start if it changes. All 3,243 product IDs are present in `train_products_*_metadata.parquet` and none in `test_*` (checked 2026-10-31). Results describe this split only.
- Strata from the manifest (queries per product × gallery images per product):

| h | stratum | N_h | sample n_h | pilot |
|---|---|---:|---:|---:|
| 1 | 1 query · 1 gallery | 1,538 | 285 | 95 |
| 2 | 2 queries · 1 gallery | 508 | 94 | 31 |
| 3 | 3+ queries · 1 gallery | 1,025 | 190 | 63 |
| 4 | 2+ gallery (any queries) | 172 | 31 | 11 |
| | total | 3,243 | 600 | 200 |

- Draw `[R19]`: product IDs sorted as integers; stratum h shuffled with `numpy.random.default_rng([20261031, h])`; the first n_h form the sample, the first pilot_h of those form the pilot. Display order for every stage is a fresh permutation of all items in that stage with `default_rng([20261032, stage])`. The 600 IDs, the 200 pilot IDs and their sha256 are committed before any review.
- Risk group (up to 40 products, never mixed into any rate): drawn from the 2,643 products **outside** the sample, ranked by highest cross-ID product score (§3 B) and same-normalised-title collisions; reported separately.
- No replacement; a product is never swapped because it is hard.

## 3. Stages — run globally, one stage at a time `[R3]`
Each stage covers all items of the current batch (pilot 200, then the rest) before the next stage starts. Earlier stages are locked when the next one opens. Titles are shown only after every image stage **and** its re-review are locked.

1. **Target (query only).** Every query image alone. The reviewer marks the target by drawing a box on the garment the query is about, or chooses ambiguous (several garments could be meant) or not visible `[R2]`. The box is saved before any gallery image of that product is ever shown.
2. **C · query to truth.** All query × gallery pairs of each product (1,310 pairs for the 600) `[R10]`, each judged against the boxed target, globally shuffled and mixed 1:1 with decoy pairs. A decoy pairs the query with the gallery image of the most similar **other** product (product score, §3 B) in the same CATEGORY, drawn with `default_rng([20261034, query])`, never a product in the same sample and never the query's own product. A decoy judged **same** is recorded as a B finding.
3. **A · inside the ID.** All remaining image pairs within each product (query–query, gallery–gallery; ≤ 9 images per product) — no single reference image `[R9]`. Splits and merges are decided from the full pair graph, never by transitivity from partial matches.
4. **B · across validation IDs** `[R11]`. Product score = max cosine between any image of the two products, frozen SigLIP2 `google/siglip2-base-patch16-224` at the revision used by the baseline, same preprocessing; the 5 highest-scoring other validation products (ties by product ID), plus up to 3 products sharing a normalised title. All gallery images of each candidate are shown; the candidate source is hidden. Model revision, embedding file sha256 and the final candidate list sha256 are committed before review.
5. **D · train–validation** `[R12]`. (a) Perceptual-hash near-duplicates between every sampled validation image and every train image (Hamming ≤ 6 on 64-bit pHash), all reviewed; (b) the 5 train products with the highest product score over train query **and** gallery images. Reviewed like B.
6. **Titles** `[R8]`. Text field `ENG_TEXT` from the train metadata parquet, joined by image URL → image sha256 (join conflicts are an error that stops the stage). Normalised = lowercase, punctuation removed, trailing image numbers removed. A product may have several normalised titles; all are kept. Titles are shown only now, and any change they cause is a separate **title-assisted** judgement; the primary analysis uses image-only judgements. Titles come from the same source as the IDs and are never independent evidence of same.

The reviewer never sees model ranks, scores, retrieval success, which pair is the labelled truth, or BLIP2 captions.

## 4. Events and measures `[R4]`
Per sampled product, after all image stages:
- **ERR** = at least one confirmed error: an A pair **different** or **variant**, or a C pair whose relation to the labelled truth is **different** or **variant** for a query with an identified target.
- **UNR** = no ERR, but at least one required A or C judgement is **undecidable**, missing, or the product is partly reviewed.
- **AMB** = at least one query whose target is ambiguous or not visible (reported on its own; its C pairs are not counted in ERR or UNR).
- **F = ERR ∪ UNR** — the primary label-quality event. Denominator per stratum = n_h as drawn, whatever was completed.
- Estimate p̂ = Σ_h W_h p̂_h with W_h = N_h / 3,243.
- **Interval `[R5]`** (Korn and Graubard 1998, stratified Clopper–Pearson): variance v = Σ_h W_h² (1 − n_h/N_h) p̂_h(1 − p̂_h)/(n_h − 1); effective n* = p̂(1 − p̂)/v, capped at 600, and n* = 600 when p̂ = 0; degrees-of-freedom adjustment n*_df = n* · (z_{0.95} / t_{0.95, 600−4})²; x* = n*_df · p̂; one-sided 95% upper = Beta⁻¹(0.95; x* + 1, n*_df − x*), lower = Beta⁻¹(0.05; x*, n*_df − x* + 1) (0 when x* = 0). Implemented in the repository with a test that reproduces these worked values (computed 2026-10-31 with SciPy):
  - 0 events in 600 → upper 0.50%
  - 1 event (stratum 1) → upper 0.79%
  - 2 events (both stratum 1) → upper 1.05% — fails the 1% rule, so the audit passes only with at most 1 failing product
  - 6 events all in stratum 4 → p̂ 1.03%, upper 2.00%, lower 0.45%; 6 events spread 3 · 1 · 2 · 0 → p̂ 1.00%, upper 1.96% — weight, not count, decides.
- Secondary, never in F: AMB rate; query-level error rate averaged within products; B and D duplicates **found by this procedure** (a discovery count, not a rate); variant pairs; decoys judged same; time per stage.

## 5. Re-review (one reviewer) `[R14]` `[R15]`
- **Pilot:** 40 of the 200 (8 per 40 drawn proportionally to strata, seed `[20261033, 1]`), at least 7 days after the pilot's last image stage, new display IDs and order, first judgements hidden. The remaining 400 start **only after** the pilot re-review is analysed.
- **Full:** 80 more from the 400 (seed `[20261033, 2]`); the full re-review set is these 80 plus the pilot 40.
- Disagreement is counted per product (any change in a core relation: same, variant, different, undecidable, or the target box/ambiguity) and per judgement; both are reported with the confusion table.
- Product-level disagreement above 10% in the pilot → stop, fix definitions in a new contract version, re-review all 200. Above 10% in the full set → the full audit cannot pass.
- Every first and final judgement is kept; re-reviewed products are not given different final rules from the rest.
- This measures one reviewer's consistency. Without a second reviewer, every conclusion is "the rate this procedure detected". If a second reviewer is available, a blind check of 50 products (including 25 random **same** judgements) is added and reported; it is not required to pass.

## 6. Branch rules — fixed now, applied in this order `[R6]` `[R7]`
| order | condition | next step |
|---|---|---|
| 1 | review incomplete (any n_h not fully judged) `[R18]` | state **incomplete**: no promotion, report counts and reason |
| 2 | target identity or the same-product rule unworkable (pilot re-review stop twice) | rewrite the task definition before any rate |
| 3 | any confirmed train–validation duplicate | quarantine the linked set from both splits; validation items never move to train; models trained on the contaminated train are flagged and their results annotated; retraining is a separate decision |
| 4 | any confirmed B duplicate (two validation IDs the same product) | promotion blocked until those IDs share one truth set and the evaluator and index are rebuilt |
| 5 | AMB one-sided 95% upper bound above 10% | the image-only track is evaluated only on target-identified queries; the rest go to a target-specified track (box or category, ladder step 2); both reported |
| 6 | F lower bound above 5% | hold same-product training and promotion; narrow the source, redefine the task or consider another dataset |
| 7 | F upper bound below 1% | fix every error found, freeze truth v2, go to ladder step 2 |
| 8 | otherwise | fix the rules behind the errors (source, variant, mapping), write truth v2, then **validate the fixed rules on a fresh sample** of 200 products drawn from the 2,643 not yet sampled (seed `[20261035, h]`, same strata proportions) with the same stages; pass = F upper bound below 1% on that sample. The original 600 are never used to confirm their own fixes. |
- 1%, 5% and 10% are working thresholds chosen now, not values from literature.
- The pilot alone never passes the audit; it checks procedure, time and definitions. A definition change after the pilot is a new contract version and the 200 are reviewed again.

## 7. Truth v2 and baselines `[R13]` `[R16]` `[R17]`
- Per query, statuses: **confirmed** (labelled truth judged same) · **corrected** (another product judged same) · **original invalid, replacement not found** · **absence confirmed** · **not reviewed**. Absence confirmed requires that the top 50 validation products by product score and all title matches were reviewed and none is same; only those queries move to a no-match track. A top-5 miss is never absence.
- Truth v2 records image-level membership (each image's product), a set of valid truth products per query (not a single ID), the status above and its evidence. Existing opaque query IDs are kept; a mapping old ID → new IDs covers splits and merges. Supporting truth sets and statuses in the schema and evaluator is part of #44 and must be done before any v2 baseline.
- Two comparisons on truth v2: (a) **same queries, same gallery, same product weights**, only the truth judgement changed; (b) the full v2 evaluation with merges, splits, exclusions and rebuilt index. Changes in the number of queries and products and in coverage are reported with both. Model comparisons only ever use one truth version.
- Validation judgements are never used as training labels. The test split stays a regression set; correcting labels does not make it unopened again.

## 8. Workload `[R18]`
- Before review the tool reports item counts per stage. For the 600: target ≈ 1,170 queries, C ≈ 1,310 pairs + 1,310 decoys, A ≈ 1,300 within-product pairs, B ≤ 4,800 candidates, D ≤ 3,000 candidates plus pHash hits.
- The pilot measures seconds per item per stage. If the projected total for the 600 exceeds 40 hours, B and D drop to the top 3 (fixed now as the only fallback), and the coverage note says so. Any other cut makes the audit **incomplete**.

## Not claimed
- Exact SKU identity (size, edition).
- A complete duplicate rate: B and D report what this procedure found.
- Reviewer accuracy: one reviewer, consistency only.
- Anything about train, test or LRVS outside this validation split.
