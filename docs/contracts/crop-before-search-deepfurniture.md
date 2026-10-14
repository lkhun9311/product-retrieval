# Crop before search on DeepFurniture scenes v1

Fixed 2026-10-14 (main session), issue #40, before any crop result on DeepFurniture is computed.
One adversarial review (Codex Astra, high) on the draft; 8 findings, all applied after checking each against the data (marked `[F#]`).

## Question
Does the detector → crop → merge pipeline fixed in `crop-before-search.md` (IKEA, exploratory) raise per-scene Recall@10 over whole-scene search on scenes it has never been run on? A ground-truth-box reference arm is reported alongside.

## What this confirms, and what it does not
- Every detector, crop and merge setting is copied unchanged from `crop-before-search.md`, whose sha256 must equal **`10e64eee0f82d4d5024b5f1cafbf097fd0d1c7bad60210c5abed485dfadce8cc`** (the hash recorded by the IKEA run). The run asserts this before anything else and also records the code commit and the executable settings (prompts, thresholds, M, padding, revision) it actually used. `[F7]`
- Nothing is tuned on DeepFurniture.
- **Overlap disclosed and removed:** 22 scenes that would otherwise qualify supplied the 22 val query crops of baseline v0 from this archive. They are excluded, so no scene here was used in an earlier result. `[F3]`
- Limits: scenes are **rendered**, not photographed; answers per scene are few (1–3); the gallery has one rendered preview per product. The verdict tests the direction of the IKEA result, not its size.

## Population (fixed now)
- Scenes: every scene in `scenes_0000.tar.gz` (the one scene archive on disk, 1,000 scenes; the first archive, not chosen by content) whose annotation has **at least one val identity and no test identity** under manifest `deepfurniture-manifest-v1`, **minus** scenes named as the source scene of any val query in `metadata/query_index.json`. Counted from annotations only: **228 scenes, 285 answers**, at most 3 per scene. The run asserts these counts.
- Scenes containing a test identity are excluded so that no test product is an answer or a query; the test split stays closed.
- Answers of a scene: the set of its distinct val identities. Instances of train identities stay in the image but are not answers and are not in the gallery.
- Gallery: the val gallery of baseline v0 (2,527 identities, one preview image each).
- The run writes the scene list, answer sets and gallery list with sha256, and asserts that all three arms see the same scenes, answers and gallery.

## Images and coordinates `[F2]`
- Images are used at their native size; sizes differ between scenes (for example 1600 × 1200 and 4000 × 3000). Annotation boxes are in native pixel coordinates.
- Before any crop, every annotated box must lie inside its image after clipping and have positive area; a box entirely outside the image is an error that stops the run (not silently dropped). The run reports image sizes found.

## Embedding model `[F6]`
- SigLIP2 `google/siglip2-base-patch16-224` at revision **`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`** for model and processor. Gallery, whole-scene and crop embeddings are all computed in this run with that revision (no cached vectors from other runs). Product score = max cosine.

## Arms
- **Whole:** the whole scene image is one query.
- **Detector:** exactly the IKEA contract's detector arm: OWLv2 `google/owlv2-base-patch16-ensemble` at revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`, the same 24 prompts, score ≥ 0.10, NMS IoU 0.5, M = 10, 10% padding, 16 px minimum, whole-image fallback when no crop survives, merge by rank depth with the same tie rules.
- **GT box (reference):** the annotated boxes of all instances in the scene regardless of split (box positions only, no identity), ordered by `areaRatio` descending; instances without `areaRatio` (14 in this population) come after all others, ordered by `numberID` ascending; ties among known values by `numberID` ascending. Cut to the first M = 10, then the same padding, minimum size and merge as the detector arm. The run reports how many instances lacked `areaRatio` and how many answers had no box within the first 10. `[F1]`
- The GT-box arm differs from the detector arm in box geometry **and** in box selection and order, so Δ (GT box − Detector) is not a measure of detector error alone. It is a reference for a capped, area-ordered set of annotated boxes and is never presented as a deployable result. `[F4]`
- All arms cut the merged list at the same K.

## Metrics and comparison
- Per scene Hit@K and Recall@K, K = 1, 5, 10, 20, 100 (definitions from `ikea-real-photo-check.md`).
- **Primary: Δ Recall@10 = Detector − Whole**, paired by scene; mean over the 228 scenes; counts of scenes with Δ > 0, = 0, < 0.
- Uncertainty: paired cluster bootstrap, B = 10,000, `numpy.random.default_rng(0)`, 95% percentile interval. Clusters link scenes whose whole-scene SigLIP2 embeddings have cosine ≥ 0.95 (computed before any crop ranking); report the cluster count.
- **Minimum clusters `[F5]`:** an inferential verdict needs **at least 20 clusters**. With fewer, report the observed mean Δ and the sentence "insufficient independent clusters".
- **Verdict sentence, fixed:** lower bound > 0 → "the crop pipeline raised Recall@10 on these DeepFurniture scenes"; upper bound < 0 → "the crop pipeline lowered Recall@10 on these DeepFurniture scenes"; otherwise → "the interval includes zero" (not evidence of equivalence).
- **Secondary, not used for the verdict:** Δ (GT box − Whole) and Δ (GT box − Detector) at K = 10 with the same bootstrap; Δ for all K and for Hit@K; boxes per scene; scenes falling back to the whole image; detector and embedding time per scene.
- **Per category `[F8]`:** for Whole, Detector and GT box, pooled Recall@10 over distinct (scene, val identity) pairs whose instance has that category (an identity with instances of two categories in one scene counts once per category), with each category's denominator. Descriptive only.

## Not claimed
- Real-photo performance; these scenes are rendered.
- That the IKEA effect size carries over; only the direction is tested by the verdict.
- A user pointing at one item.
- Scenes from the other 24 archives.
- That Δ (GT box − Detector) measures detector error alone.
