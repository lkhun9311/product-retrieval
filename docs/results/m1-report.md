# M1 report: correction-trained reranking

Milestone M1 asked whether a small reranker trained on correction labels can lift a frozen image-retrieval baseline. This report collects the question, the setup, the numbers, what failed, and the limits. Every number links to a committed result file.

## Question and preregistered target
- Hypothesis H1-rerank-r5 (D17, contract `c4-label-selection.md` v3, fixed 2026-10-04): a reranker trained on correction labels lifts product-level macro Recall@5 from the frozen baseline to **0.887** within 3,000 labels.
- Recall@K here: for each query, 1 if the correct product is in the top K; average within each product, then across products.

## Setup
| Part | Choice | Where |
|---|---|---|
| Data | LRVS-Fashion, product id = same model and colour; val 3,243 products / 6,782 queries, test 3,196 / 6,709 | `data/`, D21 |
| Retrieval | Frozen SigLIP2 base, exact FAISS, product score = max cosine over gallery images | `index/flat.py` |
| Corrections | Simulated: the simulator answers match/not-match from the truth; label budget = one judgement on one query–candidate pair | `feedback/simulate.py`, `c4-label-selection.md` |
| Labels | match → pos, not_match → neg, skip and no response → no label; conflicts resolved by in-session last judgement, cross-session majority, ties dropped | `c4-event-labels.md` |
| Reranker | v2: MLP over [q⊙g, \|q−g\|, 10 scalar features], top 20 only | `c5-reranker-v2.md` |

## Results

### Development validation set (curve-v1, 108 runs, all completed)
Primary condition (stratified, noise 0), mean of three seeds:

| Labels | 100 | 300 | 1,000 | 2,000 | 3,000 |
|---|---:|---:|---:|---:|---:|
| R@5 | 0.8515 | 0.8557 | 0.8556 | 0.8569 | 0.8664 |

Baseline 0.8571. No seed reached 0.887 at any preregistered point. Only at 3,000 labels was the improvement interval above zero for all seeds. Source: `docs/results/curve-v1/`.

### Test set, opened once (`docs/results/m1-test-check.md`)
Mean R@5 of the three fixed models 0.8696 against a test baseline of 0.8600 (Δ +0.0096; target Δ +0.03). **Not reached.** The test gain matches the validation gain at 3,000 labels.

**Verdict on H1-rerank-r5: not reached** on either split.

## What failed and what was learned
1. **Score-derived features cannot reorder.** Rerankers v0 (5 features from scores and ranks) and v1 (+5 image-level similarity statistics) stayed at R@5 0.856–0.858 even with 100,000 simulated labels. Every feature came from the same embedding similarity the baseline already ranks by (PR #17, #19).
2. **Embedding interaction helps, slowly.** v2 is the first reranker whose R@5 rises with labels (0.894 with 100,000 labels as a diagnostic), but at the 3,000-label budget the gain is about +0.01 (PR #21, #22).
3. **R@1 does not improve.** On test all three seeds have Δ R@1 below zero with intervals including zero.
4. **Noisy corrections are dangerous without a gate.** With 20% flipped answers, one seed fell to R@5 0.30 at 100 labels and the 3,000-label mean was 0.844, below the baseline. A deployment gate that refuses a model worse than the current one is required before any correction-trained model ships.
5. **A leak was found and closed before any experiment.** The old query id `source:product_id:sha12` exposed the truth for 100% of 53,882 training queries; ids are now opaque (PR #14).

## Limits
- All corrections are simulated from the truth. No real-user effect is claimed.
- The validation set also selected v2 over v0/v1, so validation numbers carry selection bias; the test check is the unbiased number and it agrees.
- Gallery sizes are 3,196–3,243 products. Recall drops as galleries grow (200 → 3,243 products moved R@1 from 0.916 to 0.697), so real catalogues will score lower.
- The source data comes from a small number of online shops, so styles and photo conditions are narrower than real use.

## Beyond M1 (M2, started)
- DeepFurniture rendered-scene baseline: R@5 0.799 on 2,527 identities (`docs/results/deepfurniture-baseline-v0.md`).
- Real IKEA room photos: whole-photo search finds few listed items (Recall@10 0.118); a detector → crop → merge pipeline doubles that to 0.248, Δ +0.130 [+0.112, +0.148] (exploratory, `docs/results/crop-before-search-v1.md`).

## Next options
- Label efficiency: uncertainty-based selection (the follow-up contract named in c4-v3), or partial fine-tuning of the embedder (D17 comparison). Both need a new preregistered version, reported next to this one.
- Pointing at one item (click or segmentation) on real photos, the step the crop result points to.
