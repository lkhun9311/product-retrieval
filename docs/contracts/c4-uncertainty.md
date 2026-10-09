# C4 · Uncertainty label selection v1

Follow-up contract required by `c4-label-selection.md` v3 §3 ("uncertainty: progression rule, batch size, retraining and tie-breaking fixed before running"). Fixed 2026-10-09 before any uncertainty-policy result is computed.
One adversarial review (Codex Astra, high); 5 findings, all applied (marked `[U#]`).

## Question
At the same label budget, does choosing which candidates to label by the current reranker's uncertainty give higher validation R@5 than the preregistered stratified selection?

## Selection history (disclosed)
- The stratified curves for seeds 0–2 (curve-v1) are known; the uncertainty policy has never been run. The comparison rule below is fixed before any uncertainty result.
- This is a new hypothesis (`H2-uncertainty`), not a change to H1. H1's judgement stays as recorded.

## 1. Fixed from c4 v3 and c5 v2 (unchanged)
- Candidate pool: top 20 of every training query in the frozen SigLIP2 train rankings (sha256 recorded); about 1.08 million pairs.
- Answers and noise: c4 v3 §5. This contract runs **noise p = 0 only**.
- Shared initial labels: I_s = the first 100 of the stratified order for seed s, with their answers (c4 v3 §4).
- Reranker: c5-rerank-v2, same hyperparameters, training seed = s, holdout early stopping on the train split only.
- Evaluation: development validation split, product-level macro R@5, frozen baseline 0.857, target 0.887; measurement points 100, 200, 300, 500, 700, 1,000, 1,500, 2,000, 3,000.
- Seeds: 0, 1, 2.
- Evaluation population, frozen: the same 6,782 validation queries over 3,243 products as curve-v1 and the deployment gate, manifest sha256 `92665a22adcfc069901ecb91ed684606d203124e9fc2dcca25b92b71318b11a9` (computed as in `c6-deploy-gate.md` §1). Every rankings file used for a metric or a comparison must match it; subsets, duplicates and mismatches are errors before any metric. `[U2]`
- Failures and retries: c4-learning-curve rules carry over. A failed key keeps its record; at most one retry with the same settings, only for a non-deterministic error. A seed with a failed earlier point does not get a "first reach" at a later point ("판정 불가"). `[U3]`

## 2. Progression rule
One path per seed. Start with L = I_s.
At each measurement point B_j (j = 1..8, B_0 = 100):
1. Train the reranker on L (all labels so far, from scratch, training seed s). Call it M_j−1. The model trained at L = I_s is identical to the stratified B = 100 model of the same seed.
2. Evaluate M_j−1 on validation; this is the uncertainty curve's point at |L|.
3. Score every pool pair not yet in L with M_j−1: p = sigmoid(logit).
4. Uncertainty u = |p − 0.5|. Order unlabeled pairs by (u ascending, query_id ascending, position ascending).
5. Walk that order and take pairs until B_j − |L| are taken, skipping a pair whose query already has 3 judgements in L ∪ (taken this round). Ask the simulator for their answers and add them to L.
After the last batch (|L| = 3,000) train and evaluate once more, so every measurement point 100 … 3,000 has exactly one model trained on exactly that many labels.
Batch sizes are therefore the gaps between measurement points (100, 100, 200, 200, 300, 500, 500, 1,000). Larger batches make the policy less adaptive than one-at-a-time selection; that is a property of this v1, reported with the result.

The selector reads only: pool rankings (query_id, product ids, positions, scores), candidate statistics, embedding vectors, the reranker, and the answers already in L. It never reads truth_product_id (c4 v3 §2). `[U1]`
- Selector signature: the selection function takes those inputs as arguments and has no access to the simulator, the rankings' truth field or any manifest.
- Truth-blindness test (one round): hold L, its answers, the model weights and the public inputs fixed; replace or delete every truth_product_id in the rankings passed to the selector; the selected pairs must be identical.

## 3. Outputs
- Per seed: the full selection order (FeedbackEvent JSONL, policy_id `uncertainty`, in selection order), the model at each point, and a run record per measurement point with the same fields as curve-v1 records plus `policy = "uncertainty"`, `round`, `n_pos`, `n_neg`.
- Hypothesis id `H2-uncertainty`, its own run id. H1 records are not touched.
- Field names: comparisons against the frozen SigLIP2 baseline are stored as `delta_vs_baseline`. The H2 comparison is a separate record per seed and point, `h2_comparison`: uncertainty model id, stratified model id, Δ = uncertainty − stratified for R@1 and R@5, 95% interval, P(Δ ≤ 0), population manifest sha256, bootstrap B and seed. Only `h2_comparison` records feed the verdict. `[U5]`

## 4. Comparison rule (fixed now)
- **Primary:** at B = 3,000, for each seed s, Δ_s = R@5(uncertainty, s) − R@5(stratified, s) on validation, with the paired product-level bootstrap (B = 1,000, seed 0) between the two reranked rankings.
- **Comparator, pinned `[U4]`:** stratified rankings are regenerated from the saved curve-v1 models (B = 3,000, noise 0): seed 0 `cadfa7929f8e`, seed 1 `b43a45cb35c5`, seed 2 `6bdb7fec4b7b`, using the curve-v1 input hashes recorded in `reports/curve/curve-v1/plan.json`. Before any comparison, each regenerated macro R@5 must equal the recorded value within 1e-9: 0.8687032876672101, 0.8683824501123298, 0.8622586376517922. On a mismatch the comparison aborts. The sha256 of each regenerated rankings file is recorded. The same check applies at every other measurement point used for secondary Δ.
- **Completeness `[U3]`:** a verdict needs three completed, finite endpoint comparisons. Otherwise the result is "undetermined: incomplete" and names the missing seeds.
- **Verdict sentence, fixed:**
  - every seed's 95% lower bound > 0 → "uncertainty selection raised R@5 over stratified at 3,000 labels on these seeds";
  - every seed's 95% upper bound < 0 → "uncertainty selection lowered R@5 below stratified at 3,000 labels on these seeds";
  - three complete comparisons and neither of the above → "no consistent difference between the two policies at 3,000 labels" (not evidence of equivalence).
- **Secondary, reported, not used for the verdict:** Δ at every measurement point; the first point at which each seed reaches 0.887 (c4 v3 wording); R@1; positive-label share per round; time per round.

## 5. Not claimed
- Noise conditions: only p = 0 is run.
- Uncertainty + diversity: not run; a later contract.
- That one-label-at-a-time selection would behave the same; batches here are 100–1,000.
- Real users: answers are simulated.
