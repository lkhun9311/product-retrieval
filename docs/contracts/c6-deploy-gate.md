# C6 · Deployment gate v1

Fixed 2026-10-09 (main session), issue #36, before any gate decision is computed.
Implements D20 `eval.gate(new, current) -> pass/fail + reasons` for the reranker stage. D19 P1 target: block 3/3 bad models.
One adversarial review (Codex Astra, high) on the draft; 12 findings, all applied (marked `[A#]`).

## Question
Given the current bundle and a retrained candidate, should the candidate be deployed? The gate answers pass or block, with every failing check named. It does not choose between two passing candidates.

## Selection history (disclosed)
- The checks and margins below were chosen **after** seeing the 108 curve-v1 runs (`docs/results/curve-v1/`), including the noise runs that motivated the gate. Applying the gate to those runs is exploratory.
- The fresh-seed check (§6b) uses seeds no gate setting has seen. It is a functional check on the same validation set, not an independent confirmation (§6c). `[A8]`

## 1. Inputs
- `current`: rankings JSONL of the deployed bundle on the gate evaluation set.
- `candidate`: rankings JSONL of the candidate on the same set.
- `candidate_model`: the candidate's `model.json`. The gate reads `n_pos`, `n_neg`, `label_version`, `reranker_version` and `label_policy` from it; counts are never typed in by hand. `[A2]`
- Evaluation set for v1, frozen: the development validation split, 6,782 queries over 3,243 truth products. Its manifest is the sorted list of `query_id<TAB>truth_product_id` lines (UTF-8, `\n` after every line) with sha256 `92665a22adcfc069901ecb91ed684606d203124e9fc2dcca25b92b71318b11a9`, taken from the frozen-SigLIP2 baseline rankings `reports/eval/f924c1c9…__val.rankings.jsonl`. `[A1]`
- The test split is never used by the gate.

## 2. Checks
All three are evaluated every time; the gate passes only if none fails.

| id | check | fails when |
|---|---|---|
| G1 | R@5 non-inferiority | the 95% lower bound of Δ macro R@5 (candidate − current) is below **−0.005** |
| G2 | R@1 non-inferiority | the 95% lower bound of Δ macro R@1 is below **−0.03** |
| G3 | positive-label share | `n_pos / (n_pos + n_neg)` is above **0.10** (= 2 × p_ref, §3) |

- Δ and its interval come from the existing paired product-level bootstrap (`eval.bootstrap.paired_bootstrap`) called with `ks=(1, 5)`, B = 1,000, seed 0, percentile interval. `[A5]`
- G1 and G2 pass only when the **lower confidence bound** meets the margin. A candidate whose point estimate is inside the margin can still be blocked for insufficient evidence; on curve-v1, stratified seed 2 at 500 labels had ΔR@5 −0.0021 and was blocked by G1 (lower bound −0.0053). `[A11]`
- G2's margin is wider than G1's because R@1 moves more between seeds. On curve-v1 the noise-free 3,000-label runs had ΔR@1 from −0.033 to −0.002 (stratified only: −0.012 to −0.002) while R@5 rose or held. `[A12]`
- G3 reads the training labels, before any evaluation. It is the one check that can block a contaminated model without looking at the evaluation set.

## 3. p_ref and scope of G3
- v1 fixes `p_ref = 0.05`; the threshold is 0.10. It is a constant of this contract version, not a CLI option. `[A3]`
- Source: noise-free **stratified** runs of curve-v1 had 3–6 positives per 100 labels and 133–176 per 3,000 (4.4%–5.9%), on the train split. Under clean labels with share 0.05 the chance that 100 labels exceed 0.10 is 1.1% (binomial tail, P(X ≥ 11)).
- **Scope:** p_ref is calibrated only for the c4 stratified policy over the top-20 candidate pool. The gate requires `label_policy == "stratified"` in `candidate_model`; any other or missing value is an input error. Other policies need their own reference in a new contract version. `[A6]`
- With real corrections, p_ref must come from an independently audited clean sample. A candidate passing the gate never makes its own labels the new reference. `[A7]`

## 4. Failure modes (fail loudly, never pass)
An error is neither block nor pass; the output names it as an error, exit code 1.
- Either rankings file's `(query_id, truth_product_id)` set differs from the frozen manifest (hash mismatch), or has a duplicate query id. `[A1]`
- Any query in either file has fewer than 5 unique product ids in `top_k_product_ids`. `[A5]`
- `n_pos`, `n_neg` missing, not integers (booleans excluded), negative, or summing to 0. `[A4]`
- `label_policy` missing or not `stratified`. `[A6]`
- `candidate_model` unreadable or missing `label_version` / `reranker_version`. `[A2]`

## 5. Output
One JSON object: `decision` (`pass` / `block`), `checks` (per id: value, threshold, failed; for G1/G2 also delta, ci_low, ci_high), `reasons` (failed ids in id order), `inputs` (paths and sha256 of both rankings files and of `candidate_model`, manifest sha256, n_pos, n_neg, label_version, reranker_version, label_policy, B, seed), `contract` = `c6-deploy-gate-v1`.
CLI: `pr gate --current A --candidate B --candidate-model model.json [--out path]`. Exit code 0 pass, 2 block, 1 error. No option changes a threshold. `[A3]`
`pr curve` writes `label_policy` into each run's `model.json` from this version on (additive field). `[A6]`

## 6. Checks of the gate itself
### Outcome classes, defined now on validation point estimates `[A9]`
- **harmful:** ΔR@5 < −0.01 or ΔR@1 < −0.05.
- **useful:** ΔR@5 ≥ +0.005 and ΔR@1 ≥ −0.03.
- **neutral:** everything else.
"Clean" (noise 0) is a training condition, not an outcome class; a clean run can be harmful.

### 6a. Retrospective (exploratory)
Apply G1–G3 to all 108 curve-v1 runs using the stored deltas and label counts. Report every run with its decision, failing checks and outcome class. G3 is reported only for stratified runs; random-policy runs are out of G3's scope (§3) and shown with G1/G2 only. `[A6]`

### 6b. Fresh-seed functional check
- Runs: stratified policy, noise p ∈ {0, 0.1, 0.2}, budget B ∈ {100, 3000}, seeds **3, 4, 5** → 18 runs, same training and evaluation code path as curve-v1.
- `pr curve` must accept a plan (keys) and a hypothesis id as inputs, and the summarizer must select records by that plan; the run uses hypothesis id `C6-gate-check` and its own run id. H1's plan and records stay untouched. `[A10]`
- **Pass criteria for the gate, fixed now `[A8]`:**
  1. every harmful run is blocked;
  2. at least 2 of every 3 useful runs pass (useful runs blocked ≤ ⌊useful/3⌋);
  3. a gate that blocks all 18 or passes all 18 fails this check outright.
- Reported with explicit denominators: harmful blocked / harmful; useful passed / useful; neutral blocked / neutral; clean runs blocked / 6 at both budgets; runs blocked by G3 alone. `[A9]`
- If fewer than 3 harmful or fewer than 3 useful runs appear, report the counts and say the criterion is untested; no runs are added.

### 6c. What this check can and cannot show
- G1 and G2 read the same validation set that defines "harmful" and "useful", so blocking harmful runs through G1/G2 is expected by construction. The informative parts are G3 (independent of evaluation) and how many useful runs are blocked.
- The validation set was also used to choose the reranker: v0, v1 and v2 were compared on it before v2 was picked. Claims of independent confirmation need a separately held evaluation set. `[A8]`

## 7. Not claimed
- That these margins suit real users or a production golden set. They come from simulated corrections on one split.
- Coverage of latency, cost or risk–coverage checks listed in D19; v1 covers ranking quality and label contamination only.
- Protection against biased (non-random) correction errors. Flips in curve-v1 are symmetric and random.
- Protection against slow label drift; that depends on the audited reference in §3.
