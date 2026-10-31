# C6 · Deployment gate v2

Drafted 2026-10-31 (main session), issue #46, milestone M5, before truth v2 exists and before any v2 gate decision is computed.
One adversarial review (Codex Astra, high) on the draft: 11 findings (4 critical, 7 important), all applied (marked `[B#]`).
Replaces gate v1 (`c6-deploy-gate.md`) once fixed. v1 stays the record for the 2026-10 runs.

## Question
v1 blocked every harmful run (8/8), but its cost was untested: only 2 useful runs appeared, and 1 was blocked. It also judged "useful" and "harmful" on the same validation set its checks read. v2 asks: on truth v2, how often does the gate block a candidate that is useful on a half of the split the gate's checks do not read? It also adds the checks D19 lists that v1 left out (slices, latency and memory, reproducibility).

## What is fixed now and what waits
Every **rule** below is fixed now, before any truth v2 data or v2 number exists. Only **measurements** wait:
- for an audit release (§1),
- for the evaluator's truth-set support (#48),
- for the prerequisites in §7.

No margin, slice rule, threshold, salt or run is chosen after a v2 number is seen. Any change is v3.

## Selection history (disclosed) `[B1]`
- G1 and G2 margins were chosen after seeing curve-v1 on the **whole** validation split, which includes the future O half. The reranker was also selected on the whole split.
- O is therefore not untouched data. The false-block study in §5 is a **conditional functional check**: it tests the gate's checks on products those checks do not read in this run. It is not independent confirmation.
- Independent confirmation needs products no earlier choice has seen. This split has none left.

## 1. Evaluation set and truth `[B3]`
- The gate runs only on an **audit release**: truth v2 issued after the ID audit reached branch 7, or reached branch 8 and passed its fresh-600 check (`ladder-1-id-audit` v1.1 §6). Truth v2 from an audit that is incomplete, stopped, or has not passed branch 8 is an input error.
- The release ships an **eligibility manifest**. The gate uses it as is and never derives eligibility itself. A query is excluded when any of these holds:
  - its status is "original invalid, replacement not found" or "absence confirmed";
  - it is in a quarantined train–validation set (audit branch 3);
  - it was routed to the target-specified track (audit branch 5, when that branch fired).
- Every other query is eligible, including queries the audit never reviewed and compatibility-only (CONS) queries.
- The gate evaluates audit comparison **§7(b)**: the full truth v2 evaluation with merges, splits, exclusions and the rebuilt index.

## 2. Groups and the two halves `[B2]` `[B8]`
- **Groups.** Link two products when they appear together in any eligible query's set of valid truth products. A group is a connected component of that graph; a product alone is its own group. Each eligible query belongs to the one group that holds its valid products, and macro weighting and bootstrap resampling use these groups. A group with no eligible query is dropped and counted.
- **Hits.** A query is a hit at K when any product in its valid set appears in the top K.
- **Halves.** Group key = the smallest original product ID in the group, compared as integers. A group is in **G** when the first hex digit of `sha256("c6-gate-v2|" + group key)` is even, and in **O** otherwise.
- **Salts.** The salt `c6-gate-v2` is fixed now. Any v3 uses the salt `c6-gate-v3`, fixed now too. No other salt is tried.
- **Canonical v2 manifest.** It replaces v1's `(query_id, truth_product_id)` manifest. One UTF-8 line per eligible query, sorted by query ID: `query_id<TAB>group_key<TAB>valid product IDs, sorted and comma-joined<TAB>half<TAB>slice`. Its sha256 is frozen with the audit release, and both rankings files must match it.
- G1, G2 and G4 read only G. Outcome classes (§5) are computed only on O.
- Cost, stated now: each half has about half the groups, so intervals widen by about √2. False blocks are measured under that condition.

## 3. Checks
All checks are evaluated every time. The gate passes only if none fails. Insufficient evidence blocks, as in v1.

| id | check | fails when |
|---|---|---|
| G1 | R@5 non-inferiority on G | lower end of the central 95% interval of Δ macro R@5 (candidate − current) below **−0.005** |
| G2 | R@1 non-inferiority on G | lower end of the central 95% interval of Δ macro R@1 below **−0.03** |
| G3 | positive-label share | `n_pos / (n_pos + n_neg)` above **0.10** (p_ref = 0.05, v1 §3) |
| G4 | slice non-inferiority on G | for any slice of §3a, lower end of the central 95% interval of Δ macro R@5 below **−0.02** |
| G5 | latency and memory | candidate p95 latency above **50 ms**, or candidate peak resident memory above current + **512 MiB** |
| G6 | bundle reproducibility | the canonical rankings hash of the rebuilt candidate differs from the submitted one |

- G1 and G2 keep v1's margins unchanged.
- Bootstrap for G1, G2 and G4: v1's paired product-level bootstrap (B = 1,000, seed 0, percentile), resampling the groups of §2. G4 runs its own bootstrap for each slice.

### 3a. Slices for G4 `[B7]` `[B11]`
- **Slices.** The `CATEGORY` values of the LRVS metadata, plus CONS as one slice. `CATEGORY` exists only in the simple metadata parquet. Products without it form the slice `none`, and products with more than one value form `mixed`. Each product's slice is fixed in the canonical manifest.
- **Inclusion.** A slice is included when it has at least 300 eligible queries **and** at least 150 groups in G. The list is fixed from the manifest before any candidate is scored. Excluded slices are listed with their counts; an empty slice is never tested.
- **Error controlled.** The risk G4 protects against is passing a candidate that hurts some slice. Since every slice must pass, adding slices can only increase blocking. So each slice uses the same central 95% interval with no Bonferroni widening, and the number of slices tested is reported with the decision.

### 3b. G3 scope
- p_ref = 0.05 holds only for **simulated** corrections under the c4 stratified policy (v1 §3). A noise-free simulated label is consistent with the simulator's truth by construction.
- `label_source` is a new required field in `model.json`, with values `simulated` or `real`. Real corrections have no p_ref until an independently audited clean sample of train-side labels exists, and the ID audit does not provide one. Until then `label_source = real` is an input error.

### 3c. G5 and G6 protocols `[B10]` `[B11]`
- **Inputs.** Both bundles are inputs, pinned by directory sha256: `--current-bundle` and `--candidate-bundle`.
- **Measured operation.** From a precomputed query embedding to the final top-20 product list: index search plus rerank, if the bundle has a reranker. A bundle without a reranker measures index search only.
- **Latency.**
  - Machine: one CPU, `OMP_NUM_THREADS=1`, the same session for both bundles.
  - Queries: the 1,000 eligible G queries with the smallest query IDs.
  - Each bundle is warmed up on 100 other G queries.
  - Then 3 repeats, ordered current, candidate, candidate, current, current, candidate.
  - Value: the median of the three p95 values.
- **Memory.** Peak RSS of a fresh process per bundle doing the latency run.
- **G6 rebuild.**
  - Same `uv.lock` sha256, same Python minor version and the same code commit as the candidate's `model.json`. A mismatch is an input error.
  - Canonical rankings hash: the sha256 of lines `query_id<TAB>top-20 product IDs, comma-joined`, sorted by query ID. File formatting never matters.

## 4. Failure modes (fail loudly, never pass)
These replace the manifest checks of v1 §4. The other v1 checks stay: duplicate query IDs, fewer than 5 unique products, bad label counts, a missing `label_version` or `reranker_version`, and `label_policy` not `stratified`.
- No audit release, or no eligibility manifest; either rankings file mismatches the canonical v2 manifest.
- A G/O assignment differs from the §2 rule when recomputed, or the slice list differs from the frozen list.
- `label_source` is missing, unknown or `real`.
- A bundle is missing, its sha256 mismatches, or G5 or G6 cannot be measured. A missing measurement is an error, never a pass.

## 5. False-block study (checks of the gate itself)
### Outcome classes on O, point estimates
- **harmful:** ΔR@5 < −0.01 or ΔR@1 < −0.05.
- **useful:** ΔR@5 ≥ +0.005 and ΔR@1 ≥ −0.03.
- **neutral:** everything else.

The thresholds are v1's, measured on O.

### Candidate plan `[B5]` `[B9]`
- A full grid: stratified policy, budget 3,000 labels, noise {0, 0.1, 0.2} × seeds **6–21**, giving **48 runs**. Hypothesis id `C6-gate-v2-check`, with its own run id.
- Seeds 0–5 were used by earlier work. Seeds 6–21 have not been used by any run, config or result in this repository (checked 2026-10-31).
- Noise 0 is the only condition where useful runs appeared before: 5 of 6 stratified runs at 3,000 labels, seeds 0–2 in curve-v1 and 3–5 in gate-check-v1, on the whole split under truth v1. On O, under truth v2, the count can differ. Enriching the plan for it is deliberate. The pooled false-block rate therefore describes this mixture, not candidates in general.
- No run is added after results.

### Terminal status and errors `[B6]`
- Each of the 48 runs ends in exactly one of pass, block or error.
- A run that errors gets the curve runner's bounded retry (at most 2), with the same seed.
- A run still in error is counted as a **block** in every criterion: a useful candidate that the gate cannot clear is a false block.
- More than 2 runs in error after retries make the study **incomplete**: counts are reported and no criterion is judged.

### Acceptance criteria `[B4]`
This is a finite-plan acceptance check on 48 non-independent runs, not a powered test of a population rate.
1. Every harmful run is blocked.
2. Useful runs: at least **8** must appear, or the criterion is reported as **untested**. If they do, it is met when the useful runs blocked are at most ⌊useful/3⌋. Eight is the smallest count for which zero blocks put the exact one-sided 95% upper bound below one third (8 → 31.2%; 6 → 39.3%).
3. A gate that blocks all 48 or passes all 48 fails outright.

### Reporting
- Report by noise condition and pooled, always with denominators:
  - harmful blocked / harmful,
  - useful passed / useful,
  - neutral blocked / neutral.
- For every blocked useful run, list the checks that failed. Also count the runs blocked only by G4, G5 or G6.
- The false-block rate among useful runs is reported with its exact one-sided 95% upper bound. A seed-cluster version treats each seed's three runs as one unit, because the conditions share seeds.

### If criterion 2 fails
- The result is reported as measured.
- Margins change only in v3, checked on seeds 22 and above with salt `c6-gate-v3`.
- v3's O half overlaps v2's products, so v3 is a functional check too (Selection history).

## 6. Output
v1 §5, plus:
- the audit release id and the canonical manifest sha256;
- the G/O group and query counts;
- per slice: Δ, the interval, and included or excluded with counts;
- G5 values with the machine description and both bundle sha256s;
- G6: the rebuilt canonical hash, the `uv.lock` sha256 and the Python version.

`contract` = `c6-deploy-gate-v2`. CLI: `pr gate --current A --candidate B --candidate-model model.json --release R --current-bundle D1 --candidate-bundle D2 [--out path]`. Exit codes stay 0 pass, 2 block, 1 error, and no option changes a threshold.

## 7. Prerequisites before any v2 number `[B7]` `[B9]`
- An audit release with the eligibility manifest and the canonical v2 manifest (ID audit, #44 and #45 onward).
- Evaluator truth-set support and group-based bootstrap (#48).
- Curve runner: accept the 48-run plan with exact planned-key completeness checks, and read population counts from the canonical manifest instead of v1's fixed counts.
- Bundle loading for G5 and G6, and the measurement harness of §3c.
- Each prerequisite lands with tests before the plan runs.

## 8. Not claimed
- Independent confirmation on held-out data (Selection history).
- That 50 ms and 512 MiB suit a production service. They are working limits for one machine.
- That G3 protects against real-correction contamination (§3b).
- That the false-block rate generalises beyond the 48-run mixture (§5).
- Anything about the test split. It is never used by the gate.
