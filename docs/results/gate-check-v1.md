# Deployment gate fresh-seed check v1

Gate `docs/contracts/c6-deploy-gate.md` v1, §6b. Run `c6-gate-check-v1` (`reports/curve/c6-gate-check-v1/`), hypothesis id `C6-gate-check`, plan `configs/c6-gate-check-plan.json`: stratified policy, noise 0 / 0.1 / 0.2, budgets 100 and 3,000, seeds 3, 4, 5. 18 attempted, 18 completed, 0 failed. Same training and evaluation path as curve-v1; "current" is the frozen SigLIP2 baseline on the development validation split.

Decisions use the deltas and intervals stored in each run record (same bootstrap as the gate, B = 1,000, seed 0; the retrospective cross-check showed `pr gate` reproduces them exactly) and the label counts, checked against each run's `model.json` (`label_policy: stratified`).

## Against the pass criteria fixed in §6b

| criterion | result | status |
|---|---|---|
| 1. every harmful run blocked | 8 / 8 | met |
| 2. at least 2 of every 3 useful runs pass | 1 / 2 passed | **untested** — fewer than 3 useful runs appeared |
| 3. not all blocked, not all passed | 17 blocked, 1 passed | met |

Other counts with denominators: neutral blocked 8 / 8; clean runs blocked 3 / 3 at 100 labels and 2 / 3 at 3,000; runs blocked by G3 alone 0 / 18.

Criterion 2 is untested by the contract's rule, so no runs were added. One of the two useful runs (noise 0, seed 3, 3,000 labels; ΔR@5 +0.0055) was blocked by G2 alone: its ΔR@1 lower bound was −0.0323 against the −0.03 margin. That is the cost the gate is meant to keep small, and it is recorded here as such. The margin is not changed after seeing it; a change would be a new contract version checked on new seeds.

## Every run

| noise | seed | labels | ΔR@5 [95%] | ΔR@1 [95%] | positive share | class | decision | failed |
|---:|---:|---:|---|---|---|---|---|---|
| 0.0 | 3 | 100 | -0.0034 [-0.0079, +0.0008] | -0.0278 [-0.0355, -0.0208] | 5/100 = 0.050 | neutral | block | G1,G2 |
| 0.0 | 3 | 3000 | +0.0055 [+0.0022, +0.0085] | -0.0225 [-0.0323, -0.0129] | 140/3000 = 0.047 | useful | block | G2 |
| 0.0 | 4 | 100 | -0.0020 [-0.0061, +0.0018] | -0.0428 [-0.0520, -0.0338] | 6/100 = 0.060 | neutral | block | G1,G2 |
| 0.0 | 4 | 3000 | -0.0040 [-0.0073, -0.0010] | -0.0124 [-0.0174, -0.0075] | 150/3000 = 0.050 | neutral | block | G1 |
| 0.0 | 5 | 100 | -0.0337 [-0.0407, -0.0268] | -0.1582 [-0.1696, -0.1462] | 6/100 = 0.060 | harmful | block | G1,G2 |
| 0.0 | 5 | 3000 | +0.0091 [+0.0036, +0.0146] | +0.0042 [-0.0039, +0.0125] | 161/3000 = 0.054 | useful | pass | — |
| 0.1 | 3 | 100 | -0.0030 [-0.0067, +0.0002] | -0.0190 [-0.0256, -0.0128] | 17/100 = 0.170 | neutral | block | G1,G3 |
| 0.1 | 3 | 3000 | +0.0019 [-0.0013, +0.0050] | -0.0411 [-0.0505, -0.0313] | 431/3000 = 0.144 | neutral | block | G2,G3 |
| 0.1 | 4 | 100 | -0.0295 [-0.0358, -0.0229] | -0.0701 [-0.0815, -0.0588] | 15/100 = 0.150 | harmful | block | G1,G2,G3 |
| 0.1 | 4 | 3000 | -0.0025 [-0.0063, +0.0011] | -0.0249 [-0.0324, -0.0172] | 415/3000 = 0.138 | neutral | block | G1,G2,G3 |
| 0.1 | 5 | 100 | -0.0342 [-0.0410, -0.0272] | -0.1619 [-0.1739, -0.1503] | 15/100 = 0.150 | harmful | block | G1,G2,G3 |
| 0.1 | 5 | 3000 | -0.0120 [-0.0172, -0.0071] | -0.0115 [-0.0163, -0.0070] | 448/3000 = 0.149 | harmful | block | G1,G3 |
| 0.2 | 3 | 100 | -0.0049 [-0.0085, -0.0015] | -0.0176 [-0.0236, -0.0117] | 29/100 = 0.290 | neutral | block | G1,G3 |
| 0.2 | 3 | 3000 | -0.0102 [-0.0158, -0.0047] | -0.0502 [-0.0612, -0.0400] | 719/3000 = 0.240 | harmful | block | G1,G2,G3 |
| 0.2 | 4 | 100 | -0.7843 [-0.7978, -0.7696] | -0.6721 [-0.6873, -0.6548] | 26/100 = 0.260 | harmful | block | G1,G2,G3 |
| 0.2 | 4 | 3000 | -0.0366 [-0.0442, -0.0296] | -0.0269 [-0.0347, -0.0190] | 721/3000 = 0.240 | harmful | block | G1,G2,G3 |
| 0.2 | 5 | 100 | -0.0339 [-0.0409, -0.0270] | -0.1583 [-0.1699, -0.1465] | 30/100 = 0.300 | harmful | block | G1,G2,G3 |
| 0.2 | 5 | 3000 | -0.0025 [-0.0063, +0.0012] | -0.0279 [-0.0359, -0.0201] | 707/3000 = 0.236 | neutral | block | G1,G2,G3 |

## Reading
- Every harmful run was blocked. G1 and G2 read the same validation set that defines "harmful", so this is a functional check, not independent confirmation (contract §6c).
- G3 did not block anything on its own this time: every noisy run also failed G1 or G2. In the curve-v1 retrospective it was the only failing check for 10 runs.
- With noise 0, results varied by seed: seed 5 at 100 labels was harmful (ΔR@1 −0.158), seed 4 at 3,000 labels was below baseline on R@5 (−0.0040). These fresh seeds are not part of H1 and do not change its judgement.
- The gate blocked 17 of 18 runs. Most blocked runs were harmful or neutral; one useful run was blocked.
