# H2 · Uncertainty label selection v1

Contract `docs/contracts/c4-uncertainty.md` (fixed 2026-10-09 before any result). Run `h2-uncertainty-v1` (`reports/uncertainty/h2-uncertainty-v1/`), hypothesis id `H2-uncertainty`, noise 0, seeds 0–2, 27 points attempted, 27 completed, 0 failed. Comparison `pr h2-compare` against the pinned curve-v1 stratified models; every regenerated R@5 matched its record within 1e-9, the evaluation population matched the frozen manifest, and both runs used the same training inputs.

**Verdict (contract wording): no consistent difference between the two policies at 3,000 labels.**

At 3,000 labels the three seeds point in different directions: seed 2 higher (+0.0088, interval above 0), seed 1 lower (−0.0304, interval below 0), seed 0 interval includes 0 (−0.0027). This is not evidence that the two policies are equivalent. No seed reached the 0.887 target under either policy.

Delta = uncertainty - stratified, macro recall, paired product-level bootstrap (B=1000, seed 0), 95% interval, P = share of replicates with delta <= 0. Only the 3,000-label rows feed the verdict.

| seed | labels | stratified R@5 | uncertainty R@5 | dR@5 [95% CI] | P(d<=0) | dR@1 [95% CI] |
|---:|---:|---:|---:|---|---:|---|
| 0 | 100 | 0.8533 | 0.8533 | +0.0000 [+0.0000, +0.0000] | 1.000 | +0.0000 [+0.0000, +0.0000] |
| 0 | 200 | 0.8510 | 0.8549 | +0.0039 [+0.0001, +0.0075] | 0.021 | +0.0368 [+0.0285, +0.0455] |
| 0 | 300 | 0.8575 | 0.8542 | -0.0033 [-0.0064, -0.0003] | 0.986 | +0.0215 [+0.0142, +0.0290] |
| 0 | 500 | 0.8559 | 0.8541 | -0.0018 [-0.0042, +0.0005] | 0.944 | +0.0139 [+0.0081, +0.0200] |
| 0 | 700 | 0.8557 | 0.8547 | -0.0010 [-0.0032, +0.0013] | 0.811 | +0.0137 [+0.0077, +0.0199] |
| 0 | 1000 | 0.8561 | 0.8623 | +0.0062 [+0.0036, +0.0090] | 0.000 | +0.0155 [+0.0096, +0.0212] |
| 0 | 1500 | 0.8554 | 0.8659 | +0.0105 [+0.0072, +0.0140] | 0.000 | +0.0167 [+0.0119, +0.0214] |
| 0 | 2000 | 0.8564 | 0.8642 | +0.0078 [+0.0051, +0.0110] | 0.000 | +0.0163 [+0.0106, +0.0223] |
| 0 | 3000 | 0.8687 | 0.8661 | -0.0027 [-0.0061, +0.0007] | 0.945 | +0.0047 [-0.0031, +0.0123] |
| 1 | 100 | 0.8449 | 0.8449 | +0.0000 [+0.0000, +0.0000] | 1.000 | +0.0000 [+0.0000, +0.0000] |
| 1 | 200 | 0.8551 | 0.8466 | -0.0085 [-0.0118, -0.0056] | 1.000 | -0.0010 [-0.0058, +0.0037] |
| 1 | 300 | 0.8558 | 0.8390 | -0.0168 [-0.0219, -0.0120] | 1.000 | +0.0014 [-0.0061, +0.0090] |
| 1 | 500 | 0.8558 | 0.8380 | -0.0178 [-0.0246, -0.0115] | 1.000 | -0.0062 [-0.0155, +0.0028] |
| 1 | 700 | 0.8563 | 0.8372 | -0.0191 [-0.0247, -0.0133] | 1.000 | -0.0005 [-0.0094, +0.0081] |
| 1 | 1000 | 0.8558 | 0.8310 | -0.0249 [-0.0315, -0.0180] | 1.000 | -0.0093 [-0.0180, -0.0004] |
| 1 | 1500 | 0.8557 | 0.8336 | -0.0221 [-0.0295, -0.0146] | 1.000 | +0.0046 [-0.0042, +0.0131] |
| 1 | 2000 | 0.8556 | 0.8332 | -0.0223 [-0.0296, -0.0147] | 1.000 | -0.0083 [-0.0170, +0.0007] |
| 1 | 3000 | 0.8684 | 0.8379 | -0.0304 [-0.0372, -0.0237] | 1.000 | -0.0075 [-0.0165, +0.0012] |
| 2 | 100 | 0.8563 | 0.8563 | +0.0000 [+0.0000, +0.0000] | 1.000 | +0.0000 [+0.0000, +0.0000] |
| 2 | 200 | 0.8556 | 0.8549 | -0.0007 [-0.0026, +0.0012] | 0.780 | +0.0059 [+0.0018, +0.0102] |
| 2 | 300 | 0.8537 | 0.8586 | +0.0049 [+0.0020, +0.0081] | 0.003 | +0.0048 [+0.0016, +0.0079] |
| 2 | 500 | 0.8550 | 0.8571 | +0.0021 [-0.0001, +0.0043] | 0.033 | +0.0059 [+0.0030, +0.0092] |
| 2 | 700 | 0.8557 | 0.8632 | +0.0075 [+0.0041, +0.0110] | 0.000 | +0.0135 [+0.0091, +0.0175] |
| 2 | 1000 | 0.8550 | 0.8664 | +0.0114 [+0.0075, +0.0151] | 0.000 | +0.0207 [+0.0161, +0.0253] |
| 2 | 1500 | 0.8544 | 0.8677 | +0.0132 [+0.0090, +0.0173] | 0.000 | +0.0232 [+0.0181, +0.0286] |
| 2 | 2000 | 0.8588 | 0.8682 | +0.0094 [+0.0057, +0.0129] | 0.000 | +0.0216 [+0.0162, +0.0268] |
| 2 | 3000 | 0.8623 | 0.8711 | +0.0088 [+0.0052, +0.0124] | 0.000 | +0.0336 [+0.0262, +0.0408] |

First point with R@5 >= 0.887 (secondary, not used for the verdict):

| seed | uncertainty | stratified |
|---:|---|---|
| 0 | 지정 지점 3,000까지 도달 관측 없음 | 지정 지점 3,000까지 도달 관측 없음 |
| 1 | 지정 지점 3,000까지 도달 관측 없음 | 지정 지점 3,000까지 도달 관측 없음 |
| 2 | 지정 지점 3,000까지 도달 관측 없음 | 지정 지점 3,000까지 도달 관측 없음 |

## Positive labels selected (secondary)

| seed | labels | uncertainty positives | stratified positives |
|---:|---:|---:|---:|
| 0 | 100 | 3 (3.0%) | 3 (3.0%) |
| 0 | 1,000 | 452 (45.2%) | 47 (4.7%) |
| 0 | 3,000 | 1373 (45.8%) | 133 (4.4%) |
| 1 | 100 | 5 (5.0%) | 5 (5.0%) |
| 1 | 1,000 | 442 (44.2%) | 61 (6.1%) |
| 1 | 3,000 | 1320 (44.0%) | 176 (5.9%) |
| 2 | 100 | 6 (6.0%) | 6 (6.0%) |
| 2 | 1,000 | 430 (43.0%) | 55 (5.5%) |
| 2 | 3,000 | 1347 (44.9%) | 163 (5.4%) |

## Reading
- Uncertainty selection picks pairs the reranker scores near 0.5, and about 40–46% of those are correct products by 3,000 labels, against about 5% under stratified selection. The reranker sees many more positives, but the same candidates near its own boundary.
- Seed 1 fell below stratified from 200 labels onward and stayed there; seeds 0 and 2 were above stratified between 1,000 and 2,000 labels. With batches of 100–1,000, one early batch shapes the rest of the path.
- R@1 was higher under uncertainty for seeds 0 and 2 at most points; the verdict uses R@5 only.
- These uncertainty-trained models would all fail the deployment gate's G3 (positive share above 0.10), and they are outside G3's scope by design (c6 §3 covers stratified only). A gate for this policy needs its own reference.
- Time: 7 min 14 s wall for 3 seeds (about 102 min CPU across threads). Scoring the pool and selecting took about 15 s per round on average.

## Not claimed
- Noise conditions (only p = 0 was run), uncertainty + diversity, one-at-a-time selection, real users.
