# Deployment gate on curve-v1 (retrospective, exploratory)

Gate: `docs/contracts/c6-deploy-gate.md` v1, applied to the 108 completed curve-v1 runs (`reports/curve/curve-v1/runs.jsonl`, contract c4-v3). "Current" is always the frozen SigLIP2 baseline on the development validation split.

**Exploratory:** the gate's margins were chosen after seeing these runs. This shows the gate's behaviour on them, not evidence that the margins generalise. The fresh-seed check (contract §6b) is separate.

Computation: the deltas and 95% intervals stored in each run record come from the same paired product-level bootstrap the gate uses (B = 1,000, seed 0), so they are reused here. G3 applies to stratified runs only (contract §3); for random-policy runs it is shown as n/a and does not count.

Outcome classes (contract §6, on point estimates): harmful = ΔR@5 < −0.01 or ΔR@1 < −0.05; useful = ΔR@5 ≥ +0.005 and ΔR@1 ≥ −0.03; neutral = otherwise.

## Summary by outcome class (all 108 runs)

| class | blocked | passed | total |
|---|---:|---:|---:|
| harmful | 23 | 0 | 23 |
| useful | 0 | 3 | 3 |
| neutral | 61 | 21 | 82 |

Runs blocked by G3 alone: 10.

## By condition

| policy | noise | harmful blocked | useful passed | neutral blocked |
|---|---:|---:|---:|---:|
| stratified | 0.0 | 1/1 | 3/3 | 9/23 |
| stratified | 0.1 | 7/7 | 0/0 | 20/20 |
| stratified | 0.2 | 12/12 | 0/0 | 15/15 |
| random | 0.0 | 3/3 | 0/0 | 17/24 |

## Every run

| policy | noise | seed | labels | ΔR@5 [95%] | ΔR@1 [95%] | positive share | class | decision | failed |
|---|---:|---:|---:|---|---|---|---|---|---|
| stratified | 0.0 | 0 | 100 | -0.0038 [-0.0076, -0.0000] | -0.0431 [-0.0522, -0.0339] | 3/100 = 0.030 | neutral | block | G1,G2 |
| stratified | 0.0 | 0 | 200 | -0.0061 [-0.0100, -0.0020] | -0.0478 [-0.0576, -0.0384] | 9/200 = 0.045 | neutral | block | G1,G2 |
| stratified | 0.0 | 0 | 300 | +0.0004 [-0.0024, +0.0032] | -0.0268 [-0.0350, -0.0184] | 12/300 = 0.040 | neutral | block | G2 |
| stratified | 0.0 | 0 | 500 | -0.0011 [-0.0042, +0.0016] | -0.0182 [-0.0246, -0.0117] | 23/500 = 0.046 | neutral | pass | — |
| stratified | 0.0 | 0 | 700 | -0.0014 [-0.0047, +0.0016] | -0.0176 [-0.0240, -0.0112] | 32/700 = 0.046 | neutral | pass | — |
| stratified | 0.0 | 0 | 1000 | -0.0010 [-0.0042, +0.0020] | -0.0186 [-0.0252, -0.0117] | 47/1000 = 0.047 | neutral | pass | — |
| stratified | 0.0 | 0 | 1500 | -0.0017 [-0.0048, +0.0012] | -0.0150 [-0.0207, -0.0092] | 71/1500 = 0.047 | neutral | pass | — |
| stratified | 0.0 | 0 | 2000 | -0.0007 [-0.0038, +0.0022] | -0.0168 [-0.0239, -0.0099] | 90/2000 = 0.045 | neutral | pass | — |
| stratified | 0.0 | 0 | 3000 | +0.0116 [+0.0075, +0.0157] | -0.0024 [-0.0102, +0.0057] | 133/3000 = 0.044 | useful | pass | — |
| stratified | 0.0 | 1 | 100 | -0.0122 [-0.0174, -0.0068] | -0.0260 [-0.0345, -0.0176] | 5/100 = 0.050 | harmful | block | G1,G2 |
| stratified | 0.0 | 1 | 200 | -0.0020 [-0.0052, +0.0009] | -0.0160 [-0.0218, -0.0100] | 11/200 = 0.055 | neutral | block | G1 |
| stratified | 0.0 | 1 | 300 | -0.0013 [-0.0043, +0.0016] | -0.0197 [-0.0266, -0.0132] | 21/300 = 0.070 | neutral | pass | — |
| stratified | 0.0 | 1 | 500 | -0.0013 [-0.0043, +0.0013] | -0.0203 [-0.0271, -0.0132] | 28/500 = 0.056 | neutral | pass | — |
| stratified | 0.0 | 1 | 700 | -0.0008 [-0.0037, +0.0020] | -0.0200 [-0.0272, -0.0122] | 42/700 = 0.060 | neutral | pass | — |
| stratified | 0.0 | 1 | 1000 | -0.0012 [-0.0043, +0.0016] | -0.0190 [-0.0257, -0.0124] | 61/1000 = 0.061 | neutral | pass | — |
| stratified | 0.0 | 1 | 1500 | -0.0014 [-0.0044, +0.0014] | -0.0232 [-0.0307, -0.0153] | 91/1500 = 0.061 | neutral | block | G2 |
| stratified | 0.0 | 1 | 2000 | -0.0015 [-0.0045, +0.0014] | -0.0166 [-0.0228, -0.0103] | 118/2000 = 0.059 | neutral | pass | — |
| stratified | 0.0 | 1 | 3000 | +0.0113 [+0.0073, +0.0156] | -0.0032 [-0.0105, +0.0043] | 176/3000 = 0.059 | useful | pass | — |
| stratified | 0.0 | 2 | 100 | -0.0008 [-0.0031, +0.0017] | -0.0130 [-0.0187, -0.0077] | 6/100 = 0.060 | neutral | pass | — |
| stratified | 0.0 | 2 | 200 | -0.0015 [-0.0041, +0.0009] | -0.0076 [-0.0117, -0.0034] | 11/200 = 0.055 | neutral | pass | — |
| stratified | 0.0 | 2 | 300 | -0.0034 [-0.0067, -0.0002] | -0.0086 [-0.0127, -0.0046] | 19/300 = 0.063 | neutral | block | G1 |
| stratified | 0.0 | 2 | 500 | -0.0021 [-0.0053, +0.0012] | -0.0104 [-0.0150, -0.0058] | 29/500 = 0.058 | neutral | block | G1 |
| stratified | 0.0 | 2 | 700 | -0.0014 [-0.0043, +0.0015] | -0.0121 [-0.0173, -0.0067] | 41/700 = 0.059 | neutral | pass | — |
| stratified | 0.0 | 2 | 1000 | -0.0021 [-0.0051, +0.0010] | -0.0117 [-0.0167, -0.0068] | 55/1000 = 0.055 | neutral | block | G1 |
| stratified | 0.0 | 2 | 1500 | -0.0026 [-0.0056, +0.0004] | -0.0117 [-0.0166, -0.0067] | 81/1500 = 0.054 | neutral | block | G1 |
| stratified | 0.0 | 2 | 2000 | +0.0017 [-0.0009, +0.0046] | -0.0076 [-0.0136, -0.0021] | 112/2000 = 0.056 | neutral | pass | — |
| stratified | 0.0 | 2 | 3000 | +0.0052 [+0.0017, +0.0086] | -0.0118 [-0.0211, -0.0030] | 163/3000 = 0.054 | useful | pass | — |
| stratified | 0.1 | 0 | 100 | -0.0223 [-0.0282, -0.0161] | -0.0687 [-0.0803, -0.0579] | 14/100 = 0.140 | harmful | block | G1,G2,G3 |
| stratified | 0.1 | 0 | 200 | -0.0089 [-0.0132, -0.0049] | -0.0298 [-0.0383, -0.0216] | 27/200 = 0.135 | neutral | block | G1,G2,G3 |
| stratified | 0.1 | 0 | 300 | -0.0005 [-0.0035, +0.0025] | -0.0188 [-0.0255, -0.0124] | 37/300 = 0.123 | neutral | block | G3 |
| stratified | 0.1 | 0 | 500 | -0.0018 [-0.0050, +0.0011] | -0.0241 [-0.0314, -0.0167] | 68/500 = 0.136 | neutral | block | G1,G2,G3 |
| stratified | 0.1 | 0 | 700 | -0.0019 [-0.0054, +0.0012] | -0.0238 [-0.0315, -0.0164] | 88/700 = 0.126 | neutral | block | G1,G2,G3 |
| stratified | 0.1 | 0 | 1000 | -0.0005 [-0.0036, +0.0025] | -0.0334 [-0.0419, -0.0246] | 129/1000 = 0.129 | neutral | block | G2,G3 |
| stratified | 0.1 | 0 | 1500 | -0.0172 [-0.0228, -0.0111] | -0.0408 [-0.0505, -0.0321] | 196/1500 = 0.131 | harmful | block | G1,G2,G3 |
| stratified | 0.1 | 0 | 2000 | -0.0064 [-0.0108, -0.0017] | -0.0435 [-0.0534, -0.0346] | 269/2000 = 0.135 | neutral | block | G1,G2,G3 |
| stratified | 0.1 | 0 | 3000 | -0.0029 [-0.0068, +0.0012] | -0.0372 [-0.0462, -0.0284] | 402/3000 = 0.134 | neutral | block | G1,G2,G3 |
| stratified | 0.1 | 1 | 100 | -0.0116 [-0.0161, -0.0069] | -0.1145 [-0.1264, -0.1032] | 14/100 = 0.140 | harmful | block | G1,G2,G3 |
| stratified | 0.1 | 1 | 200 | -0.0052 [-0.0088, -0.0013] | -0.0474 [-0.0566, -0.0388] | 29/200 = 0.145 | neutral | block | G1,G2,G3 |
| stratified | 0.1 | 1 | 300 | -0.0042 [-0.0080, -0.0003] | -0.0681 [-0.0788, -0.0573] | 48/300 = 0.160 | harmful | block | G1,G2,G3 |
| stratified | 0.1 | 1 | 500 | -0.0028 [-0.0063, +0.0008] | -0.0593 [-0.0699, -0.0484] | 80/500 = 0.160 | harmful | block | G1,G2,G3 |
| stratified | 0.1 | 1 | 700 | -0.0013 [-0.0044, +0.0019] | -0.0367 [-0.0456, -0.0270] | 111/700 = 0.159 | neutral | block | G2,G3 |
| stratified | 0.1 | 1 | 1000 | -0.0027 [-0.0061, +0.0004] | -0.0276 [-0.0357, -0.0196] | 163/1000 = 0.163 | neutral | block | G1,G2,G3 |
| stratified | 0.1 | 1 | 1500 | +0.0004 [-0.0043, +0.0051] | -0.0446 [-0.0547, -0.0351] | 235/1500 = 0.157 | neutral | block | G2,G3 |
| stratified | 0.1 | 1 | 2000 | -0.0021 [-0.0059, +0.0017] | -0.0521 [-0.0627, -0.0420] | 308/2000 = 0.154 | harmful | block | G1,G2,G3 |
| stratified | 0.1 | 1 | 3000 | +0.0009 [-0.0036, +0.0055] | -0.0385 [-0.0484, -0.0292] | 462/3000 = 0.154 | neutral | block | G2,G3 |
| stratified | 0.1 | 2 | 100 | -0.0617 [-0.0707, -0.0527] | -0.4500 [-0.4680, -0.4303] | 16/100 = 0.160 | harmful | block | G1,G2,G3 |
| stratified | 0.1 | 2 | 200 | -0.0014 [-0.0037, +0.0008] | -0.0087 [-0.0133, -0.0041] | 30/200 = 0.150 | neutral | block | G3 |
| stratified | 0.1 | 2 | 300 | -0.0027 [-0.0056, -0.0000] | -0.0097 [-0.0143, -0.0052] | 48/300 = 0.160 | neutral | block | G1,G3 |
| stratified | 0.1 | 2 | 500 | -0.0022 [-0.0053, +0.0008] | -0.0121 [-0.0172, -0.0071] | 74/500 = 0.148 | neutral | block | G1,G3 |
| stratified | 0.1 | 2 | 700 | -0.0024 [-0.0056, +0.0006] | -0.0137 [-0.0187, -0.0088] | 103/700 = 0.147 | neutral | block | G1,G3 |
| stratified | 0.1 | 2 | 1000 | -0.0038 [-0.0074, -0.0004] | -0.0146 [-0.0204, -0.0093] | 147/1000 = 0.147 | neutral | block | G1,G3 |
| stratified | 0.1 | 2 | 1500 | +0.0027 [-0.0009, +0.0066] | -0.0105 [-0.0165, -0.0048] | 212/1500 = 0.141 | neutral | block | G3 |
| stratified | 0.1 | 2 | 2000 | +0.0004 [-0.0037, +0.0044] | -0.0110 [-0.0156, -0.0066] | 283/2000 = 0.141 | neutral | block | G3 |
| stratified | 0.1 | 2 | 3000 | +0.0007 [-0.0022, +0.0038] | -0.0150 [-0.0211, -0.0089] | 424/3000 = 0.141 | neutral | block | G3 |
| stratified | 0.2 | 0 | 100 | -0.2276 [-0.2407, -0.2158] | -0.1497 [-0.1623, -0.1368] | 20/100 = 0.200 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 0 | 200 | -0.0330 [-0.0393, -0.0259] | -0.0554 [-0.0655, -0.0457] | 43/200 = 0.215 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 0 | 300 | -0.0098 [-0.0143, -0.0056] | -0.0427 [-0.0517, -0.0329] | 63/300 = 0.210 | neutral | block | G1,G2,G3 |
| stratified | 0.2 | 0 | 500 | -0.0068 [-0.0110, -0.0028] | -0.0335 [-0.0416, -0.0260] | 112/500 = 0.224 | neutral | block | G1,G2,G3 |
| stratified | 0.2 | 0 | 700 | -0.0110 [-0.0155, -0.0065] | -0.0342 [-0.0423, -0.0264] | 146/700 = 0.209 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 0 | 1000 | -0.0007 [-0.0042, +0.0028] | -0.0376 [-0.0464, -0.0286] | 213/1000 = 0.213 | neutral | block | G2,G3 |
| stratified | 0.2 | 0 | 1500 | -0.0630 [-0.0718, -0.0547] | -0.0525 [-0.0606, -0.0441] | 316/1500 = 0.211 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 0 | 2000 | -0.0222 [-0.0286, -0.0159] | -0.0410 [-0.0505, -0.0320] | 433/2000 = 0.216 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 0 | 3000 | -0.0325 [-0.0400, -0.0253] | -0.0439 [-0.0527, -0.0352] | 642/3000 = 0.214 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 1 | 100 | -0.0178 [-0.0232, -0.0122] | -0.1177 [-0.1286, -0.1072] | 21/100 = 0.210 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 1 | 200 | -0.0035 [-0.0068, -0.0002] | -0.0308 [-0.0394, -0.0228] | 50/200 = 0.250 | neutral | block | G1,G2,G3 |
| stratified | 0.2 | 1 | 300 | -0.0073 [-0.0114, -0.0029] | -0.0942 [-0.1057, -0.0831] | 76/300 = 0.253 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 1 | 500 | -0.0062 [-0.0103, -0.0021] | -0.1210 [-0.1333, -0.1082] | 126/500 = 0.252 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 1 | 700 | -0.0055 [-0.0093, -0.0020] | -0.0630 [-0.0726, -0.0539] | 170/700 = 0.243 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 1 | 1000 | -0.0031 [-0.0064, +0.0002] | -0.0262 [-0.0336, -0.0194] | 248/1000 = 0.248 | neutral | block | G1,G2,G3 |
| stratified | 0.2 | 1 | 1500 | +0.0004 [-0.0027, +0.0035] | -0.0321 [-0.0405, -0.0235] | 373/1500 = 0.249 | neutral | block | G2,G3 |
| stratified | 0.2 | 1 | 2000 | -0.0017 [-0.0051, +0.0017] | -0.0369 [-0.0452, -0.0281] | 487/2000 = 0.243 | neutral | block | G1,G2,G3 |
| stratified | 0.2 | 1 | 3000 | -0.0057 [-0.0102, -0.0011] | -0.0369 [-0.0455, -0.0284] | 728/3000 = 0.243 | neutral | block | G1,G2,G3 |
| stratified | 0.2 | 2 | 100 | -0.5585 [-0.5748, -0.5411] | -0.6184 [-0.6365, -0.5971] | 19/100 = 0.190 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 2 | 200 | -0.1778 [-0.1903, -0.1658] | -0.4235 [-0.4383, -0.4059] | 45/200 = 0.225 | harmful | block | G1,G2,G3 |
| stratified | 0.2 | 2 | 300 | +0.0003 [-0.0019, +0.0027] | -0.0163 [-0.0228, -0.0097] | 68/300 = 0.227 | neutral | block | G3 |
| stratified | 0.2 | 2 | 500 | -0.0007 [-0.0035, +0.0021] | -0.0193 [-0.0260, -0.0127] | 116/500 = 0.232 | neutral | block | G3 |
| stratified | 0.2 | 2 | 700 | -0.0021 [-0.0053, +0.0008] | -0.0190 [-0.0251, -0.0127] | 163/700 = 0.233 | neutral | block | G1,G3 |
| stratified | 0.2 | 2 | 1000 | -0.0014 [-0.0042, +0.0014] | -0.0148 [-0.0208, -0.0087] | 232/1000 = 0.232 | neutral | block | G3 |
| stratified | 0.2 | 2 | 1500 | -0.0015 [-0.0045, +0.0015] | -0.0182 [-0.0246, -0.0120] | 345/1500 = 0.230 | neutral | block | G3 |
| stratified | 0.2 | 2 | 2000 | -0.0023 [-0.0051, +0.0005] | -0.0159 [-0.0221, -0.0100] | 457/2000 = 0.229 | neutral | block | G1,G3 |
| stratified | 0.2 | 2 | 3000 | +0.0001 [-0.0024, +0.0028] | -0.0194 [-0.0262, -0.0126] | 696/3000 = 0.232 | neutral | block | G3 |
| random | 0.0 | 0 | 100 | -0.0110 [-0.0153, -0.0068] | -0.0868 [-0.0966, -0.0781] | 6/100 = 0.060 (n/a) | harmful | block | G1,G2 |
| random | 0.0 | 0 | 200 | -0.0031 [-0.0062, -0.0002] | -0.0122 [-0.0177, -0.0071] | 9/200 = 0.045 (n/a) | neutral | block | G1 |
| random | 0.0 | 0 | 300 | +0.0001 [-0.0032, +0.0031] | -0.0493 [-0.0595, -0.0395] | 16/300 = 0.053 (n/a) | neutral | block | G2 |
| random | 0.0 | 0 | 500 | +0.0006 [-0.0022, +0.0033] | -0.0079 [-0.0129, -0.0029] | 19/500 = 0.038 (n/a) | neutral | pass | — |
| random | 0.0 | 0 | 700 | -0.0027 [-0.0070, +0.0014] | -0.0613 [-0.0721, -0.0509] | 22/700 = 0.031 (n/a) | harmful | block | G1,G2 |
| random | 0.0 | 0 | 1000 | -0.0003 [-0.0031, +0.0026] | -0.0187 [-0.0252, -0.0120] | 38/1000 = 0.038 (n/a) | neutral | pass | — |
| random | 0.0 | 0 | 1500 | -0.0021 [-0.0052, +0.0012] | -0.0223 [-0.0291, -0.0154] | 62/1500 = 0.041 (n/a) | neutral | block | G1 |
| random | 0.0 | 0 | 2000 | +0.0047 [+0.0008, +0.0086] | -0.0238 [-0.0335, -0.0143] | 71/2000 = 0.035 (n/a) | neutral | block | G2 |
| random | 0.0 | 0 | 3000 | +0.0017 [-0.0024, +0.0062] | -0.0102 [-0.0159, -0.0043] | 122/3000 = 0.041 (n/a) | neutral | pass | — |
| random | 0.0 | 1 | 100 | -0.0020 [-0.0052, +0.0011] | -0.0074 [-0.0113, -0.0034] | 5/100 = 0.050 (n/a) | neutral | block | G1 |
| random | 0.0 | 1 | 200 | -0.0041 [-0.0080, +0.0000] | -0.0397 [-0.0486, -0.0307] | 3/200 = 0.015 (n/a) | neutral | block | G1,G2 |
| random | 0.0 | 1 | 300 | -0.0024 [-0.0059, +0.0008] | -0.0234 [-0.0304, -0.0164] | 6/300 = 0.020 (n/a) | neutral | block | G1,G2 |
| random | 0.0 | 1 | 500 | -0.0018 [-0.0051, +0.0016] | -0.0390 [-0.0482, -0.0300] | 18/500 = 0.036 (n/a) | neutral | block | G1,G2 |
| random | 0.0 | 1 | 700 | +0.0009 [-0.0026, +0.0044] | -0.0437 [-0.0531, -0.0345] | 24/700 = 0.034 (n/a) | neutral | block | G2 |
| random | 0.0 | 1 | 1000 | +0.0007 [-0.0025, +0.0036] | -0.0234 [-0.0308, -0.0157] | 36/1000 = 0.036 (n/a) | neutral | block | G2 |
| random | 0.0 | 1 | 1500 | -0.0039 [-0.0089, +0.0011] | -0.0422 [-0.0529, -0.0309] | 66/1500 = 0.044 (n/a) | neutral | block | G1,G2 |
| random | 0.0 | 1 | 2000 | -0.0026 [-0.0059, +0.0005] | -0.0173 [-0.0235, -0.0110] | 74/2000 = 0.037 (n/a) | neutral | block | G1 |
| random | 0.0 | 1 | 3000 | +0.0009 [-0.0020, +0.0038] | -0.0331 [-0.0419, -0.0239] | 100/3000 = 0.033 (n/a) | neutral | block | G2 |
| random | 0.0 | 2 | 100 | -0.0050 [-0.0084, -0.0018] | -0.0093 [-0.0127, -0.0064] | 1/100 = 0.010 (n/a) | neutral | block | G1 |
| random | 0.0 | 2 | 200 | -0.0047 [-0.0083, -0.0012] | -0.0225 [-0.0294, -0.0154] | 8/200 = 0.040 (n/a) | neutral | block | G1 |
| random | 0.0 | 2 | 300 | +0.0014 [-0.0015, +0.0046] | -0.0180 [-0.0250, -0.0108] | 14/300 = 0.047 (n/a) | neutral | pass | — |
| random | 0.0 | 2 | 500 | -0.0004 [-0.0039, +0.0029] | -0.0506 [-0.0612, -0.0409] | 20/500 = 0.040 (n/a) | harmful | block | G2 |
| random | 0.0 | 2 | 700 | -0.0022 [-0.0047, +0.0003] | -0.0119 [-0.0173, -0.0065] | 35/700 = 0.050 (n/a) | neutral | pass | — |
| random | 0.0 | 2 | 1000 | -0.0089 [-0.0133, -0.0042] | -0.0459 [-0.0566, -0.0349] | 35/1000 = 0.035 (n/a) | neutral | block | G1,G2 |
| random | 0.0 | 2 | 1500 | -0.0001 [-0.0031, +0.0027] | -0.0212 [-0.0290, -0.0133] | 47/1500 = 0.031 (n/a) | neutral | pass | — |
| random | 0.0 | 2 | 2000 | +0.0031 [-0.0006, +0.0068] | -0.0413 [-0.0517, -0.0307] | 83/2000 = 0.042 (n/a) | neutral | block | G2 |
| random | 0.0 | 2 | 3000 | +0.0002 [-0.0027, +0.0029] | -0.0123 [-0.0179, -0.0063] | 117/3000 = 0.039 (n/a) | neutral | pass | — |

## Reading
- All 23 harmful runs are blocked, and all 3 useful runs (stratified, noise 0, 3,000 labels) pass.
- 61 of 82 neutral runs are blocked, mostly by G1/G2: their lower bounds fall below the margins even when the point estimate is near zero. Blocking a model that does not improve the current one costs little; the cost that matters is a useful model blocked, which did not happen here.
- Every noise run (p = 0.1 and 0.2) is blocked. In 10 of them G3 is the only failing check, so the label-share check stopped models that the evaluation checks would have let through.
- With only 3 useful runs, the "useful runs pass" side rests on very few cases.

## Cross-check with `pr gate`
Three runs were reranked again on the validation split with their saved models and passed through `pr gate` (copies of `model.json` with `label_policy: stratified` added, since curve-v1 predates that field). The frozen manifest hash matched the real file, and every G1/G2 lower bound equalled the stored run record.

| run | decision | failed | G1 lower | G2 lower | G3 share |
|---|---|---|---:|---:|---:|
| stratified, p 0, seed 0, 3,000 | pass | — | +0.007550 | −0.010244 | 0.0443 |
| stratified, p 0.1, seed 2, 3,000 | block | G3 | −0.002221 | −0.021056 | 0.1413 |
| stratified, p 0.2, seed 2, 100 | block | G1, G2, G3 | −0.574784 | −0.636480 | 0.1900 |
