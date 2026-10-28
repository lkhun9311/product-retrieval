# Crop before search on DeepFurniture scenes v1 (confirmation)

Contract `docs/contracts/crop-before-search-deepfurniture.md` (fixed before any result; Codex Astra review, 8 findings applied). Run `df-crop-v1`, code commit 57c2c1a (clean), 456 s wall. The IKEA crop contract hash was asserted unchanged (`10e64eee…`), so every detector, crop and merge setting is the one fixed for IKEA; nothing was tuned on DeepFurniture. Population checked against the contract before the run: 228 scenes, 285 answers, 2,527 gallery products, 187 clusters.

**Verdict (contract wording): the crop pipeline raised Recall@10 on these DeepFurniture scenes.**

Δ Recall@10 (Detector − Whole) = **+0.2844**, 95% cluster-bootstrap interval [+0.2087, +0.3602]; scenes with Δ > 0 / = 0 / < 0: 84 / 135 / 9. On IKEA (exploratory) the same pipeline gave +0.1295 [+0.1116, +0.1475]. The direction carries over; the verdict does not test the size, and the two populations differ (rendered vs photographed rooms, 1–3 vs a median of 13 answers per room).

## Mean metrics by arm

| K | Hit whole | Hit detector | Hit GT | Recall whole | Recall detector | Recall GT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0614 | 0.1140 | 0.1009 | 0.0570 | 0.1001 | 0.0899 |
| 5 | 0.1096 | 0.3860 | 0.4430 | 0.1031 | 0.3326 | 0.4057 |
| 10 | 0.1447 | 0.4605 | 0.5702 | 0.1316 | 0.4159 | 0.5190 |
| 20 | 0.1798 | 0.5395 | 0.6886 | 0.1645 | 0.5000 | 0.6308 |
| 100 | 0.3202 | 0.6667 | 0.8509 | 0.2836 | 0.6199 | 0.8026 |

## Secondary deltas (not used for the verdict)

| Comparison | Metric | K | Mean delta | 95% interval |
|---|---|---:|---:|---|
| detector - whole | recall | 1 | +0.0431 | [-0.0205, +0.1029] |
| detector - whole | recall | 5 | +0.2295 | [+0.1521, +0.3081] |
| detector - whole | recall | 10 | +0.2844 | [+0.2087, +0.3602] |
| detector - whole | recall | 20 | +0.3355 | [+0.2586, +0.4125] |
| detector - whole | recall | 100 | +0.3363 | [+0.2574, +0.4132] |
| detector - whole | hit | 1 | +0.0526 | [-0.0136, +0.1154] |
| detector - whole | hit | 5 | +0.2763 | [+0.1883, +0.3652] |
| detector - whole | hit | 10 | +0.3158 | [+0.2321, +0.4008] |
| detector - whole | hit | 20 | +0.3596 | [+0.2759, +0.4434] |
| detector - whole | hit | 100 | +0.3465 | [+0.2622, +0.4292] |
| gt - whole | recall | 1 | +0.0329 | [-0.0114, +0.0753] |
| gt - whole | recall | 5 | +0.3026 | [+0.2325, +0.3741] |
| gt - whole | recall | 10 | +0.3874 | [+0.3124, +0.4623] |
| gt - whole | recall | 20 | +0.4664 | [+0.3899, +0.5441] |
| gt - whole | recall | 100 | +0.5190 | [+0.4421, +0.5929] |
| gt - whole | hit | 1 | +0.0395 | [-0.0083, +0.0841] |
| gt - whole | hit | 5 | +0.3333 | [+0.2593, +0.4089] |
| gt - whole | hit | 10 | +0.4254 | [+0.3422, +0.5061] |
| gt - whole | hit | 20 | +0.5088 | [+0.4253, +0.5882] |
| gt - whole | hit | 100 | +0.5307 | [+0.4460, +0.6092] |
| gt - detector | recall | 1 | -0.0102 | [-0.0684, +0.0471] |
| gt - detector | recall | 5 | +0.0731 | [+0.0093, +0.1370] |
| gt - detector | recall | 10 | +0.1031 | [+0.0437, +0.1636] |
| gt - detector | recall | 20 | +0.1308 | [+0.0779, +0.1841] |
| gt - detector | recall | 100 | +0.1827 | [+0.1210, +0.2469] |
| gt - detector | hit | 1 | -0.0132 | [-0.0759, +0.0485] |
| gt - detector | hit | 5 | +0.0570 | [-0.0268, +0.1342] |
| gt - detector | hit | 10 | +0.1096 | [+0.0506, +0.1704] |
| gt - detector | hit | 20 | +0.1491 | [+0.0949, +0.2037] |
| gt - detector | hit | 100 | +0.1842 | [+0.1223, +0.2489] |

Delta (GT box - Detector) is not a measure of detector error alone: the GT arm differs in box geometry, selection and order. The GT arm is a reference, never a deployable result.

## Pooled Recall@10 per category (descriptive)

| Category | Denominator | Whole | Detector | GT box |
|---|---:|---:|---:|---:|
| Bed | 21 | 0.3810 | 0.4762 | 0.4762 |
| Cabinet#Shelf | 63 | 0.2540 | 0.3968 | 0.5556 |
| Chair#Stool | 17 | 0.2353 | 0.5882 | 0.6471 |
| Curtain | 30 | 0.0000 | 0.3667 | 0.0667 |
| Decoration | 32 | 0.0000 | 0.6250 | 0.8438 |
| Door | 43 | 0.0698 | 0.0698 | 0.2093 |
| Home-appliance | 7 | 0.0000 | 0.0000 | 0.5714 |
| Lamp | 41 | 0.0244 | 0.7073 | 0.7317 |
| Plant | 5 | 0.0000 | 0.4000 | 0.6000 |
| Sofa | 11 | 0.0909 | 0.4545 | 0.5455 |
| Table | 15 | 0.0000 | 0.4667 | 0.6667 |

## Alongside

- Image sizes: {'1600x1200': 223, '4000x3000': 5}.
- Boxes per scene: detector {'min': 1.0, 'median': 10.0, 'mean': 9.574561403508772, 'max': 10.0}, GT {'min': 2.0, 'median': 6.0, 'mean': 6.219298245614035, 'max': 10.0}.
- Scenes falling back to the whole image: {'detector': 0, 'gt': 0}.
- Instances without areaRatio: 14 in 6 scenes; answers with no box in the first 10: 3 (after the minimum size rule: 5).
- Mean seconds per scene (detector / embedding): whole 0.00 / 0.04; detector 1.01 / 0.27; gt 0.00 / 0.18.
- Total run seconds: 456.

## Reading
- Whole-scene search finds few of the answers (Recall@10 0.132): a scene embedding is dominated by the room, not the one to three val products in it.
- The detector pipeline closes much of the gap to the GT-box reference: Recall@10 0.416 against 0.519. The remaining Δ (GT box − Detector) +0.1031 mixes box geometry with box selection and order, so it is not detector error alone (contract §Arms).
- By category, lamps (0.024 → 0.707) and decorations (0.000 → 0.625) gain most from the detector; doors gain nothing (0.070 → 0.070), and "door" is not among the 24 detector prompts. Curtains are found by the detector (0.367) but rarely with GT boxes (0.067); category rows are descriptive, with small denominators.
- Detector cost on CPU: about 1.0 s per scene.

## Not claimed
- Real photographs: these scenes are rendered.
- That the IKEA effect size carries over.
- A user pointing at one item.
- Scenes from the other 24 archives.
