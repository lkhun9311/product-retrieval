# IKEA real-photo check v1

Fixed 2026-10-06, before any IKEA number is computed. Issue #29.

## Why
All DeepFurniture queries are rendered. IKEA Interior gives real room photos with the products that appear in them, so it is a small check that the frozen baseline is not only working on renders.

## Data
- Source: IKEA Interior (IvonaTau, commit 6f316e3f), `text_data/item_to_room.p`.
- Gallery: every product image listed in `item_to_room.p` whose file exists (2,175). One image per product; product id = article number from the file name.
- Queries: every room photo referenced by the mapping whose file exists (205). The correct answers for a room are all products the mapping lists for it whose image exists.
- Rooms with no existing product are dropped and counted.
- v1.1 (2026-10-06, before any IKEA number was computed): the clone files some products under two folders. When both files are byte-identical they are one image. Two basenames (`090.319.12.jpg`, `190.265.47.jpg`) have different bytes in two folders; there is no way to tell which is the product, so both are treated as missing (dropped from the gallery and from every room's answers) and listed in the report.
- No split: the whole set is evaluation only. Nothing here is used for training or model choice.

## Query form
Whole room photo, no crop (there are no boxes). This favours large items; the result is a lower bound for a system that lets the user point at one item.

## Metrics (per room, then mean over rooms)
- **Hit@K**: 1 if at least one correct product is in the top K, else 0.
- **Recall@K**: number of correct products in the top K divided by the number of correct products for that room.
- K = 1, 5, 10, 20, 100. Ranking is by max cosine over the product's image (one image each), same frozen SigLIP2 model as the other baselines.
- Room-level bootstrap, 1,000 resamples, seed 0, 95% percentile interval.
- Also report: number of rooms, products per room (min / median / max), gallery size.

## Not claimed
- Not comparable with the single-answer product-level Recall@K of LRVS or DeepFurniture.
- Not a measure of pointing at one item in a real photo.
