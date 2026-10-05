# Furniture sources (M2 stage 2)

Checked 2026-10-05. Raw files live outside the repo in `~/data/product-retrieval/`.

## DeepFurniture

- Source: Hugging Face `byliu/DeepFurniture` (dataset card license: `afl-3.0`). `data/metadata/dataset_info.json` still says `"license": "Add license here"`, so the card is the only license statement.
- Scenes are photo-realistic **renders**, not photos.
- Downloaded so far: metadata, furniture previews, query crops (1.4 GB). Not downloaded: rendered scenes (42 GB, 25 archives).
- sha256: `furnitures.jsonl` 94ea185be3f840ea…, `query_index.json` 55bea02ff4bd78cd… (full list of 40 files recorded locally).

| Item | Count |
|---|---:|
| Furniture identities (one preview image each) | 24,742 |
| Query crops | 7,264 |
| Distinct identities among queries | 3,478 (at most 3 queries each) |
| Distinct scenes the queries come from | 5,869 |
| Categories | 11: cabinet/shelf, table, chair/stool, lamp, door, bed, sofa, plant, decoration, curtain, home appliance |

Query file names look like `<furniture_id>_<n>_<scene>.jpg`. The first field matches a furniture preview for all 7,264 queries, so it is used as the truth identity. The second field is not the category id (it only agrees with it for 704 queries and goes up to 29); it looks like an instance number. **The naming is inferred, not documented.** Before using it as ground truth, check it against the scene annotations (`numberID` and identity per instance), which needs at least one scene archive.

## IKEA Interior (IvonaTau)

- Source: `github.com/IvonaTau/ikea`, commit 6f316e3f. README: "all images are property of IKEA.COM and are only allowed for non-commercial use". That is the repository author's statement; no license from IKEA itself was found.
- Room scenes are real photos. There are no bounding boxes.

| Item | Count |
|---|---:|
| Product images in `item_to_room.p` | 2,178 (2,175 found on disk) |
| Rooms referenced by the mapping | 585 (205 found on disk) |
| Usable product–room pairs (both files present) | 2,671 |
| Product folders | bed 54, chair 106, clock 107, couch 40, dining table 117, plant pot 34, other objects 1,859 |

`room_to_items.p` maps rooms to product **names** (for example "quilt cover and pillowcase"), not product ids. `item_to_room.p` maps product images (IKEA article numbers such as `500.210.76`) to rooms, so it is the usable ground truth: a room photo is a query and every product in it is a correct answer. Bedding items appear here, which DeepFurniture does not have.

## What this means for stage 2

- DeepFurniture fits the existing shape directly: query crop → gallery of identities → truth identity. Results on it are on rendered images only.
- IKEA rooms contain many products and no boxes, so a room query has several correct answers. It needs either manual boxes or a multi-answer metric (for example recall of any listed item). Kept as a small real-photo check.
