"""Pure box logic of the crop-before-search check, on hand-made boxes (no model)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from product_retrieval.crop import owl
from product_retrieval.crop.boxes import (
    build_crops,
    clip_boxes,
    crops_from_predictions,
    exif_orientation,
    expand_and_round,
    greedy_nms,
    iou,
    load_exif_rgb,
    max_sigmoid,
    select_detections,
)


def test_max_sigmoid_is_the_max_over_prompts_per_box():
    # box 0: sigmoid(0) = 0.5, sigmoid(-10) = 4.54e-5            -> 0.5
    # box 1: sigmoid(-2) = 0.119203, sigmoid(2) = 0.880797       -> 0.880797
    got = max_sigmoid(np.array([[0.0, -10.0], [-2.0, 2.0]]))
    assert got == pytest.approx([0.5, 0.8807970779778823])


@pytest.mark.parametrize("bad", [np.array([[np.nan, 0.0]]), np.zeros((3,)), np.zeros((2, 0))])
def test_max_sigmoid_rejects_broken_logits(bad):
    with pytest.raises(ValueError):
        max_sigmoid(bad)


def test_clip_boxes():
    got = clip_boxes(np.array([[-5.0, -5.0, 50.0, 50.0]]), width=40, height=30)
    assert got.tolist() == [[0.0, 0.0, 40.0, 30.0]]


def test_iou_hand_values():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    assert iou(a, np.array([0.0, 0.0, 10.0, 5.0])) == 0.5  # inter 50, union 100
    assert iou(a, np.array([20.0, 20.0, 30.0, 30.0])) == 0.0
    assert iou(np.array([3.0, 3.0, 3.0, 3.0]), np.array([3.0, 3.0, 3.0, 3.0])) == 0.0  # empty union


def test_nms_iou_exactly_half_survives_and_slightly_more_is_suppressed():
    top = [0.0, 0.0, 10.0, 10.0]
    # IoU(top, [0,0,10,5]) = 50 / 100 = 0.5 exactly: not > 0.5, kept.
    assert greedy_nms(np.array([top, [0.0, 0.0, 10.0, 5.0]]), np.array([0.9, 0.8])) == [0, 1]
    # IoU(top, [0,0,10,5.1]) = 51 / 100 = 0.51 > 0.5: suppressed.
    assert greedy_nms(np.array([top, [0.0, 0.0, 10.0, 5.1]]), np.array([0.9, 0.8])) == [0]


def test_nms_orders_by_score_and_breaks_equal_scores_by_index():
    boxes = np.array([[0, 0, 1, 1], [10, 0, 11, 1], [20, 0, 21, 1]], dtype=float)
    assert greedy_nms(boxes, np.array([0.5, 0.9, 0.5])) == [1, 0, 2]
    # identical boxes, equal scores: the lower index wins, the other is suppressed.
    same = np.array([[0, 0, 10, 10]] * 6, dtype=float)
    s = np.full(6, 0.6)
    assert greedy_nms(same, s) == [0]
    # candidate at index 3 beats the one at index 5 even though 5 appears later in the input.
    two = np.array([[0, 0, 5, 5]] * 2 + [[100, 100, 105, 105]] * 0, dtype=float)
    assert greedy_nms(two, np.array([0.7, 0.7])) == [0]


def test_nms_matches_a_vectorised_reference_on_random_boxes():
    rng = np.random.default_rng(3)
    n = 60
    xy = rng.uniform(0, 80, size=(n, 2))
    wh = rng.uniform(5, 40, size=(n, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1)
    scores = np.round(rng.uniform(0, 1, n), 1)  # many ties

    # reference: pairwise IoU matrix, then suppress by walking the (-score, index) order
    ix = np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    iy = np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    mat = inter / (area[:, None] + area[None, :] - inter)
    order = np.lexsort((np.arange(n), -scores))
    alive = np.ones(n, dtype=bool)
    ref = []
    for i in order:
        if alive[i]:
            ref.append(int(i))
            alive &= ~(mat[i] > 0.5)
            alive[i] = False
    assert greedy_nms(boxes, scores) == ref


def test_select_detections_threshold_is_inclusive_and_order_is_by_score():
    boxes = np.array([[0, 0, 5, 5], [10, 0, 15, 5], [20, 0, 25, 5], [30, 0, 35, 5]], dtype=float)
    scores = np.array([0.1, 0.0999, 0.5, 0.3])
    got = select_detections(boxes, scores, width=100, height=100)
    assert [d.index for d in got] == [2, 3, 0]  # 0.1 kept (>=), 0.0999 dropped
    assert [d.score for d in got] == [0.5, 0.3, 0.1]


def test_select_detections_takes_top_m_after_nms():
    n = 13
    boxes = np.array([[10 * i, 0, 10 * i + 5, 5] for i in range(n)], dtype=float)
    boxes[1] = boxes[0]  # box 1 duplicates box 0 and is suppressed by it
    scores = 0.95 - 0.01 * np.arange(n)
    got = select_detections(boxes, scores, width=200, height=50)
    # 13 boxes - 1 suppressed = 12 survive NMS; the first M=10 by score are indices 0, 2..10.
    assert [d.index for d in got] == [0, *range(2, 11)]


def test_select_detections_clips_before_nms_and_empty_when_nothing_passes():
    got = select_detections(np.array([[-5.0, -5.0, 50.0, 50.0]]), np.array([0.9]), width=40, height=30)
    assert got[0].box == (0.0, 0.0, 40.0, 30.0)
    assert select_detections(np.array([[0, 0, 5, 5]], dtype=float), np.array([0.05]), 40, 30) == []
    assert select_detections(np.zeros((0, 4)), np.zeros(0), 40, 30) == []


def test_select_detections_rejects_broken_inputs():
    with pytest.raises(ValueError):
        select_detections(np.array([[0, 0, 5, 5]], dtype=float), np.array([np.nan]), 40, 30)
    with pytest.raises(ValueError):
        select_detections(np.array([[0, 0, 5, 5]], dtype=float), np.array([0.5, 0.5]), 40, 30)
    with pytest.raises(ValueError):
        select_detections(np.array([[0, 0, np.inf, 5]]), np.array([0.5]), 40, 30)
    with pytest.raises(ValueError):
        select_detections(np.array([[0, 0, 5, 5]], dtype=float), np.array([0.5]), 0, 30)


def test_expand_and_round_hand_values():
    # w = h = 40, 10% = 4.0 each side: exactly (6, 16, 54, 64)
    assert expand_and_round((10.0, 20.0, 50.0, 60.0), 100, 100) == (6, 16, 54, 64)
    # x: 10.5 - 4 = 6.5 -> floor 6 ; 50.5 + 4 = 54.5 -> ceil 55 ; y: 16.2 -> 16 ; 64.2 -> 65
    assert expand_and_round((10.5, 20.2, 50.5, 60.2), 100, 100) == (6, 16, 55, 65)
    # expansion past the border is clipped: 30 wide, 3 each side -> [-3, 33] -> [0, 32]
    assert expand_and_round((0.0, 0.0, 30.0, 30.0), 32, 32) == (0, 0, 32, 32)
    # outward, not to nearest: width 10.01 -> 1.001 each side -> [18.999, 31.011] -> floor 18, ceil 32
    assert expand_and_round((20.0, 0.0, 30.01, 40.0), 100, 100)[::2] == (18, 32)


def test_build_crops_drops_under_16_px_boundary_values():
    def one(box):
        return build_crops(
            select_detections(np.array([box], dtype=float), np.array([0.9]), 200, 200), 200, 200
        )

    # width 12 -> expanded to [18.8, 33.2] -> [18, 34] = 16 px: kept (not narrower than 16)
    assert [c.box for c in one((20, 0, 32, 50))] == [(18, 0, 34, 55)]
    # width 11 -> [18.9, 32.1] -> [18, 33] = 15 px: dropped
    assert one((20, 0, 31, 50)) == []
    # height 11 -> 15 px: dropped, even though the width is fine
    assert one((0, 20, 50, 31)) == []
    # width 13 -> [18.7, 34.3] -> [18, 35] = 17 px: kept
    assert one((20, 0, 33, 50))[0].box == (18, 0, 35, 55)


def test_zero_area_box_uses_a_top_m_slot_and_is_removed_by_the_size_rule():
    # Documented reading: NMS and top-M come before the 16 px rule.
    boxes = [[5.0, 5.0, 5.0, 5.0]] + [[20 * i + 30, 0.0, 20 * i + 60, 60.0] for i in range(10)]
    scores = [0.99] + [0.9 - 0.01 * i for i in range(10)]
    crops = crops_from_predictions(np.array(boxes), np.array(scores), 400, 100)
    assert len(crops) == 9
    assert 0 not in [c.index for c in crops]


def test_crops_from_predictions_zero_boxes_gives_empty_list_for_the_caller_to_fall_back():
    assert crops_from_predictions(np.array([[0, 0, 5, 5]], dtype=float), np.array([0.01]), 100, 100) == []


def test_exif_transpose_then_rgb(tmp_path):
    p = tmp_path / "a.jpg"
    im = Image.new("L", (30, 10), color=128)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees to display
    im.save(p, format="JPEG", exif=exif)
    assert exif_orientation(p) == 6
    out = load_exif_rgb(p)
    assert out.mode == "RGB" and out.size == (10, 30)
    plain = tmp_path / "b.jpg"
    Image.new("RGB", (30, 10)).save(plain, format="JPEG")
    assert exif_orientation(plain) == 1 and load_exif_rgb(plain).size == (30, 10)


# --- OWLv2 wrapper without the model -------------------------------------------------------------


def test_owl_constants_are_the_contract_values():
    assert owl.OWL_MODEL_ID == "google/owlv2-base-patch16-ensemble"
    assert owl.OWL_REVISION == "cfd3195ba4ea9592eec887ded089f4c08eff231d"
    assert len(owl.PROMPTS) == 24 and len(set(owl.PROMPTS)) == 24
    assert owl.PROMPTS[:3] == ("bed", "sofa", "armchair") and owl.PROMPTS[-1] == "storage box"
    assert "chest of drawers" in owl.PROMPTS and "ceiling lamp" in owl.PROMPTS


def test_owl_detector_pins_revision_and_uses_a_square_target_size(monkeypatch):
    import transformers

    proc, model = MagicMock(), MagicMock()
    proc_loader, model_loader = MagicMock(return_value=proc), MagicMock(return_value=model)
    monkeypatch.setattr(transformers.Owlv2Processor, "from_pretrained", proc_loader)
    monkeypatch.setattr(transformers.Owlv2ForObjectDetection, "from_pretrained", model_loader)
    model.to.return_value = model
    model.eval.return_value = model

    det = owl.OwlDetector()
    assert proc_loader.call_args.kwargs["revision"] == owl.OWL_REVISION
    assert model_loader.call_args.kwargs["revision"] == owl.OWL_REVISION
    assert proc_loader.call_args.args[0] == model_loader.call_args.args[0] == owl.OWL_MODEL_ID
    assert [c.args[0] for c in model.to.call_args_list] == ["cpu", torch.float32]
    assert det.revision == owl.OWL_REVISION

    proc.return_value = {"pixel_values": torch.zeros(1)}
    proc.post_process_grounded_object_detection.return_value = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 90.0, 40.0], [0.0, 0.0, 5.0, 5.0]]),
            "scores": torch.tensor([0.8, 0.5]),
            "labels": torch.tensor([0, 1]),
        }
    ]
    crops = det.detect(Image.new("RGB", (100, 50)))  # W=100, H=50 -> square side 100
    assert proc.call_args.kwargs["text"] == [list(owl.PROMPTS)]
    assert proc.post_process_grounded_object_detection.call_args.kwargs["target_sizes"] == [(100, 100)]
    # box 1 is 5 px: expanded to 6 px, under 16 -> dropped. Box 0: 80 x 30 -> [2, 98] x [7, 43]
    assert [(c.index, c.box) for c in crops] == [(0, (2, 7, 98, 43))]
