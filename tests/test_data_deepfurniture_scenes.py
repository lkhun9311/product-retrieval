"""Population, GT box ordering and native-coordinate handling (contract: crop-before-search-deepfurniture)."""

import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from product_retrieval.data import deepfurniture_scenes as ds
from product_retrieval.data.deepfurniture_scenes import (
    Instance,
    SceneAnnotation,
    SceneBoxError,
    ScenePopulationError,
)


def inst(number_id, identity="V1", category="Sofa", area=None, box=(10.0, 20.0, 110.0, 120.0)):
    return Instance(number_id, box, identity, category, area)


def scene(scene_id, *instances):
    return SceneAnnotation(scene_id, tuple(instances))


def raw_instance(number_id, identity, box=(1, 2, 30, 40), area="absent", category="Sofa"):
    d = {
        "numberID": number_id,
        "boundingBox": {"xMin": box[0], "yMin": box[1], "xMax": box[2], "yMax": box[3]},
        "identityID": identity,
        "categoryName": category,
    }
    if area != "absent":
        d["areaRatio"] = area
    return d


# ------------------------------------------------------------------------------------ eligibility

SPLITS = {"V1": "val", "V2": "val", "V3": "val", "T1": "test", "R1": "train", "R2": "train"}


def annotations():
    return {
        "s1": scene("s1", inst(1, "V1"), inst(2, "R1")),  # eligible, answers [V1]
        "s2": scene("s2", inst(1, "V3"), inst(2, "V2"), inst(3, "V2")),  # eligible, answers [V2, V3]
        "s3": scene("s3", inst(1, "V1"), inst(2, "T1")),  # val + test: excluded
        "s4": scene("s4", inst(1, "R1"), inst(2, "R2")),  # no val identity: excluded
        "s5": scene("s5", inst(1, "V1")),  # would qualify, but is a val query source
    }


def test_eligibility_rules_and_counts():
    pop = ds.select_population(annotations(), SPLITS, {"s5"}, expected_scenes=2, expected_answers=3)
    assert pop.scenes == ["s1", "s2"]
    assert pop.answers == {"s1": ["V1"], "s2": ["V2", "V3"]}
    assert pop.excluded == {"no_val_identity": 1, "test_identity": 1, "val_query_source": 1}
    assert pop.excluded_query_source_scenes == ["s5"]


def test_test_identity_scene_is_never_selected():
    ann = {"s3": scene("s3", inst(1, "V1"), inst(2, "T1"))}
    with pytest.raises(ScenePopulationError, match="0 scenes"):
        ds.select_population(ann, SPLITS, set(), expected_scenes=1, expected_answers=1)


def test_val_query_source_scene_is_excluded():
    with pytest.raises(ScenePopulationError, match="2 scenes / 3 answers"):
        # without the exclusion s5 would be counted: 3 scenes, 4 answers
        ds.select_population(annotations(), SPLITS, {"s5"}, expected_scenes=3, expected_answers=4)
    pop = ds.select_population(annotations(), SPLITS, set(), expected_scenes=3, expected_answers=4)
    assert pop.scenes == ["s1", "s2", "s5"]


@pytest.mark.parametrize("scenes,answers", [(3, 3), (2, 4), (0, 0)])
def test_wrong_counts_stop_the_run(scenes, answers):
    with pytest.raises(ScenePopulationError, match="contract says"):
        ds.select_population(annotations(), SPLITS, {"s5"}, expected_scenes=scenes, expected_answers=answers)


def test_default_counts_are_the_contract_counts():
    assert (ds.EXPECTED_SCENES, ds.EXPECTED_ANSWERS, ds.EXPECTED_GALLERY) == (228, 285, 2527)
    with pytest.raises(ScenePopulationError, match="228 / 285"):
        ds.select_population(annotations(), SPLITS, {"s5"})


def test_identity_missing_from_manifest_stops_the_run():
    ann = {"s1": scene("s1", inst(1, "V1"), inst(2, "ZZ"))}
    with pytest.raises(ScenePopulationError, match="not in the manifest"):
        ds.select_population(ann, SPLITS, set(), expected_scenes=1, expected_answers=1)


def test_val_query_source_scenes_only_counts_val_identities():
    index = {
        "queries": [
            {"query_name": "V1_1_s5.jpg"},
            {"query_name": "T1_2_s3.jpg"},  # test query: its scene is not excluded by this rule
            {"query_name": "R1_1_s4.jpg"},
        ]
    }
    assert ds.val_query_source_scenes(index, SPLITS) == {"s5"}


@pytest.mark.parametrize("name", ["V1_1.jpg", "V1_1_s5.png", "V1_s5", "_1_s5.jpg"])
def test_bad_query_name_is_an_error(name):
    with pytest.raises(ScenePopulationError, match="query name"):
        ds.val_query_source_scenes({"queries": [{"query_name": name}]}, SPLITS)


def test_gallery_is_the_val_rows_sorted():
    rows = [
        SimpleNamespace(product_id="V2", split="val", gallery=["b" * 64]),
        SimpleNamespace(product_id="T1", split="test", gallery=["t" * 64]),
        SimpleNamespace(product_id="V1", split="val", gallery=["a" * 64]),
    ]
    assert ds.select_gallery(rows, expected=2) == [("V1", "a" * 64), ("V2", "b" * 64)]
    with pytest.raises(ScenePopulationError, match="contract says 2527"):
        ds.select_gallery(rows)
    rows[0].gallery = ["b" * 64, "c" * 64]
    with pytest.raises(ScenePopulationError, match="not 1"):
        ds.select_gallery(rows, expected=2)


# ---------------------------------------------------------------------------------------- parsing


def test_parse_annotation_reads_fields():
    obj = {
        "scene": {"sceneTaskID": "s1"},
        "instances": [raw_instance(7, 2880315, area=0.25), raw_instance(8, "V1", area=None)],
    }
    a = ds.parse_annotation("s1", obj)
    assert a.instances[0] == Instance(7, (1.0, 2.0, 30.0, 40.0), "2880315", "Sofa", 0.25)
    assert a.instances[1].area_ratio is None  # null counts as missing


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda o: o["instances"][0].pop("numberID"), "numberID"),
        (lambda o: o["instances"][0].pop("boundingBox"), "boundingBox"),
        (lambda o: o["instances"][0].pop("identityID"), "identityID"),
        (lambda o: o["instances"][0].update(areaRatio=float("nan")), "not finite"),
        (lambda o: o["instances"][0].update(areaRatio="big"), "areaRatio"),
        (lambda o: o["instances"][0]["boundingBox"].update(xMin=float("inf")), "not finite"),
        (lambda o: o["instances"].append(raw_instance(1, "V2")), "duplicate numberID"),
        (lambda o: o["scene"].update(sceneTaskID="other"), "sceneTaskID"),
        (lambda o: o.update(instances=[]), "no instances"),
        (lambda o: o.pop("instances"), "instances"),
    ],
)
def test_parse_annotation_rejects_bad_input(mutate, match):
    obj = {"scene": {"sceneTaskID": "s1"}, "instances": [raw_instance(1, "V1", area=0.1)]}
    mutate(obj)
    with pytest.raises(ScenePopulationError, match=match):
        ds.parse_annotation("s1", obj)


def _tar(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_archive_streaming_reads_annotations_and_only_requested_images(tmp_path):
    ann = lambda sid: json.dumps(  # noqa: E731
        {"scene": {"sceneTaskID": sid}, "instances": [raw_instance(1, "V1")]}
    ).encode()
    path = tmp_path / "scenes.tar.gz"
    _tar(
        path,
        {
            "a/image.jpg": b"A",
            "a/annotation.json": ann("a"),
            "a/depth.png": b"D",
            "b/annotation.json": ann("b"),
            "b/image.jpg": b"B",
        },
    )
    assert sorted(ds.read_annotations(path)) == ["a", "b"]
    assert ds.read_scene_images(path, {"b"}) == {"b": b"B"}
    with pytest.raises(ScenePopulationError, match="no image.jpg"):
        ds.read_scene_images(path, {"a", "zz"})


def test_truncated_or_empty_archive_is_an_error(tmp_path):
    path = tmp_path / "scenes.tar.gz"
    _tar(path, {"a/annotation.json": json.dumps({"instances": [raw_instance(1, "V1")]}).encode()})
    cut = tmp_path / "cut.tar.gz"
    cut.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(ScenePopulationError):
        ds.read_annotations(cut)
    empty = tmp_path / "empty.tar.gz"
    _tar(empty, {"a/image.jpg": b"A"})
    with pytest.raises(ScenePopulationError, match="no annotation.json"):
        ds.read_annotations(empty)


# ---------------------------------------------------------------------------------------- GT order


def test_gt_order_area_desc_missing_last_ties_by_number():
    ins = [
        inst(5, area=None),
        inst(2, area=0.30),
        inst(9, area=0.50),
        inst(1, area=None),
        inst(4, area=0.30),  # ties with #2: numberID ascending
        inst(3, area=0.0),  # a known 0.0 still beats any missing value
    ]
    assert [i.number_id for i in ds.gt_order(ins)] == [9, 2, 4, 3, 1, 5]


def test_gt_order_does_not_depend_on_input_order():
    ins = [inst(5), inst(2, area=0.3), inst(9, area=0.5), inst(1), inst(4, area=0.3)]
    assert [i.number_id for i in ds.gt_order(reversed(ins))] == [i.number_id for i in ds.gt_order(ins)]


def test_gt_crops_cut_to_m_then_padding_in_order():
    ins = [inst(n, area=1.0 / n, box=(10.0 * n, 0.0, 10.0 * n + 40, 40.0)) for n in range(1, 13)]
    res = ds.gt_crops(scene("s", *ins), 800, 200, m=10)
    assert [i.number_id for i in res.first_m] == list(range(1, 11))  # 11 and 12 are cut
    assert [c.index for c in res.crops] == list(range(1, 11))
    # 10% of 40 = 4 on each side, clipped at 0: #1 box (10,0,50,40) -> (6,0,54,44)
    assert res.crops[0].box == (6, 0, 54, 44)


def test_gt_crops_use_native_pixel_coordinates_on_a_large_image():
    big = scene("s", inst(1, area=0.2, box=(1200.0, 900.0, 2800.0, 2100.0)))
    res = ds.gt_crops(big, 4000, 3000)
    # 10% of 1600 = 160, of 1200 = 120: no rescaling to any model size
    assert res.crops[0].box == (1040, 780, 2960, 2220)
    # the same annotation on a 1600x1200 image is clipped to (1200, 900, 1600, 1200) first, then padded
    small = ds.gt_crops(big, 1600, 1200).crops[0].box
    assert small == (1160, 870, 1600, 1200)


def test_gt_box_partially_outside_is_clipped_not_an_error():
    s = scene("s", inst(1, box=(-30.0, 50.0, 60.0, 5000.0)))
    assert ds.clipped_boxes(s, 400, 300) == {1: (0.0, 50.0, 60.0, 300.0)}


@pytest.mark.parametrize(
    "box,match",
    [
        ((500.0, 10.0, 600.0, 100.0), "outside"),  # entirely right of a 400 px wide image
        ((-80.0, 10.0, -5.0, 100.0), "outside"),
        ((10.0, 300.0, 100.0, 400.0), "outside"),  # touching the bottom edge only: zero area after clipping
        ((100.0, 10.0, 100.0, 50.0), "no positive area"),
        ((100.0, 60.0, 50.0, 10.0), "no positive area"),
    ],
)
def test_unusable_box_stops_the_run(box, match):
    with pytest.raises(SceneBoxError, match=match):
        ds.gt_crops(scene("s", inst(1, box=(10.0, 10.0, 50.0, 50.0)), inst(2, box=box)), 400, 300)


def test_a_bad_box_outside_the_first_m_is_still_an_error():
    ins = [inst(n, area=1.0 - n / 100) for n in range(1, 12)]
    ins.append(inst(12, area=0.0, box=(900.0, 900.0, 950.0, 950.0)))  # last in GT order, far outside
    with pytest.raises(SceneBoxError, match="instance 12"):
        ds.gt_crops(scene("s", *ins), 400, 300, m=10)


def test_small_boxes_are_dropped_by_the_minimum_size_rule_not_by_error():
    s = scene("s", inst(1, area=0.5, box=(10.0, 10.0, 20.0, 20.0)), inst(2, area=0.4))
    res = ds.gt_crops(s, 400, 300)
    assert [c.index for c in res.crops] == [2]  # 10 px box + 10% padding is 12 px < 16
    assert [i.number_id for i in res.first_m] == [1, 2]


def test_answers_without_box_and_category_pairs():
    s = scene(
        "s",
        inst(1, "V1", "Sofa"),
        inst(2, "V2", "Table"),
        inst(3, "V2", "Chair"),
        inst(4, "R1", "Lamp"),
    )
    assert ds.answers_without_box(s, ["V1", "V2"], [2, 4]) == ["V1"]
    assert ds.answers_without_box(s, ["V1", "V2"], [1, 2, 3]) == []
    assert ds.answer_categories(s, ["V1", "V2"]) == [("V1", "Sofa"), ("V2", "Chair"), ("V2", "Table")]


def test_gt_crops_default_cut_is_ten():
    ins = [inst(n, area=1.0 / n, box=(10.0 * n, 0.0, 10.0 * n + 40, 40.0)) for n in range(1, 13)]
    res = ds.gt_crops(scene("s", *ins), 800, 200)
    assert ds.TOP_M == 10 and [c.index for c in res.crops] == list(range(1, 11))
    assert [i.number_id for i in res.first_m] == list(range(1, 11))
