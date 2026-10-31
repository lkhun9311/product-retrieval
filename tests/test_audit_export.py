import copy
import json

import pytest

from product_retrieval.audit.export import (
    ExportError,
    append_record,
    judged_groups,
    read_export,
    validate_record,
)
from product_retrieval.audit.sample import CONTRACT_VERSION


def _sha(c: str) -> str:
    return c * 64


def _rec(left, right, relation, stage="A", **over):
    base = {
        "contract_version": CONTRACT_VERSION,
        "stage": stage,
        "display_id": f"a{left[:4]}{(right or '')[:4]}",
        "query_id": None,
        "left_sha": left,
        "right_sha": right,
        "target": None,
        "left_unit": "single",
        "right_unit": "single",
        "relation": relation,
        "evidence": {"tags": ["print"], "note": ""},
        "round": "first",
        "timestamp": "2026-11-01T09:00:00+00:00",
        "seconds": 12.5,
    }
    base.update(over)
    return base


def _target(sha, target):
    return _rec(sha, None, None, stage="target", target=target, right_unit=None, query_id="lrvs:abc")


def test_valid_pair_and_target_records():
    assert validate_record(_rec(_sha("a"), _sha("b"), "same")).relation == "same"
    box = {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.7}
    assert validate_record(_target(_sha("a"), box)).target.w == 0.5
    assert validate_record(_target(_sha("a"), "ambiguous")).target == "ambiguous"
    assert validate_record(_target(_sha("a"), "not_visible")).target == "not_visible"


@pytest.mark.parametrize(
    "key",
    list(_rec(_sha("a"), _sha("b"), "same")),
)
def test_missing_field_rejected(key):
    raw = _rec(_sha("a"), _sha("b"), "same")
    del raw[key]
    with pytest.raises(ExportError, match=key):
        validate_record(raw)


@pytest.mark.parametrize(
    "over",
    [
        {"relation": "identical"},
        {"stage": "E"},
        {"left_unit": "pair"},
        {"right_unit": "box"},
        {"round": "second"},
        {"contract_version": "ladder-1-id-audit-v1"},
        {"left_sha": "xyz"},
        {"left_sha": "A" * 64},
        {"timestamp": "yesterday"},
        {"timestamp": "2026-11-01T09:00:00"},
        {"seconds": -1},
        {"seconds": float("nan")},
        {"display_id": ""},
        {"evidence": {"tags": [""], "note": ""}},
        {"evidence": {"tags": [], "note": ""}, "extra": 1},
        {"surprise": 1},
        {"relation": None},
        {"right_sha": None},
    ],
)
def test_unknown_or_bad_values_rejected(over):
    raw = {**_rec(_sha("a"), _sha("b"), "same"), **over}
    with pytest.raises(ExportError):
        validate_record(raw)


@pytest.mark.parametrize(
    "box",
    [
        {"x": -0.01, "y": 0, "w": 0.5, "h": 0.5},
        {"x": 0, "y": 1.01, "w": 0.5, "h": 0.5},
        {"x": 0, "y": 0, "w": 1.5, "h": 0.5},
        {"x": 0.6, "y": 0, "w": 0.5, "h": 0.5},
        {"x": 0, "y": 0.6, "w": 0.5, "h": 0.5},
        {"x": 0, "y": 0, "w": 0, "h": 0.5},
        {"x": 0, "y": 0, "w": 0.5},
        {"x": 0, "y": 0, "w": 0.5, "h": 0.5, "z": 1},
        "maybe",
        None,
    ],
)
def test_bad_box_rejected(box):
    with pytest.raises(ExportError):
        validate_record(_target(_sha("a"), box))


def test_box_on_the_edge_is_accepted():
    validate_record(_target(_sha("a"), {"x": 0, "y": 0, "w": 1, "h": 1}))
    validate_record(_target(_sha("a"), {"x": 0.5, "y": 0.25, "w": 0.5, "h": 0.75}))


def test_stage_rules():
    with pytest.raises(ExportError):  # target record may not carry a relation
        validate_record(_rec(_sha("a"), None, "same", stage="target", target="ambiguous", right_unit=None))
    with pytest.raises(ExportError):  # C record needs the target it was judged against
        validate_record(_rec(_sha("a"), _sha("b"), "same", stage="C", query_id="lrvs:abc"))
    validate_record(_rec(_sha("a"), _sha("b"), "same", stage="C", query_id="lrvs:abc", target="ambiguous"))


def test_input_not_an_object():
    with pytest.raises(ExportError):
        validate_record([1, 2])  # type: ignore[arg-type]


def test_append_only(tmp_path):
    path = tmp_path / "export.jsonl"
    append_record(path, _rec(_sha("a"), _sha("b"), "same"))
    before = path.read_bytes()
    append_record(path, _rec(_sha("a"), _sha("c"), "different"))
    after = path.read_bytes()
    assert after.startswith(before) and after.count(b"\n") == 2
    # the invalid record leaves the file untouched
    with pytest.raises(ExportError):
        append_record(path, _rec(_sha("a"), _sha("c"), "bogus"))
    assert path.read_bytes() == after
    assert len(read_export(path)) == 2


def test_append_refuses_torn_file(tmp_path):
    path = tmp_path / "export.jsonl"
    path.write_bytes(b'{"half": ')
    with pytest.raises(ExportError, match="newline"):
        append_record(path, _rec(_sha("a"), _sha("b"), "same"))
    assert path.read_bytes() == b'{"half": '


def test_read_reports_line_number(tmp_path):
    path = tmp_path / "export.jsonl"
    append_record(path, _rec(_sha("a"), _sha("b"), "same"))
    with open(path, "a") as f:
        f.write(json.dumps({"stage": "A"}) + "\n")
    with pytest.raises(ExportError, match=r":2:"):
        read_export(path)
    path.write_text("not json\n")
    with pytest.raises(ExportError, match=r":1:"):
        read_export(path)


def _roundtrip(tmp_path, raws):
    path = tmp_path / "export.jsonl"
    for raw in raws:
        append_record(path, raw)
    return read_export(path)


def _grouping(groups):
    return sorted(sorted(g) for g in groups["groups"])


def test_split_is_representable(tmp_path):
    # one ID owns images a, b, c; the reviewer finds a-b the same product, c a different one
    a, b, c = _sha("a"), _sha("b"), _sha("c")
    recs = _roundtrip(
        tmp_path,
        [_rec(a, b, "same"), _rec(a, c, "different"), _rec(b, c, "different")],
    )
    got = judged_groups(recs)
    assert _grouping(got) == [[a, b], [c]]
    assert got["different"] == {frozenset((a, c)), frozenset((b, c))}


def test_merge_is_representable(tmp_path):
    # image a belongs to ID P, image x to ID Q; they are judged the same product (C pair, with decoy)
    a, g, x = _sha("a"), _sha("b"), _sha("c")
    recs = _roundtrip(
        tmp_path,
        [
            _target(a, {"x": 0, "y": 0, "w": 0.5, "h": 0.5}),
            _rec(a, g, "same", stage="C", query_id="lrvs:q", target={"x": 0, "y": 0, "w": 0.5, "h": 0.5}),
            _rec(a, x, "same", stage="B", query_id="lrvs:q"),
        ],
    )
    got = judged_groups(recs)
    assert _grouping(got) == [sorted([a, g, x])]


def test_unresolved_replacement_is_representable(tmp_path):
    # the labelled truth g is judged different; candidate h is undecidable; nothing joins
    a, g, h = _sha("a"), _sha("b"), _sha("c")
    target = {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.4}
    recs = _roundtrip(
        tmp_path,
        [
            _rec(a, g, "different", stage="C", query_id="lrvs:q", target=target),
            _rec(
                a, h, "undecidable", stage="B", query_id="lrvs:q", evidence={"tags": ["cropped"], "note": "x"}
            ),
        ],
    )
    got = judged_groups(recs)
    assert _grouping(got) == [[a], [g], [h]]
    assert got["undecidable"] == {frozenset((a, h))}
    assert got["different"] == {frozenset((a, g))}


def test_groups_use_no_transitivity_from_other_relations(tmp_path):
    a, b, c = _sha("a"), _sha("b"), _sha("c")
    recs = _roundtrip(tmp_path, [_rec(a, b, "consistent"), _rec(b, c, "variant")])
    got = judged_groups(recs)
    assert _grouping(got) == [[a], [b], [c]]
    assert got["consistent"] == {frozenset((a, b))} and got["variant"] == {frozenset((b, c))}


def test_rereview_kept_apart_and_latest_wins(tmp_path):
    a, b = _sha("a"), _sha("b")
    first = _rec(a, b, "same")
    later_first = _rec(a, b, "different", timestamp="2026-11-02T09:00:00+00:00")
    rr = _rec(a, b, "variant", round="rereview", timestamp="2026-11-20T09:00:00+00:00")
    recs = _roundtrip(tmp_path, [first, later_first, rr])
    assert len(recs) == 3  # every judgement stays in the file
    assert judged_groups(recs)["different"] == {frozenset((a, b))}
    assert judged_groups(recs, "rereview")["variant"] == {frozenset((a, b))}
    assert len(judged_groups(recs, None)["variant"]) == 1


def test_record_does_not_hold_product_ids():
    raw = _rec(_sha("a"), _sha("b"), "same", query_id="lrvs:0123456789abcdef")
    assert not any("product" in k for k in raw)
    assert copy.deepcopy(raw) == raw
