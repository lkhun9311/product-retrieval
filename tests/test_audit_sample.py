import json

import pytest
from typer.testing import CliRunner

from product_retrieval.audit.sample import (
    DEFAULT_DESIGN,
    AuditDesign,
    AuditSampleError,
    display_plan,
    draw_fresh,
    draw_rereview,
    draw_sample,
    ids_sha256,
    largest_remainder,
    run_audit_sample,
    stage_counts,
    stage_items,
    stratum_of,
)
from product_retrieval.cli import app
from product_retrieval.core.ids import sha256_file
from product_retrieval.data.manifest import ManifestRow, load_manifest

runner = CliRunner()
SIZES = {1: 40, 2: 20, 3: 30, 4: 10}
DESIGN = AuditDesign(
    sample={1: 8, 2: 4, 3: 6, 4: 2},
    pilot={1: 4, 2: 2, 3: 3, 4: 1},
    rereview_pilot=4,
    rereview_full=8,
    fresh=20,
    expected_population=SIZES,
)


def _sha(n: int) -> str:
    return f"{n:064x}"


def _make_rows(sizes=SIZES, start=900_000, shuffle=False):
    shape = {1: (1, 1), 2: (2, 1), 3: (4, 1), 4: (2, 3)}
    rows, n, pid = [], 1, start
    for h, count in sizes.items():
        for _ in range(count):
            nq, ng = shape[h]
            rows.append(
                {
                    "source": "lrvs",
                    "product_id": str(pid),
                    "split": "val",
                    "query": [_sha(n + i) for i in range(nq)],
                    "gallery": [_sha(n + nq + i) for i in range(ng)],
                }
            )
            n += nq + ng
            pid += 7
    rows.append(
        {"source": "lrvs", "product_id": "5", "split": "train", "query": [_sha(n)], "gallery": [_sha(n + 1)]}
    )
    if shuffle:
        rows.reverse()
    return rows


def _write(tmp_path, rows, name="m.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _val(rows):
    return [ManifestRow.model_validate(r) for r in rows if r["split"] == "val"]


def _flat(d):
    return [i for h in (1, 2, 3, 4) for i in d[h]]


def test_default_design_matches_contract():
    assert DEFAULT_DESIGN.expected_population == {1: 1538, 2: 508, 3: 1025, 4: 172}
    assert sum(DEFAULT_DESIGN.sample.values()) == 600
    assert sum(DEFAULT_DESIGN.pilot.values()) == 200


def test_stratum_boundaries():
    def row(nq, ng):
        return ManifestRow.model_validate(
            {
                "source": "lrvs",
                "product_id": "1",
                "split": "val",
                "query": [_sha(i + 1) for i in range(nq)],
                "gallery": [_sha(100 + i) for i in range(ng)],
            }
        )

    assert [stratum_of(row(*a)) for a in [(1, 1), (2, 1), (3, 1), (9, 1), (1, 2), (5, 4)]] == [
        1,
        2,
        3,
        3,
        4,
        4,
    ]
    for bad in [(0, 1), (1, 0), (0, 0)]:
        with pytest.raises(AuditSampleError):
            stratum_of(row(*bad))


def test_strata_sizes_and_nesting():
    sample, pilot, n_h = draw_sample(_val(_make_rows()), DESIGN)
    assert n_h == SIZES
    assert {h: len(v) for h, v in sample.items()} == DESIGN.sample
    for h in sample:
        assert pilot[h] == sample[h][: len(pilot[h])]
        assert len(set(sample[h])) == len(sample[h])
    assert set(_flat(pilot)) <= set(_flat(sample))


def test_unexpected_population_fails():
    rows = _val(_make_rows({1: 41, 2: 20, 3: 30, 4: 10}))
    with pytest.raises(AuditSampleError, match="differ"):
        draw_sample(rows, DESIGN)


def test_sample_larger_than_stratum_fails():
    design = AuditDesign(sample={1: 41, 2: 4, 3: 6, 4: 2}, pilot={1: 1, 2: 1, 3: 1, 4: 1})
    with pytest.raises(AuditSampleError):
        draw_sample(_val(_make_rows()), design)


def test_pilot_larger_than_sample_fails():
    design = AuditDesign(sample={1: 2, 2: 4, 3: 6, 4: 2}, pilot={1: 3, 2: 1, 3: 1, 4: 1})
    with pytest.raises(AuditSampleError):
        draw_sample(_val(_make_rows()), design)


def test_seeds_reproducible_and_order_independent():
    a = draw_sample(_val(_make_rows()), DESIGN)
    b = draw_sample(_val(_make_rows()), DESIGN)
    c = draw_sample(_val(_make_rows(shuffle=True)), DESIGN)
    assert a == b == c


def test_sample_changes_with_manifest_ids():
    a, _, _ = draw_sample(_val(_make_rows()), DESIGN)
    b, _, _ = draw_sample(_val(_make_rows(start=100)), DESIGN)
    assert _flat(a) != _flat(b)


def test_ids_sorted_as_integers_not_strings():
    # 9 sorts after 10 as text; the draw must start from the integer order.
    rows = _val(_make_rows({1: 12, 2: 4, 3: 4, 4: 4}, start=5))
    design = AuditDesign(sample={1: 12, 2: 4, 3: 4, 4: 4}, pilot={1: 1, 2: 1, 3: 1, 4: 1})
    sample, _, _ = draw_sample(rows, design)
    assert sorted(sample[1], key=int) == [str(5 + 7 * i) for i in range(12)]


def test_non_integer_and_duplicate_ids_fail():
    rows = _make_rows()
    rows[0]["product_id"] = "abc"
    with pytest.raises(AuditSampleError, match="integer"):
        draw_sample(_val(rows), DESIGN)
    rows = _make_rows()
    rows[1]["product_id"] = rows[0]["product_id"]
    with pytest.raises(AuditSampleError, match="twice"):
        draw_sample(_val(rows), DESIGN)


def test_largest_remainder():
    assert largest_remainder(40, [95, 31, 63, 11]) == [19, 6, 13, 2]
    assert largest_remainder(80, [190, 63, 127, 20]) == [38, 13, 25, 4]
    for total in range(0, 23):
        out = largest_remainder(total, [7, 3, 11, 1, 0])
        assert sum(out) == total and out[4] == 0
    assert largest_remainder(2, [1, 1, 1]) == [1, 1, 0]  # tie goes to the lower index
    assert largest_remainder(0, [3, 4]) == [0, 0]
    assert largest_remainder(7, [3, 4]) == [3, 4]


def test_largest_remainder_failures():
    with pytest.raises(AuditSampleError):
        largest_remainder(1, [0, 0])
    with pytest.raises(AuditSampleError):
        largest_remainder(8, [3, 4])
    with pytest.raises(AuditSampleError):
        largest_remainder(-1, [3, 4])
    with pytest.raises(AuditSampleError):
        largest_remainder(1, [3, -4])


def test_rereview_sets():
    sample, pilot, _ = draw_sample(_val(_make_rows()), DESIGN)
    first, extra = draw_rereview(sample, pilot, DESIGN)
    assert len(_flat(first)) == 4 and len(_flat(extra)) == 8
    assert set(_flat(first)) <= set(_flat(pilot))
    assert set(_flat(extra)) <= set(_flat(sample)) - set(_flat(pilot))
    assert not set(_flat(first)) & set(_flat(extra))
    assert len(set(_flat(extra))) == 8
    assert draw_rereview(sample, pilot, DESIGN) == (first, extra)


def test_rereview_proportional_to_strata():
    sample, pilot, _ = draw_sample(_val(_make_rows()), DESIGN)
    first, extra = draw_rereview(sample, pilot, DESIGN)
    assert [len(first[h]) for h in (1, 2, 3, 4)] == largest_remainder(4, [4, 2, 3, 1])
    rest = [len(sample[h]) - len(pilot[h]) for h in (1, 2, 3, 4)]
    assert [len(extra[h]) for h in (1, 2, 3, 4)] == largest_remainder(8, rest)


def test_fresh_sample_disjoint_and_proportional():
    rows = _val(_make_rows())
    sample, _, _ = draw_sample(rows, DESIGN)
    risk = [r.product_id for r in rows if r.product_id not in set(_flat(sample))][:5]
    fresh = draw_fresh(rows, _flat(sample), DESIGN, risk_group=risk)
    flat = _flat(fresh)
    assert len(flat) == 20 == len(set(flat))
    assert not set(flat) & set(_flat(sample))
    assert not set(flat) & set(risk)
    pool = {h: SIZES[h] - DESIGN.sample[h] for h in SIZES}
    removed = {h: sum(1 for r in rows if r.product_id in risk and stratum_of(r) == h) for h in SIZES}
    sizes = [pool[h] - removed[h] for h in (1, 2, 3, 4)]
    assert [len(fresh[h]) for h in (1, 2, 3, 4)] == largest_remainder(20, sizes)
    assert draw_fresh(rows, _flat(sample), DESIGN, risk_group=risk) == fresh


def test_fresh_sample_too_big_fails():
    rows = _val(_make_rows())
    sample, _, _ = draw_sample(rows, DESIGN)
    big = AuditDesign(sample=DESIGN.sample, pilot=DESIGN.pilot, fresh=500)
    with pytest.raises(AuditSampleError):
        draw_fresh(rows, _flat(sample), big)


def test_stage_counts_hand_computed():
    rows = _val(_make_rows({1: 1, 2: 1, 3: 1, 4: 1}))
    ids = [r.product_id for r in rows]
    # queries 1+2+4+2 = 9; C = 1*1 + 2*1 + 4*1 + 2*3 = 13; A = 0 + 1 + 6 + (1 + 3) = 11
    assert stage_counts(rows, ids) == {"target": 9, "C_pairs": 13, "C_decoys": 13, "A_pairs": 11}
    assert stage_counts(rows, []) == {"target": 0, "C_pairs": 0, "C_decoys": 0, "A_pairs": 0}
    items = stage_items(rows, ids)
    assert (len(items[1]), len(items[2]), len(items[3])) == (9, 26, 11)


def test_display_ids_opaque():
    rows = _val(_make_rows())
    ids = [r.product_id for r in rows]
    items = stage_items(rows, ids)
    for stage in (1, 2, 3):
        plan = display_plan(items[stage], stage)
        assert len(plan) == len(items[stage])
        assert sorted(map(repr, (i for _, i in plan))) == sorted(map(repr, items[stage]))
        shown = [d for d, _ in plan]
        assert len(set(shown)) == len(shown)
        for d in shown:
            assert not any(pid in d for pid in ids)
        # item order is not product order; display ids sorted do not give display order
        assert [i for _, i in plan] != items[stage]
        assert shown != sorted(shown)
        assert display_plan(items[stage], stage) == plan
    assert display_plan(items[1], 1) != display_plan(items[1], 2)


def test_display_order_is_not_product_order_or_its_reverse():
    rows = _val(_make_rows())
    items = stage_items(rows, [r.product_id for r in rows])[1]
    shown = [i for _, i in display_plan(items, 1)]
    assert shown != items and shown != items[::-1]
    # product ids ascend along the input; along the display they do not
    pos = {i: k for k, i in enumerate(shown)}
    assert [pos[i] for i in items] != sorted(pos[i] for i in items)


def test_display_empty_stage():
    assert display_plan([], 1) == []


def test_run_writes_output_and_hashes(tmp_path):
    path = _write(tmp_path, _make_rows())
    sha = sha256_file(path)
    out = tmp_path / "sub" / "audit.json"
    result = run_audit_sample(path, sha, out, DESIGN)
    data = json.loads(out.read_text())
    assert data["manifest_sha256"] == sha
    assert data["N_h"] == {"1": 40, "2": 20, "3": 30, "4": 10}
    assert data["sets_sha256"]["sample"] == ids_sha256(data["sets"]["sample"])
    assert len(data["sets"]["sample"]) == 20 and len(data["sets"]["pilot"]) == 10
    assert set(data["sets"]["pilot"]) <= set(data["sets"]["sample"])
    assert not set(data["sets"]["fresh"]) & set(data["sets"]["sample"])
    assert data["stage_counts"]["sample"]["target"] == len(data["display"]["target"])
    assert data["stage_counts"]["sample"]["C_pairs"] * 2 == len(data["display"]["C"])
    assert result["sets"] == data["sets"]
    # same inputs, same bytes
    out2 = tmp_path / "audit2.json"
    run_audit_sample(path, sha, out2, DESIGN)
    assert out.read_bytes() == out2.read_bytes()


def test_manifest_sha_mismatch_fails_loudly(tmp_path):
    path = _write(tmp_path, _make_rows())
    out = tmp_path / "audit.json"
    with pytest.raises(AuditSampleError, match="differs from expected"):
        run_audit_sample(path, "0" * 64, out, DESIGN)
    assert not out.exists()
    res = runner.invoke(
        app, ["audit-sample", "--manifest", str(path), "--expect-sha", "0" * 64, "--out", str(out)]
    )
    assert res.exit_code == 1
    assert "differs from expected" in res.output
    assert not out.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    path = _write(tmp_path, _make_rows())
    out = tmp_path / "audit.json"
    out.write_text("keep")
    with pytest.raises(AuditSampleError, match="exists"):
        run_audit_sample(path, sha256_file(path), out, DESIGN)
    assert out.read_text() == "keep"


def test_no_val_products_fails(tmp_path):
    rows = [r for r in _make_rows() if r["split"] != "val"]
    path = _write(tmp_path, rows)
    with pytest.raises(AuditSampleError, match="no val"):
        run_audit_sample(path, sha256_file(path), tmp_path / "o.json", DESIGN)


def test_cli_with_default_design_rejects_wrong_population(tmp_path):
    path = _write(tmp_path, _make_rows())
    out = tmp_path / "audit.json"
    res = runner.invoke(
        app, ["audit-sample", "--manifest", str(path), "--expect-sha", sha256_file(path), "--out", str(out)]
    )
    assert res.exit_code == 1
    assert "differ from expected" in res.output
    assert not out.exists()


def test_train_products_are_not_sampled(tmp_path):
    path = _write(tmp_path, _make_rows())
    data = run_audit_sample(path, sha256_file(path), tmp_path / "o.json", DESIGN)
    assert "5" not in data["sets"]["sample"]
    assert len(load_manifest(path)) == 101
