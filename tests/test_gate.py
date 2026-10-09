"""Deployment gate, contract c6-deploy-gate-v1 (docs/contracts/c6-deploy-gate.md).

Most cases use fixtures where every product behaves the same way (one query per product, the truth at a
fixed rank), so the paired bootstrap delta is the same in every replicate and the interval collapses to the
hand-computed point: delta = R@K(candidate) - R@K(current), ci_low = ci_high = delta. One case checks the
bootstrap interval of a mixed fixture against a loop-based recomputation, and the comparison operators are
checked at their boundaries with the bootstrap replaced by fixed intervals.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from product_retrieval.cli import app
from product_retrieval.eval import gate
from product_retrieval.eval.gate import GateError, evaluate_gate, manifest_sha256
from product_retrieval.eval.retrieval import QueryResult

REPO = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()
MODEL = {
    "n_pos": 5,
    "n_neg": 95,
    "label_version": "lv-test",
    "reranker_version": "rv-test",
    "label_policy": "stratified",
}


def make(n_products: int, pos, *, per_product: int = 1, n_ids: int = 10) -> list[QueryResult]:
    """One QueryResult per query; ``pos(i)`` is the 0-based rank position of product i's truth."""
    out = []
    for i in range(n_products):
        for j in range(per_product):
            qid, truth = f"q{i:03d}-{j}", f"p{i:03d}"
            ids = [f"x{qid}-{r}" for r in range(n_ids)]
            ids[pos(i, j) if callable(pos) else pos] = truth
            out.append(QueryResult(qid, truth, tuple(ids)))
    return out


def sha_of(results: list[QueryResult]) -> str:
    text = "".join(sorted(f"{r.query_id}\t{r.truth_product_id}\n" for r in results))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_gate(current, candidate, model=MODEL):
    return evaluate_gate(current, candidate, model, manifest_sha=sha_of(current))


# ---- manifest hash ------------------------------------------------------------------------------


def test_manifest_sha256_hand_written_example():
    # `printf 'q1\tpA\nq2\tpB\n' | sha256sum`, entered out of order on purpose
    results = [QueryResult("q2", "pB", ("a",)), QueryResult("q1", "pA", ("a",))]
    assert manifest_sha256(results) == "c533fd02eefea254de88243be678e260ce54a1302dfa20c8e66a59f8fcafde8e"
    assert manifest_sha256(results[::-1]) == manifest_sha256(results)


def test_manifest_sha256_sorts_whole_lines_and_encodes_utf8():
    # `printf 'a\tX\na1\tX\n' | sha256sum`: "a<TAB>" sorts before "a1" because TAB < "1"
    results = [QueryResult("a1", "X", ("a",)), QueryResult("a", "X", ("a",))]
    assert manifest_sha256(results) == "2f5396d09108a17a9d01d8d1fef7e4f9e8f2a5e7073fdd7eddc3e3557e8c9af0"
    # `printf '상품1\t제품A\n상품2\t제품B\n' | sha256sum`
    korean = [QueryResult("상품2", "제품B", ("a",)), QueryResult("상품1", "제품A", ("a",))]
    assert manifest_sha256(korean) == "2abaad97b427c613e6d865da4ee0b3e5ecfa80c329622f95c26667c409dbbdc4"


def test_manifest_sha256_sees_every_part_of_a_line():
    base = manifest_sha256([QueryResult("q1", "p1", ("a",))])
    assert base == "c7d054f9642f60ef1a26a6d8579d69c798dc06492d40dc5ba23b067279a71f2a"  # `printf 'q1\tp1\n'`
    assert manifest_sha256([QueryResult("q1", "p2", ("a",))]) != base
    assert manifest_sha256([QueryResult("q2", "p1", ("a",))]) != base
    assert manifest_sha256([QueryResult("q1", "p1", ("a",)), QueryResult("q2", "p1", ("a",))]) != base


def test_contract_constants_match_the_contract_text():
    text = (REPO / "docs" / "contracts" / "c6-deploy-gate.md").read_text(encoding="utf-8")
    assert gate.MANIFEST_SHA256 in text and gate.CONTRACT_ID in text
    assert (gate.G1_MARGIN, gate.G2_MARGIN, gate.G3_MAX_SHARE, gate.P_REF) == (-0.005, -0.03, 0.10, 0.05)
    assert gate.G3_MAX_SHARE == 2 * gate.P_REF
    assert (gate.KS, gate.BOOTSTRAP_B, gate.BOOTSTRAP_SEED) == ((1, 5), 1000, 0)
    assert gate.REQUIRED_LABEL_POLICY == "stratified" and gate.MIN_UNIQUE_IDS == 5


def test_no_argument_changes_a_threshold():
    assert list(inspect.signature(evaluate_gate).parameters) == [
        "current",
        "candidate",
        "candidate_model",
        "manifest_sha",
    ]


# ---- decisions: each check alone, all pass, several together -------------------------------------


def test_all_pass_identical_rankings():
    cur = make(10, 2)
    out = run_gate(cur, make(10, 2))
    assert out["decision"] == "pass" and out["reasons"] == [] and out["contract"] == "c6-deploy-gate-v1"
    g1, g2, g3 = (out["checks"][c] for c in ("G1", "G2", "G3"))
    assert (g1["delta"], g1["ci_low"], g1["ci_high"], g1["failed"]) == (0.0, 0.0, 0.0, False)
    assert (g2["delta"], g2["ci_low"], g2["ci_high"], g2["failed"]) == (0.0, 0.0, 0.0, False)
    assert g1["threshold"] == -0.005 and g2["threshold"] == -0.03 and g3["threshold"] == 0.10
    assert g3["value"] == pytest.approx(0.05) and g3["failed"] is False
    assert g1["value"] == g1["ci_low"] and g2["value"] == g2["ci_low"]


def test_all_pass_when_the_candidate_is_better():
    # current: truth at rank 7 (R@1 = R@5 = 0); candidate: rank 1 (R@1 = R@5 = 1)
    out = run_gate(make(10, 6), make(10, 0))
    assert out["decision"] == "pass"
    assert out["checks"]["G1"]["delta"] == 1.0 and out["checks"]["G2"]["delta"] == 1.0


def test_g1_fails_alone():
    # current: rank 3 (R@1 = 0, R@5 = 1); candidate: rank 7 (R@1 = 0, R@5 = 0)
    out = run_gate(make(10, 2), make(10, 6))
    assert out["decision"] == "block" and out["reasons"] == ["G1"]
    g1, g2, g3 = (out["checks"][c] for c in ("G1", "G2", "G3"))
    assert (g1["delta"], g1["ci_low"], g1["ci_high"], g1["failed"]) == (-1.0, -1.0, -1.0, True)
    assert (g2["delta"], g2["failed"]) == (0.0, False)
    assert g3["failed"] is False


def test_g2_fails_alone():
    # current: rank 1 (R@1 = R@5 = 1); candidate: rank 4 (R@1 = 0, R@5 = 1)
    out = run_gate(make(10, 0), make(10, 3))
    assert out["decision"] == "block" and out["reasons"] == ["G2"]
    g1, g2 = out["checks"]["G1"], out["checks"]["G2"]
    assert (g2["delta"], g2["ci_low"], g2["ci_high"], g2["failed"]) == (-1.0, -1.0, -1.0, True)
    assert (g1["delta"], g1["failed"]) == (0.0, False)
    assert out["checks"]["G3"]["failed"] is False


def test_g3_fails_alone_and_the_boundary_is_inclusive():
    cur = make(10, 2)
    over = run_gate(cur, make(10, 2), {**MODEL, "n_pos": 11, "n_neg": 89})
    assert over["decision"] == "block" and over["reasons"] == ["G3"]
    assert over["checks"]["G3"]["value"] == pytest.approx(0.11) and over["checks"]["G3"]["failed"] is True
    assert not over["checks"]["G1"]["failed"] and not over["checks"]["G2"]["failed"]
    at = run_gate(cur, make(10, 2), {**MODEL, "n_pos": 10, "n_neg": 90})  # 10 / 100 is exactly 0.10
    assert at["checks"]["G3"]["value"] == 0.1 and at["checks"]["G3"]["failed"] is False
    assert at["decision"] == "pass"


def test_g3_with_zero_positives_passes():
    out = run_gate(make(10, 2), make(10, 2), {**MODEL, "n_pos": 0, "n_neg": 100})
    assert out["decision"] == "pass" and out["checks"]["G3"]["value"] == 0.0


def test_several_fail_together_with_reasons_in_id_order():
    cur = make(10, 0)  # R@1 = R@5 = 1
    bad = make(10, 6)  # R@1 = R@5 = 0
    out = run_gate(cur, bad, {**MODEL, "n_pos": 30, "n_neg": 70})
    assert out["decision"] == "block" and out["reasons"] == ["G1", "G2", "G3"]
    assert [out["checks"][c]["failed"] for c in ("G1", "G2", "G3")] == [True, True, True]
    two = run_gate(cur, bad)
    assert two["reasons"] == ["G1", "G2"]
    # G1 and G3 without G2: R@1 stays 0 on both sides
    mixed = run_gate(make(10, 2), make(10, 6), {**MODEL, "n_pos": 30, "n_neg": 70})
    assert mixed["reasons"] == ["G1", "G3"]
    # G2 and G3 without G1
    mixed = run_gate(make(10, 0), make(10, 3), {**MODEL, "n_pos": 30, "n_neg": 70})
    assert mixed["reasons"] == ["G2", "G3"]


def test_point_estimate_inside_the_margin_is_still_blocked_by_the_lower_bound():
    # 200 products; only product 0 loses its R@5 hit: delta = -1/200 = -0.005, exactly the margin
    cur = make(200, 2)
    cand = make(200, lambda i, j: 6 if i == 0 else 2)
    out = run_gate(cur, cand)
    g1 = out["checks"]["G1"]
    assert g1["delta"] == pytest.approx(-0.005, abs=1e-12)
    assert g1["delta"] >= gate.G1_MARGIN - 1e-12  # the point estimate is not below the margin
    assert g1["ci_low"] < gate.G1_MARGIN and g1["failed"] is True
    assert out["decision"] == "block" and out["reasons"] == ["G1"]


def test_bootstrap_interval_matches_a_loop_based_recomputation():
    # 30 products x 2 queries; current and candidate hit R@5 / R@1 on different, uneven subsets
    n = 30

    def cur_pos(i, j):
        return [0, 3, 8, 2][(i + j) % 4]

    def cand_pos(i, j):
        return [3, 0, 8, 7][(i * 2 + j) % 4]

    cur, cand = make(n, cur_pos, per_product=2), make(n, cand_pos, per_product=2)

    def macro(pos_fn, k):
        per_product = [np.mean([1.0 if pos_fn(i, j) < k else 0.0 for j in range(2)]) for i in range(n)]
        return np.array(per_product)

    idx = np.random.default_rng(0).integers(0, n, size=(1000, n))
    out = run_gate(cur, cand)
    for check_id, k in (("G1", 5), ("G2", 1)):
        a, b = macro(cur_pos, k), macro(cand_pos, k)
        reps = np.array([(b[row] - a[row]).mean() for row in idx])
        lo, hi = np.percentile(reps, [2.5, 97.5])
        c = out["checks"][check_id]
        assert c["delta"] == pytest.approx(b.mean() - a.mean(), abs=1e-12)
        assert c["ci_low"] == pytest.approx(lo, abs=1e-12) and c["ci_high"] == pytest.approx(hi, abs=1e-12)


def test_comparison_operators_at_the_boundary(monkeypatch):
    def cell(low):
        return {"delta": low + 0.01, "ci": (low, low + 0.02), "p_le_0": 0.5}

    def fake(low1, low5):
        def paired(results_a, results_b, ks, **kw):
            assert tuple(ks) == (1, 5) and kw == {"b": 1000, "seed": 0}
            return {"macro": {1: cell(low1), 5: cell(low5)}, "n_queries": 10, "n_products": 10}

        return paired

    cur = make(10, 2)
    sha = sha_of(cur)

    def decide(low1, low5):
        monkeypatch.setattr(gate, "paired_bootstrap", fake(low1, low5))
        return evaluate_gate(cur, cur, MODEL, manifest_sha=sha)

    assert decide(-0.03, -0.005)["decision"] == "pass"  # lower bound equal to the margin is allowed
    assert decide(-0.03, -0.0050001)["reasons"] == ["G1"]
    assert decide(-0.0300001, -0.005)["reasons"] == ["G2"]
    assert decide(-0.0300001, -0.0050001)["reasons"] == ["G1", "G2"]
    with pytest.raises(GateError) as err:
        decide(float("nan"), 0.0)
    assert err.value.kind == "non_finite"


def test_inputs_block_carries_counts_versions_policy_and_bootstrap_settings():
    cur = make(10, 2)
    out = run_gate(cur, make(10, 2))
    assert out["inputs"] == {
        "manifest_sha256": sha_of(cur),
        "n_queries": 10,
        "n_products": 10,
        "n_pos": 5,
        "n_neg": 95,
        "label_version": "lv-test",
        "reranker_version": "rv-test",
        "label_policy": "stratified",
        "B": 1000,
        "seed": 0,
    }


# ---- failure modes (section 4): errors, never pass, never block ----------------------------------


def expect_error(kind, current, candidate, model=MODEL, **kw):
    with pytest.raises(GateError) as err:
        evaluate_gate(current, candidate, model, **kw)
    assert err.value.kind == kind, str(err.value)
    return str(err.value)


def test_manifest_mismatch_in_either_file():
    cur = make(10, 2)
    sha = sha_of(cur)
    changed_truth = [QueryResult(cur[0].query_id, "other", cur[0].ranked_product_ids), *cur[1:]]
    missing = cur[:-1]
    extra = [*cur, QueryResult("q-extra", "p-extra", tuple(f"y{i}" for i in range(6)))]
    renamed = [QueryResult("renamed", cur[0].truth_product_id, cur[0].ranked_product_ids), *cur[1:]]
    for bad in (changed_truth, missing, extra, renamed):
        assert "candidate" in expect_error("manifest_mismatch", cur, bad, manifest_sha=sha)
        assert "current" in expect_error("manifest_mismatch", bad, cur, manifest_sha=sha)
    expect_error("manifest_mismatch", cur, cur, manifest_sha="0" * 64)
    expect_error("manifest_mismatch", cur, cur)  # the default is the frozen v1 set, which a fixture is not
    expect_error("manifest_mismatch", [], [], manifest_sha=sha)


def test_duplicate_query_id_in_either_file():
    cur = make(10, 2)
    dup = [*cur, cur[0]]
    assert "candidate" in expect_error("duplicate_query_id", cur, dup, manifest_sha=sha_of(cur))
    assert "current" in expect_error("duplicate_query_id", dup, cur, manifest_sha=sha_of(cur))


def test_a_query_with_fewer_than_five_unique_ids_is_an_error():
    cur = make(10, 2)
    sha = sha_of(cur)
    short = [QueryResult(cur[0].query_id, cur[0].truth_product_id, cur[0].ranked_product_ids[:4]), *cur[1:]]
    assert "candidate" in expect_error("too_few_candidates", cur, short, manifest_sha=sha)
    assert "current" in expect_error("too_few_candidates", short, cur, manifest_sha=sha)
    five = [QueryResult(r.query_id, r.truth_product_id, r.ranked_product_ids[:5]) for r in make(10, 2)]
    assert run_gate(five, five)["decision"] == "pass"  # exactly 5 ids is enough
    with pytest.raises(ValueError):  # duplicate ids inside one ranking cannot even be constructed
        QueryResult("q", "p", ("a", "a", "b", "c", "d"))


@pytest.mark.parametrize(
    "patch",
    [
        {"n_pos": None},
        {"n_neg": None},
        {"n_pos": "5"},
        {"n_pos": 5.0},
        {"n_neg": 95.5},
        {"n_pos": True},
        {"n_neg": False},
        {"n_pos": -1},
        {"n_neg": -1},
        {"n_pos": 0, "n_neg": 0},
        {"n_pos": [5]},
    ],
)
def test_invalid_label_counts_are_errors(patch):
    cur = make(10, 2)
    expect_error("invalid_label_counts", cur, make(10, 2), {**MODEL, **patch}, manifest_sha=sha_of(cur))


@pytest.mark.parametrize("field", ["n_pos", "n_neg"])
def test_missing_label_counts_are_errors(field):
    cur = make(10, 2)
    model = {k: v for k, v in MODEL.items() if k != field}
    expect_error("invalid_label_counts", cur, make(10, 2), model, manifest_sha=sha_of(cur))


@pytest.mark.parametrize("policy", ["random", "Stratified", "", None, 1])
def test_label_policy_must_be_stratified(policy):
    cur = make(10, 2)
    sha = sha_of(cur)
    expect_error(
        "invalid_label_policy", cur, make(10, 2), {**MODEL, "label_policy": policy}, manifest_sha=sha
    )
    model = {k: v for k, v in MODEL.items() if k != "label_policy"}
    expect_error("invalid_label_policy", cur, make(10, 2), model, manifest_sha=sha)


@pytest.mark.parametrize("field", ["label_version", "reranker_version"])
def test_missing_versions_are_errors(field):
    cur = make(10, 2)
    sha = sha_of(cur)
    expect_error(
        "invalid_model", cur, make(10, 2), {k: v for k, v in MODEL.items() if k != field}, manifest_sha=sha
    )
    expect_error("invalid_model", cur, make(10, 2), {**MODEL, field: ""}, manifest_sha=sha)
    expect_error("invalid_model", cur, make(10, 2), {**MODEL, field: None}, manifest_sha=sha)


def test_a_model_that_is_not_an_object_is_an_error():
    cur = make(10, 2)
    expect_error("invalid_model", cur, make(10, 2), [MODEL], manifest_sha=sha_of(cur))


def test_an_error_is_not_confused_with_a_block_even_when_a_check_would_fail():
    cur = make(10, 0)
    bad_model = {**MODEL, "label_policy": "random", "n_pos": 90, "n_neg": 10}
    expect_error("invalid_label_policy", cur, make(10, 6), bad_model, manifest_sha=sha_of(cur))


# ---- CLI: exit codes, output, inputs block ---------------------------------------------------------


def write_rankings(path: Path, results: list[QueryResult]) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": r.query_id,
                    "truth_product_id": r.truth_product_id,
                    "top_k_product_ids": list(r.ranked_product_ids),
                }
            )
            + "\n"
            for r in results
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def files(tmp_path, monkeypatch):
    cur = make(10, 2)
    # the CLI uses the frozen v1 manifest; the fixture has its own, so bind that for the exit-code tests
    monkeypatch.setattr(
        "product_retrieval.cli.evaluate_gate", functools.partial(evaluate_gate, manifest_sha=sha_of(cur))
    )

    def build(candidate_pos=2, model=MODEL, current_pos=2):
        paths = {
            "current": write_rankings(tmp_path / "cur.jsonl", make(10, current_pos)),
            "candidate": write_rankings(tmp_path / "cand.jsonl", make(10, candidate_pos)),
            "model": tmp_path / "model.json",
            "out": tmp_path / "out" / "gate.json",
        }
        paths["model"].write_text(json.dumps(model), encoding="utf-8")
        return paths

    return build


def invoke(p, *extra, out=True):
    args = [
        "gate",
        "--current",
        str(p["current"]),
        "--candidate",
        str(p["candidate"]),
        "--candidate-model",
        str(p["model"]),
    ]
    if out:
        args += ["--out", str(p["out"])]
    return RUNNER.invoke(app, [*args, *extra])


def test_cli_exit_0_on_pass_with_inputs_block(files):
    p = files()
    res = invoke(p)
    assert res.exit_code == 0, res.output
    body = json.loads(res.stdout)
    assert body == json.loads(p["out"].read_text(encoding="utf-8"))
    assert body["decision"] == "pass" and body["reasons"] == [] and body["contract"] == "c6-deploy-gate-v1"
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()  # noqa: E731
    inputs = body["inputs"]
    for name in ("current", "candidate"):
        assert inputs[name] == {"path": str(p[name]), "sha256": sha(p[name])}
    assert inputs["candidate_model"] == {"path": str(p["model"]), "sha256": sha(p["model"])}
    assert (inputs["n_pos"], inputs["n_neg"], inputs["B"], inputs["seed"]) == (5, 95, 1000, 0)
    assert (inputs["label_version"], inputs["reranker_version"], inputs["label_policy"]) == (
        "lv-test",
        "rv-test",
        "stratified",
    )
    assert inputs["manifest_sha256"] == sha_of(make(10, 2)) and inputs["n_queries"] == 10


def test_cli_exit_2_on_block_names_the_failing_checks(files):
    p = files(candidate_pos=6)
    res = invoke(p)
    assert res.exit_code == 2, res.output
    body = json.loads(res.stdout)
    assert body["decision"] == "block" and body["reasons"] == ["G1"]
    assert json.loads(p["out"].read_text(encoding="utf-8"))["decision"] == "block"
    assert invoke(files(model={**MODEL, "n_pos": 20, "n_neg": 80}), out=False).exit_code == 2


def test_cli_exit_1_on_the_frozen_manifest_mismatch_without_the_test_binding(tmp_path):
    cur = write_rankings(tmp_path / "cur.jsonl", make(10, 2))
    model = tmp_path / "m.json"
    model.write_text(json.dumps(MODEL), encoding="utf-8")
    res = RUNNER.invoke(
        app, ["gate", "--current", str(cur), "--candidate", str(cur), "--candidate-model", str(model)]
    )
    assert res.exit_code == 1
    body = json.loads(res.stdout)
    assert body["decision"] == "error" and body["error"]["kind"] == "manifest_mismatch"
    assert gate.MANIFEST_SHA256 in body["error"]["message"]


def test_cli_exit_1_on_input_errors_and_the_output_says_error(files):
    cases = {
        "invalid_label_policy": {**MODEL, "label_policy": "random"},
        "invalid_label_counts": {**MODEL, "n_pos": True},
        "invalid_model": {k: v for k, v in MODEL.items() if k != "reranker_version"},
    }
    for kind, model in cases.items():
        res = invoke(files(model=model))
        assert res.exit_code == 1, res.output
        body = json.loads(res.stdout)
        assert body["decision"] == "error" and body["error"]["kind"] == kind and "checks" not in body


def test_cli_error_overwrites_a_stale_out_file(files):
    p = files()
    assert invoke(p).exit_code == 0
    assert json.loads(p["out"].read_text(encoding="utf-8"))["decision"] == "pass"
    p["model"].write_text(json.dumps({**MODEL, "label_policy": "random"}), encoding="utf-8")
    assert invoke(p).exit_code == 1
    assert json.loads(p["out"].read_text(encoding="utf-8"))["decision"] == "error"


def test_cli_exit_1_on_unreadable_inputs(files, tmp_path):
    p = files()
    p["model"].write_text("{not json", encoding="utf-8")
    res = invoke(p)
    assert res.exit_code == 1 and json.loads(res.stdout)["error"]["kind"] == "invalid_model"
    p["model"].write_text("[1, 2]", encoding="utf-8")
    assert invoke(p).exit_code == 1
    p = files()
    p["candidate"].write_text("", encoding="utf-8")  # no rows
    assert invoke(p).exit_code == 1
    p = files()
    p["current"].write_text('{"query_id": "q"}\n', encoding="utf-8")  # missing fields
    assert invoke(p).exit_code == 1
    p = files()
    p["current"].write_text("[1, 2]\n", encoding="utf-8")  # not an object
    assert invoke(p).exit_code == 1
    p = files()
    p["model"].unlink()
    res = invoke(p)
    assert res.exit_code == 1 and json.loads(res.stdout)["error"]["kind"] == "input_not_found"


def test_cli_exit_1_on_duplicate_query_ids_and_duplicate_product_ids(files):
    p = files()
    rows = p["candidate"].read_text(encoding="utf-8").splitlines()
    p["candidate"].write_text("\n".join([*rows, rows[0]]) + "\n", encoding="utf-8")
    assert invoke(p).exit_code == 1
    p = files()
    row = json.loads(p["candidate"].read_text(encoding="utf-8").splitlines()[0])
    row["top_k_product_ids"] = ["a", "a", "b", "c", "d", "e"]
    p["candidate"].write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert invoke(p).exit_code == 1


def test_cli_missing_option_is_exit_1_not_the_parsers_exit_2(files):
    p = files()
    res = RUNNER.invoke(app, ["gate", "--current", str(p["current"])])
    assert res.exit_code == 1
    assert json.loads(res.stdout)["error"]["kind"] == "missing_option"
    assert RUNNER.invoke(app, ["gate"]).exit_code == 1


def test_cli_has_no_threshold_option():
    out = RUNNER.invoke(app, ["gate", "--help"]).output
    for word in ("margin", "threshold", "p-ref", "p_ref", "--b ", "--seed", "--policy"):
        assert word not in out
    assert math.isclose(gate.G3_MAX_SHARE, 0.1)
