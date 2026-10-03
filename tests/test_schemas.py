from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from product_retrieval.core.schemas import (
    BBox,
    Bundle,
    Candidate,
    CropSpec,
    EvalReport,
    FeedbackEvent,
    Gate,
    Label,
    Product,
    Query,
    SearchResult,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


# --- Product -----------------------------------------------------------------


def test_product_valid():
    p = Product(
        product_id="p1",
        source="lrvs",
        split="train",
        gallery_shas=[SHA_A, SHA_B],
        brand="acme",
        title="widget",
        attrs={"color": "red"},
    )
    assert p.gallery_shas == [SHA_A, SHA_B]
    with pytest.raises(ValidationError):
        p.product_id = "p2"  # frozen


def test_product_bad_source():
    with pytest.raises(ValidationError):
        Product(product_id="p1", source="ebay", split="train")


def test_product_bad_gallery_sha():
    with pytest.raises(ValidationError):
        Product(product_id="p1", source="lrvs", split="train", gallery_shas=["not-a-sha"])


# --- Query ---------------------------------------------------------------


def test_query_valid():
    q = Query(query_id="q1", image_sha=SHA_A, truth_product_id="p1", source="capture")
    assert q.image_sha == SHA_A


def test_query_bad_image_sha_uppercase():
    with pytest.raises(ValidationError):
        Query(query_id="q1", image_sha="A" * 64, truth_product_id="p1", source="capture")


def test_query_bad_image_sha_length():
    with pytest.raises(ValidationError):
        Query(query_id="q1", image_sha="a" * 63, truth_product_id="p1", source="capture")


# --- CropSpec --------------------------------------------------------------


def test_cropspec_full_valid():
    spec = CropSpec(kind="full")
    assert spec.box is None
    assert spec.mask_sha is None


def test_cropspec_box_valid():
    spec = CropSpec(kind="box", box=BBox(x1=0, y1=0, x2=10, y2=10))
    assert spec.box is not None


def test_cropspec_box_missing_box_is_invalid():
    with pytest.raises(ValidationError):
        CropSpec(kind="box")


def test_cropspec_full_with_box_is_invalid():
    with pytest.raises(ValidationError):
        CropSpec(kind="full", box=BBox(x1=0, y1=0, x2=10, y2=10))


def test_cropspec_mask_valid():
    spec = CropSpec(kind="mask", mask_sha=SHA_A, model="sam21")
    assert spec.mask_sha == SHA_A


def test_cropspec_mask_missing_mask_sha_is_invalid():
    with pytest.raises(ValidationError):
        CropSpec(kind="mask")


def test_cropspec_box_with_mask_sha_is_invalid():
    with pytest.raises(ValidationError):
        CropSpec(kind="box", box=BBox(x1=0, y1=0, x2=10, y2=10), mask_sha=SHA_A)


def test_cropspec_bad_mask_sha_format():
    with pytest.raises(ValidationError):
        CropSpec(kind="mask", mask_sha="zz" * 32)


# --- Candidate / SearchResult ----------------------------------------------


def test_candidate_valid():
    c = Candidate(product_id="p1", image_sha=SHA_A, score_retrieval=0.9, rank=1)
    assert c.score_rerank is None


def test_candidate_bad_image_sha():
    with pytest.raises(ValidationError):
        Candidate(product_id="p1", image_sha="short", score_retrieval=0.9, rank=1)


def test_search_result_valid():
    sr = SearchResult(
        list_id="l1",
        bundle_id="b1",
        decision="same",
        candidates=[Candidate(product_id="p1", image_sha=SHA_A, score_retrieval=0.9, rank=1)],
    )
    assert sr.decision == "same"
    assert sr.latency_ms.crop is None


def test_search_result_bad_decision():
    with pytest.raises(ValidationError):
        SearchResult(list_id="l1", bundle_id="b1", decision="maybe", candidates=[])


# --- FeedbackEvent / Label ---------------------------------------------------


def test_feedback_event_valid():
    ev = FeedbackEvent(
        event_id="e1",
        ts=datetime.now(UTC),
        session_id="s1",
        list_id="l1",
        query_id="q1",
        product_id="p1",
        action="match",
        actor="human",
    )
    assert ev.actor == "human"


def test_feedback_event_bad_action():
    with pytest.raises(ValidationError):
        FeedbackEvent(
            event_id="e1",
            ts=datetime.now(UTC),
            session_id="s1",
            list_id="l1",
            query_id="q1",
            product_id="p1",
            action="click",
            actor="human",
        )


def test_feedback_event_bad_actor():
    with pytest.raises(ValidationError):
        FeedbackEvent(
            event_id="e1",
            ts=datetime.now(UTC),
            session_id="s1",
            list_id="l1",
            query_id="q1",
            product_id="p1",
            action="match",
            actor="robot",
        )


def test_label_valid():
    label = Label(query_id="q1", product_id="p1", kind="pos", weight=1.0, label_version="v1")
    assert label.origin_events == []


def test_label_bad_kind():
    with pytest.raises(ValidationError):
        Label(query_id="q1", product_id="p1", kind="maybe", weight=1.0, label_version="v1")


# --- Bundle / EvalReport ------------------------------------------------------


def test_bundle_valid():
    b = Bundle(
        bundle_id="bid1",
        embed_model_id="m1",
        crop_policy="full",
        index_id="idx1",
        created_at=datetime.now(UTC),
    )
    assert b.reranker_version is None


def test_eval_report_valid():
    report = EvalReport(
        bundle_id="bid1",
        dataset_manifest_sha=SHA_A,
        metrics={"R@1": 0.5},
        ci={"R@1": [0.4, 0.6]},
        gate=Gate.model_validate({"pass": True, "reasons": []}),
    )
    assert report.gate.passed is True


def test_eval_report_bad_dataset_manifest_sha():
    with pytest.raises(ValidationError):
        EvalReport(
            bundle_id="bid1",
            dataset_manifest_sha="not-a-sha",
            metrics={},
            ci={},
            gate=Gate.model_validate({"pass": False, "reasons": ["worse"]}),
        )
