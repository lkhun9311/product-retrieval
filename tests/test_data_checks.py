from product_retrieval.data.checks import check_manifest
from product_retrieval.data.manifest import ManifestRow
from product_retrieval.data.store import ImageStore

SHA_Q1 = "1" * 64
SHA_Q2 = "2" * 64
SHA_G1 = "3" * 64
SHA_G2 = "4" * 64
SHA_MISSING = "5" * 64


def _put(root, source: str, sha: str) -> None:
    image_dir = root / source / "images" / sha[:2]
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{sha}.img").write_bytes(b"fake-bytes")


def _row(product_id, split="test", query=(), gallery=()) -> ManifestRow:
    return ManifestRow.model_validate(
        {
            "source": "lrvs",
            "product_id": product_id,
            "split": split,
            "query": list(query),
            "gallery": list(gallery),
        }
    )


def test_ok_when_all_images_present_and_no_leakage(tmp_path):
    for sha in (SHA_Q1, SHA_G1):
        _put(tmp_path, "lrvs", sha)
    rows = [_row("p1", query=(SHA_Q1,), gallery=(SHA_G1,))]
    store = ImageStore(tmp_path, "lrvs")

    report = check_manifest(rows, store)

    assert report.ok is True
    assert report.missing_image_shas == []
    assert report.cross_split_products == []
    assert report.counts_by_split["test"].products == 1
    assert report.counts_by_split["test"].queries == 1
    assert report.counts_by_split["test"].gallery_images == 1


def test_catches_missing_image(tmp_path):
    _put(tmp_path, "lrvs", SHA_G1)  # query image SHA_Q1 is never written
    rows = [_row("p1", query=(SHA_Q1,), gallery=(SHA_G1,))]
    store = ImageStore(tmp_path, "lrvs")

    report = check_manifest(rows, store)

    assert report.ok is False
    assert report.missing_image_shas == [SHA_Q1]


def test_catches_shared_sha_across_products(tmp_path):
    for sha in (SHA_Q1, SHA_G1, SHA_G2):
        _put(tmp_path, "lrvs", sha)
    rows = [
        _row("p1", query=(SHA_Q1,), gallery=(SHA_G1,)),
        _row("p2", query=(SHA_Q2,), gallery=(SHA_G1,)),  # SHA_G1 shared with p1
    ]
    for sha in (SHA_Q1, SHA_Q2, SHA_G1):
        _put(tmp_path, "lrvs", sha)
    store = ImageStore(tmp_path, "lrvs")

    report = check_manifest(rows, store)

    assert report.duplicate_shas == [SHA_G1]
    # a shared sha alone (no cross-split product) does not fail the gate
    assert report.cross_split_products == []
    assert report.ok is True


def test_catches_cross_split_product(tmp_path):
    for sha in (SHA_Q1, SHA_Q2, SHA_G1, SHA_G2):
        _put(tmp_path, "lrvs", sha)
    rows = [
        _row("p1", split="train", query=(SHA_Q1,), gallery=(SHA_G1,)),
        _row("p1", split="test", query=(SHA_Q2,), gallery=(SHA_G2,)),
    ]
    store = ImageStore(tmp_path, "lrvs")

    report = check_manifest(rows, store)

    assert report.cross_split_products == ["p1"]
    assert report.ok is False
