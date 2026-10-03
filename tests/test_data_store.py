import io

import pytest
from PIL import Image

from product_retrieval.core.ids import sha256_bytes
from product_retrieval.data.store import ImageIntegrityError, ImageStore, ShaFormatError


def _png_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    img = Image.new("RGB", (4, 4), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_image(root, source: str, data: bytes) -> str:
    sha = sha256_bytes(data)
    image_dir = root / source / "images" / sha[:2]
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{sha}.img").write_bytes(data)
    return sha


def test_path_follows_content_addressed_layout(tmp_path):
    store = ImageStore(tmp_path, "lrvs")
    sha = "a" * 64
    assert store.path(sha) == tmp_path / "lrvs" / "images" / "aa" / f"{sha}.img"


def test_path_rejects_non_hex_sha(tmp_path):
    store = ImageStore(tmp_path, "lrvs")
    with pytest.raises(ShaFormatError):
        store.path("not-hex" + "0" * 57)


def test_path_rejects_short_sha(tmp_path):
    store = ImageStore(tmp_path, "lrvs")
    with pytest.raises(ShaFormatError):
        store.path("a" * 63)


def test_exists_false_when_missing(tmp_path):
    store = ImageStore(tmp_path, "lrvs")
    assert store.exists("a" * 64) is False


def test_read_bytes_round_trip(tmp_path):
    data = _png_bytes()
    sha = _write_image(tmp_path, "lrvs", data)
    store = ImageStore(tmp_path, "lrvs")
    assert store.exists(sha) is True
    assert store.read_bytes(sha) == data


def test_read_bytes_raises_on_hash_mismatch(tmp_path):
    data = _png_bytes()
    sha = _write_image(tmp_path, "lrvs", data)
    store = ImageStore(tmp_path, "lrvs")
    # Corrupt the file in place so its bytes no longer match the sha-named path.
    store.path(sha).write_bytes(data + b"corrupt")
    with pytest.raises(ImageIntegrityError):
        store.read_bytes(sha)


def test_open_image_returns_rgb_pil_image(tmp_path):
    data = _png_bytes()
    sha = _write_image(tmp_path, "lrvs", data)
    store = ImageStore(tmp_path, "lrvs")
    img = store.open_image(sha)
    assert img.mode == "RGB"
    assert img.size == (4, 4)
