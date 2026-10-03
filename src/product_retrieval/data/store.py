"""Content-addressed image storage (D20 section 5).

Layout: ``<root>/<source>/images/<sha[:2]>/<sha>.img`` where ``sha`` is the
lowercase hex sha256 digest of the raw image bytes (JPEG/PNG/WEBP; the file
extension is literally ``.img``).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from PIL import Image

from product_retrieval.core.ids import sha256_bytes

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ShaFormatError(ValueError):
    """Raised when a value is not a well-formed 64-character lowercase hex sha256 digest."""


class ImageIntegrityError(ValueError):
    """Raised when the bytes read from the store do not hash to the requested sha."""


def validate_sha(sha: str) -> str:
    """Return ``sha`` unchanged if it is a well-formed sha256 hex digest, else raise."""
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        raise ShaFormatError(f"expected a 64-character lowercase hex sha256 digest, got {sha!r}")
    return sha


class ImageStore:
    """Read-only, content-addressed access to images under ``root/source/images/``."""

    def __init__(self, root: Path | str, source: str) -> None:
        self.root = Path(root)
        self.source = source

    def path(self, sha: str) -> Path:
        """Return the on-disk path for ``sha`` (does not check that it exists)."""
        sha = validate_sha(sha)
        return self.root / self.source / "images" / sha[:2] / f"{sha}.img"

    def exists(self, sha: str) -> bool:
        """Return whether a file for ``sha`` is present on disk."""
        return self.path(sha).is_file()

    def read_bytes(self, sha: str) -> bytes:
        """Read the raw bytes for ``sha`` and verify they hash back to it.

        Raises ``ImageIntegrityError`` if the file's content does not match ``sha``.
        """
        path = self.path(sha)
        data = path.read_bytes()
        actual = sha256_bytes(data)
        if actual != sha:
            raise ImageIntegrityError(f"{path}: expected sha256 {sha}, got {actual}")
        return data

    def open_image(self, sha: str) -> Image.Image:
        """Read, verify, and decode ``sha`` into an RGB ``PIL.Image``."""
        data = self.read_bytes(sha)
        return Image.open(io.BytesIO(data)).convert("RGB")
