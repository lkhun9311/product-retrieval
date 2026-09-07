#!/usr/bin/env python3
"""사진을 스캔해 manifest 뼈대를 만들고, 돌리기 전에 검증한다.

manifest 를 손으로 쓰면 20건에서 반드시 틀린다 — 파일명 오타, bbox 범위 밖, truth 누락.
그런 오류는 실행 중에야 드러나거나, 더 나쁘게는 **조용히 잘못된 측정**이 된다.

사용:
    python3 probe/prepare.py            # 스캔 → manifest 뼈대 생성/갱신 (기존 항목은 보존)
    python3 probe/prepare.py --check    # 검증만. 문제 있으면 exit 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

PROBE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = PROBE_DIR / "photos"
MANIFEST = PHOTOS_DIR / "manifest.json"

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
HEIC_EXTS = {".heic", ".heif"}


def scan() -> tuple[list[Path], list[Path]]:
    """읽을 수 있는 사진과 못 읽는 사진을 나눠 돌려준다."""
    if not PHOTOS_DIR.exists():
        print(f"[실패] {PHOTOS_DIR} 가 없습니다.", file=sys.stderr)
        sys.exit(1)
    ok, bad = [], []
    for p in sorted(PHOTOS_DIR.iterdir()):
        if not p.is_file() or p.name.startswith(".") or p.name == "manifest.json":
            continue
        if p.suffix.lower() in HEIC_EXTS:
            bad.append(p)
            continue
        if p.suffix.lower() not in EXTS:
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            ok.append(p)
        except Exception:
            bad.append(p)
    return ok, bad


def dimensions(p: Path) -> tuple[int, int]:
    """EXIF 회전을 적용한 뒤의 크기. bbox 는 이 좌표계로 적어야 한다."""
    with Image.open(p) as im:
        return ImageOps.exif_transpose(im).size


def build(paths: list[Path]) -> int:
    existing = {}
    if MANIFEST.exists():
        try:
            existing = {i["file"]: i for i in json.loads(MANIFEST.read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[실패] 기존 manifest.json 을 읽을 수 없습니다: {e}\n"
                  "  손으로 고치거나 지운 뒤 다시 실행하세요. 덮어쓰지 않았습니다.", file=sys.stderr)
            return 1

    # 신규 id 는 **기존 최대값 다음**에서 이어간다.
    # 정렬 순번을 쓰면 사전순으로 앞서는 파일이 추가될 때 기존 id 와 충돌한다
    # (b.jpg=p01 뒤 a.jpg 추가 → 둘 다 p01). 채점은 id 사전이라 서로 다른 사진이
    # 같은 판정을 공유하게 된다.
    used = set()
    for it in existing.values():
        if isinstance(it.get("id"), str):
            used.add(it["id"])
    next_n = 1

    def new_id() -> str:
        nonlocal next_n
        while f"p{next_n:02d}" in used:
            next_n += 1
        tag = f"p{next_n:02d}"
        used.add(tag)
        return tag

    items, added = [], 0
    for p in paths:
        if p.name in existing:
            items.append(existing[p.name])
            continue
        w, h = dimensions(p)
        items.append({
            "id": new_id(),
            "file": p.name,
            "_size": [w, h],            # 참고용. bbox 를 적을 때 범위 확인에 쓴다
            "bbox": None,               # [x, y, 너비, 높이] 또는 null (전체 이미지)
            "difficulty": "easy",       # easy | hard
            "truth": {"brand": None, "model": None, "url": None},
            "manual_search": {"seconds": None, "success": None},
        })
        added += 1

    MANIFEST.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest.json — 전체 {len(items)}건 (신규 {added}, 기존 보존 {len(items)-added})")
    if added:
        print("\n채워야 할 것: difficulty · truth.brand · truth.model · manual_search")
        print("bbox 는 생략 가능(null 이면 전체 이미지)")
    return 0


def check() -> int:
    if not MANIFEST.exists():
        print(f"[실패] {MANIFEST} 가 없습니다. 먼저 인자 없이 실행하세요.", file=sys.stderr)
        return 1
    items = json.loads(MANIFEST.read_text(encoding="utf-8"))

    problems: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for it in items:
        tag = it.get("id", "?")
        if tag in seen_ids:
            problems.append(f"{tag}: id 중복")
        seen_ids.add(tag)

        src = PHOTOS_DIR / it.get("file", "")
        if not src.exists():
            problems.append(f"{tag}: 파일 없음 — {it.get('file')}")
            continue

        if bbox := it.get("bbox"):
            if not (isinstance(bbox, list) and len(bbox) == 4):
                problems.append(f"{tag}: bbox 는 [x, y, 너비, 높이] 4개여야 함")
            else:
                x, y, w, h = bbox
                iw, ih = dimensions(src)
                if w <= 0 or h <= 0:
                    problems.append(f"{tag}: bbox 너비·높이가 0 이하 ({w}x{h})")
                elif x < 0 or y < 0 or x + w > iw or y + h > ih:
                    problems.append(f"{tag}: bbox 가 이미지 밖 (원본 {iw}x{ih}, bbox {bbox})")

        if it.get("difficulty") not in ("easy", "hard"):
            problems.append(f"{tag}: difficulty 는 easy 또는 hard")

        truth = it.get("truth") or {}
        if not truth.get("brand") and not truth.get("model"):
            problems.append(f"{tag}: truth 가 비었음 — 정답을 모르면 표본에서 빼세요")

        ms = it.get("manual_search") or {}
        if ms.get("success") is None:
            warnings.append(f"{tag}: manual_search 없음 — '사람보다 나은가' 를 못 잰다")

    easy = sum(1 for i in items if i.get("difficulty") == "easy")
    hard = sum(1 for i in items if i.get("difficulty") == "hard")

    print(f"검증 {len(items)}건 — easy {easy} · hard {hard}")
    if easy < 10 or hard < 10:
        warnings.append(f"권장은 easy 10 + hard 10 (현재 {easy}/{hard}) — 표본이 작으면 결론도 약하다")

    for w in warnings:
        print(f"  [주의] {w}")
    for p in problems:
        print(f"  [오류] {p}", file=sys.stderr)

    if problems:
        print(f"\n오류 {len(problems)}건. 고치기 전엔 실행하지 마세요.", file=sys.stderr)
        return 1
    print("\n통과. 실행 가능:")
    print("  python3 probe/identify.py --model gemini-3.1-flash-lite --no-grounding")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="프로브 사진 준비·검증")
    ap.add_argument("--check", action="store_true", help="검증만 수행")
    args = ap.parse_args()

    ok, bad = scan()
    if bad:
        print(f"[주의] 읽을 수 없는 파일 {len(bad)}개:", file=sys.stderr)
        for p in bad[:10]:
            print(f"    {p.name}", file=sys.stderr)
        if any(p.suffix.lower() in HEIC_EXTS for p in bad):
            print("  HEIC 입니다. 이 환경엔 변환 도구가 없습니다 [확인: 2026-09-06].\n"
                  "  아이폰 설정 > 카메라 > 포맷 > '높은 호환성' 으로 바꾸고 다시 촬영하는 게 가장 쉽습니다.",
                  file=sys.stderr)
        print(file=sys.stderr)

    if args.check:
        sys.exit(check())

    if not ok:
        print(f"[실패] {PHOTOS_DIR} 에 읽을 수 있는 사진이 없습니다.", file=sys.stderr)
        sys.exit(1)
    print(f"읽을 수 있는 사진 {len(ok)}장")
    rc = build(ok)
    if rc == 0:
        print("\n다음: manifest 를 채운 뒤  python3 probe/prepare.py --check")
    sys.exit(rc)


if __name__ == "__main__":
    main()
