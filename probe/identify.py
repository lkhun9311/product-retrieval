#!/usr/bin/env python3
"""제품 식별 프로브 — crop 을 VLM+검색 그라운딩에 보내 {브랜드, 모델명, 속성, 링크} 를 받는다.

측정하려는 것: "사진에서 지정한 제품이 무엇인지" 를 클라우드 VLM 이 얼마나 맞히는가.
이 숫자가 온디바이스 경로(D-8)의 **상한**이다. 여기서 안 나오면 온디바이스는 볼 것도 없다.

이 스크립트는 manifest 의 truth/manual_search 필드를 **절대 읽지 않는다.** 채점은 score.py 가 한다.

사용:
    export GEMINI_API_KEY=...
    python3 probe/identify.py --list-models          # 키로 쓸 수 있는 모델 확인
    python3 probe/identify.py --model <model-id>     # 실행
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
PROBE_DIR = Path(__file__).resolve().parent
PHOTOS_DIR = PROBE_DIR / "photos"
RESULTS_DIR = PROBE_DIR / "results"

# 모델에 보내는 지시. 정답을 지어내지 말라는 것이 핵심이다 —
# 근거 URL 없는 제품명은 score.py 에서 실패로 친다.
PROMPT = """이 이미지에 있는 제품 하나를 식별하세요. 웹 검색을 사용하세요.

다음 JSON 형식으로만 답하세요. 다른 말은 쓰지 마세요.

{
  "category": "제품 종류 (예: 크로스백)",
  "attributes": ["관찰한 시각적 특징만. 추론하지 말 것"],
  "visible_text": "제품에 실제로 보이는 글자/로고. 없으면 빈 문자열",
  "brand": "브랜드명. 확신 없으면 null",
  "model": "구체 모델/제품명. 확신 없으면 null",
  "links": ["실제로 검색해서 찾은 판매/제품 페이지 URL"],
  "confidence": "high | medium | low",
  "reasoning": "왜 그렇게 판단했는지 한 문장"
}

중요: 확실하지 않으면 brand/model 을 null 로 두세요.
추측한 제품명보다 null 이 낫습니다. links 는 검색으로 실제 확인한 것만 넣으세요."""


def die(msg: str, code: int = 1) -> None:
    """실패는 조용히 지나가지 않는다. stderr 로 크게 알리고 종료한다."""
    print(f"\n[실패] {msg}", file=sys.stderr)
    sys.exit(code)


def api_key() -> str:
    """환경변수 → 리포 루트 .env 순으로 찾는다.

    .env 는 .gitignore 되어 있다. 키를 대화·커밋·로그 어디에도 남기지 않기 위한 경로다.
    이 함수는 키를 절대 출력하지 않는다.
    """
    if key := (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return key

    # 리포 밖(~/.config)을 먼저 본다 — 실수로 커밋되거나 glob 에 걸릴 여지가 없다.
    candidates = [
        Path.home() / ".config" / "product-retrieval" / ".env",
        PROBE_DIR.parent / ".env",
    ]
    for env_file in candidates:
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                if value := value.strip().strip("'\""):
                    return value

    die("API 키가 없습니다. 찾아본 곳:\n"
        + "".join(f"    {p}\n" for p in candidates)
        + "  1) https://aistudio.google.com/apikey 에서 발급 (카드 등록 불필요)\n"
        "  2) 본인 터미널에서 — 이 대화에 붙여넣지 마세요:\n"
        f"       printf 'GEMINI_API_KEY=%s\\n' '발급받은키' > {candidates[0]}\n"
        f"       chmod 600 {candidates[0]}")


def list_models(key: str) -> None:
    """키로 실제 쓸 수 있는 모델을 서버에 물어본다. 모델 id 를 추측하지 않기 위해서다."""
    r = requests.get(f"{API_ROOT}/models", params={"key": key}, timeout=30)
    if r.status_code != 200:
        die(f"모델 목록 조회 실패 HTTP {r.status_code}\n{r.text[:800]}")
    models = [m for m in r.json().get("models", [])
              if "generateContent" in m.get("supportedGenerationMethods", [])]
    if not models:
        die("generateContent 를 지원하는 모델이 없습니다. 키 권한을 확인하세요.")
    print(f"generateContent 지원 모델 {len(models)}개:\n")
    for m in sorted(models, key=lambda x: x["name"]):
        name = m["name"].removeprefix("models/")
        print(f"  {name:<45} {m.get('displayName','')}")
    print("\n--model <위 이름 중 하나> 로 실행하세요.")


def load_manifest() -> list[dict]:
    path = PHOTOS_DIR / "manifest.json"
    if not path.exists():
        die(f"{path} 가 없습니다. probe/README.md 의 '2. 사진 준비' 를 보세요.")
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        die(f"manifest.json 파싱 실패: {e}")
    if not isinstance(items, list) or not items:
        die("manifest.json 이 비었거나 리스트가 아닙니다.")
    return items


def open_photo(path: Path) -> Image.Image:
    """사진을 열고 EXIF 회전을 실제 픽셀에 적용한다.

    아이폰 사진은 회전 정보가 EXIF 에만 있다. 이걸 적용하지 않으면 모델이 **옆으로 누운 사진**을
    보게 되고, 정확도가 떨어져도 원인을 알 수 없다 — 조용히 나빠지는 유형이라 여기서 막는다.
    """
    if not path.exists():
        die(f"사진이 없습니다: {path}")
    try:
        img = Image.open(path)
    except Exception as e:
        if path.suffix.lower() in (".heic", ".heif"):
            die(f"HEIC 를 읽을 수 없습니다: {path.name}\n"
                "  이 환경엔 pillow-heif 도 변환 도구도 없습니다 [확인: 2026-09-06].\n"
                "  가장 쉬운 해결: 아이폰 설정 > 카메라 > 포맷 > '높은 호환성' 으로 바꾸고 다시 촬영\n"
                "  (이미 찍었다면 맥/아이폰에서 JPEG 로 내보내 옮기세요)")
        die(f"{path.name} 을 열 수 없습니다: {e}")
    return ImageOps.exif_transpose(img).convert("RGB")


def prepare_crop(item: dict, max_side: int = 1024) -> tuple[bytes, tuple[int, int]]:
    """manifest 의 bbox 로 자르고 긴 변을 max_side 로 줄인다.

    bbox 가 없으면 전체 이미지를 쓴다 — 이 경우 "분할이 기여하는가" 는 측정되지 않는다.
    """
    img = open_photo(PHOTOS_DIR / item["file"])
    if bbox := item.get("bbox"):
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            die(f"{item['id']}: bbox 의 너비·높이가 0 이하입니다 — [x, y, 너비, 높이] 형식입니다")
        # PIL 의 crop 은 범위를 벗어나면 **검은 픽셀로 채워서** 돌려준다.
        # 크기만 검사하면 완전히 빈 crop 이 통과한다 — 모델은 검은 사각형을 보고,
        # 정확도가 0 이어도 원인을 알 수 없다. 좌표를 직접 검사한다.
        iw, ih = img.size
        if x < 0 or y < 0 or x + w > iw or y + h > ih:
            die(f"{item['id']}: bbox 가 이미지 밖입니다 (원본 {iw}x{ih}, bbox {bbox}).\n"
                "  python3 probe/prepare.py --check 로 전체를 먼저 검증하세요.")
        img = img.crop((x, y, x + w, y + h))
        if min(img.size) < 8:
            die(f"{item['id']}: crop 이 너무 작습니다 ({img.size})")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), img.size


def extract_json(text: str) -> dict | None:
    """응답에서 첫 JSON 객체를 꺼낸다. 그라운딩을 켜면 JSON 모드를 못 써서 필요하다.

    **문자열 안의 중괄호를 세면 안 된다.** `{"reasoning": "로고에 { 모양"}` 같은 정상 응답이
    파싱 실패로 버려진다 — 모델이 한국어로 이유를 쓰면 실제로 일어난다.
    """
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        start = None
    return None


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def git_describe() -> str:
    """사전등록 태그를 기록한다. 실행이 어느 기준에 묶였는지 나중에 확인하기 위해서다."""
    try:
        r = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                           cwd=PROBE_DIR.parent, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def identify(key: str, model: str, jpeg: bytes, grounding: bool) -> dict:
    """한 장을 모델에 보낸다. 지연·토큰·원문을 전부 남긴다."""
    body: dict = {
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(jpeg).decode()}},
            ]
        }],
        "generationConfig": {"temperature": 0.0},
    }
    if grounding:
        body["tools"] = [{"google_search": {}}]

    t0 = time.monotonic()
    try:
        r = requests.post(f"{API_ROOT}/models/{model}:generateContent",
                          params={"key": key}, json=body, timeout=180)
    except requests.RequestException as e:
        # 네트워크 실패도 결과다. 예외로 죽으면 앞서 성공한 건까지 전부 잃는다.
        return {"ok": False, "http": None, "error": f"네트워크 오류: {type(e).__name__}: {e}",
                "latency_s": round(time.monotonic() - t0, 2)}
    elapsed = time.monotonic() - t0

    if r.status_code != 200:
        # 그라운딩 툴 이름이 모델마다 다를 수 있다. 원문을 그대로 보여준다.
        return {"ok": False, "http": r.status_code, "error": r.text[:1500],
                "latency_s": round(elapsed, 2)}

    data = r.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError):
        return {"ok": False, "http": 200, "error": f"응답 구조 예상 밖: {json.dumps(data)[:1500]}",
                "latency_s": round(elapsed, 2)}

    usage = data.get("usageMetadata", {})
    grounded = data["candidates"][0].get("groundingMetadata")
    return {
        "ok": True,
        "latency_s": round(elapsed, 2),
        "parsed": extract_json(text),
        "raw_text": text,
        "grounding_used": bool(grounded),
        "grounding_queries": (grounded or {}).get("webSearchQueries", []),
        "tokens": {"in": usage.get("promptTokenCount"),
                   "out": usage.get("candidatesTokenCount"),
                   "total": usage.get("totalTokenCount")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="제품 식별 프로브")
    ap.add_argument("--model", help="모델 id (--list-models 로 확인)")
    ap.add_argument("--list-models", action="store_true", help="쓸 수 있는 모델 나열")
    ap.add_argument("--no-grounding", action="store_true",
                    help="웹검색 없이 실행 — 그라운딩 기여도를 재려면 두 번 돌린다")
    ap.add_argument("--only", help="특정 id 하나만 실행")
    args = ap.parse_args()

    key = api_key()
    if args.list_models:
        list_models(key)
        return
    if not args.model:
        die("--model 이 필요합니다. --list-models 로 먼저 확인하세요.")

    items = load_manifest()
    if args.only:
        items = [i for i in items if i["id"] == args.only]
        if not items:
            die(f"id '{args.only}' 가 manifest 에 없습니다.")

    grounding = not args.no_grounding
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"run-{stamp}.json"

    # 무엇을 무엇에 대고 쟀는지 고정한다. 이게 없으면 나중에 표본·프롬프트가 바뀌어도
    # 같은 실행처럼 보인다 — 사전등록 태그가 입력에 묶이지 않으면 무의미하다.
    manifest_raw = (PHOTOS_DIR / "manifest.json").read_bytes()
    provenance = {
        "prompt_sha": sha256_of(PROMPT.encode()),
        "manifest_sha": sha256_of(manifest_raw),
        "subset": args.only or "all",
        "prereg_tag": git_describe(),
    }

    def save() -> None:
        out.write_text(json.dumps({
            "model": args.model, "grounding": grounding, "timestamp": stamp,
            "n": len(results), "provenance": provenance, "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"모델 {args.model} · 그라운딩 {'ON' if grounding else 'OFF'} · {len(items)}장")
    print(f"사전등록 {provenance['prereg_tag']} · manifest {provenance['manifest_sha']}\n")
    results, failures = [], 0
    for n, item in enumerate(items, 1):
        jpeg, size = prepare_crop(item)
        print(f"[{n}/{len(items)}] {item['id']} ({size[0]}x{size[1]}, {len(jpeg)//1024}KB) ... ",
              end="", flush=True)
        res = identify(key, args.model, jpeg, grounding)
        # truth 는 결과에 넣지 않는다. 채점은 score.py 가 별도로 한다.
        results.append({"id": item["id"], "file": item["file"],
                        "difficulty": item.get("difficulty"),
                        "photo_sha": sha256_of((PHOTOS_DIR / item["file"]).read_bytes()),
                        "bbox": item.get("bbox"),
                        "crop_px": list(size), "crop_bytes": len(jpeg), **res})
        save()   # 매 건마다 저장한다. 중간에 죽어도 앞선 결과를 잃지 않는다
        if res["ok"]:
            p = res.get("parsed") or {}
            print(f"{res['latency_s']}s · {p.get('brand') or '브랜드?'} / "
                  f"{p.get('model') or '모델?'} · 링크 {len(p.get('links') or [])}개")
        else:
            failures += 1
            print(f"실패 (HTTP {res['http']})")
            print(f"    {res['error'][:300]}", file=sys.stderr)

    ok = len(results) - failures
    print(f"\n성공 {ok}/{len(results)} · 저장 {out}")
    if failures:
        print(f"실패 {failures}건 — 위 stderr 메시지를 확인하세요.", file=sys.stderr)
    print(f"\n다음: python3 probe/score.py {out.name}")
    if failures == len(results):
        sys.exit(2)  # 전부 실패면 종료코드로도 알린다


if __name__ == "__main__":
    main()
