#!/usr/bin/env python3
"""프로브 채점 — 모델 출력을 사람이 판정하고 집계한다.

판정은 3분류다 (codex 권고):
    1 동일 제품    정확히 그 제품. 근거 URL 로 확인됨
    2 비슷할 뿐    같은 종류지만 다른 제품
    3 근거 부족    답을 못 했거나, 제품명은 댔는데 근거 URL 이 없다

"근거 URL 없이 댄 제품명" 은 통과가 아니라 실패다. 지어낸 답이 통과하면 이 측정은 무의미해진다.

사용:
    python3 probe/score.py run-20260905-201500.json
    python3 probe/score.py run-20260905-201500.json --report   # 이미 매긴 판정만 집계
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROBE_DIR / "results"
PHOTOS_DIR = PROBE_DIR / "photos"

# 채점 기준은 실행 종류에 따라 다르다.
#
# 그라운딩 ON  = Q-B "실제 제품 페이지를 찾았는가" — 근거 URL 이 판정의 일부다
# 그라운딩 OFF = Q-A "이 텍스트로 사람이 검색하면 찾겠는가" — 링크가 없는 게 정상이다
#
# 이걸 구분하지 않으면 무그라운딩 실행이 전부 '근거 부족' 으로 찍혀 1단계를 잴 수 없다.
VERDICTS = {"1": "hit", "2": "partial", "3": "miss"}

RUBRIC = {
    True: {  # 그라운딩 ON
        "title": "Q-B 식별 — 실제 제품을 찾았는가",
        "labels": {"hit": "동일 제품", "partial": "비슷할 뿐", "miss": "근거 부족"},
        "prompt": "1 동일(근거 URL 확인됨) / 2 비슷할 뿐 / 3 근거부족",
        "metric": "정확 식별률",
    },
    False: {  # 그라운딩 OFF
        "title": "Q-A 서술 — 이 텍스트로 검색하면 찾겠는가",
        "labels": {"hit": "충분", "partial": "부분적", "miss": "쓸모없음"},
        "prompt": "1 충분(브랜드+모델 또는 결정적 OCR) / 2 부분적(종류·속성만) / 3 쓸모없음(틀렸거나 너무 일반적)",
        "metric": "검색 가능 서술률",
    },
}


def die(msg: str) -> None:
    print(f"\n[실패] {msg}", file=sys.stderr)
    sys.exit(1)


def load_truth() -> dict[str, dict]:
    path = PHOTOS_DIR / "manifest.json"
    if not path.exists():
        die(f"{path} 가 없습니다.")
    return {i["id"]: i for i in json.loads(path.read_text(encoding="utf-8"))}


def show(res: dict, truth: dict, grounding: bool) -> None:
    p = res.get("parsed") or {}
    print("\n" + "=" * 68)
    print(f"  {res['id']}   난이도 {res.get('difficulty', '?')}   {res['file']}")
    print("=" * 68)
    if not res.get("ok"):
        print(f"  호출 실패: HTTP {res.get('http')}")
        print(f"  {(res.get('error') or '')[:400]}")
        return
    print(f"  종류    {p.get('category') or '-'}")
    print(f"  특징    {', '.join(p.get('attributes') or []) or '-'}")
    print(f"  보이는 글자  {p.get('visible_text') or '-'}")
    print(f"  브랜드  {p.get('brand') or 'null'}")
    print(f"  모델    {p.get('model') or 'null'}")
    print(f"  확신    {p.get('confidence') or '-'}")
    print(f"  근거    {p.get('reasoning') or '-'}")
    links = p.get("links") or []
    if grounding:
        print(f"  링크 {len(links)}개")
        for u in links[:5]:
            print(f"      {u}")
        if res.get("grounding_queries"):
            print(f"  검색어  {', '.join(res['grounding_queries'][:4])}")
    print(f"  지연 {res.get('latency_s')}s · 토큰 {(res.get('tokens') or {}).get('total')}")

    # 지어낸 답 경고 — 검색을 켰는데 제품명만 대고 근거가 없는 경우.
    # 무그라운딩 실행에서는 링크가 없는 게 정상이므로 경고하지 않는다.
    if grounding and (p.get("brand") or p.get("model")) and not links:
        print("  ⚠️  제품명을 댔지만 근거 URL 이 없다 → 3 이 기본값이다")

    t = truth.get(res["id"], {}).get("truth") or {}
    print(f"\n  [정답] {t.get('brand') or '?'} / {t.get('model') or '?'}")
    if t.get("url"):
        print(f"         {t['url']}")


def judge(results: list[dict], truth: dict, prior: dict, grounding: bool) -> dict:
    rub = RUBRIC[grounding]
    verdicts = dict(prior)
    print(f"\n채점 기준: {rub['title']}\n  {rub['prompt']}")
    for res in results:
        if res["id"] in verdicts:
            continue
        show(res, truth, grounding)
        while True:
            a = input(f"\n  판정 [{rub['prompt']} / s 건너뜀 / q 저장후종료]: ").strip().lower()
            if a == "q":
                return verdicts
            if a == "s":
                break
            if a in VERDICTS:
                verdicts[res["id"]] = VERDICTS[a]
                break
            print("  1, 2, 3, s, q 중 하나를 입력하세요.")
    return verdicts


def report(run: dict, results: list[dict], truth: dict, verdicts: dict) -> None:
    judged = [r for r in results if r["id"] in verdicts]
    if not judged:
        die("판정된 항목이 없습니다.")

    grounding = bool(run.get("grounding"))
    rub = RUBRIC[grounding]
    counts = {v: 0 for v in rub["labels"]}
    for r in judged:
        counts[verdicts[r["id"]]] += 1
    n = len(judged)

    print("\n" + "=" * 68)
    print(f"  {rub['title']}")
    print(f"  {run['model']} · 그라운딩 {'ON' if grounding else 'OFF'}")
    print("=" * 68)
    print(f"  판정 {n}/{len(results)}건\n")
    for k, lab in rub["labels"].items():
        c = counts[k]
        bar = "█" * round(c / n * 30)
        print(f"  {lab:<10} {c:>3}건  {c/n*100:>5.1f}%  {bar}")
    print(f"\n  {rub['metric']}  {counts['hit']/n*100:.1f}%   ← 핵심 지표")

    # 난이도별
    for d in ("easy", "hard"):
        sub = [r for r in judged if r.get("difficulty") == d]
        if sub:
            hit = sum(verdicts[r["id"]] == "hit" for r in sub)
            print(f"    {d:<6} {hit}/{len(sub)}  ({hit/len(sub)*100:.0f}%)")

    ok = [r for r in judged if r.get("ok")]
    if ok:
        lat = sorted(r["latency_s"] for r in ok)
        tok = [(r.get("tokens") or {}).get("total") or 0 for r in ok]
        grounded = sum(1 for r in ok if r.get("grounding_used"))
        print(f"\n  지연   중앙 {statistics.median(lat):.1f}s · 최대 {lat[-1]:.1f}s")
        print(f"  토큰   평균 {statistics.mean(tok):.0f}/건")
        print(f"  검색   {grounded}/{len(ok)}건에서 실제 그라운딩 발생")

    # 손검색 대비 — 이게 없으면 "쓸모 있는가" 를 답할 수 없다
    manual = [(r, truth[r["id"]]["manual_search"])
              for r in judged
              if r["id"] in truth and truth[r["id"]].get("manual_search")]
    if manual:
        ms = sum(1 for _, m in manual if m.get("success"))
        secs = [m["seconds"] for _, m in manual if m.get("seconds")]
        auto = sum(1 for r, _ in manual if verdicts[r["id"]] == "hit")
        print(f"\n  손검색 대비 ({len(manual)}건)")
        print(f"    사람   {ms}/{len(manual)} 성공" + (f" · 평균 {statistics.mean(secs):.0f}초" if secs else ""))
        print(f"    모델   {auto}/{len(manual)} 성공")
        if grounding and auto < ms:
            print("    → 사람보다 못하다. 이 상태로는 이식할 이유가 없다.")
    else:
        print("\n  손검색 비교 없음 — manifest 의 manual_search 를 채우면 '쓸모 있는가' 를 답할 수 있다.")

    print("\n" + "=" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description="프로브 채점")
    ap.add_argument("run", help="results/ 안의 run-*.json 파일명")
    ap.add_argument("--report", action="store_true", help="판정 입력 없이 집계만")
    args = ap.parse_args()

    run_path = RESULTS_DIR / Path(args.run).name
    if not run_path.exists():
        die(f"{run_path} 가 없습니다.")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    results = run["results"]
    truth = load_truth()

    score_path = RESULTS_DIR / f"score-{run_path.stem.removeprefix('run-')}.json"
    prior = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else {}

    verdicts = prior if args.report else judge(results, truth, prior, bool(run.get("grounding")))
    if not args.report and verdicts != prior:
        score_path.write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n판정 저장 {score_path}")

    report(run, results, truth, verdicts)


if __name__ == "__main__":
    main()
