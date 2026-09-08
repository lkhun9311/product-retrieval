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


def forced_miss(res: dict, grounding: bool) -> str | None:
    """사람이 고를 수 없는 경우를 돌려준다. 없으면 None.

    경고만 띄우면 눌러서 지나갈 수 있다 — 실제로 호출 실패도 '동일 제품' 으로 찍혔다.
    판정 자체를 막는다.
    """
    if not res.get("ok"):
        return "호출이 실패했다. 응답이 없으므로 판정 대상이 아니다"
    if (res.get("parsed") or {}) == {} or res.get("parsed") is None:
        return "JSON 파싱 실패. 판정할 내용이 없다"
    if grounding and not ((res.get("parsed") or {}).get("links")):
        return "근거 URL 이 0개다. 제품명을 댔더라도 확인 불가 = 근거 부족"
    return None


def judge(results: list[dict], truth: dict, prior: dict, grounding: bool) -> dict:
    rub = RUBRIC[grounding]
    verdicts = dict(prior)
    print(f"\n채점 기준: {rub['title']}\n  {rub['prompt']}")
    for res in results:
        if res["id"] in verdicts:
            continue
        if reason := forced_miss(res, grounding):
            verdicts[res["id"]] = "miss"
            print(f"\n  {res['id']}: 자동 {rub['labels']['miss']} — {reason}")
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

    # 분모는 **실행 전체**다. 판정 안 한 건을 빼면 1건만 판정하고 100% 가 나온다.
    n = len(results)
    unjudged = n - len(judged)

    print("\n" + "=" * 68)
    print(f"  {rub['title']}")
    print(f"  {run['model']} · 그라운딩 {'ON' if grounding else 'OFF'}")
    if prov := run.get("provenance"):
        print(f"  사전등록 {prov.get('prereg_tag')} · manifest {prov.get('manifest_sha')} "
              f"· 대상 {prov.get('subset')}")
    print("=" * 68)
    if (run.get("provenance") or {}).get("smoke"):
        print("  ⚠️  스모크 실행이다. 배관 확인용이며 **판정·집계 대상이 아니다.**")
        print("      D-11 판정에 쓰지 마라. 평가는 별도 표본으로 다시 실행한다.\n")
    print(f"  실행 {n}건 · 판정 {len(judged)}건 · 미판정 {unjudged}건\n")
    for k, lab in rub["labels"].items():
        c = counts[k]
        print(f"  {lab:<10} {c:>3}건  {c/n*100:>5.1f}%  {'█' * round(c / n * 30)}")
    if unjudged:
        print(f"  {'미판정':<10} {unjudged:>3}건  {unjudged/n*100:>5.1f}%  "
              f"{'░' * round(unjudged / n * 30)}")

    print(f"\n  {rub['metric']}  {counts['hit']/n*100:.1f}%   ← 핵심 지표 (분모 = 실행 전체 {n})")
    if unjudged:
        print(f"  ⚠️  미판정 {unjudged}건이 있다. 이 수치는 **하한**이며 D-11 판정에 쓸 수 없다.")
        print("      전부 판정한 뒤 다시 집계하세요.")

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

    # 기준선 대비 — 이게 없으면 "쓸모 있는가" 를 답할 수 없다.
    # **측정하지 않은 기준선을 패배로 세지 않는다.** success 가 null 이면 비교에서 뺀다 —
    # 안 그러면 Lens 를 안 잰 것이 "사람 0/20" 이 되어 우리 쪽에 유리하게 편향된다.
    paired = [(r, (truth.get(r["id"]) or {}).get("manual_search") or {}) for r in judged]
    measured = [(r, m) for r, m in paired if m.get("success") is not None]
    missing = len(paired) - len(measured)

    print(f"\n  기준선(Google Lens) 대비")
    if measured:
        base_ok = sum(1 for _, m in measured if m["success"])
        secs = [m["seconds"] for _, m in measured if m.get("seconds")]
        auto = sum(1 for r, _ in measured if verdicts[r["id"]] == "hit")
        print(f"    Lens   {base_ok}/{len(measured)} 성공"
              + (f" · 평균 {statistics.mean(secs):.0f}초" if secs else ""))
        print(f"    모델   {auto}/{len(measured)} 성공")
        # 짝별 승패 — n=20 에서 비율 차이보다 이쪽이 읽을 만하다
        win = sum(1 for r, m in measured if verdicts[r["id"]] == "hit" and not m["success"])
        lose = sum(1 for r, m in measured if verdicts[r["id"]] != "hit" and m["success"])
        tie = len(measured) - win - lose
        print(f"    짝별   승 {win} · 패 {lose} · 무 {tie}")
        if grounding and auto < base_ok:
            print("    → Lens 보다 못하다. 이 상태로는 이식할 이유가 없다 (D-11 중단 조건).")
    else:
        print("    측정된 기준선 0건 — **비교 불가.** D-11 의 '손검색보다 낮으면 중단' 을 판정할 수 없다.")
    if missing:
        print(f"    ⚠️  기준선 미측정 {missing}건은 비교에서 제외했다(패배로 세지 않음).")

    print("\n" + "=" * 68)


def selftest() -> int:
    """채점기가 **실패를 실패로 찍는지** 확인한다.

    "안 걸림" 과 "통과" 를 구분 못 하는 검증은 없는 것보다 나쁘다 — 있다고 믿게 만들기 때문이다.
    그래서 일부러 나쁜 입력을 넣어 막히는지 본다. 여기서 하나라도 실패하면 채점 결과는 무효다.
    """
    cases = [
        ("호출 실패는 판정 불가",
         {"id": "x", "ok": False, "http": 500}, True, True),
        ("Q-B: 링크 0개는 판정 불가",
         {"id": "x", "ok": True, "parsed": {"brand": "Nike", "links": []}}, True, True),
        ("Q-B: 링크 있으면 판정 가능",
         {"id": "x", "ok": True, "parsed": {"brand": "Nike", "links": ["http://a"]}}, True, False),
        ("Q-A: 링크 없어도 판정 가능",
         {"id": "x", "ok": True, "parsed": {"category": "가방", "links": []}}, False, False),
        ("파싱 실패는 판정 불가",
         {"id": "x", "ok": True, "parsed": None}, False, True),
    ]
    failed = 0
    print("채점기 자기검사")
    for name, res, grounding, want_blocked in cases:
        blocked = forced_miss(res, grounding) is not None
        ok = blocked == want_blocked
        failed += not ok
        print(f"  {'OK  ' if ok else '실패'} {name}")

    # 분모 검사: 20건 중 1건만 판정하면 핵심 지표가 100% 여선 안 된다
    import io as _io
    import contextlib
    run = {"model": "t", "grounding": True,
           "results": [{"id": f"p{i:02d}", "file": "f", "ok": True, "difficulty": "easy",
                        "latency_s": 1.0, "tokens": {"total": 1},
                        "parsed": {"links": ["http://a"]}} for i in range(20)]}
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(run, run["results"], {}, {"p00": "hit"})
    out = buf.getvalue()
    denom_ok = "5.0%" in out and "미판정 19건" in out
    failed += not denom_ok
    print(f"  {'OK  ' if denom_ok else '실패'} 1/20 판정 시 핵심 지표가 5%(하한)로 나오는가")

    # 기준선 미측정이 패배로 집계되지 않는가
    truth = {f"p{i:02d}": {"manual_search": {"seconds": None, "success": None}} for i in range(20)}
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(run, run["results"], truth, {f"p{i:02d}": "hit" for i in range(20)})
    out = buf.getvalue()
    base_ok = "측정된 기준선 0건" in out and "비교에서 제외" in out
    failed += not base_ok
    print(f"  {'OK  ' if base_ok else '실패'} 미측정 기준선을 패배로 세지 않는가")

    print(f"\n{'통과' if not failed else f'{failed}건 실패 — 채점 결과를 신뢰하지 마세요'}")
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="프로브 채점")
    ap.add_argument("run", nargs="?", help="results/ 안의 run-*.json 파일명")
    ap.add_argument("--report", action="store_true", help="판정 입력 없이 집계만")
    ap.add_argument("--selftest", action="store_true", help="채점기가 실패를 잡는지 검사")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.run:
        die("run 파일명이 필요합니다 (또는 --selftest)")

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
