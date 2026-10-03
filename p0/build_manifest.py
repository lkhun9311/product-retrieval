"""P0 — 수집 결과로 상품 단위 매니페스트와 학습/검증/시험 분할을 만든다.

입력: fetch_images.py 의 JSONL 결과들(url, product_id, kind, status, sha256).
출력: 상품 한 줄 JSONL — query(연출·착용 사진)와 gallery(상품 단독 사진)의 sha256 목록과 split.

규칙
- split 은 product_id 의 해시로 정한다(실행할 때마다 같다): 0–79 train, 80–89 val, 90–99 test.
- query 와 gallery 가 모두 1장 이상인 상품만 남긴다.
- 같은 이미지(sha256)가 서로 다른 상품에 쓰이면 두 상품 모두 뺀다(분할 간 누수 방지).
- 마지막에 매니페스트 자체의 sha256 을 출력한다 — 결과 보고에 이 값을 남긴다.
"""

import argparse
import collections
import hashlib
import json
import sys

QUERY_KINDS = {"complex", "review"}
GALLERY_KINDS = {"simple", "catalog"}


def split_of(product_id):
    bucket = int(hashlib.sha256(str(product_id).encode()).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="데이터 출처 이름(예: lrvs, amazon2023)")
    ap.add_argument("out_jsonl")
    ap.add_argument("results", nargs="+")
    args = ap.parse_args()

    products = collections.defaultdict(lambda: {"query": set(), "gallery": set()})
    owners = collections.defaultdict(set)
    seen = failed = 0
    for path in args.results:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            seen += 1
            if r.get("status") != "ok":
                failed += 1
                continue
            side = "query" if r["kind"] in QUERY_KINDS else "gallery" if r["kind"] in GALLERY_KINDS else None
            if side is None:
                sys.exit(f"unknown kind {r['kind']!r} in {path}")
            products[r["product_id"]][side].add(r["sha256"])
            owners[r["sha256"]].add(r["product_id"])

    shared = {pid for sha, pids in owners.items() if len(pids) > 1 for pid in pids}
    kept = {pid: p for pid, p in products.items() if p["query"] and p["gallery"] and pid not in shared}

    counts = collections.Counter()
    with open(args.out_jsonl, "w") as out:
        for pid in sorted(kept, key=str):
            p = kept[pid]
            split = split_of(pid)
            counts[split] += 1
            out.write(
                json.dumps(
                    {
                        "source": args.source,
                        "product_id": pid,
                        "split": split,
                        "query": sorted(p["query"]),
                        "gallery": sorted(p["gallery"]),
                    }
                )
                + "\n"
            )

    digest = hashlib.sha256(open(args.out_jsonl, "rb").read()).hexdigest()
    print(f"rows read {seen}, not ok {failed}")
    print(
        f"products seen {len(products)}, sharing an image with another product {len(shared)}, "
        f"missing a side {sum(1 for p in products.values() if not (p['query'] and p['gallery']))}"
    )
    print(f"kept {len(kept)}: " + ", ".join(f"{k} {counts[k]}" for k in ("train", "val", "test")))
    print(f"manifest sha256 {digest}")


if __name__ == "__main__":
    main()
