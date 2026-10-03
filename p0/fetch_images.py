"""P0 — 공개 데이터셋의 이미지 URL이 실제로 받아지는지 이미지 단위로 잰다.

입력은 duckdb 로 뽑은 CSV(url, product_id, kind)다. 호스트마다 요청 간격을 두고,
robots.txt 가 막은 경로는 요청하지 않으며, 결과를 URL 단위로 JSONL 에 남긴다.
--save 를 주면 받은 바이트를 sha256 이름으로 저장한다(리포 밖 경로만 허용).

표준 라이브러리만 쓴다. 이 머신에는 pip 가 없다.
"""

import argparse
import collections
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import urllib.robotparser
from urllib.parse import urlsplit

USER_AGENT = "product-retrieval-research/0.1 (+personal research; contact via github lkhun9311)"
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"GIF8")


class HostGate:
    """호스트별 최소 요청 간격과 robots.txt 판정을 한곳에서 관리한다."""

    def __init__(self, interval):
        self.interval = interval
        self.last = collections.defaultdict(float)
        self.locks = collections.defaultdict(threading.Lock)
        self.robots = {}
        self.robots_lock = threading.Lock()

    def allowed(self, url):
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        with self.robots_lock:
            rp = self.robots.get(base)
            if rp is None:
                rp = urllib.robotparser.RobotFileParser(base + "/robots.txt")
                try:
                    rp.read()
                except Exception as exc:  # robots.txt 를 못 읽으면 기록하고 허용 쪽으로 둔다
                    print(f"robots.txt unreadable for {base}: {exc!r}", file=sys.stderr)
                    rp.allow_all = True
                self.robots[base] = rp
        return rp.can_fetch(USER_AGENT, url)

    def wait(self, host):
        with self.locks[host]:
            gap = self.last[host] + self.interval - time.monotonic()
            if gap > 0:
                time.sleep(gap)
            self.last[host] = time.monotonic()


def fetch(row, gate, save_dir, timeout):
    url = row["url"]
    host = urlsplit(url).netloc
    out = {"url": url, "host": host, "product_id": row.get("product_id"), "kind": row.get("kind")}
    if not gate.allowed(url):
        out["status"] = "robots_disallow"
        return out
    gate.wait(host)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "image/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        out["status"] = f"http_{exc.code}"
        return out
    except Exception as exc:
        out["status"] = type(exc).__name__
        return out
    if not body.startswith(IMAGE_MAGIC):
        out["status"] = "not_image"
        return out
    digest = hashlib.sha256(body).hexdigest()
    out.update(status="ok", bytes=len(body), sha256=digest)
    if save_dir:
        path = os.path.join(save_dir, digest[:2], digest + ".img")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "wb") as fh:
                fh.write(body)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out_jsonl")
    ap.add_argument("--interval", type=float, default=0.5, help="호스트별 최소 요청 간격(초)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=15)
    ap.add_argument("--save", help="이미지 저장 디렉터리(리포 밖)")
    args = ap.parse_args()

    if args.save and os.path.realpath(args.save).startswith(
        os.path.realpath(os.path.dirname(__file__) + "/..")
    ):
        sys.exit("--save 는 리포 밖이어야 한다")

    rows = list(csv.DictReader(open(args.csv)))
    gate = HostGate(args.interval)
    counts = collections.defaultdict(collections.Counter)
    with open(args.out_jsonl, "w") as out, cf.ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(lambda r: fetch(r, gate, args.save, args.timeout), rows):
            out.write(json.dumps(res) + "\n")
            counts[res["host"]][res["status"]] += 1
    for host, c in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(c.values())
        print(f"{host}\tok {c['ok']}/{total} = {100 * c['ok'] / total:.1f}%\t{dict(c)}")


if __name__ == "__main__":
    main()
