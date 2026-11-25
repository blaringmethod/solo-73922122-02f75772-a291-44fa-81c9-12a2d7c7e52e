"""Large-manuscript walkthrough against a running server.

Usage: python tests/big_manuscript.py http://127.0.0.1:8000 [box_count]

It imports a manuscript with hundreds of one-character boxes (so a release has
hundreds of source sets), then exercises the browser flow that used to fail on
SQLite's per-statement parameter limit:

  * open the manuscript page (provenance rendering),
  * revise a range that spans every box (the replacement inherits the full
    union of box intervals),
  * look up that replacement: complete detail for every box, not a preview,
  * verify the hover preview still shows exactly eight entries plus "…",
  * verify the earlier release is untouched and still readable.

The server's SQLite limit is whatever the platform reports; the point is that
neither the number of spans nor the size of the inherited source set appears
as a parameter count in any statement.
"""

from __future__ import annotations

import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
BOX_COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 600
CODE = f"BIG-{uuid.uuid4().hex[:8]}"

BOXES = [
    {
        "page_no": 1 + i // 40,
        "box_no": 1 + i % 40,
        "rect": [40, 40 + (i % 40) * 30, 500, 26],
        "text": chr(0x4E00 + i),
    }
    for i in range(BOX_COUNT)
]
BODY = "".join(b["text"] for b in BOXES)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


opener = urllib.request.build_opener(NoRedirect)


def request(method: str, path: str, data: dict | None = None, json_body=None):
    url = BASE + path
    headers = {}
    body = None
    if json_body is not None:
        body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif data is not None:
        body = urllib.parse.urlencode(data, doseq=True).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with opener.open(req, timeout=60) as resp:
            return resp.status, resp.headers, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read().decode("utf-8")


def check(label: str, condition: bool, detail: str = "") -> None:
    print(("PASS  " if condition else "FAIL  ") + label + (f"  [{detail}]" if detail else ""))
    if not condition:
        raise SystemExit(f"big-manuscript walkthrough failed at: {label}")


def body_of(page: str) -> str:
    match = re.search(r'<div id="body"[^>]*>(.*?)</div>', page, re.S)
    if not match:
        raise SystemExit("no body div in page")
    return html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))


def base_release_id(page: str) -> int:
    match = re.search(r'name="base_release_id" value="(\d+)"', page)
    if not match:
        raise SystemExit("no base_release_id input")
    return int(match.group(1))


print(f"== target {BASE} · {BOX_COUNT} boxes ==")

# 1. Import the large manuscript.
status, headers, page = request(
    "POST",
    "/import",
    data={"code": CODE, "title": "大稿演示", "boxes": json.dumps(BOXES, ensure_ascii=False)},
)
check("大稿导入后重定向", status in (302, 303), f"HTTP {status}")

# 2. Open the manuscript page: this renders one provenance span per box and
#    used to raise "too many SQL variables".
status, _, page = request("GET", f"/m/{CODE}")
check("稿件页可打开", status == 200, f"HTTP {status}")
check("初稿完整", body_of(page) == BODY, f"{len(body_of(page))} 码点")
title_match = re.search(r'<span class="prov"[^>]*title="([^"]*)"', page)
check("悬停提示存在", title_match is not None)
first_title = html.unescape(title_match.group(1))
check("单片悬停只列本框来源", first_title.count("页") == 1, first_title)

# 3. Revise the whole body: the new character inherits every box's interval as
#    one source set, which used to exceed the parameter limit on write.
base1 = base_release_id(page)
status, headers, page = request(
    "POST",
    f"/m/{CODE}/revise",
    data={
        "base_release_id": base1,
        "start": 0,
        "end": len(BODY),
        "replacement": "整",
        "note": f"跨 {BOX_COUNT} 个框整版替换",
    },
)
check("跨大量识别框校订后重定向", status in (302, 303), f"HTTP {status}")
status, _, page = request("GET", headers["Location"])
check("新版正文为换入字", body_of(page) == "整", body_of(page)[:40])

# 4. Hover preview of the inserted run: 8 entries, then an ellipsis.
hover = re.search(r'<span class="prov"[^>]*title="([^"]*)"', page)
check("换入字有悬停提示", hover is not None)
hover_title = html.unescape(hover.group(1))
check(
    "悬停按原顺序列前 8 条并加省略号",
    hover_title.count("页") == 8 and hover_title.endswith("；…"),
    hover_title,
)
check(
    "前 8 条为页 1 框 1 至框 8",
    "页1 框1 " in hover_title and "页1 框8 " in hover_title and "框9 " not in hover_title,
)

# 5. Full-detail lookup must still return every contributing box.
status, _, text = request("GET", f"/api/m/{CODE}/lookup?start=0&end=1")
payload = json.loads(text)
check("查源接口成功", status == 200, f"HTTP {status}")
check("查源返回完整来源（全部框）", len(payload["sources"]) == BOX_COUNT, str(len(payload["sources"])))
check("来源按阅读顺序", [s["box_no"] for s in payload["sources"][:40]] == list(range(1, 41)))
check("换入字标为非精确并展示整框区间",
      payload["sources"][0]["exact"] is False
      and payload["sources"][0]["box_intervals"] == [{"start": 0, "end": 1}])

# 6. Re-edit the inserted character: the full inherited union is reused.
base2 = base_release_id(page)
status, headers, page = request(
    "POST",
    f"/m/{CODE}/revise",
    data={"base_release_id": base2, "start": 0, "end": 1, "replacement": "整版", "note": "再改"},
)
check("再次校订成功", status in (302, 303), f"HTTP {status}")
status, _, page = request("GET", headers["Location"])
check("再次校订后正文", body_of(page) == "整版", body_of(page))
status, _, text = request("GET", f"/api/m/{CODE}/lookup?start=0&end=2")
payload = json.loads(text)
check("再次校订沿用完整来源并集", len(payload["sources"]) == BOX_COUNT, str(len(payload["sources"])))

# 7. The first release is still an untouched, readable snapshot.
status, _, page = request("GET", f"/m/{CODE}?release=1")
check("历史版可打开", status == 200, f"HTTP {status}")
check("历史版正文仍是初稿", body_of(page) == BODY)
check("历史版只读", 'name="base_release_id"' not in page)
status, _, text = request("GET", f"/api/m/{CODE}/releases")
releases = json.loads(text)["releases"]
check("三版均保留", [r["release_no"] for r in releases] == [1, 2, 3], str(len(releases)))
check("初版正文原样", releases[0]["body"] == BODY)

print("\nBIG_MANUSCRIPT_OK")
