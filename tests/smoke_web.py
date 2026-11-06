r"""网页 Demo 的冒烟测试：自己起服务、跑一遍完整流程、检查结果。

跑法（在项目根目录）：
    & "D:\envs\embodied\python.exe" scripts\server.py --port 8123    # 终端 A
    & "D:\envs\embodied\python.exe" tests\smoke_web.py               # 终端 B

不这么测的话，很容易"代码没报错但页面点不动"——之前就踩过两次：
首页路由没处理查询串导致 404，以及页面脚本里有个字符串换行导致语法错误，
整个 <script> 直接不执行，按钮全是死的。这两个用眼睛看是看不出来的。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

PLATE = (0.150, -0.140)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8123")
    ap.add_argument("--timeout", type=float, default=45.0)
    args = ap.parse_args()
    base = args.base.rstrip("/")

    def get(path: str):
        with urllib.request.urlopen(base + path, timeout=8) as r:
            return r.status, r.read()

    def post(path: str, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(base + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()

    # 等服务起来（最多 20 秒）
    for _ in range(40):
        try:
            get("/api/status")
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    else:
        print(f"服务没起来：{base}")
        return 1

    ok = True

    st, body = get("/api/status")
    s = json.loads(body)
    print(f"1) /api/status  {st}  message={s['message']}  cube={s['cube']}")

    st, page = get("/")
    html = page.decode("utf-8", "replace")
    print(f"2) /  {st}  {len(page)} bytes")
    for name, cond in {
        "首页返回 200": st == 200,
        "有输入表单": "<form" in html,
        "有视频流": "/stream" in html,
        "有脚本块": "<script" in html,
    }.items():
        print(f"     {'OK  ' if cond else 'FAIL'} {name}")
        ok &= cond

    st, body = post("/api/command", {"text": "把红色方块放到盘子上"})
    plan = json.loads(body)
    good = bool(plan.get("ok"))
    print(f"3) /api/command  {st}  ok={good}  target={plan.get('plan', {}).get('target')}")
    ok &= good

    t0 = time.time()
    last = -1
    s = json.loads(get("/api/status")[1])
    while time.time() - t0 < args.timeout:
        s = json.loads(get("/api/status")[1])
        if s["step"] != last:
            last = s["step"]
            print(f"   [{time.time()-t0:5.1f}s] {s['step']}/{s['total_steps']}  {s['message']}")
        if s["total_steps"] and s["step"] >= s["total_steps"]:
            break
        time.sleep(0.3)

    cube = s["cube"]
    d = math.hypot(cube[0] - PLATE[0], cube[1] - PLATE[1])
    placed = d < 0.045 and 0.02 <= cube[2] <= 0.045
    print(f"4) 完成 {s['step']}/{s['total_steps']}，方块 {cube}，距盘心 {d*1000:.1f} mm")
    print(f"     {'OK  ' if placed else 'FAIL'} 方块落到盘上")
    ok &= placed

    for path in ("/api/reset", "/api/view_reset"):
        st, _ = post(path, {})
        print(f"5) {path}  {st}")
        ok &= st == 200

    print()
    print("全部通过" if ok else "有失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
