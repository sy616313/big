r"""本地 Web Demo：浏览器里看仿真 + 打字指挥机械臂（真的抓起来放到盘子上）。

跑法：
    & "D:\envs\embodied\python.exe" sim\server.py
    浏览器打开 http://127.0.0.1:8000

说明：
    - 只用标准库 + mujoco + Pillow，不额外装依赖
    - 仿真在后台线程跑，页面通过 MJPEG 拉流
    - 指令解析复用 src/llm（没配 key 自动落到规则兜底）
    - 动作通过逆运动学走关键点：悬停→下降→合爪→抬起→移到盘上→下降→松爪
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENE = ROOT / "sim" / "arm_scene.xml"

from sim.kinematics import ik  # noqa: E402
from src.llm import parse as parse_instruction  # noqa: E402

W, H = 800, 600
FPS = 24
GRIP_OPEN = (0.030, -0.030)
GRIP_CLOSED = (0.000, 0.000)


class Sim:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(SCENE.read_text(encoding="utf-8"))
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip")
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = [0.15, -0.03, 0.10]
        self.cam.distance = 0.80
        self.cam.azimuth = 140.0
        self.cam.elevation = -22.0

        self.lock = threading.RLock()
        self.frames = 0
        self.started_at = time.time()
        self._jpeg = self._blank()
        self._running = True

        self.plan: dict | None = None
        self.message = "等待指令"
        self.waypoints: list[dict] = []
        self.wp_index = 0

        self.ctrl_target = np.array([0.0, -0.30, 0.90, -0.60, 0.0, *GRIP_OPEN])
        for _ in range(400):
            self.data.ctrl[:] = self.ctrl_target
            mujoco.mj_step(self.model, self.data)

    @staticmethod
    def _blank() -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (W, H), (232, 236, 243)).save(buf, format="JPEG")
        return buf.getvalue()

    def tip(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    def body_xyz(self, name: str) -> list[float]:
        return [float(v) for v in self.data.body(name).xpos]

    # ---------- 指令 ----------
    def command(self, text: str) -> dict:
        body_map = {"red_cube": "cube", "blue_cube": "cube", "plate": "plate"}
        try:
            plan = parse_instruction(text)
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.message = "没听懂这句话"
            return {"ok": False, "error": str(e).splitlines()[0]}

        with self.lock:
            cube = np.array(self.body_xyz(body_map.get(plan.target, "cube")))
            plate = np.array(self.body_xyz(body_map.get(plan.place, "plate")))
            plan.target_xy = [float(cube[0]), float(cube[1])]
            plan.place_xy = [float(plate[0]), float(plate[1])]

            self.plan = plan.to_dict()
            self.message = f"目标 {plan.target} → 放到 {plan.place}"
            self.waypoints = [
                {"p": cube + [0, 0, 0.060],  "grip": GRIP_OPEN,   "note": "1/7 悬停到方块上方"},
                {"p": cube + [0, 0, 0.002],  "grip": GRIP_OPEN,   "note": "2/7 下降对准方块"},
                {"p": cube + [0, 0, 0.002],  "grip": GRIP_CLOSED, "note": "3/7 合爪"},
                {"p": cube + [0, 0, 0.070],  "grip": GRIP_CLOSED, "note": "4/7 抬起"},
                {"p": plate + [0, 0, 0.075], "grip": GRIP_CLOSED, "note": "5/7 移到盘子上方"},
                {"p": plate + [0, 0, 0.042], "grip": GRIP_CLOSED, "note": "6/7 下降放盘"},
                {"p": plate + [0, 0, 0.042], "grip": GRIP_OPEN,   "note": "7/7 松爪"},
            ]
            self.wp_index = 0
        return {"ok": True, "plan": plan.to_dict()}

    def _advance(self) -> None:
        if self.wp_index >= len(self.waypoints):
            return
        wp = self.waypoints[self.wp_index]
        target = np.asarray(wp["p"], dtype=float)
        if np.linalg.norm(target - self.tip()) < 0.008:
            self.message = wp["note"]
            self.wp_index += 1
            if self.wp_index >= len(self.waypoints):
                self.message = "完成：方块已放到盘子上"

    def loop(self) -> None:
        renderer = mujoco.Renderer(self.model, height=H, width=W)
        step = self.model.opt.timestep
        per_frame = max(1, int(round((1.0 / FPS) / step)))
        while self._running:
            t0 = time.time()
            with self.lock:
                self._advance()
                if self.wp_index < len(self.waypoints):
                    wp = self.waypoints[self.wp_index]
                    q = ik(wp["p"][0], wp["p"][1], wp["p"][2])
                    self.ctrl_target = np.array([*q, 0.0, wp["grip"][0], wp["grip"][1]])
                    self.data.ctrl[:] = self.ctrl_target

            for _ in range(per_frame):
                mujoco.mj_step(self.model, self.data)

            renderer.update_scene(self.data, camera=self.cam)
            buf = io.BytesIO()
            Image.fromarray(renderer.render()).save(buf, format="JPEG", quality=82)
            with self.lock:
                self._jpeg = buf.getvalue()
                self.frames += 1

            sleep = (1.0 / FPS) - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def jpeg(self) -> bytes:
        with self.lock:
            return self._jpeg

    def status(self) -> dict:
        with self.lock:
            return {
                "message": self.message,
                "plan": self.plan,
                "step": self.wp_index,
                "total_steps": len(self.waypoints),
                "cube": [round(v, 3) for v in self.body_xyz("cube")],
                "frames": self.frames,
                "uptime": round(time.time() - self.started_at, 1),
            }


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>轻量具身智能体 · 本地 Demo</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0f1419; color:#e6edf3;
         font-family:"Microsoft YaHei",system-ui,sans-serif; }
  .wrap { max-width:900px; margin:24px auto; padding:0 16px; }
  h1 { font-size:18px; font-weight:600; margin:0 0 4px; }
  .sub { color:#8b949e; font-size:13px; margin-bottom:16px; }
  .stage { position:relative; border-radius:10px; overflow:hidden;
           background:#000; box-shadow:0 8px 24px rgba(0,0,0,.4); }
  .stage img { display:block; width:100%; }
  .badge { position:absolute; left:12px; top:12px; background:rgba(0,0,0,.55);
           padding:4px 10px; border-radius:999px; font-size:12px; color:#9fe0a8; }
  .state { position:absolute; left:12px; bottom:12px; background:rgba(0,0,0,.55);
           padding:4px 10px; border-radius:8px; font-size:12px; color:#e6edf3; }
  form { display:flex; gap:8px; margin-top:14px; }
  input { flex:1; padding:11px 14px; border-radius:8px; border:1px solid #30363d;
          background:#161b22; color:#e6edf3; font-size:15px; }
  button { padding:11px 20px; border-radius:8px; border:0; cursor:pointer;
           background:#2f6feb; color:#fff; font-size:15px; font-weight:600; }
  button:disabled { opacity:.5; cursor:default; }
  .log { margin-top:12px; font-size:13px; color:#8b949e; min-height:44px;
         background:#161b22; border:1px solid #21262d; border-radius:8px; padding:10px 12px;
         white-space:pre-wrap; font-family:Consolas,monospace; }
  .hint { margin-top:10px; font-size:12px; color:#6e7681; }
</style></head>
<body><div class="wrap">
  <h1>轻量具身智能体 · 本地 Demo</h1>
  <div class="sub">仿真中：六轴机械臂 / 红方块 / 蓝盘子 · 跑在 127.0.0.1</div>
  <div class="stage">
    <div class="badge" id="badge">LIVE</div>
    <div class="state" id="state">等待指令</div>
    <img src="/stream" alt="sim">
  </div>
  <form id="f">
    <input id="q" autocomplete="off"
           placeholder="说一句话，比如：把红色方块放到盘子上">
    <button id="b">执行</button>
  </form>
  <div class="log" id="log">等待指令……</div>
  <div class="hint">这里只能下「机器人指令」，不是终端。没配 API key 时会走规则解析，同样听得懂「红色方块」「盘子」。</div>
</div>
<script>
const badge=document.getElementById('badge'), state=document.getElementById('state');
setInterval(async ()=>{
  try{
    const s=await (await fetch('/api/status')).json();
    badge.textContent = 'LIVE · ' + s.frames + ' 帧 · ' + Math.round(s.uptime) + 's';
    state.textContent = s.message;
  }catch(e){ badge.textContent='LIVE · 连接断开'; }
}, 1000);
const f=document.getElementById('f'), q=document.getElementById('q'),
      b=document.getElementById('b'), log=document.getElementById('log');
f.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const text=q.value.trim(); if(!text) return;
  b.disabled=true; log.textContent='解析中……';
  try{
    const r=await fetch('/api/command',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text})});
    const j=await r.json();
    log.textContent = j.ok
      ? '解析结果：'+JSON.stringify(j.plan,null,2)
      : '没听懂：'+j.error;
  }catch(err){ log.textContent='请求失败：'+err; }
  b.disabled=false;
});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    sim: Sim

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/status":
            self._send(json.dumps(self.sim.status(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = self.sim.jpeg()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                    self.wfile.write(jpg + b"\r\n")
                    time.sleep(1.0 / FPS)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/command":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        result = self.sim.command(str(payload.get("text", "")))
        self._send(json.dumps(result, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    sim = Sim()
    threading.Thread(target=sim.loop, daemon=True).start()
    Handler.sim = sim

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"本地 Demo 已启动： http://127.0.0.1:{args.port}")
    print("按 Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sim._running = False
        print("\n已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
