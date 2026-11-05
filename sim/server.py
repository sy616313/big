r"""本地 Web Demo：浏览器里看仿真 + 打字指挥机械臂。

跑法：
    & "D:\envs\embodied\python.exe" sim\server.py
    然后浏览器打开 http://127.0.0.1:8000

说明：
    - 只用标准库（http.server + mujoco + Pillow），不额外装依赖
    - 仿真在主线程外的后台线程里跑，页面通过 MJPEG 拉流
    - 指令解析复用 src/llm（没配 API key 会自动落到规则兜底）
"""
from __future__ import annotations

import argparse
import io
import json
import math
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

from src.llm import parse as parse_instruction  # noqa: E402

W, H = 800, 600
FPS = 24


class Sim:
    """跑仿真 + 提供最新一帧 JPEG。"""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(SCENE.read_text(encoding="utf-8"))
        self.data = mujoco.MjData(self.model)
        # 渲染器不在这里建：GL 上下文与线程绑定，放到 loop() 里创建
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = [0.17, 0.0, 0.12]
        self.cam.distance = 0.95
        self.cam.azimuth = 132.0
        self.cam.elevation = -20.0

        self.frames = 0
        self.started_at = time.time()
        self.lock = threading.RLock()
        self._jpeg = self._blank()
        self._running = True
        self.plan: dict | None = None
        self.message = "等待指令"
        self._t0 = 0.0            # 本次动作开始时间
        self._phase = "idle"      # idle / reach / grasp / lift

        # 关节默认姿势（收拢）
        self._home = np.array([0.0, -0.30, 0.90, -0.60, 0.0, 0.030, -0.030])
        self.data.ctrl[:] = self._home
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _blank() -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (W, H), (232, 236, 243)).save(buf, format="JPEG")
        return buf.getvalue()

    # ---------- 指令 ----------
    def command(self, text: str) -> dict:
        try:
            plan = parse_instruction(text)
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.message = f"没听懂：{e}"
            return {"ok": False, "error": str(e)}

        # 坐标以仿真里的真实位置为准（config.json 里那份只是离线参考值）
        body_map = {"red_cube": "cube", "blue_cube": "cube", "plate": "plate"}
        with self.lock:
            tb = body_map.get(plan.target, "cube")
            pb = body_map.get(plan.place, "plate")
            txy = [float(v) for v in self.data.body(tb).xpos[:2]]
            pxy = [float(v) for v in self.data.body(pb).xpos[:2]]
        plan.target_xy, plan.place_xy = txy, pxy
        yaw = math.atan2(txy[1], txy[0])
        with self.lock:
            self.plan = plan.to_dict()
            self.message = f"目标 {plan.target} → 放到 {plan.place}"
            self._phase = "reach"
            self._t0 = time.time()
            self._yaw = max(-3.0, min(3.0, yaw))
        return {"ok": True, "plan": plan.to_dict()}

    # ---------- 主循环 ----------
    def _targets(self) -> np.ndarray:
        """按阶段给出 7 个执行器的目标值。"""
        with self.lock:
            phase = self._phase
            age = time.time() - self._t0
            yaw = getattr(self, "_yaw", 0.0)

        reach = np.array([yaw, -0.88, 1.32, -0.45, 0.0, 0.030, -0.030])
        grasp = reach.copy()
        grasp[5], grasp[6] = 0.004, -0.004
        lift = np.array([yaw, -0.55, 1.05, -0.55, 0.0, 0.004, -0.004])

        if phase == "reach":
            return reach if age < 1.6 else grasp
        if phase == "grasp":
            return grasp
        if phase == "lift":
            return lift
        return self._home

    def loop(self) -> None:
        # GL 上下文是线程绑定的，渲染器必须在本线程创建
        renderer = mujoco.Renderer(self.model, height=H, width=W)
        step = self.model.opt.timestep
        per_frame = max(1, int(round((1.0 / FPS) / step)))
        while self._running:
            frame_start = time.time()
            with self.lock:
                tgt = self._targets()
                # 阶段推进
                age = time.time() - self._t0
                if self._phase == "reach" and age > 1.6:
                    self._phase = "grasp"
                    self._t0 = time.time()
                elif self._phase == "grasp" and age > 0.45:
                    self._phase = "lift"
                    self._t0 = time.time()
                elif self._phase == "lift" and age > 1.2:
                    self._phase = "idle"
                    self.message = "完成（当前还不会真抓取，下一步接逆运动学）"
                self.ctrl_target = tgt

            for _ in range(per_frame):
                self.data.ctrl[:] = self.ctrl_target
                mujoco.mj_step(self.model, self.data)

            renderer.update_scene(self.data, camera=self.cam)
            img = renderer.render()
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=82)
            with self.lock:
                self._jpeg = buf.getvalue()
                self.frames += 1

            sleep = (1.0 / FPS) - (time.time() - frame_start)
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
                "phase": self._phase,
                "cube": [round(float(v), 3) for v in self.data.body("cube").xpos],
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
  <div class="stage"><div class="badge" id="badge">LIVE</div><img src="/stream" alt="sim"></div>
  <form id="f">
    <input id="q" autocomplete="off"
           placeholder="说一句话，比如：把红色方块放到盘子上">
    <button id="b">执行</button>
  </form>
  <div class="log" id="log">等待指令……</div>
  <div class="hint">提示：没配 API key 时会走规则解析，同样能听懂「红色方块」「盘子」这类说法。</div>
</div>
<script>
const badge=document.getElementById('badge');
setInterval(async ()=>{
  try{
    const s=await (await fetch('/api/status')).json();
    badge.textContent = 'LIVE · ' + s.frames + ' 帧 · ' + Math.round(s.uptime) + 's';
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
      : '失败：'+j.error;
  }catch(err){ log.textContent='请求失败：'+err; }
  b.disabled=false;
});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    sim: Sim

    def log_message(self, *a):  # 静默访问日志
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
        if self.path == "/api/status":
            body = json.dumps(self.sim.status(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
        body = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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
