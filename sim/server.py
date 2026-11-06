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

from sim.env import HOME_XYZ  # noqa: E402
from sim.kinematics import Unreachable, ik  # noqa: E402
from src.llm import parse as parse_instruction  # noqa: E402

W, H = 800, 600
FPS = 24
GRIP_OPEN = (0.018, -0.018)
# 和 sim/env.py、scripts/grasp_demo.py 保持一致：闭合到 0 会硬挤方块
GRIP_CLOSED = (0.004, -0.004)

HOME_Q = ik(*HOME_XYZ)
HOME_CTRL = np.array([*HOME_Q, 0.0, *GRIP_OPEN])


class Sim:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(SCENE.read_text(encoding="utf-8"))
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip")
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.view = {"az": 138.0, "el": -20.0, "dist": 0.72, "look": [0.17, -0.04, 0.09]}
        self._apply_view()

        self.lock = threading.RLock()
        self.frames = 0
        self.started_at = time.time()
        self.version = time.strftime('%m-%d %H:%M:%S')
        self._jpeg = self._blank()
        self._running = True

        self.plan: dict | None = None
        self.message = "等待指令"
        self.waypoints: list[dict] = []
        self.wp_index = 0
        self._cmd_p = self.tip().copy()      # 当前下发给 IK 的位置（限速用）
        self._wp_t0 = time.time()            # 到达当前关键点的时刻

        self.ctrl_target = HOME_CTRL.copy()
        for _ in range(400):
            self.data.ctrl[:] = self.ctrl_target
            mujoco.mj_step(self.model, self.data)

    @staticmethod
    def _blank() -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (W, H), (232, 236, 243)).save(buf, format="JPEG")
        return buf.getvalue()

    def _apply_view(self) -> None:
        v = self.view
        self.cam.azimuth = v["az"]
        self.cam.elevation = v["el"]
        self.cam.distance = v["dist"]
        self.cam.lookat[:] = v["look"]

    def camera(self, d: dict) -> dict:
        """处理来自页面的视角增量。

        daz / del：旋转（度）；ddist：缩放（倍率增量）；dx / dy：平移（像素）
        """
        with self.lock:
            v = self.view
            v["az"] = (v["az"] + float(d.get("daz", 0.0))) % 360.0
            v["el"] = max(-88.0, min(88.0, v["el"] + float(d.get("del", 0.0))))
            v["dist"] = max(0.20, min(1.15, v["dist"] * (1.0 + float(d.get("ddist", 0.0)))))

            dx = float(d.get("dx", 0.0))
            dy = float(d.get("dy", 0.0))
            if dx or dy:
                azr = np.radians(v["az"])
                elr = np.radians(v["el"])
                fwd = np.array([np.cos(elr) * np.cos(azr), np.cos(elr) * np.sin(azr), np.sin(elr)])
                right = np.array([-np.sin(azr), np.cos(azr), 0.0])
                up = np.cross(right, fwd)
                k = v["dist"] * 0.0016
                v["look"] = list(np.array(v["look"]) - right * dx * k + up * dy * k)
                # 平移后统一夹紧，防止看到房间外
                v["look"] = list(np.clip(v["look"], [-0.12, -0.30, -0.02], [0.46, 0.26, 0.36]))
            self._apply_view()
            return dict(v)

    def reset_view(self) -> dict:
        with self.lock:
            self.view = {"az": 138.0, "el": -20.0, "dist": 0.72, "look": [0.17, -0.04, 0.09]}
            self._apply_view()
            return dict(self.view)

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
                {"p": cube + [0, 0, 0.002],  "grip": GRIP_CLOSED, "note": "3/7 合爪", "dwell": 0.8},
                {"p": cube + [0, 0, 0.070],  "grip": GRIP_CLOSED, "note": "4/7 抬起"},
                {"p": plate + [0, 0, 0.075], "grip": GRIP_CLOSED, "note": "5/7 移到盘子上方"},
                {"p": plate + [0, 0, 0.042], "grip": GRIP_CLOSED, "note": "6/7 下降放盘"},
                {"p": plate + [0, 0, 0.042], "grip": GRIP_OPEN,   "note": "7/7 松爪", "dwell": 0.8},
            ]
            self.wp_index = 0
            self._cmd_p = self.tip().copy()
            self._wp_t0 = time.time()
        return {"ok": True, "plan": plan.to_dict()}

    def reset(self) -> None:
        """把场景恢复初始状态，方便反复演示。"""
        with self.lock:
            self.data = mujoco.MjData(self.model)
            self.waypoints = []
            self.wp_index = 0
            self.plan = None
            self.message = "等待指令"
            self.ctrl_target = HOME_CTRL.copy()
            self.data.ctrl[:] = self.ctrl_target
            for _ in range(500):
                self.data.ctrl[:] = self.ctrl_target
                mujoco.mj_step(self.model, self.data)

    def _advance(self) -> None:
        if self.wp_index >= len(self.waypoints):
            return
        wp = self.waypoints[self.wp_index]
        target = np.asarray(wp["p"], dtype=float)
        dwell = float(wp.get("dwell", 0.25))
        arrived = np.linalg.norm(target - self.tip()) < 0.010
        waited = time.time() - self._wp_t0 > dwell
        if arrived and waited:
            self.message = wp["note"]
            self.wp_index += 1
            self._wp_t0 = time.time()
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
                    dest = np.asarray(wp["p"], dtype=float)
                    dt = per_frame * step
                    delta = dest - self._cmd_p
                    dist = float(np.linalg.norm(delta))
                    speed = 0.08                      # m/s，慢到能看清
                    if dist > 1e-9:
                        self._cmd_p += delta * min(1.0, speed * dt / dist)
                    try:
                        q = ik(self._cmd_p[0], self._cmd_p[1], self._cmd_p[2])
                    except Unreachable as e:
                        # 这里绝对不能让它冒出去：异常会杀死后台循环线程，
                        # 网页画面就永久冻住了。停下来报个信，比静默死掉好。
                        self.waypoints = []
                        self.wp_index = 0
                        self.message = f"目标够不着：{e}"
                        continue
                    self.ctrl_target = np.array([*q, 0.0, wp["grip"][0], wp["grip"][1]])
                    self.data.ctrl[:] = self.ctrl_target

            for _ in range(per_frame):
                mujoco.mj_step(self.model, self.data)

            with self.lock:
                self._apply_view()
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
                "version": self.version,
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
  .stage { cursor: grab; }
  .stage.dragging { cursor: grabbing; }
  button.alt { background:#21262d; border:1px solid #30363d; font-weight:500; }
  .build { margin-top:10px; font-size:12px; color:#6e7681; font-family:Consolas,monospace; }
</style></head>
<body><div class="wrap">
  <h1>轻量具身智能体 · 本地 Demo</h1>
  <div class="sub">仿真中：六轴机械臂 / 红方块 / 蓝盘子 · 跑在 127.0.0.1</div>
  <div class="stage">
    <div class="badge" id="badge">LIVE</div>
    <div class="state" id="state">等待指令</div>
    <img id="sim" src="/stream" alt="sim">
  </div>
  <form id="f">
    <input id="q" autocomplete="off"
           placeholder="说一句话，比如：把红色方块放到盘子上">
    <button id="b">执行</button>
    <button type="button" id="demo" class="alt">一键演示</button>
    <button type="button" id="reset" class="alt">重置</button>
    <button type="button" id="viewreset" class="alt">复位视角</button>
  </form>
  <div class="log" id="log">等待指令……</div>
  <div class="hint">拖动旋转 · 滚轮缩放 · 右键平移 · 双击复位视角｜下面只能下机器人指令，不是终端</div>
  <div class="build" id="build">正在连接服务……</div>
</div>
<script>
const $id = (x) => document.getElementById(x);
const badge = $id('badge'), stateEl = $id('state'), logEl = $id('log'), buildEl = $id('build');
const form = $id('f'), input = $id('q'), btnRun = $id('b');
const btnDemo = $id('demo'), btnReset = $id('reset');

async function post(path, body) {
  const opt = body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : { method: 'POST' };
  const r = await fetch(path, opt);
  return r.json();
}

function show(j) {
  if (j.ok && j.plan) {
    const p = j.plan;
    logEl.textContent = `已收到指令
目标：${p.target}
放到：${p.place}
坐标：${JSON.stringify(p.target_xy)}`;
  } else {
    logEl.textContent = '没听懂：' + (j.error || '未知原因');
  }
}

async function run(text) {
  btnRun.disabled = true;
  logEl.textContent = '解析中……';
  try { show(await post('/api/command', { text })); }
  catch (e) { logEl.textContent = '请求失败：' + e; }
  btnRun.disabled = false;
}

// ---------- 自由视角 ----------
const stage = document.querySelector('.stage');
const viewReset = $id('viewreset');
let drag = null;
let pending = { daz: 0, del: 0, ddist: 0, dx: 0, dy: 0 };
let flushTimer = null;

function flushCamera() {
  flushTimer = null;
  const p = pending;
  pending = { daz: 0, del: 0, ddist: 0, dx: 0, dy: 0 };
  fetch('/api/camera', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(p),
  }).catch(() => {});
}

function queueCamera(patch) {
  for (const k in patch) pending[k] += patch[k];
  if (!flushTimer) flushTimer = setTimeout(flushCamera, 45);
}

stage.addEventListener('mousedown', (e) => {
  e.preventDefault();
  drag = { x: e.clientX, y: e.clientY, btn: e.button, shift: e.shiftKey };
  stage.classList.add('dragging');
});

window.addEventListener('mouseup', () => {
  drag = null;
  stage.classList.remove('dragging');
});

window.addEventListener('mousemove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x;
  const dy = e.clientY - drag.y;
  drag.x = e.clientX;
  drag.y = e.clientY;
  if (drag.btn === 0 && !drag.shift) queueCamera({ daz: dx * 0.35, del: -dy * 0.35 });
  else queueCamera({ dx, dy });
});

stage.addEventListener('wheel', (e) => {
  e.preventDefault();
  queueCamera({ ddist: e.deltaY * 0.0012 });
}, { passive: false });

stage.addEventListener('contextmenu', (e) => e.preventDefault());
stage.addEventListener('dblclick', () => viewReset.click());

viewReset.addEventListener('click', async () => {
  await fetch('/api/view_reset', { method: 'POST' });
});

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const t = input.value.trim();
  if (t) run(t);
});

btnDemo.addEventListener('click', async () => {
  input.value = '把红色方块放到盘子上';
  btnDemo.disabled = true;
  try {
    await post('/api/reset');
    await new Promise((r) => setTimeout(r, 700));
    await run(input.value);
  } finally {
    btnDemo.disabled = false;
  }
});

btnReset.addEventListener('click', async () => {
  await post('/api/reset');
  logEl.textContent = '已重置，可以再来一次。';
});

setInterval(async () => {
  try {
    const s = await (await fetch('/api/status')).json();
    badge.textContent = 'LIVE · ' + s.frames + ' 帧 · ' + Math.round(s.uptime) + 's';
    stateEl.textContent = s.message;
    buildEl.textContent = '服务启动于 ' + s.version + '　步骤 ' + s.step + '/' + s.total_steps;
  } catch (e) {
    badge.textContent = 'LIVE · 连接断开';
  }
}, 1000);
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
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("", "/") or path.startswith("/index"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/status":
            self._send(json.dumps(self.sim.status(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/stream":
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
        path = self.path.split("?", 1)[0]
        if path == "/api/camera":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:  # noqa: BLE001
                payload = {}
            self._send(json.dumps(self.sim.camera(payload), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/api/view_reset":
            self._send(json.dumps(self.sim.reset_view(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if path == "/api/reset":
            self.sim.reset()
            self._send(b'{"ok":true}', "application/json")
            return
        if path != "/api/command":
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
