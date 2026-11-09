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
import socket
import sys
import threading
import time
import webbrowser
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

# ---- 策略接管（把训练好的 ACT 接到网页上）----
# 训练时的画面是 320x240 / 名为 front 的相机；这里用同一个相机渲染再降采样，
# 保证模型看到的画面和训练时一致（分辨率不同不影响视场角）。
POLICY_IMG = (320, 240)
POLICY_CAMERA = "front"
POLICY_EXEC = 10          # 每执行多少步重新预测一次（实测 10 是这批数据的最优点）
POLICY_MAX_STEPS = 400
ARM_JOINTS = ["j1_yaw", "j2_shoulder", "j3_elbow", "j4_wrist", "grip_l"]



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

        # 策略接管相关（默认关闭，--policy 指向权重目录才启用）
        self.arm_qadr = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in ARM_JOINTS
        ]
        self.policy = None
        self.policy_pre = None
        self.policy_post = None
        self.policy_device = None
        self.policy_queue: list[np.ndarray] = []
        self.policy_consumed = 0
        self.policy_steps = 0

        # 智能体模式（--agent）：大模型出计划，执行器逐步跑，失败回传重规划
        self.agent_state = "idle"          # idle | running | done
        self.agent_text = ""
        self.agent_plan: dict | None = None
        self.agent_trace: list[dict] = []
        self.agent_result: dict | None = None
        self.agent_enabled = False
        self._renderer = None              # 由 loop() 建好后挂上来，策略取画面要用
        self._per_frame = 1

        self.plate_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self.plate_mid = int(self.model.body_mocapid[self.plate_body])
        self._plate_home = np.array(self.data.body("plate").xpos, dtype=float).copy()

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

    # ---------- 策略接管 ----------
    def load_policy(self, ckpt: Path) -> None:
        """加载训练好的策略（act / diffusion 都认）。

        注意：这里不复用 scripts/eval_act.py 的加载函数——那个模块在 import 时会调用
        ensure_utf8()，在已经跑着的服务里会重启进程，只能自己写一遍。
        """
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(str(ckpt))
        if cfg.type == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy as Cls
        elif cfg.type == "diffusion":
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as Cls
        else:
            raise SystemExit(f"网页还不支持 policy.type={cfg.type}")
        policy = Cls.from_pretrained(str(ckpt), config=cfg)
        policy.eval()
        pre, post = make_pre_post_processors(cfg, pretrained_path=str(ckpt))
        with self.lock:
            self.policy, self.policy_pre, self.policy_post = policy, pre, post
            self.policy_device = next(policy.parameters()).device
            self.policy_queue = []
            self.policy_steps = 0
            self.message = f"策略已加载：{Path(ckpt).parent.parent.parent.name}"

    def _policy_batch(self, renderer) -> dict:
        """按训练时的相机和尺寸取一帧，拼成策略要的 batch。"""
        import torch

        renderer.update_scene(self.data, camera=POLICY_CAMERA)
        frame = renderer.render()                       # 800x600，视场角与训练一致
        small = np.asarray(Image.fromarray(frame).resize(POLICY_IMG, Image.BILINEAR))
        img = torch.from_numpy(small).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.policy_device)
        state = torch.tensor([[float(self.data.qpos[a]) for a in self.arm_qadr]],
                             dtype=torch.float32, device=self.policy_device)
        return {"observation.images.front": img, "observation.state": state}

    def _query_policy(self, renderer) -> None:
        """取一整块动作放进队列。select_action 内部有队列，连调 N 次只算一次前向。"""
        import torch

        batch = self.policy_pre(self._policy_batch(renderer))
        n = int(getattr(self.policy.config, "n_action_steps", 40))
        with torch.no_grad():
            self.policy.reset()
            acts = [self.policy_post(self.policy.select_action(batch)).squeeze(0)
                    .float().cpu().numpy() for _ in range(n)]
        self.policy_queue = [np.asarray(a, dtype=np.float64) for a in acts]
        self.policy_consumed = 0

    def _policy_done(self) -> bool:
        """只判"方块在盘子上且夹爪松开了"，和训练/评估的判据一致。"""
        c = np.array(self.body_xyz("cube"))
        p = np.array(self.body_xyz("plate"))
        grip = float(self.data.qpos[self.arm_qadr[4]])
        return (float(np.hypot(c[0] - p[0], c[1] - p[1])) < 0.045
                and 0.020 <= c[2] <= 0.045 and grip > 0.008)

    # ---------- 给智能体当"环境"用的接口 ----------
    # src/agent.py 的 Executor 只要求下面这几个方法，名字和 PickPlaceEnv 保持一致，
    # 这样同一套执行器既能跑命令行（无界面）、也能跑网页。
    def cube_xyz(self) -> np.ndarray:
        return np.array(self.body_xyz("cube"), dtype=float)

    def plate_xyz(self) -> np.ndarray:
        return np.array(self.body_xyz("plate"), dtype=float)

    def state(self) -> np.ndarray:
        return np.array([float(self.data.qpos[a]) for a in self.arm_qadr], dtype=np.float32)

    def success(self) -> bool:
        return self._policy_done()

    def _apply_action(self, act: np.ndarray) -> None:
        grip = float(np.clip(act[4], 0.0, GRIP_OPEN[0]))
        self.data.ctrl[0:4] = act[0:4]
        self.data.ctrl[4] = 0.0
        self.data.ctrl[5] = grip
        self.data.ctrl[6] = -grip
        self.ctrl_target = np.array([*act[0:4], 0.0, grip, -grip])

    def step(self, act) -> dict:
        """智能体线程专用的推进：加锁走一帧物理，睡一小会儿让网页看得清。"""
        with self.lock:
            self._apply_action(np.asarray(act, dtype=float))
            for _ in range(self._per_frame):
                mujoco.mj_step(self.model, self.data)
        time.sleep(1.0 / (FPS * 2))
        return {"state": self.state()}

    def render_policy_frame(self) -> np.ndarray:
        """策略要的画面：主循环每帧顺手渲好的缓存。

        不在这个函数里现渲——智能体在自己的线程里跑，而 MuJoCo 的 Renderer
        不能两个线程同时用（实测会取到烂图，策略就完全抓不到东西）。
        """
        with self.lock:
            frame = getattr(self, "_policy_frame", None)
        if frame is None:
            return np.zeros((POLICY_IMG[1], POLICY_IMG[0], 3), dtype=np.uint8)
        return frame

    def obs(self) -> dict:
        return {"state": self.state(), "image": self.render_policy_frame()}

    # ---------- 智能体模式 ----------
    def start_agent(self, text: str) -> dict:
        """收到一句话就开一个线程跑「大模型拆计划 → 执行 → 失败回传重规划」。"""
        with self.lock:
            if self.policy is None:
                return {"ok": False, "error": "智能体模式要先加载策略（--policy）"}
            if self.agent_state == "running":
                return {"ok": False, "error": "上一条指令还在执行"}
            self.agent_state = "running"
            self.agent_text = text
            self.agent_trace = []
            self.agent_plan = None
            self.agent_result = None
            self.message = "大模型正在拆解指令…"
        threading.Thread(target=self._agent_worker, args=(text,), daemon=True).start()
        return {"ok": True}

    def _agent_worker(self, text: str) -> None:
        from src.agent import Executor

        try:
            ex = Executor(self, self.policy, self.policy_pre, self.policy_post, self.policy_device)
            original_run = ex.run

            def run_and_trace(plan, verbose: bool = True):
                with self.lock:
                    self.agent_plan = plan.to_dict()
                    self.message = "计划：" + " → ".join(s.tool for s in plan.steps)
                result = original_run(plan, verbose=verbose)
                with self.lock:
                    self.agent_trace.extend(result.get("trace", []))
                    row = (result.get("trace") or [{}])[-1]
                    ok = row.get("result", {}).get("ok")
                    self.message = f"{row.get('tool', '?')} {'成功' if ok else '失败，把结果回传给大模型'}"
                return result

            ex.run = run_and_trace                      # 让页面能看到每一步
            result = ex.run_until_done(text, max_turns=2, verbose=False)
            with self.lock:
                self.agent_result = result
                self.message = "智能体完成：成功" if result.get("ok") else "智能体完成：没做成"
        except Exception as e:  # noqa: BLE001 - 网页线程绝不能把异常漏出去
            with self.lock:
                self.message = f"智能体出错：{str(e)[:80]}"
                self.agent_result = {"ok": False, "error": str(e)}
        finally:
            with self.lock:
                self.agent_state = "done"

    def reset(self, seed: int | None = None) -> dict:   # noqa: ARG002 - 网页场景固定，seed 只为兼容
        self.do_reset()
        return {"state": self.state()}


    def body_xyz(self, name: str) -> list[float]:
        return [float(v) for v in self.data.body(name).xpos]

    # ---------- 指令 ----------
    def command(self, text: str) -> dict:
        if self.agent_enabled:
            # 智能体模式：一句话交给大模型拆计划，执行过程异步跑，结果看 /api/status
            return self.start_agent(text)
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
            if self.policy is not None:
                # 策略模式：不用逆解轨迹，清空队列让策略从当前状态重新规划
                self.waypoints = []
                self.wp_index = 0
                self.policy_queue = []
                self.policy_consumed = 0
                self.policy_steps = 0
                self.message = "策略接管：正在看画面执行"
                return {"ok": True, "plan": plan.to_dict(), "mode": "policy"}
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

    def do_reset(self) -> None:
        """把场景恢复初始状态，方便反复演示（HTTP 的 /api/reset 和智能体的 home 都走这里）。"""
        with self.lock:
            self.data = mujoco.MjData(self.model)
            self.waypoints = []
            self.wp_index = 0
            self.plan = None
            if self.agent_state != "running":
                # 智能体正在跑的时候不要把它的话盖掉（它自己会把进展写进 message）
                self.message = "等待指令"
            self.ctrl_target = HOME_CTRL.copy()
            self.policy_queue = []
            self.policy_consumed = 0
            self.policy_steps = 0
            if self.policy is not None:
                self.policy.reset()
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
        with self.lock:
            self._renderer = renderer        # 策略/智能体取画面要用同一个渲染器
            self._per_frame = per_frame
        while self._running:
            t0 = time.time()
            with self.lock:
                # 智能体线程自己在推进物理，主循环这时只负责渲染和看守
                agent_busy = self.agent_state == "running"
                policy_mode = self.policy is not None and self.plan is not None
                if agent_busy:
                    pass          # 物理由智能体线程推进（step() 里加锁自走），这里只渲染
                elif policy_mode:
                    # 策略接管：每 POLICY_EXEC 步重新预测一次，其余步执行队列里的动作
                    if not self.policy_queue or self.policy_consumed >= POLICY_EXEC:
                        self._query_policy(renderer)
                    act = self.policy_queue.pop(0) if self.policy_queue else HOME_CTRL[0:5]
                    self.policy_consumed += 1
                    self.policy_steps += 1
                    grip = float(np.clip(act[4], 0.0, GRIP_OPEN[0]))
                    self.data.ctrl[0:4] = act[0:4]
                    self.data.ctrl[4] = 0.0
                    self.data.ctrl[5] = grip
                    self.data.ctrl[6] = -grip
                    self.ctrl_target = np.array([*act[0:4], 0.0, grip, -grip])
                    if self._policy_done():
                        self.message = "完成：策略把方块放到盘子上了"
                        self.plan = None
                        self.policy_queue = []
                    elif self.policy_steps >= POLICY_MAX_STEPS:
                        self.message = "超时：策略没能在规定步数内完成"
                        self.plan = None
                        self.policy_queue = []
                else:
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

            if not agent_busy:
                # 智能体线程自己推进物理，主循环别插手，否则两边各走一遍，动作会翻倍
                for _ in range(per_frame):
                    mujoco.mj_step(self.model, self.data)

            with self.lock:
                self._apply_view()
            renderer.update_scene(self.data, camera=self.cam)
            buf = io.BytesIO()
            Image.fromarray(renderer.render()).save(buf, format="JPEG", quality=82)
            # 顺手给策略渲一帧（训练同款相机）。渲染只在这一个线程里做，
            # 智能体线程只读缓存——两个线程抢一个 Renderer 会取到烂图。
            renderer.update_scene(self.data, camera=POLICY_CAMERA)
            policy_frame = np.asarray(Image.fromarray(renderer.render()).resize(
                POLICY_IMG, Image.BILINEAR)).copy()
            with self.lock:
                self._jpeg = buf.getvalue()
                self._policy_frame = policy_frame
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
                "agent_state": self.agent_state,
                "agent_text": self.agent_text,
                "agent_plan": self.agent_plan,
                "agent_trace": self.agent_trace[-8:],
                "agent_result": self.agent_result,
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
    // 智能体模式：把「计划 → 每步结果」画出来，这是演示的主角
    if (s.agent_text) {
      const plan = s.agent_plan
        ? s.agent_plan.steps.map((x, i) => (i + 1) + '. ' + x.tool).join('　')
        : '（大模型正在拆解…）';
      let html = '<b>你说</b>：' + s.agent_text + '<br><b>计划</b>：' + plan;
      const tr = (s.agent_trace || []).slice(-6);
      if (tr.length) {
        html += '<br><b>执行</b>：';
        for (const row of tr) {
          const r = row.result || {};
          const ok = r.ok === true || (r.error === undefined && r.ok === undefined);
          const bad = r.error || r.ok === false;
          const detail = r.dist_mm !== undefined ? (' ' + r.dist_mm + ' mm')
                        : (r.error ? (' ' + r.error) : '');
          const mark = bad ? '❌' : (r.ok === true ? '✅' : '·');
          html += '<br>　' + mark + ' ' + row.tool + detail;
        }
      }
      buildEl.innerHTML = html;
    }
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
        if path == "/api/agent":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {}
            result = self.sim.start_agent(str(payload.get("text", "")))
            self._send(json.dumps(result, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
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


def port_in_use(port: int) -> bool:
    """端口上是否已经有人在服务。

    Windows 的 SO_REUSEADDR 允许两个进程绑同一个端口（Python 的
    ThreadingHTTPServer 默认开着），光靠 bind 失败判断不出来，会变成两个
    服务抢连接。所以先连一下，连得上就说明演示已经在跑。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true",
                    help="启动后自动用默认浏览器打开页面（双击 启动网页.cmd 用）")
    ap.add_argument("--policy", type=Path, default=None,
                    help="指向 .../pretrained_model 目录：让网页跑训练好的 ACT，"
                         "而不是逆解轨迹。不传就用逆解。")
    ap.add_argument("--agent", action="store_true",
                    help="智能体模式：网页里每一句话先让大模型拆成工具计划，执行时逐步记录，"
                         "失败会把真实结果回传给大模型重新出方案。需要同时给 --policy。")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"

    if port_in_use(args.port):
        # 最常见的原因：演示已经开着了，又双击了一次启动器
        print(f"端口 {args.port} 已经有服务在跑，不再重复启动。")
        print(f"直接打开 {url} 即可。")
        if args.open:
            webbrowser.open(url)
        return 0

    sim = Sim()
    if args.policy is not None:
        if not (args.policy / "config.json").exists():
            print(f"找不到策略：{args.policy}")
            return 1
        sim.load_policy(args.policy)
        print(f"策略模式：{args.policy}")
    if args.agent:
        if args.policy is None:
            print("``--agent`` 需要同时给 ``--policy``（智能体靠策略动手）")
            return 1
        sim.agent_enabled = True
        print("智能体模式已开：网页会走「大模型拆计划 → 执行 → 失败回传重规划」")
    threading.Thread(target=sim.loop, daemon=True).start()
    Handler.sim = sim

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        # 端口被占最常见的原因：演示已经开着了，再双击了一次启动器
        print(f"端口 {args.port} 已被占用：{exc}")
        print(f"如果演示已经在运行，直接打开 {url} 即可。")
        if args.open:
            webbrowser.open(url)
        return 1

    print(f"本地 Demo 已启动： {url}")
    print("按 Ctrl+C 停止")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sim._running = False
        print("\n已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
