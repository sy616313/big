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

from sim.env import (  # noqa: E402
    CUBE_REST_Z,
    HOME_XYZ,
    PLATE_Z,
    SUCCESS_TOL_XY,
    UPRIGHT_MIN,
    sample_scene,
)
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
        self.policy_alternates: list = []   # 主策略失败后依次换的备用策略
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

        # 方块是自由体：记住它的 qpos/qvel 地址，换场景时直接改
        self.cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_qadr = int(self.model.jnt_qposadr[self.model.body_jntadr[self.cube_body]])
        self.cube_vadr = int(self.model.jnt_dofadr[self.model.body_jntadr[self.cube_body]])
        self.scene = {
            "cube": tuple(float(v) for v in self.data.qpos[self.cube_qadr:self.cube_qadr + 2]),
            "plate": tuple(float(v) for v in self.data.body("plate").xpos[:2]),
        }
        self.scene_count = 0

        # 网页控制：暂停 / 模式（智能体 or 手动）/ 手动点动
        self.paused = False
        self.pause_gate = threading.Event()
        self.pause_gate.set()                 # set = 正在跑；clear = 冻结
        self.mode = "agent" if self.agent_enabled else "auto"
        self.agent_abort = False
        self.manual_target = self.tip().copy()
        self.manual_grip = float(GRIP_OPEN[0])

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
    def load_policy(self, ckpts, retry_with=None, relative=False) -> None:
        """加载训练好的策略：给一个就单模型，给多个就集成（动作取平均）。

        集成逻辑走 sim/ensemble.py —— 和 scripts/eval_ensemble.py 是同一份代码，
        所以网页里跑出来的就是评估里测过的那个东西，不会出现「网页 76%、报告 84%」。
        """
        from sim.ensemble import EnsemblePolicy

        if isinstance(ckpts, (str, Path)):
            ckpts = [ckpts]
        ens = EnsemblePolicy.from_paths(ckpts, relative=relative)
        # 主策略失败后依次换的备用策略。可以是别的模型（--retry-with），
        # 不能只是"同一个模型再来一次"——策略是确定性的，同一局重放等于重放，白跑。
        alternates = []
        for p in (retry_with or []):
            others = EnsemblePolicy.from_paths([p])
            alternates.append(EnsemblePolicy([others.members[0]]))
        with self.lock:
            self.policy = ens
            self.policy_alternates = alternates or ([EnsemblePolicy([m]) for m in ens.members]
                                                    if len(ens.members) > 1 else [])
            self.policy_pre = None            # 每个成员自己的归一化在 act() 里做
            self.policy_post = None
            self.policy_device = ens.device
            self.policy_queue = []
            self.policy_consumed = 0
            self.policy_steps = 0
            tag = " + ".join(ens.names)
            extra = ("（备用：" + "、".join("+".join(a.names) for a in self.policy_alternates) + "）"
                     if self.policy_alternates else "")
            self.message = f"策略已加载：{tag}" + ("（集成）" if len(ens.members) > 1 else "") + extra

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
        """取一整块动作放进队列。连调 N 次只算一次前向（靠策略内部的队列）。"""
        batch = self._policy_batch(renderer)      # 原始 batch，归一化交给各成员自己
        n = int(getattr(self.policy.config, "n_action_steps", 40))
        acts = self.policy.act(batch, n)
        # 换块处做两步过渡，否则每 0.42 秒目标跳一下，看着像抽搐
        self.policy_queue = [np.asarray(a, dtype=np.float64)
                             for a in crossfade(getattr(self, "_last_act", None), acts)]
        self.policy_consumed = 0

    def _policy_done(self) -> bool:
        """方块真的落在盘子中间、放平了、夹爪也松开了——和 sim/env.py 的判据一致。

        阈值 37 mm 是"方块整个落在盘面上"的几何上限（盘半径 58 - 方块半宽 21）。
        用 45 mm 会出现一半悬在盘沿外还判成功的情况，演示时一眼就能看出来。
        """
        c = np.array(self.body_xyz("cube"))
        p = np.array(self.body_xyz("plate"))
        grip = float(self.data.qpos[self.arm_qadr[4]])
        if float(np.hypot(c[0] - p[0], c[1] - p[1])) > SUCCESS_TOL_XY:
            return False
        if not (0.020 <= c[2] <= 0.045) or grip <= 0.008:
            return False
        rot = np.asarray(self.data.body("cube").xmat, dtype=float).reshape(3, 3)
        # 立方体绕 x/y 转 90° 依然是"平放在某个面上"，三个局部轴里有一个竖直就行
        return float(np.max(np.abs(rot[2, :]))) >= UPRIGHT_MIN

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
        self.pause_gate.wait()               # 暂停时卡在这里，物理不动
        if self.agent_abort:
            from src.agent import Aborted
            raise Aborted("用户切到手动模式")
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
            if self.mode == "manual":
                return {"ok": False, "error": "现在是手动模式，先切回智能体模式"}
            if self.agent_state == "running":
                return {"ok": False, "error": "上一条指令还在执行"}
            if self.paused:                   # 暂停状态下开会一直卡在 step() 里
                self.paused = False
                self.pause_gate.set()
            self.agent_state = "running"
            self.agent_abort = False
            self.agent_text = text
            self.agent_trace = []
            self.agent_plan = None
            self.agent_result = None
            self.message = "大模型正在拆解指令…"
        threading.Thread(target=self._agent_worker, args=(text,), daemon=True).start()
        return {"ok": True}

    def _agent_worker(self, text: str) -> None:
        from src.agent import Aborted, Executor

        try:
            ex = Executor(self, self.policy, self.policy_pre, self.policy_post,
                          self.policy_device, alternates=self.policy_alternates)
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
            result = ex.run_until_done(text, max_turns=3, verbose=False)
            with self.lock:
                self.agent_result = result
                self.message = "智能体完成：成功" if result.get("ok") else "智能体完成：没做成"
        except Aborted:
            with self.lock:
                self.message = "已切到手动模式，智能体的动作中止了"
                self.agent_result = {"ok": False, "aborted": True}
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


    def set_message(self, text: str) -> None:
        """给执行器用的进展播报（网页状态条）。"""
        with self.lock:
            self.message = text

    def body_xyz(self, name: str) -> list[float]:
        return [float(v) for v in self.data.body(name).xpos]

    # ---------- 场景 / 暂停 / 模式 ----------
    def _apply_scene(self) -> None:
        """按 self.scene 里的坐标摆盘子和方块（方块是自由体，直接写 qpos）。"""
        px, py = self.scene["plate"]
        cx, cy = self.scene["cube"]
        self.data.mocap_pos[self.plate_mid] = [px, py, PLATE_Z]
        self.data.qpos[self.cube_qadr:self.cube_qadr + 7] = [cx, cy, CUBE_REST_Z + 0.003,
                                                             1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.cube_vadr:self.cube_vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def new_scene(self, seed: int | None = None) -> dict:
        """换一套随机的方块/盘子位置。

        采样的规则和训练数据完全一致（sim.env.sample_scene），所以演示里随机出来的场景
        和训练/评估是同一个分布——评委看到的不是预先摆好的位置。
        """
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        cube, plate = sample_scene(rng)
        with self.lock:
            self.scene = {"cube": tuple(cube), "plate": tuple(plate)}
            self.do_reset()
            self.scene_count += 1
            self.message = (f"新场景 #{self.scene_count}：方块 ({cube[0]:.2f}, {cube[1]:.2f})"
                            f"　盘子 ({plate[0]:.2f}, {plate[1]:.2f})")
            return {"ok": True, "cube": [round(v, 3) for v in cube],
                    "plate": [round(v, 3) for v in plate], "scene": self.scene_count}

    def set_paused(self, paused: bool) -> dict:
        """暂停/继续：暂停时物理不动、智能体线程也会卡在 step() 里不动。"""
        with self.lock:
            self.paused = bool(paused)
            if self.paused:
                self.pause_gate.clear()
                self.message = "已暂停（物理冻结，可以切模式）"
            else:
                # 暂停期间手臂可能被别的线程动过（比如智能体刚被中止），
                # 继续之前把下发点重新对齐到真实位置，避免恢复的那一帧跳一下
                tip = self.tip().copy()
                self._cmd_p = tip
                if self.mode == "manual":
                    self.manual_target = tip
                self.pause_gate.set()
                self.message = "继续"
            return {"ok": True, "paused": self.paused}

    def set_mode(self, mode: str) -> dict:
        """切换智能体模式 / 手动模式。切到手动时会把还在跑的智能体中止掉。"""
        if mode not in ("agent", "manual", "auto"):
            return {"ok": False, "error": f"不认识的模式：{mode}"}
        with self.lock:
            if mode == "manual" and self.agent_state == "running":
                self.agent_abort = True      # 让智能体线程在下一步抛 Aborted 退出
            self.mode = mode
            if mode == "manual":
                self.plan = None
                self.policy_queue = []
                self.policy_consumed = 0
                self.policy_steps = 0
                self.waypoints = []
                self.wp_index = 0
                # 关键：把"当前下发点"重新锚到机械臂真实位置，并沿用当前夹爪开度。
                # 不锚的话 _cmd_p 还停在进智能体模式之前的位置，手动模式第一帧就会
                # 让手臂瞬移过去——手里夹着的方块直接被甩飞（实测踩到过）。
                tip = self.tip().copy()
                self.manual_target = tip
                self._cmd_p = tip
                # 沿用当前**指令**上的夹爪开度，不是量出来的指头位置：
                # 夹着方块时指头被撑开在 0.011 左右，而指令是 0.004（一直捏着）。
                # 拿测量值当指令会把指头松开 7 mm，方块直接掉下去（实测踩到过）。
                self.manual_grip = float(np.clip(self.data.ctrl[5], 0.0, GRIP_OPEN[0]))
                self.message = ("手动模式" + ("（还暂停着，点「继续」才能动）" if self.paused
                                             else "：方向键移动，空格开合爪子"))
            elif mode == "agent":
                self.message = ("智能体模式" + ("（还暂停着，点「继续」才会动）" if self.paused
                                               else "：打一句话让它自己干"))
            return {"ok": True, "mode": self.mode}

    def jog(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
        """手动点动：把目标点挪一点，主循环按 0.1 m/s 限速走过去。"""
        with self.lock:
            t = self.manual_target + np.array([dx, dy, dz], dtype=float)
            # 夹在够得着的范围里，否则逆解一直报错
            t[0] = float(np.clip(t[0], 0.10, 0.30))
            t[1] = float(np.clip(t[1], -0.22, 0.22))
            t[2] = float(np.clip(t[2], 0.006, 0.26))
            self.manual_target = t
            return {"ok": True, "target": [round(float(v), 3) for v in t]}

    def set_grip(self, open_: bool) -> dict:
        with self.lock:
            self.manual_grip = float(GRIP_OPEN[0]) if open_ else 0.0
            return {"ok": True, "grip": self.manual_grip}

    # ---------- 指令 ----------
    def command(self, text: str) -> dict:
        if self.mode == "manual":
            return {"ok": False, "error": "现在是手动模式，先切回智能体模式再下指令"}
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
        """把场景恢复初始状态，方便反复演示（HTTP 的 /api/reset 和智能体的 home 都走这里）。

        注意：会重新摆回 self.scene 里记着的那一组坐标，所以"重置"是同一局重来，
        "换场景"才会抽新的方块/盘子位置。
        """
        with self.lock:
            self.data = mujoco.MjData(self.model)
            self._apply_scene()               # 新 MjData 会把方块和盘子放回模型默认位置
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
            self.manual_target = self.tip().copy()

    def _manual_step(self, per_frame: int) -> None:
        """手动模式：按 0.1 m/s 朝点动目标走，夹爪用 self.manual_grip。"""
        dt = per_frame * self.model.opt.timestep
        tip = self.tip()
        # 手臂被外部动过（智能体中止、暂停期间被推）就跟丢了下发点，先重新对齐，
        # 否则这一步会算出"从旧下发点飞向目标"的大跳变，夹着的东西会被甩出去
        if float(np.linalg.norm(tip - self._cmd_p)) > 0.03:
            self._cmd_p = tip.copy()
            self.manual_target = tip.copy()
        delta = self.manual_target - self._cmd_p
        dist = float(np.linalg.norm(delta))
        if dist > 1e-9:
            self._cmd_p = self._cmd_p + delta * min(1.0, 0.10 * dt / dist)
        try:
            q = ik(self._cmd_p[0], self._cmd_p[1], self._cmd_p[2])
        except Unreachable:
            self.manual_target = self._cmd_p.copy()   # 别再往外走了，停在够得着的边上
            return
        g = float(self.manual_grip)
        self.ctrl_target = np.array([*q, 0.0, g, -g])
        self.data.ctrl[:] = self.ctrl_target

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
        # 只用一个 Renderer。试过"主画面 800x600 + 策略画面 320x240 各一个"，
        # 跑几局之后 mujoco.dll 直接访问违例崩掉（0xc0000005，GL 上下文多实例共享不稳）。
        # 稳定性优先：一个渲染器，出两帧（主画面 + 降到策略尺寸的帧）。
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
                if self.paused:
                    pass          # 冻结：什么都不推进，只把当前画面继续推给浏览器
                elif agent_busy:
                    pass          # 物理由智能体线程推进（step() 里加锁自走），这里只渲染
                elif self.mode == "manual":
                    self._manual_step(per_frame)
                elif policy_mode:
                    # 策略接管：每 POLICY_EXEC 步重新预测一次，其余步执行队列里的动作
                    if not self.policy_queue or self.policy_consumed >= POLICY_EXEC:
                        self._query_policy(renderer)
                    act = self.policy_queue.pop(0) if self.policy_queue else HOME_CTRL[0:5]
                    self._last_act = act
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

            if not agent_busy and not self.paused:
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
            # 用 BOX（面积平均）缩图：比双线性更接近"原生低分辨率渲染"的样子，
            # 减少喂给策略的画面和训练分布的差别。
            renderer.update_scene(self.data, camera=POLICY_CAMERA)
            policy_frame = np.asarray(Image.fromarray(renderer.render()).resize(
                POLICY_IMG, Image.BOX)).copy()
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
                "plate": [round(v, 3) for v in self.body_xyz("plate")],
                "tip": [round(float(v), 3) for v in self.tip()],
                "mode": self.mode,
                "paused": self.paused,
                "grip": round(float(np.clip(self.data.qpos[self.arm_qadr[4]], 0.0,
                                            GRIP_OPEN[0])), 4),
                "scene": self.scene_count,
                "success": bool(self._policy_done()),
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
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>轻量具身智能体 · 本地 Demo</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0f1419; color:#e6edf3;
         font-family:"Microsoft YaHei",system-ui,sans-serif; }
  .app { max-width:940px; margin:20px auto 44px; padding:0 16px; }
  header { display:flex; align-items:baseline; gap:10px; margin-bottom:12px; flex-wrap:wrap; }
  h1 { font-size:18px; margin:0; font-weight:600; }
  header .sub { color:#8b949e; font-size:12.5px; }
  .stage { position:relative; border-radius:12px; overflow:hidden; background:#000;
           box-shadow:0 10px 30px rgba(0,0,0,.45); cursor:grab; }
  .stage.dragging { cursor:grabbing; }
  .stage img { display:block; width:100%; }
  .chip { position:absolute; padding:5px 11px; border-radius:999px; font-size:12px;
          background:rgba(13,17,23,.72); }
  #live { left:12px; top:12px; color:#7ee787; letter-spacing:.5px; }
  #live.off { color:#f0883e; }
  #status { left:12px; bottom:12px; max-width:74%; }
  #meta { right:12px; bottom:12px; color:#9aa4b2; font-family:Consolas,monospace; font-size:11.5px; }
  .bar { display:flex; gap:8px; align-items:center; margin-top:14px; flex-wrap:wrap; }
  .seg { display:flex; background:#161b22; border:1px solid #30363d; border-radius:9px; overflow:hidden; }
  .seg button { background:transparent; border:0; color:#8b949e; padding:10px 16px;
                font-size:14px; cursor:pointer; font-weight:600; }
  .seg button.on { background:#1f6feb; color:#fff; }
  .bar .sp { flex:1; }
  .alt { background:#21262d; border:1px solid #30363d; color:#e6edf3; border-radius:9px;
         padding:10px 14px; font-size:14px; cursor:pointer; }
  .alt:hover { background:#30363d; }
  form { display:flex; gap:8px; margin-top:12px; }
  input { flex:1; padding:12px 14px; border-radius:9px; border:1px solid #30363d;
          background:#161b22; color:#e6edf3; font-size:15px; }
  input:focus { outline:2px solid #1f6feb; outline-offset:-1px; }
  .primary { padding:12px 22px; border-radius:9px; border:0; background:#2f6feb;
             color:#fff; font-size:15px; font-weight:600; cursor:pointer; }
  .primary:disabled { opacity:.45; cursor:default; }
  .panel { margin-top:12px; background:#161b22; border:1px solid #21262d;
           border-radius:11px; padding:14px 16px; }
  .tl { font-size:13.5px; line-height:1.85; white-space:pre-wrap; color:#c9d1d9; }
  .manual { display:flex; gap:34px; align-items:center; justify-content:center; flex-wrap:wrap; }
  .pad { display:grid; grid-template-columns:repeat(3,60px); grid-template-rows:repeat(3,46px); gap:8px; }
  .pad button, .vpad button { border-radius:10px; border:1px solid #30363d; background:#21262d;
                              color:#e6edf3; font-size:13px; cursor:pointer; user-select:none; }
  .pad button:active, .vpad button:active { background:#1f6feb; }
  .vpad { display:flex; flex-direction:column; gap:10px; }
  .vpad button { padding:12px 18px; }
  .grip { padding:14px 18px; border-radius:11px; border:1px solid #30363d; background:#21262d;
          color:#e6edf3; font-size:14px; cursor:pointer; min-width:132px; }
  .grip.open { background:#1a7f37; border-color:#2ea043; }
  .hint { color:#6e7681; font-size:12px; text-align:center; margin-top:12px; }
  footer { margin-top:14px; color:#6e7681; font-size:11.5px; font-family:Consolas,monospace;
           display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }
</style></head>
<body><div class="app">

<header>
  <h1>轻量具身智能体</h1>
  <span class="sub">六轴机械臂 · 红方块 · 蓝盘子 · 本地实时仿真（不是录屏）</span>
</header>

<div class="stage" id="stage">
  <img id="sim" src="/stream" alt="sim">
  <div class="chip" id="live">LIVE</div>
  <div class="chip" id="status">正在连接…</div>
  <div class="chip" id="meta">—</div>
</div>

<div class="bar">
  <div class="seg" id="modes">
    <button data-mode="agent" class="on">智能体模式</button>
    <button data-mode="manual">手动模式</button>
  </div>
  <button class="alt" id="pause">暂停</button>
  <span class="sp"></span>
  <button class="alt" id="scene">换个场景</button>
  <button class="alt" id="reset">同一局重来</button>
</div>

<section id="panel-agent">
  <form id="f">
    <input id="q" autocomplete="off" placeholder="说一句话，比如：把红色方块放到盘子上">
    <button class="primary" id="run">执行</button>
  </form>
  <div class="panel"><div class="tl" id="tl">说一句话，大模型会把它拆成「看一眼 → 挪盘子 → 抓放」，每一步的真实结果都显示在这里。</div></div>
</section>

<section id="panel-manual" hidden>
  <div class="panel">
    <div class="manual">
      <div class="pad">
        <span></span><button data-k="x+">远 X+</button><span></span>
        <button data-k="y-">左 Y−</button><span></span><button data-k="y+">右 Y+</button>
        <span></span><button data-k="x-">近 X−</button><span></span>
      </div>
      <div class="vpad">
        <button data-k="z+">升 Z+</button>
        <button class="grip open" id="grip">爪子：张开</button>
        <button data-k="z-">降 Z−</button>
      </div>
    </div>
    <div class="hint">方向键 = 前后左右　PageUp / PageDown = 升降　空格 = 开合爪子（按住可以连续动）</div>
  </div>
</section>

<footer><span id="foot">—</span><span>拖动旋转 · 滚轮缩放 · 双击复位视角</span></footer>
</div>
<script>
const $ = (id) => document.getElementById(id);
const liveEl = $('live'), statusEl = $('status'), metaEl = $('meta'), footEl = $('foot');
const tlEl = $('tl'), pauseBtn = $('pause');
const panelAgent = $('panel-agent'), panelManual = $('panel-manual');
const modeBtns = Array.from(document.querySelectorAll('#modes button'));
const S = { mode: 'agent', paused: false, gripOpen: true };

async function post(path, body) {
  try {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    return await r.json();
  } catch (e) { return {}; }
}

function setMode(m) {
  S.mode = m;
  modeBtns.forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
  panelAgent.hidden = m !== 'agent';
  panelManual.hidden = m !== 'manual';
}

modeBtns.forEach((b) => b.addEventListener('click', async () => {
  setMode(b.dataset.mode);
  await post('/api/mode', { mode: b.dataset.mode });
}));

pauseBtn.addEventListener('click', async () => {
  const r = await post('/api/pause', { paused: !S.paused });
  S.paused = !!r.paused;
  pauseBtn.textContent = S.paused ? '继续' : '暂停';
  liveEl.textContent = S.paused ? '已暂停' : 'LIVE';
  liveEl.classList.toggle('off', S.paused);
});

$('scene').addEventListener('click', () => post('/api/scene', {}));
$('reset').addEventListener('click', () => post('/api/reset', {}));

$('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const t = $('q').value.trim();
  if (!t) return;
  if (S.mode !== 'agent') { setMode('agent'); await post('/api/mode', { mode: 'agent' }); }
  $('run').disabled = true;
  const r = await post('/api/agent', { text: t });
  if (!r.ok) tlEl.textContent = '没接上：' + (r.error || '未知原因');
  $('run').disabled = false;
});

// ---------- 手动：按住连续动 ----------
const STEP = 0.012;
const DIR = { 'x+': [STEP, 0, 0], 'x-': [-STEP, 0, 0], 'y+': [0, STEP, 0],
              'y-': [0, -STEP, 0], 'z+': [0, 0, STEP], 'z-': [0, 0, -STEP] };
const KEY2DIR = { ArrowUp: 'x+', ArrowDown: 'x-', ArrowLeft: 'y-', ArrowRight: 'y+',
                  PageUp: 'z+', PageDown: 'z-' };
const HELD = new Set();

document.querySelectorAll('[data-k]').forEach((b) => {
  const k = b.dataset.k;
  b.addEventListener('pointerdown', (e) => { e.preventDefault(); HELD.add(k); });
  ['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) =>
    b.addEventListener(ev, () => HELD.delete(k)));
});

setInterval(() => {
  if (!HELD.size || S.mode !== 'manual' || S.paused) return;
  let dx = 0, dy = 0, dz = 0;
  for (const k of HELD) { const d = DIR[k]; if (d) { dx += d[0]; dy += d[1]; dz += d[2]; } }
  post('/api/jog', { dx, dy, dz });
}, 90);

let lastGripClick = 0;
function setGripLabel(open) {
  S.gripOpen = open;
  const b = $('grip');
  b.textContent = '爪子：' + (open ? '张开' : '合上');
  b.classList.toggle('open', open);
}

$('grip').addEventListener('click', async () => {
  lastGripClick = Date.now();
  setGripLabel(!S.gripOpen);
  await post('/api/grip', { open: S.gripOpen });
});

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === ' ') { e.preventDefault(); $('grip').click(); return; }
  if (e.key === 'p' || e.key === 'P') { pauseBtn.click(); return; }
  const d = KEY2DIR[e.key];
  if (d && S.mode === 'manual') { e.preventDefault(); HELD.add(d); }
});
window.addEventListener('keyup', (e) => { const d = KEY2DIR[e.key]; if (d) HELD.delete(d); });

// ---------- 自由视角 ----------
const stage = $('stage');
let drag = null;
const pending = { daz: 0, del: 0, ddist: 0, dx: 0, dy: 0 };
let flushTimer = null;
function flushCamera() {
  flushTimer = null;
  const p = Object.assign({}, pending);
  pending.daz = pending.del = pending.ddist = pending.dx = pending.dy = 0;
  fetch('/api/camera', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                         body: JSON.stringify(p) }).catch(() => {});
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
window.addEventListener('mouseup', () => { drag = null; stage.classList.remove('dragging'); });
window.addEventListener('mousemove', (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.btn === 0 && !drag.shift) queueCamera({ daz: dx * 0.35, del: -dy * 0.35 });
  else queueCamera({ dx: dx, dy: dy });
});
stage.addEventListener('wheel', (e) => { e.preventDefault(); queueCamera({ ddist: e.deltaY * 0.0012 }); },
                      { passive: false });
stage.addEventListener('contextmenu', (e) => e.preventDefault());
stage.addEventListener('dblclick', () => fetch('/api/view_reset', { method: 'POST' }));

// ---------- 状态轮询 ----------
function renderTimeline(s) {
  let out = ['你说：' + s.agent_text];
  out.push('计划：' + (s.agent_plan
    ? s.agent_plan.steps.map((x, i) => (i + 1) + '. ' + x.tool).join('　')
    : '（大模型正在拆解…）'));
  const tr = (s.agent_trace || []).slice(-6);
  if (tr.length) {
    out.push('执行：');
    for (const row of tr) {
      const r = row.result || {};
      const bad = r.error || r.ok === false;
      const mark = bad ? '✗' : (r.ok === true ? '✓' : '·');
      let note = '';
      if (r.error) note = '　' + r.error;
      else if (r.dist_mm !== undefined) note = '　离盘心 ' + r.dist_mm + ' mm';
      if (r.attempts && r.attempts > 1) note += '（第 ' + r.attempts + ' 次尝试）';
      out.push('　' + mark + ' ' + row.tool + note);
    }
  }
  tlEl.textContent = out.join('\\n');
}

async function poll() {
  try {
    const s = await (await fetch('/api/status')).json();
    S.paused = !!s.paused;
    statusEl.textContent = s.message;
    metaEl.textContent = '方块 ' + s.cube[0].toFixed(2) + ',' + s.cube[1].toFixed(2) +
      '　盘子 ' + s.plate[0].toFixed(2) + ',' + s.plate[1].toFixed(2) +
      (s.scene ? '　场景 #' + s.scene : '');
    liveEl.textContent = s.paused ? '已暂停' : 'LIVE';
    liveEl.classList.toggle('off', s.paused);
    pauseBtn.textContent = s.paused ? '继续' : '暂停';
    footEl.textContent = s.frames + ' 帧　' + Math.round(s.uptime) + 's';
    setMode(s.mode === 'manual' ? 'manual' : 'agent');
    if (S.mode === 'manual' && s.grip !== undefined && Date.now() - lastGripClick > 1500) {
      // 从别的模式切过来时，爪子的真实开度可能和按钮显示的不一样（比如正夹着方块），
      // 以服务器为准同步一下，省得第一次点按钮反而把东西放开。
      // 阈值 0.015：夹着方块时指头被撑开在 0.010~0.012，完全张开是 0.018
      const open = s.grip > 0.015;
      if (open !== S.gripOpen) setGripLabel(open);
    }
    if (S.mode === 'agent' && s.agent_text) renderTimeline(s);
  } catch (e) {
    liveEl.textContent = '连接断开';
    liveEl.classList.add('off');
  }
}
setInterval(poll, 700);
poll();
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

        def body() -> dict:
            length = int(self.headers.get("Content-Length", 0))
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}

        def reply(obj) -> None:
            self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        if path == "/api/pause":
            payload = body()
            reply(self.sim.set_paused(bool(payload.get("paused", True))))
            return
        if path == "/api/mode":
            reply(self.sim.set_mode(str(body().get("mode", "agent"))))
            return
        if path == "/api/scene":
            seed = body().get("seed")
            reply(self.sim.new_scene(int(seed) if seed is not None else None))
            return
        if path == "/api/jog":
            payload = body()
            reply(self.sim.jog(float(payload.get("dx", 0.0)), float(payload.get("dy", 0.0)),
                               float(payload.get("dz", 0.0))))
            return
        if path == "/api/grip":
            reply(self.sim.set_grip(bool(body().get("open", True))))
            return
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
    ap.add_argument("--policy", type=Path, default=None, nargs="+",
                    help="指向 .../pretrained_model 目录：让网页跑训练好的 ACT，"
                         "而不是逆解轨迹。不传就用逆解。"
                         "给两个以上目录就是集成（每个模型各自预测，动作取平均）。")
    ap.add_argument("--relative", action="store_true",
                    help="主策略是「相对关节增量」训出来的（data\\human_fix_rel），推理时要把当前关节角加回去")
    ap.add_argument("--retry-with", type=Path, nargs="+", default=None,
                    help="主策略失败后依次换的单模型（原地重试）。注意必须是**不同**的模型："
                         "策略是确定性的，同一个模型重放一遍结果完全一样，白等")
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
        missing = [str(p) for p in args.policy if not (p / "config.json").exists()]
        if missing:
            print("找不到策略：\n  " + "\n  ".join(missing))
            return 1
        sim.load_policy(args.policy, retry_with=args.retry_with, relative=args.relative)
        print("策略模式：" + " + ".join(str(p) for p in args.policy))
    if args.agent:
        if args.policy is None:
            print("``--agent`` 需要同时给 ``--policy``（智能体靠策略动手）")
            return 1
        sim.agent_enabled = True
        sim.mode = "agent"
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
