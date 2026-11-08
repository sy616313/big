r"""遥操作采集：人手动把方块抓起来放到盘子上，过程录成示范数据。

跑法：
    双击项目根目录的「遥操作.cmd」，或者：
    & "D:\envs\embodied\python.exe" scripts\teleop.py

    停掉再重开**会接着已录的往下追加**，不会丢也不会拦你。
    想清空重来（比如换了物理参数、要从头录）：加 --force

键盘（按住不放会连续动，因为窗口会收到系统按键重复）：
    W / S       前 / 后（+x / -x）      方向键也可以
    A / D       左 / 右（+y / -y）
    Q / E       上 / 下（+z / -z）
    Tab         切换精细档（粗 5 mm / 细 1.5 mm），状态栏会显示
    空格        开 / 合夹爪
    回车        结束本局：**成功了才存**，没成功会提示并丢弃
    退格        放弃本局（做砸了就用这个，绝不会写进数据集）
    R           视角回正
    F11         全屏 / 退出全屏
    Esc         退出

    鼠标：左键拖动 = 绕场景转视角，滚轮 = 拉近/推远。
          抓手被挡住或者想从侧面确认对位时很有用。

    每条轨迹默认最多 60 秒（1200 步），到点会自动结算；嫌短可以加
    --max-steps 0 完全不限。

    注意：精细档不用 Shift。中文输入法下按住 Shift 会一直切输入法，
    操作会乱。Tab 是「按一下切一次」的开关，不需要一直按住。

和 collect.py 的关系：
    collect.py 用「会算逆解的专家」自动生成示范；本脚本让你用键盘自己开。
    两者写出的数据格式完全一样（字段都来自 sim.env.dataset_features），
    所以 train_act.py 不用改，直接能训。

为什么自己开窗口、不用 MuJoCo 自带的视窗：
    自带视窗有一堆快捷键：W 会把画面切成线框、鼠标拖动会改视角、还能改各种
    显示模式。这些和我们的操作键直接冲突（实测按 W 就把画面变线框了）。
    换成 Tkinter 之后键盘完全归我们处理，没有任何隐藏快捷键。

为什么让你控制末端坐标、而不是直接给关节角：
    人很难同时协调 4 个关节。这里你只管夹爪中心的 (x, y, z)，四个关节角由闭式
    逆解自动算出来，屏幕上看得到什么就能直接对应上。

为什么只有成功的示范才存：
    模仿学习是"照着做"。把没抓起来的、抓掉了的轨迹喂进去，等于教它做砸。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

from sim.env import (  # noqa: E402
    FPS,
    GRIP_CLOSED,
    GRIP_OPEN,
    MAX_EPISODE_STEPS,
    TASK,
    PickPlaceEnv,
    dataset_features,
)
from sim.kinematics import Unreachable, ik  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "human_v1"
SEED_BASE = 20_000          # 和训练用的 0~299、评估用的 10000 起都错开
KEY_STEP = 0.005            # 粗档：按一次 5 mm
STEP_FINE = 0.0015          # 细档：按一次 1.5 mm
MAX_KEYS_PER_TICK = 4       # 每个控制周期最多处理几次按键（防止重复率太高飞出去）
TAIL_HOLD = 10              # 按回车结算前，张爪保持多少帧（0.5 秒）再写盘

DISP_W, DISP_H = 1280, 960  # 显示渲染尺寸：只在启动时建一次，之后按窗口大小缩放
# 注意：不要在运行中重建 Renderer。实测同时存在两个离屏 GL 上下文时，
# 新建的那个会渲染出全黑（窗口拉大后画面直接黑掉）。宁可多花几毫秒缩放。

# 每条轨迹的步数上限。20 Hz 下 1200 步 = 60 秒。
# 原来定的 160 步（8 秒）是按专家节奏算的，人操作根本不够——找位置、对位、
# 犹豫都要时间。0 表示完全不限。
MAX_STEPS_DEFAULT = 1200

# 显示相机。必须显式设置：MuJoCo 默认会把镜头放在整个模型外面，
# 而我们的场景有一整间白墙房间，默认视角看到的是一片白。
VIEW = {"azimuth": 138.0, "elevation": -36.0, "distance": 0.88,
        "lookat": [0.18, -0.03, 0.07]}

MOVE_KEYS = {
    "w": (1, 0, 0), "s": (-1, 0, 0), "Up": (1, 0, 0), "Down": (-1, 0, 0),
    "a": (0, 1, 0), "d": (0, -1, 0), "Left": (0, 1, 0), "Right": (0, -1, 0),
    "q": (0, 0, 1), "e": (0, 0, -1),
}


def axis_of(keysym: str) -> tuple[int, int, int] | None:
    """把按键名翻成移动方向。

    注意别只查小写：字母键的 keysym 是 'w'，方向键是 'Up'（大写 U）。
    统一小写会把方向键全部漏掉——这个 bug 我在自检里专门盯了一条。
    """
    return MOVE_KEYS.get(keysym) or MOVE_KEYS.get(keysym.lower())


def describe(env: PickPlaceEnv, grip_open: bool, fine: bool) -> str:
    tip = env.tip()
    c = env.cube_xyz()
    p = env.plate_xyz()
    return (f"爪 ({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f})   "
            f"离方块 {np.hypot(*(c[:2] - tip[:2])) * 1000:3.0f} mm   "
            f"离盘子 {np.hypot(*(p[:2] - tip[:2])) * 1000:3.0f} mm   "
            f"夹爪{'张开' if grip_open else '合上'}   "
            f"{'精细 1.5mm' if fine else '粗调 5mm'}   步 {env.steps:3d}")


class Recorder:
    """把一局的帧攒起来，成功后一次性写入数据集。"""

    def __init__(self, dataset, env: PickPlaceEnv) -> None:
        self.ds = dataset
        self.env = env
        self.frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add(self, obs: dict, action: np.ndarray) -> None:
        self.frames.append((obs["image"],
                            np.asarray(obs["state"], dtype=np.float32),
                            np.asarray(action, dtype=np.float32)))

    def commit(self) -> tuple[bool, int]:
        good = self.env.success()
        if not good:
            self.frames.clear()
            return False, 0
        # 去掉"连续重复动作"的帧。
        # 遥操作只在按键时才改目标，两次按键之间动作完全一样。实测人类示范里
        # 有 83% 的帧是重复的，模型于是学成"什么都别做"（真发生过，闭环 0/20）。
        # 这里只保留动作真正变化的那一帧，把"保持不动"压掉。
        kept: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        prev: np.ndarray | None = None
        for img, st, act in self.frames:
            if prev is None or not np.allclose(act, prev, atol=1e-6):
                kept.append((img, st, act))
                prev = act
        # 但「松手」这一段不能压：按回车时张爪往往只录到一帧，模型学不到
        # “松开后保持张开”，会把方块一直夹着（实测 20 条里 4 条这样）。
        # 这里把最后 TAIL_HOLD 帧原样保留；finish() 会先张爪走满这些帧。
        tail = self.frames[-TAIL_HOLD:] if self.frames else []
        if len(tail) < TAIL_HOLD or any(abs(a[4] - GRIP_OPEN) > 1e-6 for _, _, a in tail):
            tail = []          # 结尾不是连续张爪，就别补（自检/异常路径）
        for img, st, act in kept:
            self.ds.add_frame({"observation.images.front": img,
                               "observation.state": st,
                               "action": act,
                               "task": TASK})
        # tail[0] 就是 kept[-1]（张爪那帧），已经写过，跳过避免重复。
        for img, st, act in tail[1:]:
            self.ds.add_frame({"observation.images.front": img,
                               "observation.state": st,
                               "action": act,
                               "task": TASK})
        self.ds.save_episode()
        n = len(kept) + max(0, len(tail) - 1)
        self.frames.clear()
        return True, n


def hold_gripper_open(env: PickPlaceEnv, rec: Recorder, obs: dict,
                      prev_q: np.ndarray, n: int = TAIL_HOLD) -> dict:
    """张爪并保持 n 帧，返回最后一帧 obs。收尾和自检共用。"""
    action = np.array([*prev_q, GRIP_OPEN], dtype=np.float64)
    for _ in range(n):
        rec.add(obs, action)
        obs = env.step(action)
    return obs


# ------------------------------------------------------------------ 主窗口
class TeleopWindow:
    def __init__(self, env: PickPlaceEnv, ds, out: Path,
                 max_steps: int = MAX_STEPS_DEFAULT) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self.tk = tk
        self.Image = Image
        self.ImageTk = ImageTk
        self.env = env
        self.ds = ds
        self.out = out
        self.rec = Recorder(ds, env)

        # 显示用的渲染器：分辨率比数据集高，只是给人看
        # 显示用的渲染器：启动时建一次，之后不再重建（重建会黑屏，见文件头注释）
        self.disp = mujoco.Renderer(env.model, height=DISP_H, width=DISP_W)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.azimuth = VIEW["azimuth"]
        self.cam.elevation = VIEW["elevation"]
        self.cam.distance = VIEW["distance"]
        self.cam.lookat[:] = VIEW["lookat"]

        self.root = tk.Tk()
        self.root.title("遥操作采集 —— 机械臂抓放  （F11 全屏）")
        self.root.configure(bg="#1b1f24")
        self.root.geometry("1100x900")     # 够大，也能继续拖大或按 F11 全屏
        self.root.minsize(720, 560)

        # 焦点指示灯：窗口没拿到键盘焦点时键是收不到的，与其让人猜，
        # 不如直接说清楚。第一次跑「按键没反应」基本都是这个原因。
        self.focus_bar = tk.Label(self.root, bg="#1b1f24", fg="#ffa657",
                                  font=("Microsoft YaHei", 11, "bold"), anchor="w",
                                  text="⚠ 窗口没有键盘焦点 —— 用鼠标点一下这个窗口再按键")
        self.focus_bar.pack(fill="x", padx=10, pady=(8, 0))

        # 图片放在 Canvas 里，不放 Label 里。
        # Label 会按图片大小反过来要空间，窗口一放大就形成"图变大→Label要更大→
        # 下面的文字被挤出视野"的反馈，实测最大化后底部那行参数会被切掉。
        # Canvas 的尺寸完全由布局决定，图片只是画在上面，不会反过来撑布局。
        self.canvas = tk.Canvas(self.root, bg="#1b1f24", highlightthickness=0, bd=0)
        self.canvas.pack(padx=8, pady=(6, 4), fill="both", expand=True)
        self._img_id: int | None = None
        self._drag: tuple[int, int] | None = None

        # 鼠标控制视角：左键拖动 = 绕场景转，滚轮 = 拉近/推远，R 键回正
        self.canvas.bind("<ButtonPress-1>", self._cam_drag_start)
        self.canvas.bind("<B1-Motion>", self._cam_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))
        self.canvas.bind("<MouseWheel>", self._cam_wheel)

        # 锁死高度：这两行文字每帧都在变（数字在动），不锁的话 Tk 每帧都要
        # 重新排版整个窗口，白白吃掉十几毫秒。宽度不能锁——锁了文字会被裁掉。
        self.info = tk.Label(self.root, text="", bg="#1b1f24", fg="#e6edf3",
                             font=("Consolas", 11), anchor="w", justify="left", height=2)
        self.info.pack(fill="x", padx=10)

        self.hint = tk.Label(self.root, bg="#1b1f24", fg="#8b949e",
                             font=("Microsoft YaHei", 10), anchor="w", justify="left",
                             text=("W/S 前后   A/D 左右   Q/E 上下   Tab 精细档   "
                                   "空格 开合夹爪   回车 存本局（自动保持松手）   退格 放弃   F11 全屏   Esc 退出"))
        self.hint.pack(fill="x", padx=10, pady=(0, 8))

        self.root.bind("<KeyPress>", self.on_key)
        self.root.bind("<FocusIn>", lambda _e: self._set_focus(True))
        self.root.bind("<FocusOut>", lambda _e: self._set_focus(False))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.lift()
        self.root.focus_force()

        self.ep = 0
        self.saved = 0
        self.discarded = 0
        self.fine = False
        self.grip_open = True
        self.max_steps = max_steps
        self._photo = None
        self._closed = False
        self._fullscreen = False
        self.ticks = 0                 # 累计刷新次数，用来算实际帧率
        self._tick_times = deque(maxlen=64)
        self.ep_t0 = time.time()
        self.start_episode()

    # ---------- 每局开始 ----------
    def start_episode(self) -> None:
        self.obs = self.env.reset(seed=SEED_BASE + self.ep)
        self.target = self.env.tip().copy()
        self.prev_q = self.env.state()[:4].copy()
        self.grip_open = True
        self.ep_t0 = time.time()

    # ---------- 焦点指示灯 ----------
    def _set_focus(self, has_focus: bool) -> None:
        if has_focus:
            self.focus_bar.configure(text="✔ 键盘已连接到这个窗口", fg="#7ee787")
        else:
            self.focus_bar.configure(text="⚠ 窗口没有键盘焦点 —— 用鼠标点一下这个窗口再按键",
                                     fg="#ffa657")

    # ---------- 键盘 ----------
    def on_key(self, event) -> str | None:
        key = event.keysym

        if key == "Tab":
            self.fine = not self.fine
            return "break"                      # 别让 Tab 把焦点移走
        if key == "space":
            self.grip_open = not self.grip_open
            return "break"
        if key in ("Return", "KP_Enter"):
            self.finish(save=True)
            return "break"
        if key == "BackSpace":
            self.finish(save=False)
            return "break"
        if key == "Escape":
            self.on_close()
            return "break"
        if key == "F11":
            self._fullscreen = not self._fullscreen
            self.root.attributes("-fullscreen", self._fullscreen)
            return "break"
        if key.lower() == "r":
            self._cam_reset()
            return "break"

        axis = axis_of(key)
        if axis is not None:
            step = STEP_FINE if self.fine else KEY_STEP
            self.target += np.array(axis, dtype=float) * step
        return "break"

    # ---------- 结束一局 ----------
    def finish(self, save: bool) -> None:
        if save:
            self._hold_open_before_save()
            ok, n = self.rec.commit()
            if ok:
                self.saved += 1
                print(f"[已保存第 {self.saved} 条，{n} 帧]")
            else:
                self.discarded += 1
                print("[没放成功，这条丢弃]")
        else:
            self.rec.frames.clear()
            self.discarded += 1
            print("[主动放弃本局]")
        self.ep += 1
        self.start_episode()

    def _hold_open_before_save(self) -> None:
        """按回车结算前，先张爪并保持 TAIL_HOLD 帧。

        遥操作是“按回车那一刻结束录制”，张爪往往只留下 1 帧；ACT 学不到
        “松开后继续保持张开”，于是把方块搬到盘子上却一直夹着。
        这里用同一套 env.step 真走 10 帧，让状态里夹爪真的张开。
        """
        if not self.rec.frames:
            return
        self.grip_open = True
        self.obs = hold_gripper_open(self.env, self.rec, self.obs, self.prev_q)

    # ---------- 主循环 ----------
    def tick(self) -> None:
        if self._closed:
            return
        t0 = time.perf_counter()

        # 逆解算不出来就原地不动，否则手臂会贴着工作区边界抖
        try:
            q = ik(self.target[0], self.target[1], self.target[2], ref=self.prev_q)
        except Unreachable:
            self.target = self.env.tip().copy()
            q = self.prev_q
        self.prev_q = q

        grip = GRIP_OPEN if self.grip_open else GRIP_CLOSED
        action = np.array([*q, grip], dtype=np.float64)

        self.rec.add(self.obs, action)
        self.obs = self.env.step(action)

        # 到了上限就自动结算（成功就存、不成功就丢），而不是像以前那样
        # 把机械臂冻住、只弹一行字——那看起来像卡死了。
        if self.max_steps and self.env.steps >= self.max_steps:
            print(f"[达到 {self.max_steps} 步上限，自动结算]")
            self.finish(save=True)

        # 刷新画面：固定尺寸渲染，再按画布大小等比缩放
        self.disp.update_scene(self.env.data, camera=self.cam)
        img = self.Image.fromarray(self.disp.render())
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw > 50 and ch > 50:
            s = min(cw / DISP_W, ch / DISP_H)
            size = (max(1, int(DISP_W * s)), max(1, int(DISP_H * s)))
            if size != img.size:
                img = img.resize(size, self.Image.BILINEAR)
        self._photo = self.ImageTk.PhotoImage(image=img)
        if self._img_id is None:
            self._img_id = self.canvas.create_image(cw // 2, ch // 2, image=self._photo)
        else:
            self.canvas.coords(self._img_id, cw // 2, ch // 2)
            self.canvas.itemconfig(self._img_id, image=self._photo)

        elapsed = time.time() - self.ep_t0
        limit = (f" / {self.max_steps / FPS:.0f}s 上限" if self.max_steps else "")
        self.ticks += 1
        self.info.configure(
            text=f"{describe(self.env, self.grip_open, self.fine)}   已用 {elapsed:4.1f}s{limit}\n"
                 f"已保存 {self.saved} 条   已丢弃 {self.discarded} 条   {self._fps():.0f} fps")

        rest = 1.0 / FPS - (time.perf_counter() - t0)
        self.root.after(max(1, int(rest * 1000)), self.tick)

    # ---------- 鼠标看视角 ----------
    def _cam_drag_start(self, event) -> None:
        self._drag = (event.x, event.y)
        self.root.focus_set()          # 拖完还能直接按键

    def _cam_drag(self, event) -> None:
        if self._drag is None:
            self._drag = (event.x, event.y)
            return
        dx = event.x - self._drag[0]
        dy = event.y - self._drag[1]
        self._drag = (event.x, event.y)
        self.cam.azimuth = (self.cam.azimuth - dx * 0.4) % 360.0
        self.cam.elevation = max(-89.0, min(89.0, self.cam.elevation + dy * 0.4))

    def _cam_wheel(self, event) -> None:
        # Windows 上 event.delta 是 ±120 的倍数
        step = 0.9 if event.delta > 0 else 1.1
        self.cam.distance = max(0.22, min(1.60, self.cam.distance * step))

    def _cam_reset(self) -> None:
        self.cam.azimuth = VIEW["azimuth"]
        self.cam.elevation = VIEW["elevation"]
        self.cam.distance = VIEW["distance"]
        self.cam.lookat[:] = VIEW["lookat"]

    def _fps(self) -> float:
        """最近若干帧的实际刷新率。低于 18 就说明窗口太大、要缩一点。"""
        now = time.perf_counter()
        self._tick_times.append(now)
        while len(self._tick_times) > 2 and now - self._tick_times[0] > 1.0:
            self._tick_times.popleft()
        span = now - self._tick_times[0]
        return (len(self._tick_times) - 1) / span if span > 1e-6 else 0.0

    def on_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.ds.finalize()
        finally:
            self.disp.close()
            self.root.destroy()
        print(f"\n结束：保存 {self.saved} 条，丢弃 {self.discarded} 条。数据在 {self.out}")

    def run(self) -> int:
        self.root.after(0, self.tick)
        self.root.mainloop()
        return 0 if self.saved else 2


# ------------------------------------------------------------------ 自检
def _selftest(env: PickPlaceEnv, ds) -> int:
    """不开窗口，用程序模拟一遍「悬停→下降→合爪→抬起→搬运→放下」。

    目的不是替代专家，而是验证遥操作这条录制链路本身是对的：
    位移 → 逆解 → 动作 → 录帧 → 成功判定 → 写数据集。
    """
    rec = Recorder(ds, env)
    obs = env.reset(seed=SEED_BASE)
    target = env.tip().copy()
    prev_q = env.state()[:4].copy()
    cube = env.cube_xyz().copy()
    plate = env.plate_xyz().copy()

    def go_to(goal, grip: bool, tol: float = 0.004) -> None:
        nonlocal obs, target, prev_q
        goal = np.asarray(goal, dtype=float)
        while env.steps < MAX_EPISODE_STEPS:
            delta = goal - target
            d = float(np.linalg.norm(delta))
            if d < tol:
                return
            target = target + delta * min(1.0, KEY_STEP / d)
            try:
                prev_q = ik(*target, ref=prev_q)
            except Unreachable:
                target = goal
                return
            action = np.array([*prev_q, GRIP_OPEN if grip else GRIP_CLOSED], dtype=np.float64)
            rec.add(obs, action)
            obs = env.step(action)

    print("\n模拟操作：")
    go_to(cube + [0, 0, 0.055], grip=True)
    print(f"  悬停   {describe(env, True, False)}")
    go_to(cube + [0, 0, 0.001], grip=True)
    print(f"  下降   {describe(env, True, False)}")
    for _ in range(8):
        action = np.array([*prev_q, GRIP_CLOSED], dtype=np.float64)
        rec.add(obs, action)
        obs = env.step(action)
    print(f"  合爪   方块 z={env.cube_xyz()[2]:.3f}")
    go_to(cube + [0, 0, 0.055], grip=False)
    print(f"  抬起   方块 z={env.cube_xyz()[2]:.3f}（应该明显抬起来了）")
    go_to([plate[0], plate[1], 0.085], grip=False)
    go_to([plate[0], plate[1], 0.042], grip=False)
    for _ in range(8):
        action = np.array([*prev_q, GRIP_OPEN], dtype=np.float64)
        rec.add(obs, action)
        obs = env.step(action)
    print(f"  放下   {describe(env, True, False)}")
    obs = hold_gripper_open(env, rec, obs, prev_q)

    ok, n = rec.commit()
    ds.finalize()
    print()
    if ok:
        print(f"自检通过：录到 {n} 帧，成功判定 True，已写入数据集 {ds.root}")
        env.close()
        return 0
    print("自检失败：这一局没放成功，录制链路或控制逻辑有问题")
    env.close()
    return 1


def count_episodes(out: Path) -> int:
    """已经录了多少条。读不出来就当 0。"""
    info = out / "meta" / "info.json"
    if not info.exists():
        return 0
    try:
        return int(json.loads(info.read_text(encoding="utf-8")).get("total_episodes") or 0)
    except Exception:  # noqa: BLE001
        return 0


def open_dataset(out: Path, force: bool = False):
    """按需要「新建 / 续写 / 清空重建」数据集。

    默认行为是**接着已录的往下追加**。以前写成"发现目录存在就拒绝启动"，
    结果每次想重开都得手动删目录——那是设计错误，不是保护。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rounds = count_episodes(out)
    repo_id = f"local/{out.name}"

    if force and out.exists():
        print(f"清空重来：删掉已有的 {rounds} 条（{out}）")
        shutil.rmtree(out)
        rounds = 0

    if rounds > 0:
        print(f"接着已有的 {rounds} 条继续录（想清空重来加 --force）")
        return LeRobotDataset.resume(repo_id=repo_id, root=out, video_backend="pyav")

    if out.exists():
        print(f"清理上次留下的空目录 {out}")
        shutil.rmtree(out)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=dataset_features(),
        root=out,
        robot_type="sim_arm_5dof",
        use_videos=True,
        video_backend="pyav",
    )
    print(f"数据写到 {out}")
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None,
                    help="数据目录，默认 data/human_v1；--selftest 时默认 data/_selftest")
    ap.add_argument("--force", action="store_true",
                    help="清空已有数据重新开始（默认是接着已录的往下追加）")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT,
                    help="每条轨迹最多多少步（20 Hz 下 1200 步 = 60 秒），0 = 不限")
    ap.add_argument("--selftest", action="store_true", help="不开窗口，用程序模拟一遍流程")
    args = ap.parse_args()

    # 自检绝对不能落到正式数据集上：老版本默认写 data/human_v1，一跑
    # --selftest 就把 20 条人工示范整个删掉重录。现在自检默认写 data/_selftest。
    out: Path = args.out or ((ROOT / "data" / "_selftest") if args.selftest else DEFAULT_OUT)
    if args.selftest and out.resolve() == DEFAULT_OUT.resolve():
        print("拒绝：--selftest 会清空输出目录，不能指向正式数据 data/human_v1。")
        print("      省略 --out 即使用 data/_selftest。")
        return 2
    if args.selftest and out.exists():
        shutil.rmtree(out)          # 自检总是从干净目录开始

    env = PickPlaceEnv(render=True)
    ds = open_dataset(out, force=args.force)

    if args.selftest:
        return _selftest(env, ds)

    print(__doc__)
    app = TeleopWindow(env, ds, out, max_steps=args.max_steps)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
