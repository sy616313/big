r"""无头仿真环境：把 arm_scene.xml 包成一个「重置 → 给动作 → 拿观测」的循环。

为什么单独抽这个文件：
    采集脚本和评估脚本必须用**同一套**观测与动作定义。两边各写一份，
    训练时喂的 state 和评估时喂的 state 一旦差一个顺序，模型会默默学歪，
    而且完全看不出错在哪。所以这里把接口钉死：

        观测 = front 相机图像 (240×320×3 uint8) + 5 个关节角 (float32)
        动作 = 5 个关节的**绝对目标角**（位置伺服，不是力矩）

    5 个量依次是：底座偏航、肩、肘、腕、夹爪开合（0 闭合 ~ 0.018 全开）。
    中间的 roll 关节不参与，恒为 0。

为什么动作取绝对角而不是末端坐标：
    末端坐标要再过一层逆解，逆解失败时动作就是错的，模型会把逆解的 bug
    学进去。直接给关节角，模型输出什么机械臂就是什么，因果关系干净。

可达范围受什么限制（踩过的坑，写在这免得再犯）：
    工具竖直向下时，手腕必须待在那个高度再往上 0.192 m 处，所以
    「手腕到肩的距离」必须落在 [0.050, 0.350]。这比 L1+L2+L3 看起来的
    0.542 m 小得多。实测采样的每个点都要逐点验证整条轨迹，不能只验证抓取点。

跑法（自检）：
    & "D:\envs\embodied\python.exe" -m sim.env --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.kinematics import Unreachable, ik, ik_chain  # noqa: E402

SCENE = ROOT / "sim" / "arm_scene.xml"

# ---------- 时间 ----------
FPS = 20
PHYS_DT = 0.002                 # 场景里 timestep 的值
SUBSTEPS = 25                   # 每个动作推进 25×0.002 = 0.05 s，正好 20 Hz
ACTION_DT = SUBSTEPS * PHYS_DT

# ---------- 观测 / 动作 ----------
IMG_W, IMG_H = 320, 240
CAMERA = "front"
STATE_DIM = 5
# 观测/动作第 5 维（夹爪开合）的名字，写在这当文档用；
# 未来若给数据集补 names 字段，直接引这个列表即可。
STATE_NAMES = ["yaw", "shoulder", "elbow", "wrist", "grip"]
ARM_JOINTS = ("j1_yaw", "j2_shoulder", "j3_elbow", "j4_wrist", "grip_l")

# ---------- 任务常量 ----------
# 夹爪开合量是单侧偏移，两侧对称。手指橡胶垫内表面在 y = ±(0.018 + 开合)，
# 方块半宽 0.021。开合 = 0 时缝隙只有 0.036，比方块还窄 6 mm，会硬挤。
# 所以闭合位取 0.004：缝隙 0.044，正好比方块宽 2 mm，轻轻夹住。
GRIP_OPEN = 0.018
GRIP_CLOSED = 0.004

CUBE_HALF = 0.021
CUBE_REST_Z = 0.0232            # 方块静止时中心高度（工作垫 0.0022 + 半宽 0.021）
PLATE_XY = (0.150, -0.140)      # 盘子的默认位置（也是 XML 里的初始位置）
PLATE_Z = 0.008                 # 盘子底面高度
PLACE_Z = 0.042                 # 松爪高度

HOME_XYZ = (0.150, 0.0, 0.125)  # 待机位。手腕距肩 0.311 m，余量充足
PICK_CLEAR_Z = 0.055            # 悬停高度 / 抬起高度
MAX_EPISODE_STEPS = 160         # 8 秒还抓不到就算失败

TASK = "把红色方块放到盘子上"

# 方块采样范围（米 / 弧度）。上限由「悬停点必须可达」倒推：
# r=0.245 时悬停高度 0.078，手腕距肩 0.333 m，距 0.350 的上限还有 17 mm 余量。
SAMPLE_R = (0.170, 0.245)
SAMPLE_TH = (-0.75, 0.75)       # 约 ±43°，都在正前方
SAMPLE_X = (0.02, 0.385)
SAMPLE_Y = (-0.100, 0.105)

# 盘子采样范围。下限避开机械臂底座（底座半径约 0.09 + 盘半径 0.058），
# 上限由「盘子上方 0.085 m 处必须可达」倒推：r=0.24 时手腕距肩 0.338 m，还有余量。
PLATE_R = (0.165, 0.240)
PLATE_TH = (-1.00, 0.60)
MIN_CUBE_PLATE_DIST = 0.115     # 方块和盘子至少隔这么远，否则方块本来就落在盘上
_COLLECTED_KEYS = ("p", "grip", "hold", "note")


def expert_waypoints(cube_xyz, plate_xy: tuple[float, float] = PLATE_XY) -> list[dict]:
    """专家轨迹的关键点。采样和采集共用同一个函数，保证「能验证的」就是「会执行的」。"""
    c = np.asarray(cube_xyz, dtype=float)[:3]
    px, py = plate_xy
    return [
        {"p": np.array([c[0], c[1], c[2] + PICK_CLEAR_Z]), "grip": GRIP_OPEN,   "hold": 2, "note": "1/7 悬停"},
        {"p": np.array([c[0], c[1], c[2] + 0.001]),        "grip": GRIP_OPEN,   "hold": 2, "note": "2/7 下降"},
        {"p": np.array([c[0], c[1], c[2] + 0.001]),        "grip": GRIP_CLOSED, "hold": 8, "note": "3/7 合爪"},
        {"p": np.array([c[0], c[1], c[2] + PICK_CLEAR_Z]), "grip": GRIP_CLOSED, "hold": 2, "note": "4/7 抬起"},
        {"p": np.array([px, py, 0.085]),                   "grip": GRIP_CLOSED, "hold": 2, "note": "5/7 移到盘上"},
        {"p": np.array([px, py, PLACE_Z]),                 "grip": GRIP_CLOSED, "hold": 3, "note": "6/7 下降"},
        {"p": np.array([px, py, PLACE_Z]),                 "grip": GRIP_OPEN,   "hold": 8, "note": "7/7 松爪"},
    ]


def trail_reachable(cube_xyz, plate_xy: tuple[float, float] = PLATE_XY) -> bool:
    """整条专家轨迹逐点求逆解，任何一点够不着或换分支都算不可用。"""
    try:
        ik_chain([w["p"] for w in expert_waypoints(cube_xyz, plate_xy)])
    except Unreachable:
        return False
    return True


class PickPlaceEnv:
    """单臂抓放环境。整个进程建议只建一个实例（OpenGL 上下文不好反复创建）。"""

    def __init__(self, render: bool = True, camera: str = CAMERA,
                 image_size: tuple[int, int] = (IMG_W, IMG_H), settle_steps: int = 200) -> None:
        self.model = mujoco.MjModel.from_xml_string(SCENE.read_text(encoding="utf-8"))
        self.data = mujoco.MjData(self.model)
        self.camera = camera
        self.image_size = image_size
        self.settle_steps = settle_steps
        self.steps = 0
        self.rng = np.random.default_rng(0)

        o = mujoco.mjtObj
        self.grip_site = mujoco.mj_name2id(self.model, o.mjOBJ_SITE, "grip")
        self.cube_body = mujoco.mj_name2id(self.model, o.mjOBJ_BODY, "cube")
        self.plate_body = mujoco.mj_name2id(self.model, o.mjOBJ_BODY, "plate")

        cube_jnt = mujoco.mj_name2id(self.model, o.mjOBJ_JOINT, "cube_free")
        self.cube_qadr = int(self.model.jnt_qposadr[cube_jnt])
        self.cube_vadr = int(self.model.jnt_dofadr[cube_jnt])

        self.arm_qadr = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, o.mjOBJ_JOINT, n)])
            for n in ARM_JOINTS
        ]

        # 待机位算不出来就别往下走了，早报错比晚出错好
        self.home_q = ik(*HOME_XYZ)

        self.renderer = None
        if render:
            self.renderer = mujoco.Renderer(self.model, height=image_size[1], width=image_size[0])
        # 建好后立刻摆到待机位，否则 data 停在「全零关节角」这个没意义的状态上，
        # 这时候读 tip() 会得到 (0.392, 0, 0.194)，很容易误判成标定错了。
        self._hold(self.home_action, self.settle_steps)

    # ---------- 基础量 ----------
    def tip(self) -> np.ndarray:
        """夹爪中心（site grip）的世界坐标。"""
        return self.data.site_xpos[self.grip_site].copy()

    def state(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.arm_qadr], dtype=np.float32)

    def cube_xyz(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def plate_xyz(self) -> np.ndarray:
        return self.data.xpos[self.plate_body].copy()

    def task(self) -> str:
        return TASK

    # ---------- 控制 ----------
    def _apply(self, action: np.ndarray) -> None:
        """5 维动作 → 7 个执行器。roll 恒 0，两根手指对称地反向给。"""
        grip = float(np.clip(action[4], 0.0, GRIP_OPEN))
        self.data.ctrl[0:4] = action[0:4]
        self.data.ctrl[4] = 0.0
        self.data.ctrl[5] = grip
        self.data.ctrl[6] = -grip

    def _hold(self, action: np.ndarray, n: int) -> None:
        self._apply(action)
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    @property
    def home_action(self) -> np.ndarray:
        return np.array([*self.home_q, GRIP_OPEN], dtype=np.float64)

    # ---------- 重置 ----------
    def sample_cube_xy(self, rng: np.random.Generator) -> tuple[float, float]:
        """在垫子上随机挑一个点，且整条轨迹都够得着。"""
        for _ in range(2000):
            r = rng.uniform(*SAMPLE_R)
            th = rng.uniform(*SAMPLE_TH)
            x, y = r * np.cos(th), r * np.sin(th)
            if not (SAMPLE_X[0] <= x <= SAMPLE_X[1]):
                continue
            if not (SAMPLE_Y[0] <= y <= SAMPLE_Y[1]):
                continue
            if np.hypot(x - PLATE_XY[0], y - PLATE_XY[1]) < 0.085:
                continue
            if not trail_reachable((x, y, CUBE_REST_Z)):
                continue
            return float(x), float(y)
        raise RuntimeError("采样失败：可达范围可能变了，检查 SAMPLE_R 与 arm_scene.xml")

    def _place_cube(self, x: float, y: float) -> None:
        self.data.qpos[self.cube_qadr:self.cube_qadr + 7] = [x, y, CUBE_REST_Z + 0.003, 1, 0, 0, 0]
        self.data.qvel[self.cube_vadr:self.cube_vadr + 6] = 0.0
        self._hold(self.home_action, 100)

    def reset(self, seed: int | None = None, cube_xy: tuple[float, float] | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self._hold(self.home_action, self.settle_steps)   # 先把臂摆到待机位
        if cube_xy is None:
            cube_xy = self.sample_cube_xy(self.rng)
        self._place_cube(*cube_xy)                        # 再把方块放上去
        return self.obs()

    # ---------- 交互 ----------
    def step(self, action: np.ndarray) -> dict:
        self._apply(np.asarray(action, dtype=np.float64))
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        return self.obs()

    def obs(self) -> dict:
        out = {"state": self.state(), "cube_xyz": self.cube_xyz().astype(np.float32),
               "success": self.success()}
        if self.renderer is not None:
            out["image"] = self.image()
        return out

    def image(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    # ---------- 判定 ----------
    def success(self, tol_xy: float = 0.045) -> bool:
        """方块落到盘子里、并且夹爪已经松开才算成功。"""
        c = self.cube_xyz()
        if np.hypot(c[0] - PLATE_XY[0], c[1] - PLATE_XY[1]) > tol_xy:
            return False
        if not (0.020 <= c[2] <= 0.045):     # 盘面 z≈0.013，方块中心≈0.034
            return False
        return float(self.state()[4]) > 0.008  # 手指张开了，说明真放下了

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class ScriptedExpert:
    """会算逆解的“老师”：知道方块真实坐标，直接走一条干净轨迹。

    它输出的是和策略**完全相同**的动作格式，所以采出来的数据可以直接拿来做
    模仿学习。策略看不到方块坐标，只能看图像——这是这套数据唯一的价值来源。
    """

    def __init__(self, env: PickPlaceEnv, speed: float = 0.16, reach_tol: float = 0.006) -> None:
        self.env = env
        self.speed = speed
        self.reach_tol = reach_tol
        self.plan: list[dict] = []
        self.i = 0
        self.hold = 0
        self.p = np.zeros(3)
        self.q = np.zeros(4)

    def reset(self) -> None:
        self.plan = expert_waypoints(self.env.cube_xyz())
        self.i = 0
        self.hold = 0
        self.p = self.env.tip().copy()
        self.q = self.env.state()[:4].copy()

    @property
    def done(self) -> bool:
        return self.i >= len(self.plan)

    @property
    def note(self) -> str:
        return self.plan[min(self.i, len(self.plan) - 1)]["note"]

    def act(self) -> np.ndarray:
        wp = self.plan[min(self.i, len(self.plan) - 1)]
        dest = np.asarray(wp["p"], dtype=np.float64)

        delta = dest - self.p
        dist = float(np.linalg.norm(delta))
        if dist > 1e-9:
            self.p = self.p + delta * min(1.0, self.speed * ACTION_DT / dist)

        # ref=self.q 让逆解挑离当前姿态最近的解，避免中途换分支把臂甩出去
        self.q = ik(self.p[0], self.p[1], self.p[2], ref=self.q)
        action = np.array([*self.q, wp["grip"]], dtype=np.float64)

        if np.linalg.norm(dest - self.env.tip()) < self.reach_tol:
            self.hold += 1
            if self.hold >= wp["hold"]:
                self.i += 1
                self.hold = 0
        else:
            self.hold = 0
        return action


# ------------------------------------------------------------------ 自检
def _selftest(n: int = 8, render: bool = False) -> int:
    env = PickPlaceEnv(render=render)
    expert = ScriptedExpert(env)

    print(f"待机位  关节角 {np.round(env.home_q, 3).tolist()}")
    print(f"        末端   {np.round(env.tip(), 3).tolist()}")
    print()

    ok, lengths = 0, []
    for ep in range(n):
        obs = env.reset(seed=ep)
        c0 = obs["cube_xyz"].copy()
        expert.reset()
        while not expert.done and env.steps < MAX_EPISODE_STEPS:
            obs = env.step(expert.act())
        good = env.success()
        ok += good
        lengths.append(env.steps)
        c1 = env.cube_xyz()
        err = np.hypot(c1[0] - PLATE_XY[0], c1[1] - PLATE_XY[1])
        src = np.hypot(c0[0], c0[1])
        print(f"ep{ep}: {'成功' if good else '失败'}  步数={env.steps:3d}  "
              f"方块起点 r={src:.3f}  ->  离盘心 {err * 1000:5.1f} mm  "
              f"z={c1[2]:.3f}  爪={env.state()[4]:.4f}")

    print()
    print(f"成功率 {ok}/{n}   平均步数 {np.mean(lengths):.1f} "
          f"（{np.mean(lengths) / FPS:.1f} 秒）  最长 {max(lengths)} 最短 {min(lengths)}")
    env.close()
    return 0 if ok == n else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--render", action="store_true", help="自检时也渲染图像（会慢很多）")
    ap.add_argument("-n", type=int, default=8)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest(args.n, args.render))
    ap.print_help()
