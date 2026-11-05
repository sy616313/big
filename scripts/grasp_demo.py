r"""第 1 步：真的抓起来，把方块放到盘子上。

跑法：
    & "D:\envs\embodied\python.exe" scripts\grasp_demo.py
    & "D:\envs\embodied\python.exe" scripts\grasp_demo.py --render   # 存关键帧
    & "D:\envs\embodied\python.exe" scripts\grasp_demo.py --selftest # 只校验运动学

关于逆运动学（大白话版）：
    "让夹爪中心到 (x,y,z)，并且爪子朝下" —— 三个条件，正好对应三个关节角
    （肩、肘、腕）。用数值方法解，比手推几何公式稳，也不容易搞错符号。

踩过的坑（写下来免得再犯）：
    1. 第一版数值 IK 只约束位置、没约束姿态，爪子斜着伸过去，夹在方块上方；
    2. 第二版手推几何公式把第二条连杆方向算错了（零位时它是水平的，不是竖直的），
       角度全错，还以为是伺服没力气；
    3. 现在用"前向运动学 + 数值求逆 + 与 MuJoCo 对拍"，--selftest 随时可复现。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim" / "arm_scene.xml"
OUT = ROOT / "runs" / "grasp"

L1, L2, L3 = 0.150, 0.200, 0.192     # 肩->肘、肘->腕、腕->夹爪中心
Z0 = 0.044                           # 肩关节高度
PSI_DOWN = math.pi / 2               # 末端方向角：pi/2 = 竖直向下
GRIP_OPEN = (0.030, -0.030)
GRIP_CLOSED = (0.000, 0.000)

JOINT_LO = np.array([-3.05, -1.75, -2.15, -2.70])
JOINT_HI = np.array([3.05, 1.35, 2.15, 2.70])


def fk_planar(t1: float, t3: float, t4: float) -> tuple[float, float]:
    """给定肩/肘/腕三个角，算夹爪中心在手臂平面里的 (水平距离, 高度)。"""
    x = L1 * math.sin(t1) + L2 * math.cos(t1 + t3) + L3 * math.cos(t1 + t3 + t4)
    z = Z0 + L1 * math.cos(t1) - L2 * math.sin(t1 + t3) - L3 * math.sin(t1 + t3 + t4)
    return x, z


def ik(x: float, y: float, z: float, psi: float = PSI_DOWN) -> np.ndarray:
    """求四个关节角：夹爪中心到 (x,y,z)，末端指向 psi。"""
    yaw = math.atan2(y, x)
    r = math.hypot(x, y)

    def residual(v):
        t1, t3, t4 = v
        px, pz = fk_planar(t1, t3, t4)
        return [px - r, pz - z, (t1 + t3 + t4) - psi]

    best = None
    for guess in ([0.3, 0.9, -1.2], [-0.2, -0.9, 1.2], [1.0, 1.4, -2.4], [-1.0, -1.4, 2.4]):
        sol = least_squares(residual, guess,
                            bounds=(JOINT_LO[1:], JOINT_HI[1:]),
                            xtol=1e-12, ftol=1e-12, max_nfev=2000)
        if best is None or sol.cost < best.cost:
            best = sol
    return np.array([yaw, *best.x])


class Arm:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(SCENE.read_text(encoding="utf-8"))
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip")
        self._settle(300)

    def _settle(self, n: int) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def tip(self) -> np.ndarray:
        return self.data.site_xpos[self.site].copy()

    def move(self, xyz, grip=GRIP_OPEN, steps=800) -> float:
        q = ik(xyz[0], xyz[1], xyz[2])
        self.data.ctrl[0:4] = q
        self.data.ctrl[5], self.data.ctrl[6] = grip
        self._settle(steps)
        err = float(np.linalg.norm(np.asarray(xyz) - self.tip()))
        flag = "OK  " if err < 0.012 else "偏差"
        print(f"    {flag} 到位误差 {err * 1000:5.1f} mm")
        return err

    def move_linear(self, xyz, grip=GRIP_OPEN, n=60, settle=6) -> None:
        """让夹爪尖沿空间直线走到目标，避免关节直插时把物体扫飞。"""
        target = np.asarray(xyz, dtype=float)
        for _ in range(n):
            cur = self.tip()
            delta = target - cur
            if np.linalg.norm(delta) < 0.003:
                break
            step = cur + delta * 0.25          # 每步走剩余的 1/4，靠近后自动减速
            q = ik(step[0], step[1], step[2])
            self.data.ctrl[0:4] = q
            self.data.ctrl[5], self.data.ctrl[6] = grip
            self._settle(settle)
        self.data.ctrl[0:4] = ik(target[0], target[1], target[2])
        self.data.ctrl[5], self.data.ctrl[6] = grip
        self._settle(300)
        err = float(np.linalg.norm(target - self.tip()))
        print(f"    {'OK  ' if err < 0.012 else '偏差'} 到位误差 {err * 1000:5.1f} mm")

    def where(self, body: str) -> np.ndarray:
        return self.data.body(body).xpos.copy()


def selftest() -> int:
    """把解析公式和 MuJoCo 的真实正运动学对拍。"""
    arm = Arm()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        t1 = float(rng.uniform(JOINT_LO[1], JOINT_HI[1]))
        t3 = float(rng.uniform(JOINT_LO[2], JOINT_HI[2]))
        t4 = float(rng.uniform(JOINT_LO[3], JOINT_HI[3]))
        arm.data.qpos[0:4] = [0.0, t1, t3, t4]
        mujoco.mj_forward(arm.model, arm.data)
        real = arm.tip()[[0, 2]]
        mine = np.array(fk_planar(t1, t3, t4))
        worst = max(worst, float(np.linalg.norm(real - mine)))
    print(f"前向运动学对拍：20 组随机姿态，最大误差 {worst * 1000:.2f} mm")
    return 0 if worst < 0.002 else 1


def run(render: bool) -> int:
    arm = Arm()
    cube0 = arm.where("cube")
    plate = arm.where("plate")
    print(f"方块 {cube0.round(3)}　盘子 {plate.round(3)}")

    hover_cube = cube0 + np.array([0.0, 0.0, 0.055])
    grasp_cube = cube0 + np.array([0.0, 0.0, 0.002])   # 对准方块中心，夹高了会把它顶翻
    hover_plate = plate + np.array([0.0, 0.0, 0.070])
    place_plate = plate + np.array([0.0, 0.0, 0.042])

    rend = cam = None
    if render:
        OUT.mkdir(parents=True, exist_ok=True)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [0.16, -0.02, 0.10]
        cam.distance = 0.78
        cam.azimuth = 138.0
        cam.elevation = -22.0
        rend = mujoco.Renderer(arm.model, height=600, width=800)

    def snap(tag: str) -> None:
        if rend is None:
            return
        rend.update_scene(arm.data, camera=cam)
        import imageio.v2 as imageio

        imageio.imwrite(OUT / f"{tag}.png", rend.render())

    print("1) 悬停到方块上方"); arm.move_linear(hover_cube); snap("1_hover")
    print("2) 竖直下降");       arm.move_linear(grasp_cube); snap("2_down")
    print("3) 合爪（缓慢合拢，避免把方块顶飞）")
    for k in range(1, 41):
        f_ = k / 40.0
        arm.data.ctrl[5] = GRIP_OPEN[0] * (1 - f_)
        arm.data.ctrl[6] = GRIP_OPEN[1] * (1 - f_)
        arm._settle(15)
    arm._settle(200); snap("3_close")

    print("4) 抬起");           arm.move_linear(hover_cube, grip=GRIP_CLOSED); snap("4_lift")
    lifted = 1000 * (arm.where("cube")[2] - cube0[2])
    print(f"    方块跟着起来 {lifted:.0f} mm")

    print("5) 移到盘子上方");   arm.move_linear(hover_plate, grip=GRIP_CLOSED); snap("5_move")
    print("6) 下降");           arm.move_linear(place_plate, grip=GRIP_CLOSED); snap("6_place")
    print("7) 松爪")
    for k in range(1, 21):
        f_ = 1 - k / 20.0
        arm.data.ctrl[5] = GRIP_OPEN[0] * (1 - f_)
        arm.data.ctrl[6] = GRIP_OPEN[1] * (1 - f_)
        arm._settle(15)
    arm.data.ctrl[5], arm.data.ctrl[6] = GRIP_OPEN
    arm._settle(500); snap("7_release")

    final = arm.where("cube")
    d = float(np.linalg.norm(final[:2] - plate[:2]))
    print(f"\n方块最终 {final.round(3)}，距盘心 {1000 * d:.0f} mm")
    ok = d < 0.06 and final[2] < 0.055
    print("结果：" + ("成功，方块落在盘子上" if ok else "没放准"))
    if rend is not None:
        rend.close()
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    return run(args.render)


if __name__ == "__main__":
    sys.exit(main())
