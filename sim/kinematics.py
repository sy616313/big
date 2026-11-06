r"""机械臂运动学：前向 + 逆解（闭式解，不是数值迭代）。

单独成文件，是因为仿真脚本（scripts）和网页服务（sim/server.py）都要用同一套公式。
两处各写一份迟早会对不上，那是最难查的 bug。

=============== 为什么逆解要用闭式解 ===============

旧版用 scipy.least_squares 迭代，实测有两个毛病：

  1. 目标够不着时它不报错，而是返回"离目标最近"的那组角。
     比如 (0.335, 0, 0.145) 实际只能做到 93 mm 外，它照样返回，
     调用方以为成功了。这类静默错误会直接污染训练数据。
  2. 即使在**可达**的点上，也有约 3% 收敛到局部极小，最多偏 15.9 mm。

现在改成解析解，顺带带来两个好处：不可达时抛 Unreachable（能证明不可达），
以及快两个数量级——采数据时一次轨迹要算上千次逆解，这个差别很实在。

=============== 几何约定 ===============

关节：yaw 绕 z，肩/肘/腕三个都绕 y 轴，所以末端在竖直平面内的朝向角就是
        psi = t1 + t3 + t4
psi = pi/2 表示工具竖直向下。

把 psi 固定后，末端（site grip）的 (水平距离 r, 高度 z) 满足

        x = L1·sin(t1) + L2·cos(t1+t3) + L3·cos(psi)
        z = Z0 + L1·cos(t1) - L2·sin(t1+t3) - L3·sin(psi)

后两项里的 L3 只和 psi 有关，移到左边就得到**手腕**的位置：

        w = (r, z + L3 - Z0)

于是问题退化成标准的两连杆（L1、L2）逆解，而两连杆的充要可达条件是

        |L1 - L2| <= |w| <= L1 + L2

这是能解析判定的，不需要迭代碰运气。再由 sin(t3) 解出肘角，回代解出肩角。
详细推导写在 ik() 里。
"""
from __future__ import annotations

import math

import numpy as np

# 从 arm_scene.xml 量出来的几何参数
L1, L2, L3 = 0.150, 0.200, 0.192     # 肩->肘、肘->腕、腕->夹爪中心
Z0 = 0.044                           # 肩关节高度
PSI_DOWN = math.pi / 2               # 末端方向角：pi/2 = 竖直向下

JOINT_LO = np.array([-3.05, -1.75, -2.15, -2.70])
JOINT_HI = np.array([3.05, 1.35, 2.15, 2.70])

W_MIN = abs(L1 - L2)                 # 手腕离肩的最近距离 0.050
W_MAX = L1 + L2                      # 手腕离肩的最远距离 0.350

IK_TOL = 1e-4                        # 闭式解应该精确到 0.1 mm 以内


class Unreachable(ValueError):
    """目标点够不着，或者被关节限位挡住。"""


def fk_planar(t1: float, t3: float, t4: float) -> tuple[float, float]:
    """给定肩/肘/腕三个角，算夹爪中心在手臂平面里的 (水平距离, 高度)。"""
    x = L1 * math.sin(t1) + L2 * math.cos(t1 + t3) + L3 * math.cos(t1 + t3 + t4)
    z = Z0 + L1 * math.cos(t1) - L2 * math.sin(t1 + t3) - L3 * math.sin(t1 + t3 + t4)
    return x, z


def wrist_radius(x: float, y: float, z: float, psi: float = PSI_DOWN) -> float:
    """手腕到肩关节的水平距离。够不够得着，只看这个值和 z+... 组成的向量长度。"""
    r = math.hypot(x, y)
    wx = r
    wy = z + L3 * math.sin(psi) - Z0     # psi=pi/2 时就是 z+L3-Z0
    return math.hypot(wx, wy)


def reachable(x: float, y: float, z: float, psi: float = PSI_DOWN) -> bool:
    """只做几何判断，不考虑关节限位。真要用请直接 try ik()。"""
    return W_MIN <= wrist_radius(x, y, z, psi) <= W_MAX


def _wrap(a: float) -> float:
    """把角度规整到 [-pi, pi]。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def ik(x: float, y: float, z: float, psi: float = PSI_DOWN,
       ref: np.ndarray | None = None) -> np.ndarray:
    """求四个关节角，使夹爪中心到 (x, y, z)、末端朝向 psi。

    参数 ref：上一帧的关节角。同一组目标位置通常有多个解，给一个参考值可以
    挑**离当前姿态最近**的那个——不然轨迹中间换个分支，机械臂会猛地甩一下。
    轨迹平滑性靠这个参数保证。

    够不着时抛 Unreachable，绝不返回一个"差不多"的错了。
    """
    r = math.hypot(x, y)
    if r < 1e-9 and abs(x) + abs(y) < 1e-9:
        yaw = 0.0
    else:
        yaw = math.atan2(y, x)
    if not (JOINT_LO[0] - 1e-9 <= yaw <= JOINT_HI[0] + 1e-9):
        raise Unreachable(f"yaw={yaw:.3f} 超出限位 ±{JOINT_HI[0]:.2f}")

    # ---- 折算到手腕位置，退化成两连杆 ----
    wx = r
    wy = z + L3 * math.sin(psi) - Z0
    w2 = wx * wx + wy * wy
    w = math.sqrt(w2)
    if not (W_MIN - 1e-9 <= w <= W_MAX + 1e-9):
        raise Unreachable(
            f"目标 ({x:.3f},{y:.3f},{z:.3f}) 手腕需在 {w:.3f} m 处，"
            f"超出可达范围 [{W_MIN:.3f}, {W_MAX:.3f}]")

    # ---- 肘角：|w|^2 = L1^2 + L2^2 - 2·L1·L2·sin(t3) ----
    s3 = (L1 * L1 + L2 * L2 - w2) / (2 * L1 * L2)
    s3 = max(-1.0, min(1.0, s3))

    # sin(t3)=s3 在 [-2.15, 2.15] 内最多三个解
    base = math.asin(s3)
    cands: list[np.ndarray] = []
    for t3 in (base, math.pi - base, -math.pi - base):
        if not (JOINT_LO[2] - 1e-9 <= t3 <= JOINT_HI[2] + 1e-9):
            continue

        # ---- 肩角：把方程写成 A·sin(t1)+B·cos(t1)=wx, A·cos(t1)-B·sin(t1)=wy ----
        a = L1 - L2 * math.sin(t3)
        b = L2 * math.cos(t3)
        d = a * a + b * b
        if d < 1e-12:
            continue
        sn = (a * wx - b * wy) / d
        cs = (a * wy + b * wx) / d
        norm = math.hypot(sn, cs)
        if norm < 1e-9:
            continue
        t1 = math.atan2(sn / norm, cs / norm)
        if not (JOINT_LO[1] - 1e-9 <= t1 <= JOINT_HI[1] + 1e-9):
            continue

        # ---- 腕角：psi 是三者之和 ----
        t4 = _wrap(psi - t1 - t3)
        if not (JOINT_LO[3] - 1e-9 <= t4 <= JOINT_HI[3] + 1e-9):
            continue

        q = np.array([yaw, t1, t3, t4])
        # 最后必须用正向运动学校验一遍——解错了宁可报错也不能放过去
        px, pz = fk_planar(t1, t3, t4)
        if abs(px - r) > IK_TOL or abs(pz - z) > IK_TOL:
            continue
        cands.append(q)

    if not cands:
        raise Unreachable(
            f"目标 ({x:.3f},{y:.3f},{z:.3f}) 几何上够得着，但被关节限位挡住")

    if ref is not None:
        ref = np.asarray(ref, dtype=float)
        return min(cands, key=lambda q: float(np.sum((q - ref) ** 2)))
    return min(cands, key=lambda q: abs(q[3]))


def ik_chain(points, psi: float = PSI_DOWN) -> list[np.ndarray]:
    """对一串点连续求逆解，自动用上一步当参考，保证不换分支。"""
    out: list[np.ndarray] = []
    ref = None
    for p in points:
        q = ik(float(p[0]), float(p[1]), float(p[2]), psi=psi, ref=ref)
        out.append(q)
        ref = q
    return out


# ------------------------------------------------------------------ 自检
def _selftest() -> int:
    rng = np.random.default_rng(0)

    print("1) 闭式逆解精度：在可达区域内随机取点，解开后用正向运动学校验")
    worst, n_ok, n_try = 0.0, 0, 0
    for _ in range(3000):
        r = rng.uniform(0.02, 0.34)
        th = rng.uniform(-2.8, 2.8)
        z = rng.uniform(0.005, 0.20)
        x, y = r * math.cos(th), r * math.sin(th)
        n_try += 1
        try:
            q = ik(x, y, z)
        except Unreachable:
            continue
        px, pz = fk_planar(q[1], q[2], q[3])
        err = math.hypot(px - r, pz - z)
        worst = max(worst, err)
        n_ok += 1
    print(f"   {n_ok}/{n_try} 个点可解，最大误差 {worst * 1e6:.2f} um")
    if worst > 1e-5:
        print("   失败：闭式解误差应该接近机器精度")
        return 1

    print()
    print("2) 不可达必须报错：给一堆明显够不着的点")
    # 注意 z 为负（桌子以下）不属于不可达——连杆不跟桌子碰撞，物理上够得着。
    # 那是任务层面的约束，留给 env.py 去卡，别混进运动学里。
    bad = [(0.335, 0.0, 0.145),   # 太高太远
           (0.60, 0.0, 0.10),     # 超过臂展
           (0.0, 0.0, 0.40),      # 举太高
           (0.25, 0.0, 0.30),     # 又高又远
           (0.03, 0.0, -0.11)]    # 离肩太近，落进手腕够不着的死区
    missed = 0
    for p in bad:
        try:
            ik(*p)
        except Unreachable as e:
            print(f"   {p} -> 正确拒绝：{e}")
        else:
            print(f"   {p} -> 竟然返回了解，错误！")
            missed += 1
    if missed:
        return 1

    print()
    print("3) 分支连续性：沿一条 40 点直线移动，关节角不应跳变")
    pts = [(0.13 + 0.003 * i, -0.05, 0.055) for i in range(40)]
    try:
        qs = np.array(ik_chain(pts))
    except Unreachable as e:
        print(f"   路径本身有问题：{e}")
        return 1
    jumps = np.abs(np.diff(qs, axis=0)).max(axis=0)
    print(f"   相邻两点最大关节变化 {np.round(jumps, 5).tolist()} rad")
    print("   " + ("OK" if jumps.max() < 0.15 else "有跳变，分支选择有问题"))
    return 0 if jumps.max() < 0.15 else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
