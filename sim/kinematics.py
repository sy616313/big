r"""机械臂运动学：前向 + 逆解。

单独成文件，是因为仿真脚本（scripts）和网页服务（sim/server.py）都要用同一套公式。
两处各写一份迟早会对不上，那是最难查的 bug。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

# 从 arm_scene.xml 量出来的几何参数
L1, L2, L3 = 0.150, 0.200, 0.192     # 肩->肘、肘->腕、腕->夹爪中心
Z0 = 0.044                           # 肩关节高度
PSI_DOWN = math.pi / 2               # 末端方向角：pi/2 = 竖直向下

JOINT_LO = np.array([-3.05, -1.75, -2.15, -2.70])
JOINT_HI = np.array([3.05, 1.35, 2.15, 2.70])


def fk_planar(t1: float, t3: float, t4: float) -> tuple[float, float]:
    """给定肩/肘/腕三个角，算夹爪中心在手臂平面里的 (水平距离, 高度)。"""
    x = L1 * math.sin(t1) + L2 * math.cos(t1 + t3) + L3 * math.cos(t1 + t3 + t4)
    z = Z0 + L1 * math.cos(t1) - L2 * math.sin(t1 + t3) - L3 * math.sin(t1 + t3 + t4)
    return x, z


def ik(x: float, y: float, z: float, psi: float = PSI_DOWN) -> np.ndarray:
    """求四个关节角：夹爪中心到 (x,y,z)，末端指向 psi。

    三个条件（x、z、朝向）对应三个关节（肩、肘、腕），yaw 单独由 atan2 给出。
    """
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


def reachable(x: float, y: float, z: float) -> bool:
    """粗判一下目标是否够得着（避免服务端直接抛异常）。"""
    try:
        q = ik(x, y, z)
    except Exception:  # noqa: BLE001
        return False
    return bool(np.all(q[1:] >= JOINT_LO[1:] - 1e-6) and np.all(q[1:] <= JOINT_HI[1:] + 1e-6))
