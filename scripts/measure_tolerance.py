r"""量抓取容差：夹爪整体偏多少毫米，方块就抓不住了。

为什么要量这个：
    44 条人类示范训出来的模型成功率 55%，人自己的示范彼此差 44 mm。
    到底是"模型不够好"，还是"这个任务要求的精度本来就超过了人能稳定达到的水平"？
    答案决定了下一步该做什么：
        - 容差大（>20 mm）→ 任务不难，是模型的问题，继续改模型
        - 容差小（<10 mm）→ 任务本身太苛刻，该改机构/尺寸，或者放宽判据

做法：
    把每条示范整条轨迹的抓取点按给定方向平移 δ 毫米（用逆解重算关节角），
    再原样执行，看方块还在不在盘子上。δ=0 必须复现原示范的成功率，
    否则说明这套"平移后重放"的流程有问题，后面的数都不能用。

跑法：
    & "<环境>\python.exe" scripts\measure_tolerance.py
    & "...python.exe" scripts\measure_tolerance.py --dataset data\human_v2 --limits 0 5 10 15 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

import mujoco  # noqa: E402

from sim.env import PickPlaceEnv  # noqa: E402
from sim.kinematics import Unreachable, ik  # noqa: E402
from diagnose_data import FK, load  # noqa: E402


def shifted_actions(fk: FK, S, A, dx: float, dy: float):
    """把整条轨迹的末端目标在 xy 上平移 (dx, dy)，用逆解换回关节角。

    平移从第一帧就加，等价于"模型的抓取点整体偏了 (dx, dy)"——
    这也是我们实测到的模型误差形态（系统性偏移，不是每步乱抖）。
    """
    out = []
    prev_q = S[0][:4].copy()
    for i in range(len(A)):
        p = fk.tcp(S[i])
        try:
            q = ik(p[0] + dx, p[1] + dy, p[2], ref=prev_q)
        except Unreachable:
            return None
        prev_q = q
        out.append(np.array([*q, A[i][4]], dtype=np.float64))
    return np.stack(out)


def replay(fk: FK, S, A, E, dx=0.0, dy=0.0, limit=40):
    ok = 0
    n = 0
    eps = np.unique(E)[:limit]
    for e in eps:
        m = np.where(E == e)[0]
        q, a = S[m], A[m]
        tr = np.where(np.abs(np.diff(a[:, -1])) > 1e-4)[0]
        if len(tr) < 2:
            continue
        cube = fk.tcp(q[min(tr[0] + 1, len(q) - 1)])[:2]
        plate = fk.tcp(q[tr[-1]])[:2]
        try:
            fk.env.reset(cube_xy=tuple(cube), plate_xy=tuple(plate))
        except RuntimeError:
            continue
        acts = shifted_actions(fk, q, a, dx, dy)
        if acts is None:
            continue
        for x in acts:
            fk.env.step(x)
        n += 1
        ok += int(fk.env.success())
    return ok, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/human_v2")
    ap.add_argument("--limits", nargs="*", type=float, default=[0, 5, 10, 15, 20, 30])
    ap.add_argument("--limit-eps", type=int, default=40)
    args = ap.parse_args()

    d, S, A, E, info, n_files = load(args.dataset)
    fk = FK()
    print(f"数据 {d.name}：{len(np.unique(E))} 条")

    print("\n== 自检 ==")
    ok0, n0 = replay(fk, S, A, E, 0, 0, args.limit_eps)
    print(f"     δ=0（不平移）复现 {ok0}/{n0} = {ok0/max(n0,1)*100:.0f}%")
    if ok0 / max(n0, 1) < 0.85:
        fk.close()
        raise SystemExit("[自检失败] 不平移都复现不了原示范，说明平移重放的流程有问题")
    print("     通过：不平移能复现原示范，可以往下量")

    print("\n== 抓取点整体偏移后的成功率 ==")
    print("   偏移(mm)   四个方向(±x/±y)              最好    最差")
    for L in args.limits:
        L_m = L / 1000.0
        res = {}
        for name, (dx, dy) in {"+x": (L_m, 0), "-x": (-L_m, 0),
                               "+y": (0, L_m), "-y": (0, -L_m)}.items():
            k, n = replay(fk, S, A, E, dx, dy, args.limit_eps)
            res[name] = (k, n, k / max(n, 1))
        best = max(r[2] for r in res.values())
        worst = min(r[2] for r in res.values())
        detail = "  ".join(f"{k}:{r[0]:2d}/{r[1]:2d}" for k, r in res.items())
        print(f"   {L:5.0f}     {detail}    {best*100:3.0f}%   {worst*100:3.0f}%")
    fk.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
