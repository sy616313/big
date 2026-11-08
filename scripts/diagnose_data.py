r"""量数据：合爪那一刻夹爪到底停没停。

为什么要有这个脚本：
    前面用临时脚本量数据，出过四次错——读 parquet 只读了第一个分片、
    毫米和米没换算、回放时场景种子对不上、把单步动作当整块预测。
    错一次就换一个结论，等于没结论。

    所以这个脚本自带三道自检，任何一道不过就直接停，不输出结论：
        A0  TCP 会随关节角变化（证明正运动学没坏）
        A1  parquet 行数 = meta 里登记的帧数（证明分片读全了）
        A2  机器示范回放必须 ≈100%（证明回放流程本身是可信的）

    三道都过，再拿它去量人类数据。机器数据的结论已知（100%），
    它就是这套工具的"金标准"。

跑法：
    & "<环境>\python.exe" scripts\diagnose_data.py
    & "<环境>\python.exe" scripts\diagnose_data.py --datasets data\human_v2 data\grasp_v2
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

import mujoco  # noqa: E402

from sim.env import PickPlaceEnv  # noqa: E402

MM = 1000.0          # 米 → 毫米
STOP_SPEED = 1.0     # 相邻帧 TCP 位移小于 1 mm 就算"停住了"


# ---------------------------------------------------------------- 读数据
def load(name_or_path: str):
    d = Path(name_or_path)
    if not d.is_absolute():
        d = ROOT / d
    files = sorted(glob.glob(str(d / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"找不到 parquet：{d}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    info = json.loads((d / "meta" / "info.json").read_text(encoding="utf-8"))
    S = np.stack(df["observation.state"].values).astype(np.float64)
    A = np.stack(df["action"].values).astype(np.float64)
    E = df["episode_index"].values.astype(int)
    return d, S, A, E, info, len(files)


def check_alignment(d: Path, S, A, E, info, n_files: int) -> None:
    """A1：分片读全了吗？行数必须等于 meta 登记的帧数。"""
    want = int(info["total_frames"])
    got = len(S)
    eps = int(info["total_episodes"])
    if got != want:
        raise SystemExit(f"[A1 失败] {d.name}: parquet {got} 行，meta 登记 {want} 帧"
                         f"（读了 {n_files} 个分片，八成漏了）")
    if len(np.unique(E)) != eps:
        raise SystemExit(f"[A1 失败] {d.name}: {len(np.unique(E))} 条轨迹，meta 登记 {eps} 条")
    if A.shape[1] != 5 or S.shape[1] != 5:
        raise SystemExit(f"[A1 失败] {d.name}: 维度是 {A.shape[1]}/{S.shape[1]}，应为 5")
    print(f"     A1 通过：{got} 帧 / {eps} 条 / {n_files} 个分片")


# ---------------------------------------------------------------- 正运动学
class FK:
    def __init__(self):
        self.env = PickPlaceEnv(render=False)

    def tcp(self, s) -> np.ndarray:
        self.env.data.qpos[self.env.arm_qadr] = np.asarray(s, dtype=np.float64)
        mujoco.mj_forward(self.env.model, self.env.data)
        return self.env.tip().copy()

    def check(self) -> None:
        """A0：换个位姿 TCP 必须跟着动，否则后面的毫米数全是假的。"""
        a = self.tcp([0, 0, 0, 0, 0])
        b = self.tcp([0.3, 0.1, -0.5, 1.0, 0.018])
        d = np.linalg.norm(a - b) * MM
        if d < 50:
            raise SystemExit(f"[A0 失败] 换位姿后 TCP 只移动 {d:.1f} mm，正运动学没生效")
        print(f"     A0 通过：换位姿 TCP 移动 {d:.0f} mm")

    def close(self):
        self.env.close()


# ---------------------------------------------------------------- 回放
def replay(fk: FK, S, A, E, limit: int | None = None) -> tuple[int, int, list[float]]:
    """把录制动作原样跑一遍。

    场景用示范自己重建：合爪时夹爪中心 = 方块位置，张开时 = 盘子位置。
    这样不用猜当时的随机种子（踩过这个坑：按种子重采样会给出错误的 15%）。
    保真度 |回放状态 - 记录状态| 一并返回，太大说明重建没对上，结论不可用。
    """
    ok = 0
    fid: list[float] = []
    eps = np.unique(E)
    if limit:
        eps = eps[:limit]
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
        qs = []
        for i in range(len(a)):
            fk.env.step(a[i])
            qs.append(fk.env.state())
        fid.append(float(np.abs(np.stack(qs) - q).mean()) * MM)
        ok += int(fk.env.success())
    return ok, len(eps), fid


# ---------------------------------------------------------------- 合爪时的速度
def grasp_motion(S, A, E, fk: FK) -> pd.DataFrame:
    """每条轨迹：合爪那一帧，夹爪还在动吗？"""
    rows = []
    for e in np.unique(E):
        m = np.where(E == e)[0]
        q, a = S[m], A[m]
        xy = np.stack([fk.tcp(s) for s in q])          # 每帧夹爪中心
        v = np.linalg.norm(np.diff(xy, axis=0), axis=1) * MM   # 每帧位移 mm
        tr = np.where(np.abs(np.diff(a[:, -1])) > 1e-4)[0]
        if len(tr) == 0:
            continue
        c = int(tr[0])                                  # 第一次夹爪跳变
        if c < 1 or c + 2 >= len(v):
            continue
        at = float(v[max(c - 1, 0):c + 2].mean())        # 合爪前后各一帧的平均速度
        # 合爪后有没有"停住"再抬：往后数连续小位移的帧数
        dwell = 0
        for k in range(c + 1, len(v)):
            if v[k] < STOP_SPEED:
                dwell += 1
            else:
                break
        rows.append({
            "episode": int(e),
            "frames": len(m),
            "speed_at_close": at,
            "median_speed": float(np.median(v)),
            "dwell_after_close": dwell,
            "close_ratio": at / max(float(np.median(v)), 1e-9),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 示范之间有多一致
def path_spread(S, A, E, fk: FK, n_grid: int = 60, max_eps: int = 60) -> float:
    """把每条示范换算到「以方块为原点、指向盘子为 x 轴」的坐标，按归一化时间重采样，
    再算任意两条轨迹之间的平均距离（mm）。

    这个数是「示范之间有多不一致」的直接度量：同一个任务，允许的走法越自由，
    数越大，ACT 回归出来的平均值就越不指向任何一条真实轨迹。
    机器示范由同一个解析规划器生成，这个数应该很小；人类示范大，就说明歧义大。
    """
    paths = []
    eps = np.unique(E)[:max_eps]
    for e in eps:
        m = np.where(E == e)[0]
        q, a = S[m], A[m]
        tr = np.where(np.abs(np.diff(a[:, -1])) > 1e-4)[0]
        if len(tr) < 2 or len(m) < 10:
            continue
        xy = np.stack([fk.tcp(s) for s in q])
        cube = fk.tcp(q[min(tr[0] + 1, len(q) - 1)])
        plate = fk.tcp(q[tr[-1]])
        d = plate[:2] - cube[:2]
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        ex = d / n
        ey = np.array([-ex[1], ex[0]])
        rel = xy[:, :2] - cube[None, :2]
        u = rel @ ex
        v = rel @ ey          # 侧向偏移
        z = xy[:, 2]
        idx = np.linspace(0, len(m) - 1, n_grid).round().astype(int)
        paths.append(np.stack([u[idx], v[idx], z[idx]], axis=1))
    if len(paths) < 2:
        return float("nan")
    P = np.stack(paths)
    gaps = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            gaps.append(np.linalg.norm(P[i] - P[j], axis=1).mean())
    return float(np.mean(gaps)) * MM


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["data/grasp_v2", "data/human_v1", "data/human_v2", "data/human_v1c"])
    ap.add_argument("--replay-limit", type=int, default=40)
    args = ap.parse_args()

    fk = FK()
    print("== 自检 ==")
    fk.check()

    out = {}
    for name in args.datasets:
        d, S, A, E, info, n_files = load(name)
        print(f"\n== {d.name} ==")
        check_alignment(d, S, A, E, info, n_files)
        g = grasp_motion(S, A, E, fk)
        out[d.name] = g
        if g.empty:
            print("     没有可分析的合爪事件")
            continue
        moving = (g["close_ratio"] > 0.5).mean()
        print(f"     合爪时速度 {g.speed_at_close.mean():5.1f} mm/帧"
              f"（整条轨迹中位数 {g.median_speed.mean():5.1f}）"
              f"  还在动的占 {moving*100:.0f}%")
        print(f"     合爪后停住帧数 中位 {g.dwell_after_close.median():.0f}"
              f"  为 0 的占 {(g.dwell_after_close==0).mean()*100:.0f}%")
        sp = path_spread(S, A, E, fk)
        print(f"     示范之间的一致性：任意两条轨迹平均相距 {sp:5.1f} mm（越小越一致）")

    # A2：金标准——机器示范回放必须接近 100%
    print("\n== A2 金标准：机器示范原样回放 ==")
    d, S, A, E, info, n_files = load("data/grasp_v2")
    ok, n, fid = replay(fk, S, A, E, limit=args.replay_limit)
    rate = ok / max(n, 1) * 100
    print(f"     grasp_v2 回放 {ok}/{n} = {rate:.0f}%   保真度 |dq| {np.mean(fid):.1f} mm")
    if rate < 90:
        fk.close()
        raise SystemExit("[A2 失败] 机器示范回放都不到 90%，回放流程本身有问题，"
                         "下面关于人类数据的结论一律不可信")
    print("     A2 通过：回放流程可信")

    print("\n== 人类数据回放（同样流程）==")
    for name in args.datasets:
        if name.endswith("grasp_v2"):
            continue
        d, S, A, E, info, n_files = load(name)
        ok, n, fid = replay(fk, S, A, E, limit=args.replay_limit)
        print(f"     {d.name:10s} 回放 {ok}/{n} = {ok/max(n,1)*100:3.0f}%"
              f"   保真度 |dq| {np.mean(fid):5.1f} mm")
    fk.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
