r"""数据集体检：训练之前先确认数据没坏。

跑法：
    & "D:\envs\embodied\python.exe" scripts\check_dataset.py
    & "D:\envs\embodied\python.exe" scripts\check_dataset.py --dataset data\grasp_v2

为什么值得单独跑一遍：
    数据集坏掉的方式都很隐蔽——视频黑屏、action 被写成了 state、某条轨迹只录了
    三帧、夹爪值全是 0。这些在训练里不会报错，只会让模型学不出东西，然后你去
    怀疑模型、怀疑超参、怀疑人生。花几秒体检，比查一天问题划算。

注意：这里**抽样**检查，不逐帧读。23750 帧全读一遍光解码就要三分钟，
而统计量看几百帧就够。轨迹长度直接读元数据，不用碰视频。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

from sim.kinematics import JOINT_HI, JOINT_LO  # noqa: E402

DEFAULT = ROOT / "data" / "grasp_v1"
N_SAMPLE = 400        # 统计关节角用的抽样帧数
N_IMAGE = 12          # 检查画面用的抽样帧数


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT)
    args = ap.parse_args()

    root: Path = args.dataset
    if not (root / "meta" / "info.json").exists():
        print(f"不是合法的数据集目录：{root}")
        return 1

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=f"local/{root.name}", root=root)
    n = len(ds)
    print(f"目录 {root}")
    print(f"{ds.meta.total_episodes} 条轨迹 / {ds.meta.total_frames} 帧 / {ds.meta.fps} Hz")
    print(f"字段 {sorted(ds.meta.features)}")
    print()

    problems: list[str] = []

    # ---- 每条轨迹的长度（读元数据，不解码视频）----
    lengths = np.asarray(ds.meta.episodes["length"])
    print(f"轨迹长度：最短 {lengths.min()}  最长 {lengths.max()}  平均 {lengths.mean():.1f} 帧"
          f"（{lengths.mean()/ds.meta.fps:.1f} 秒）")
    if lengths.min() < 10:
        problems.append("有轨迹短于 10 帧")

    # ---- 抽样看数值 ----
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, size=min(N_SAMPLE, n), replace=False))
    states = torch.stack([ds[int(i)]["observation.state"] for i in idx]).numpy()
    actions = torch.stack([ds[int(i)]["action"] for i in idx]).numpy()

    lo, hi = JOINT_LO[:4], JOINT_HI[:4]
    ok_lim = np.all(states[:, :4] >= lo - 1e-2) and np.all(states[:, :4] <= hi + 1e-2)
    print(f"前 4 个关节角在限位内          {'OK' if ok_lim else 'FAIL'}")
    if not ok_lim:
        problems.append("关节角超出限位，state 可能存错了")

    # 夹爪是软限位，闭合时会略微越界，给 1 mm 余量
    ok_grip = np.all(states[:, 4] >= -1e-3) and np.all(states[:, 4] <= 0.018 + 1e-3)
    print(f"夹爪开合在合理区间            {'OK' if ok_grip else 'FAIL'}")
    if not ok_grip:
        problems.append("夹爪值越界")

    spread = float(states[:, 4].max() - states[:, 4].min())
    ok_spread = spread > 0.005
    print(f"夹爪确实开合过（跨度 {spread:.4f}）  {'OK' if ok_spread else 'FAIL'}")
    if not ok_spread:
        problems.append("夹爪几乎没动过，抓取可能全是空的")

    diff = float(np.abs(actions - states).max())
    ok_diff = diff > 0.01
    print(f"action 与 state 不同（最大差 {diff:.3f}）  {'OK' if ok_diff else 'FAIL'}")
    if not ok_diff:
        problems.append("action 等于 state，动作列没写对")

    # ---- 画面 ----
    img_idx = np.linspace(0, n - 1, min(N_IMAGE, n)).astype(int)
    frames = [ds[int(i)]["observation.images.front"] for i in img_idx]
    stds = [float(f.std()) for f in frames]
    ok_img = min(stds) > 0.02
    print(f"画面像素标准差 {min(stds):.3f} ~ {max(stds):.3f}      {'OK' if ok_img else 'FAIL'}")
    if not ok_img:
        problems.append("有整帧接近纯色，视频可能没编码好")

    if len(frames) >= 2:
        d = float((frames[0] - frames[-1]).abs().mean())
        print(f"首尾帧差异 {d:.4f}                  {'OK' if d > 1e-3 else 'FAIL'}")
        if d <= 1e-3:
            problems.append("首尾帧几乎一样，画面可能卡住")

    print()
    if problems:
        print("发现的问题：")
        for p in problems:
            print("  -", p)
        return 1
    print("数据没问题，可以去训练了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
