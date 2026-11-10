r"""把绝对关节目标改成「相对当前关节角」的动作空间，生成一份新的数据集。

为什么试这个：
    现在策略直接回归"关节角应该是多少"。人类示范里同一个方块位置，抓取点的散布有 44 mm，
    网络学到的是一个平均动作，误差约 14 mm——正好卡在 10 mm 容差线上。
    让网络只回归"相对现在要动多少"，绝对尺度交给本体感受，不少工作报告精度更好。
    代价很小：数据只改 action 那一列，网络和训练脚本都不用动。

具体做法（关节 4 维做差，夹爪那维保持绝对）：
    第 0 帧：rel = act - state            （相对"当前关节角"）
    第 k 帧：rel = act[k] - act[k-1]      （相对"上一步的绝对动作"）
    推理时按同样规则积回去（sim/ensemble.py 里 relative=True 那段）。

    注意：早先写成"每一帧都减自己那一帧的 state"，推理时整块都加当前关节角——
    第 1 步之后全错，实测 0/80。定稿用的是上面的递推写法。

跑法：
    & "...python.exe" scripts\make_relative_dataset.py --src data\mixed_v1 --out data\mixed_rel
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from sim.env import dataset_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "mixed_v1")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "mixed_rel")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        if not args.force:
            print(f"目录已存在：{args.out}（加 --force 覆盖）")
            return 1
        shutil.rmtree(args.out)

    src = LeRobotDataset(f"local/{args.src.name}", root=args.src)
    out = LeRobotDataset.create(
        repo_id=f"local/{args.out.name}",
        fps=src.fps,
        features=dataset_features(),   # 不能直接用 src.features：那是 (C,H,W)，建库要 (H,W,C)
        root=args.out,
        robot_type="sim_arm_5dof",
        use_videos=True,
        video_backend="pyav",
    )
    print(f"{args.src.name}：{src.num_episodes} 条 → {args.out}")

    n_frames = 0
    for ep_idx in range(src.num_episodes):
        ep = src.meta.episodes[ep_idx]
        start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        prev_act = None                      # 上一步的绝对动作，用来做递推差
        for i in range(start, end):
            row = src[i]
            state = np.asarray(row["observation.state"], dtype=np.float32)
            act = np.asarray(row["action"], dtype=np.float32).copy()
            rel = act.copy()
            anchor = state if prev_act is None else prev_act
            rel[:4] = act[:4] - anchor[:4]   # 只有 4 个关节改成增量，夹爪保持绝对
            prev_act = act
            # 取出来的图是 (C,H,W) 的 float 张量，写回去要 (H,W,C) 的 uint8
            img = row["observation.images.front"]
            if hasattr(img, "permute"):
                img = img.permute(1, 2, 0)
            img = np.asarray(img)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8) \
                    if img.max() <= 1.5 else img.astype(np.uint8)
            out.add_frame({
                "observation.images.front": img,
                "observation.state": state,
                "action": rel,
                "task": row["task"],
            })
            n_frames += 1
        out.save_episode()
        if (ep_idx + 1) % 20 == 0:
            print(f"  已转换 {ep_idx + 1}/{src.num_episodes} 条（{n_frames} 帧）")
    out.finalize()
    print(f"完成：{out.num_episodes} 条 / {out.num_frames} 帧 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
