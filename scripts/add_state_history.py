r"""把「最近 N 帧关节角」拼进 observation.state，做出一个带时间信息的版本。

为什么做这个：
    RoboMimic 那篇（arXiv 2108.03298）最大的结论是「人类数据上，看历史帧很关键」，
    他们的 BC-RNN 比只看当前帧的 BC 高一倍（Transport 17% → 71%）。
    但 LeRobot 的 ACT 实现明确不支持多帧图像：
        if self.n_obs_steps != 1: raise ValueError("Multiple observation steps not handled yet.")
    所以退一步——把历史塞进 state 通道：state 从 5 维变 5N 维，
    ACT 对 state 维度没有限制，而且关节角历史就够它判断"手在动还是在停"。

用法：
    & "...python.exe" scripts/add_state_history.py --k 3
    & "...python.exe" scripts/add_state_history.py --src data\human_v2 --dst data\human_v2_h3 --k 3

注意：
    训练、评估、部署三处必须用同一种拼法，否则输入对不上。
    拼接顺序是 [最早, …, 上一帧, 当前帧]，每段 5 维；开头不足的部分用第一帧补齐。
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

from sim.env import TASK, dataset_features  # noqa: E402


def history_features(k: int) -> dict:
    """k 帧历史 → observation.state 变成 5k 维。"""
    f = dataset_features()
    base = f["observation.state"]["shape"][0]
    f["observation.state"] = {"dtype": "float32", "shape": (base * k,), "names": None}
    return f


def stack_history(states: np.ndarray, k: int) -> np.ndarray:
    """把一条轨迹的 state 序列（T×D）变成 T×(D·k)：每行是 [t-k+1 … t] 的拼接。"""
    t_len, dim = states.shape
    out = np.empty((t_len, dim * k), dtype=np.float32)
    for i in range(t_len):
        parts = [states[max(i - j, 0)] for j in range(k - 1, -1, -1)]   # 最早 → 当前
        out[i] = np.concatenate(parts)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "human_v2")
    ap.add_argument("--dst", type=Path, default=None, help="默认 data/<src名>_h<k>")
    ap.add_argument("--k", type=int, default=3, help="保留最近几帧（含当前帧）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = args.src if args.src.is_absolute() else ROOT / args.src
    dst = args.dst or (ROOT / "data" / f"{src.name}_h{args.k}")
    if dst.exists():
        if not args.force:
            print(f"目标已存在：{dst}（加 --force 覆盖）")
            return 1
        shutil.rmtree(dst)
    if args.k < 1:
        print("--k 至少是 1")
        return 1

    old = LeRobotDataset(repo_id=f"local/{src.name}", root=src)
    eps = np.unique([int(old[i]["episode_index"]) for i in range(len(old))])
    print(f"源 {src.name}：{len(eps)} 条 / {len(old)} 帧 → 目标 state 维度 {5 * args.k}")

    new = LeRobotDataset.create(
        repo_id=f"local/{dst.name}", fps=old.meta.fps, features=history_features(args.k),
        root=dst, robot_type="sim_arm_5dof", use_videos=True, video_backend="pyav",
    )

    total = 0
    for e in eps:
        idx = np.where(np.array([int(old[i]["episode_index"]) for i in range(len(old))]) == e)[0]
        states = np.stack([old[int(i)]["observation.state"].numpy() for i in idx]).astype(np.float32)
        hist = stack_history(states, args.k)
        for j, i in enumerate(idx):
            item = old[int(i)]
            img = (item["observation.images.front"].permute(1, 2, 0).numpy() * 255.0
                   ).round().clip(0, 255).astype(np.uint8)
            new.add_frame({
                "observation.images.front": img,
                "observation.state": hist[j],
                "action": item["action"],
                "task": TASK,
            })
        new.save_episode()
        total += len(idx)
    new.finalize()
    print(f"完成：{dst}  {len(eps)} 条 / {total} 帧")
    print("训练时 observation.state 是 5*k 维；评估/部署也要按同样的拼法喂（见 sim/env.py 的 history_state）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
