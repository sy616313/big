r"""按顺序尝试多个策略，任一成功即算成功：量"允许重试"时的任务成功率。

为什么要有这个脚本：
    单次成功率 76% 是"一次机会"的数字。真实系统不会一次失败就放弃——它先看结果，
    不成再换一个策略重来。这个脚本量的是"最多试 k 次"的成功率，并把"用了几次"也记下来。

    注意两次尝试之间必须复位到**同一个场景**（同种子），否则测的是"换了一道更简单的题"。

跑法：
    & "...python.exe" scripts\eval_retry.py ^
        --ckpt A_pre B_pre A_pre... 顺序即尝试顺序
        --n-episodes 40 --exec-steps 10 --seed-base 90000

输出：成功率 + 尝试次数分布。用在汇报里必须同时给出"单次成功率"。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

from sim.ensemble import EnsemblePolicy  # noqa: E402
from sim.env import MAX_EPISODE_STEPS, PickPlaceEnv  # noqa: E402
from eval_act import EVAL_SEED_BASE, to_tensor  # noqa: E402


def run_attempt(env, ens, seed: int, exec_steps: int) -> bool:
    """在一个场景上跑一次完整尝试，返回是否成功。"""
    obs = env.reset(seed=seed)          # 每次尝试都复位到同一个场景
    ens.reset()
    queue: list[np.ndarray] = []
    while env.steps < MAX_EPISODE_STEPS:
        if not queue:
            batch = {
                "observation.images.front": to_tensor(obs["image"]).to(ens.device),
                "observation.state": torch.from_numpy(obs["state"]).float()
                    .unsqueeze(0).to(ens.device),
            }
            queue = ens.act(batch, exec_steps)
        act = queue.pop(0)
        if not np.all(np.isfinite(act)):
            return False
        obs = env.step(act)
        if env.success():
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", nargs="+", required=True, action="append",
                    metavar="CKPT",
                    help="一次尝试用到的模型（多个就是集成）。写多组 = 多次尝试，"
                         "例如：--attempt A B --attempt A --attempt B")
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--exec-steps", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    args = ap.parse_args()

    plans = [EnsemblePolicy.from_paths(group, exec_steps=args.exec_steps or None)
             for group in args.attempt]
    print(f"尝试顺序：{' → '.join('+'.join(p.names) for p in plans)}")
    print(f"{args.n_episodes} 局，种子 {args.seed_base} 起，每 {args.exec_steps} 步重规划\n")

    env = PickPlaceEnv(render=True)
    ok = 0
    att = Counter()
    per_plan_ok = [0] * len(plans)
    for ep in range(args.n_episodes):
        seed = args.seed_base + ep
        used = 0
        for i, ens in enumerate(plans):
            used = i + 1
            good = run_attempt(env, ens, seed, args.exec_steps)
            if good:
                per_plan_ok[i] += 1
                break
        att[used] += 1
        ok += int(good)
        print(f"ep{ep:3d} seed={seed}  {'成功' if good else '失败'}  第 {used} 次尝试")

    n = args.n_episodes
    print(f"\n最终成功率 {ok}/{n} = {ok/n*100:.1f}%（最多 {len(plans)} 次尝试）")
    for i, c in enumerate(per_plan_ok):
        print(f"  第 {i+1} 次（{'+'.join(plans[i].names)}）成功 {c}/{n} = {c/n*100:.0f}%")
    print("尝试次数分布：" + "，".join(f"{k} 次 {v} 局" for k, v in sorted(att.items())))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
