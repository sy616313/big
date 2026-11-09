r"""多模型集成评估：N 个模型各自预测，动作取平均。

为什么这么做：
    单个模型的误差里有一部分是"随机"的（初始化、数据顺序不同带来的差异）。
    多个模型平均能把随机部分压掉，而系统性的偏差消不掉——所以它有用但有上限。
    ACT 是确定性策略（推理时 z=0），同一个模型重复问没有意义，必须是不同模型。

用法：
    & "...python.exe" scripts\eval_ensemble.py ^
        --ckpt runs\train\act_h104_aug\checkpoints\020000\pretrained_model ^
               runs\train\act_h154_aug40k\checkpoints\040000\pretrained_model ^
        --n-episodes 40 --exec-steps 10 --seed-base 40000

注意：所有模型的 chunk_size / action 维度必须一致，否则动作没法平均。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

from sim.env import FPS, MAX_EPISODE_STEPS, PickPlaceEnv  # noqa: E402
from eval_act import EVAL_SEED_BASE, load_policy, save_montage, to_tensor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True, help="2 个以上模型目录")
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--exec-steps", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    if len(args.ckpt) < 2:
        print("至少要给两个模型")
        return 1

    models = []
    for p in args.ckpt:
        cfg, policy, pre, post = load_policy(p)
        if args.exec_steps:
            policy.config.n_action_steps = args.exec_steps
        dev = next(policy.parameters()).device
        models.append({"name": p.parents[2].name, "policy": policy, "pre": pre,
                       "post": post, "dev": dev, "queue": []})
        print(f"载入 {p.parents[2].name}：每 {policy.config.n_action_steps} 步重规划")

    env = PickPlaceEnv(render=True)   # 策略要画面，必须开渲染；--render 只控制存图
    shot_dir = ROOT / "runs" / "eval"

    ok = 0
    lengths = []
    for ep in range(args.n_episodes):
        seed = args.seed_base + ep
        obs = env.reset(seed=seed)
        plate = env.plate_xyz()[:2].copy()
        for m in models:
            m["policy"].reset()
            m["queue"] = []
        peak = 0.0
        while env.steps < MAX_EPISODE_STEPS:
            acts = []
            for m in models:
                if not m["queue"]:
                    batch = {
                        "observation.images.front": to_tensor(obs["image"]).to(m["dev"]),
                        "observation.state": torch.from_numpy(obs["state"]).float()
                            .unsqueeze(0).to(m["dev"]),
                    }
                    with torch.no_grad():
                        b = m["pre"](batch)
                        m["policy"].reset()
                        n = int(m["policy"].config.n_action_steps)
                        m["queue"] = [m["post"](m["policy"].select_action(b))
                                      .squeeze(0).float().cpu().numpy() for _ in range(n)]
                acts.append(np.asarray(m["queue"].pop(0), dtype=np.float64))
            act = np.mean(acts, axis=0)          # 集成：各模型动作取平均
            if not np.all(np.isfinite(act)):
                break
            obs = env.step(act)
            peak = max(peak, float(env.cube_xyz()[2]))
            if env.success():
                break
        good = env.success()
        ok += int(good)
        lengths.append(env.steps)
        print(f"ep{ep:3d} seed={seed}  {'成功' if good else '失败'}  步数={env.steps:3d}"
              f"  最高抬到 z={peak:.3f}")

    rate = ok / args.n_episodes
    print(f"\n集成（{len(models)} 个模型）：{ok}/{args.n_episodes} = {rate*100:.0f}%"
          f"   平均 {np.mean(lengths):.0f} 步")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
