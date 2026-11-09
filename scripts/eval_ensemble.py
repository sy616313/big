r"""多模型集成评估：N 个模型各自预测，动作取平均。

为什么这么做：
    单个模型的误差里有一部分是"随机"的（初始化、数据顺序不同带来的差异）。
    多个模型平均能把随机部分压掉，而系统性的偏差消不掉——所以它有用但有上限。
    ACT 是确定性策略（推理时 z=0），同一个模型重复问没有意义，必须是不同模型。

    集成逻辑本身在 sim/ensemble.py，网页演示用的也是那一份——保证网页跑的就是这里测过的。

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

from sim.ensemble import EnsemblePolicy  # noqa: E402
from sim.env import MAX_EPISODE_STEPS, PickPlaceEnv  # noqa: E402
from eval_act import EVAL_SEED_BASE, save_montage, to_tensor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True, help="2 个以上模型目录")
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--exec-steps", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--min-success", type=float, default=0.0,
                    help="成功率低于这个值就返回非 0 退出码（默认不卡）")
    args = ap.parse_args()

    if len(args.ckpt) < 2:
        print("至少要给两个模型")
        return 1

    ens = EnsemblePolicy.from_paths(args.ckpt, exec_steps=args.exec_steps or None)
    dev = ens.device
    print(f"集成 {len(ens.members)} 个模型：{' + '.join(ens.names)}")
    print(f"每 {ens.config.n_action_steps} 步重规划\n")

    env = PickPlaceEnv(render=True)   # 策略要画面，必须开渲染；--render 只控制存图
    shot_dir = ROOT / "runs" / "eval"
    if args.render:
        shot_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    lengths = []
    reasons: dict[str, int] = {}
    for ep in range(args.n_episodes):
        seed = args.seed_base + ep
        obs = env.reset(seed=seed)
        plate = env.plate_xyz()[:2].copy()
        ens.reset()
        peak = 0.0
        frames: list[np.ndarray] = []
        queue: list[np.ndarray] = []
        while env.steps < MAX_EPISODE_STEPS:
            if not queue:
                batch = {
                    "observation.images.front": to_tensor(obs["image"]).to(dev),
                    "observation.state": torch.from_numpy(obs["state"]).float()
                        .unsqueeze(0).to(dev),
                }
                queue = ens.act(batch, int(ens.config.n_action_steps))
            act = queue.pop(0)
            if not np.all(np.isfinite(act)):
                reasons["动作出 NaN"] = reasons.get("动作出 NaN", 0) + 1
                break
            if args.render:
                frames.append(obs["image"].copy())
            obs = env.step(act)
            peak = max(peak, float(env.cube_xyz()[2]))
            if env.success():
                break
        good = env.success()
        ok += int(good)
        lengths.append(env.steps)
        if not good and env.steps >= MAX_EPISODE_STEPS:
            # 和 eval_act.py 用同一套失败分类，两个脚本的数才能放一起比
            c, p = env.cube_xyz(), env.plate_xyz()
            grip = float(env.state()[4])
            if peak < 0.050:
                why = "没夹起来"
            elif grip < 0.008:
                why = "没松手"
            elif np.hypot(c[0] - p[0], c[1] - p[1]) > 0.045:
                why = "放偏了"
            else:
                why = "高度不对"
            reasons[why] = reasons.get(why, 0) + 1
        print(f"ep{ep:3d} seed={seed}  {'成功' if good else '失败'}  步数={env.steps:3d}"
              f"  最高抬到 z={peak:.3f}")
        if args.render and frames:
            save_montage(frames, shot_dir / f"ens_{seed}.jpg")

    rate = ok / args.n_episodes
    print(f"\n集成（{len(ens.members)} 个模型）：{ok}/{args.n_episodes} = {rate*100:.0f}%"
          f"   平均 {np.mean(lengths):.0f} 步")
    if reasons:
        print("失败原因：" + "，".join(f"{k} {v}" for k, v in
                                        sorted(reasons.items(), key=lambda kv: -kv[1])))
    env.close()
    if rate < args.min_success:
        print(f"低于合格线 {args.min_success*100:.0f}%，返回 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
