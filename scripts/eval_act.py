r"""评估训练好的 ACT 策略：在仿真里真跑，统计成功率。

跑法：
    & "D:\envs\embodied\python.exe" scripts\eval_act.py                    # 用 checkpoints/last
    & "D:\envs\embodied\python.exe" scripts\eval_act.py --n-episodes 30
    & "D:\envs\embodied\python.exe" scripts\eval_act.py --render           # 另外存几张过程图

为什么不能只看训练 loss：
    ACT 是模仿学习，训练 loss 低只说明"它能复现示范动作"，不说明"它能看着方块
    调整"。loss 一路降但闭环一塌糊涂是常态。唯一的判据是让它自己在环境里抓，
    数成功率。

为什么评估用的随机种子要和训练分开：
    训练用的是 0~299 号种子对应的方块位置。评估必须换一批没见过的位置，
    否则测的是"背题能力"，不是"泛化能力"。
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import deque
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

from sim.env import (  # noqa: E402
    FPS,
    MAX_EPISODE_STEPS,
    PickPlaceEnv,
)

DEFAULT_CKPT = ROOT / "runs" / "train" / "act_grasp_v2" / "checkpoints" / "last" / "pretrained_model"
EVAL_SEED_BASE = 10_000


def to_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    """uint8 (H,W,3) → float32 (1,3,H,W)，取值范围 0~1，和数据集里存的保持一致。"""
    t = torch.from_numpy(np.ascontiguousarray(img_hwc)).float().div_(255.0)
    return t.permute(2, 0, 1).unsqueeze(0)


def save_montage(frames: list[np.ndarray], path: Path, n: int = 8, cols: int = 4) -> None:
    """抽 n 帧拼成 4×2 网格，方便贴进文档。

    只写静态图，不导出视频——演示视频自己录屏，清晰度比逐帧渲染拼出来的高。
    """
    from PIL import Image

    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)]
    rows = [np.concatenate(picks[i:i + cols], axis=1) for i in range(0, len(picks), cols)]
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def load_policy(path: Path):
    """按存档里记的 policy.type 选类，act / diffusion 都能加载。"""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(path))
    if cfg.type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as Cls
    elif cfg.type == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as Cls
    else:
        raise SystemExit(f"评估脚本还不认识 policy.type={cfg.type}")
    policy = Cls.from_pretrained(str(path), config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path))
    return cfg, policy, pre, post


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="指向 .../pretrained_model 目录")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--exec-steps", type=int, default=0,
                    help="每执行几步重新预测一次（0 = 用训练时的 n_action_steps）")
    ap.add_argument("--render", action="store_true", help="每局存一张 8 帧拼接图到 runs/eval")
    ap.add_argument("--ablate", choices=["none", "image", "state"], default="none",
                    help="消融实验：image = 把画面抹成全黑，state = 把关节角清零。"
                         "用来证明模型到底靠哪一路输入")
    ap.add_argument("--ood", action="store_true",
                    help="把方块放到训练采样范围之外（半径 0.25~0.285 m）看泛化边界。"
                         "训练范围是 0.17~0.245，所以这里全是没见过的位置")
    ap.add_argument("--min-success", type=float, default=0.7,
                    help="成功率低于这个值就返回非 0 退出码，方便当验收闸门用")
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE,
                    help="起始种子。传 20000 = 遥操作录制时用的那批场景（训练见过的位置），"
                         "用来分清「场景没覆盖」还是「示范本身不一致」")
    ap.add_argument("--state-history", type=int, default=1,
                    help="把最近 N 帧关节角拼起来喂给策略（默认 1 = 只喂当前帧）。"
                         "必须和 add_state_history.py 训练出来的模型配套")
    ap.add_argument("--ensemble", action="store_true",
                    help="ACT 论文的时序集成：每一步都重新预测一整块，对同一时刻被多次预测到的动作"
                         "做指数加权平均再执行。论文报告 +3.3%，只增加推理开销")
    ap.add_argument("--ensemble-m", type=float, default=0.01,
                    help="集成衰减系数，越小越信任最新的预测（论文默认 0.01）")
    args = ap.parse_args()

    if args.ablate != "none":
        print(f"*** 消融实验：屏蔽 {args.ablate}。成功率应该明显下降，否则说明这一路输入没被用上 ***")
        print()
        args.min_success = 0.0   # 消融的预期结果就是变差，别拿它当失败判据

    ckpt: Path = args.ckpt
    if not (ckpt / "config.json").exists():
        print(f"找不到模型：{ckpt}")
        print("路径应该指向 .../checkpoints/<步数>/pretrained_model")
        return 1

    cfg, policy, pre, post = load_policy(ckpt)
    if args.exec_steps:
        policy.config.n_action_steps = args.exec_steps
    device = next(policy.parameters()).device
    print(f"模型   {ckpt}")
    print(f"设备   {device}   每条轨迹执行 {policy.config.n_action_steps} 步后重规划")
    print()

    env = PickPlaceEnv(render=True)
    shot_dir = ROOT / "runs" / "eval"
    if args.render:
        shot_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ok = 0
    lengths: list[int] = []
    reasons: dict[str, int] = {}
    for ep in range(args.n_episodes):
        seed = args.seed_base + ep
        if args.ood:
            g = np.random.default_rng(seed)
            r = g.uniform(0.250, 0.285)
            th = g.uniform(-0.6, 0.6)
            obs = env.reset(seed=seed, cube_xy=(r * np.cos(th), r * np.sin(th)))
        else:
            obs = env.reset(seed=seed)
        src = env.cube_xyz()[:2].copy()
        plate = env.plate_xyz()[:2].copy()   # 盘子每局位置不同，先记下来
        policy.reset()

        keep = []
        peak = 0.0
        finite = True
        chunks: list[tuple[int, np.ndarray]] = []      # 时序集成用：(开始步, 整块预测)
        # 状态历史缓冲：开头用第一帧补齐，和 add_state_history.py 的做法一致
        hist: deque = deque([np.asarray(obs["state"], dtype=np.float32)] * args.state_history,
                            maxlen=args.state_history)
        H = int(getattr(policy.config, "n_action_steps", 100))
        while env.steps < MAX_EPISODE_STEPS:
            # 状态历史：和 scripts/add_state_history.py 用同一种拼法（最早 → 当前）
            hist.append(np.asarray(obs["state"], dtype=np.float32))
            state_vec = np.concatenate(list(hist)) if args.state_history > 1 else obs["state"]
            batch = {
                "observation.images.front": to_tensor(obs["image"]).to(device),
                "observation.state": torch.from_numpy(state_vec).float().unsqueeze(0).to(device),
            }
            if args.ablate == "image":
                batch["observation.images.front"] = torch.zeros_like(batch["observation.images.front"])
            elif args.ablate == "state":
                batch["observation.state"] = torch.zeros_like(batch["observation.state"])
            with torch.no_grad():
                batch = pre(batch)
                if args.ensemble:
                    # 每步都重预测一整块（reset 之后第一次调用算一次前向，剩下 H-1 次只是出队）
                    policy.reset()
                    chunk = np.stack([post(policy.select_action(batch)).squeeze(0).float().cpu().numpy()
                                      for _ in range(H)])
                    chunks.append((env.steps, chunk))
                    chunks = [(s, c) for s, c in chunks if s + H > env.steps]
                    num = np.zeros_like(chunk[0])
                    den = 0.0
                    for s, c in chunks:
                        i = env.steps - s
                        if 0 <= i < len(c):
                            w = math.exp(-args.ensemble_m * i)
                            num += w * c[i]
                            den += w
                    act = num / max(den, 1e-9)
                else:
                    action = post(policy.select_action(batch))
                    act = action.squeeze(0).float().cpu().numpy()
            if not np.all(np.isfinite(act)):
                finite = False   # 不拦的话 NaN 会把整个仿真的 qpos 污染掉，后面全是在看垃圾
                break
            if args.render:
                keep.append(obs["image"])
            obs = env.step(act)
            peak = max(peak, float(env.cube_xyz()[2]))
            if env.success():
                break

        good = env.success()
        ok += good
        lengths.append(env.steps)
        c = env.cube_xyz()
        d = float(np.hypot(c[0] - plate[0], c[1] - plate[1]))
        grip = float(env.state()[4])
        # 失败原因按「卡在哪一步」分，这样才知道该改哪一段：
        #   没夹起来      → 抓取阶段就没成
        #   没松手        → 搬到了但夹爪不放，方块还在爪子里
        #   高度不对      → 松手了但方块没落在盘面高度（掉桌上 / 举太高）
        #   放偏了        → 位置对不上盘子中心
        if good:
            why = ""          # 成功了就不要贴失败原因，否则「成功」后面跟着「没夹起来」很误导
        elif not finite:
            why = "输出 NaN"
        elif peak < 0.050:
            why = "没夹起来"
        elif grip <= 0.008:
            why = "没松手"
        elif not (0.020 <= c[2] <= 0.045):
            why = f"高度不对 z={c[2]*1000:.0f}mm"
        else:
            why = "放偏了"
        if why:
            reasons[why] = reasons.get(why, 0) + 1
        print(f"ep{ep:3d} seed={seed}  {'成功' if good else '失败'}  "
              f"步数={env.steps:3d}  方块 r={np.hypot(*src):.3f}  盘子 r={np.hypot(*plate):.3f}  "
              f"最高抬到 z={peak:.3f}  终点离盘心 {d*1000:5.1f} mm" +
              (f"  <- {why}" if why else ""))

        if keep and args.render:
            save_montage(keep, shot_dir / f"ep{ep:03d}_{'ok' if good else 'fail'}.png")

    print()
    rate = ok / args.n_episodes
    print(f"成功率 {ok}/{args.n_episodes} = {rate * 100:.0f}%   "
          f"平均 {np.mean(lengths):.1f} 步（{np.mean(lengths)/FPS:.1f}s）   "
          f"用时 {time.time()-t0:.1f}s")
    if reasons:
        print("失败原因：" + "，".join(f"{k} {v}" for k, v in
                                  sorted(reasons.items(), key=lambda kv: -kv[1])))
    env.close()
    if rate < args.min_success:
        print(f"低于合格线 {args.min_success * 100:.0f}%，返回 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
