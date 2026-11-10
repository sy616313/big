r"""让网页自己找弱点、请专家补课：把「策略做不成的场景」变成训练数据。

为什么这么做：
    演示时最烦的是"同一局反复试还是同样的错"——因为策略是确定性的，重试等于重放。
    真正的补法是把失败场景交给解析专家重做一遍，录成示范，再混进训练集。
    这就是 DAgger 的思路（自己跑 → 看哪里错 → 请专家在这些地方示范 → 再训），
    只不过我们的"专家"是仿真里的解析轨迹，随时能调。

流程：
    1. 在**勘察种子**上跑当前策略，挑出失败的局（这些种子不参与最终评估）
    2. 对每个失败局，解析专家从同一初始场景重做一遍，录成一条示范
    3. 写成一个独立数据集（默认 data/self_v1），之后和 human_v2 合并再训练

跑法：
    & "...python.exe" scripts\self_improve.py --ckpt A B --explore-base 50000 --n 40
输出：data/self_v1（只含专家成功的轨迹），以及一份失败清单 runs/self_improve_<时间>.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from sim.ensemble import EnsemblePolicy  # noqa: E402
from sim.env import (  # noqa: E402
    FPS,
    MAX_EPISODE_STEPS,
    PickPlaceEnv,
    ScriptedExpert,
    dataset_features,
)
from scripts.collect import TASK  # noqa: E402

EXEC_STEPS = 10


def classify(env) -> str:
    """失败的局卡在哪一步——和评估脚本用同一套分类。"""
    c, p = env.cube_xyz(), env.plate_xyz()
    grip = float(env.state()[4])
    if float(c[2]) < 0.050:
        return "没夹起来"
    if grip > 0.008:
        return "没松手"
    if np.hypot(c[0] - p[0], c[1] - p[1]) > 0.037:
        return "放偏了"
    return "没放平"


def run_policy(env, ens, seed: int) -> bool:
    obs = env.reset(seed=seed)
    ens.reset()
    queue: list[np.ndarray] = []
    while env.steps < MAX_EPISODE_STEPS:
        if not queue:
            import torch
            batch = {"observation.images.front": torch.from_numpy(
                np.ascontiguousarray(obs["image"])).float().div_(255.0)
                .permute(2, 0, 1).unsqueeze(0).to(ens.device),
                "observation.state": torch.from_numpy(obs["state"]).float()
                .unsqueeze(0).to(ens.device)}
            queue = ens.act(batch, EXEC_STEPS)
        obs = env.step(queue.pop(0))
        if env.success():
            return True
    return False


def record_expert(env, seed: int) -> tuple[list, bool]:
    """专家在同一场景重做一遍，返回 (帧列表, 是否成功)。"""
    obs = env.reset(seed=seed)
    expert = ScriptedExpert(env)     # 注意：构造函数不会自己 reset
    expert.reset()
    frames = []
    while not expert.done and env.steps < MAX_EPISODE_STEPS:
        act = expert.act()
        frames.append((obs["image"].copy(), np.asarray(obs["state"], dtype=np.float32),
                       np.asarray(act, dtype=np.float32)))
        obs = env.step(act)
    return frames, env.success()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, type=Path)
    ap.add_argument("--explore-base", type=int, default=50000,
                    help="勘察用的起始种子（不能和评估种子重合）")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "self_v1")
    ap.add_argument("--exec-steps", type=int, default=EXEC_STEPS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out: Path = args.out
    if out.exists():
        if not args.force:
            print(f"目录已存在：{out}（加 --force 覆盖）")
            return 1
        shutil.rmtree(out)

    ens = EnsemblePolicy.from_paths(args.ckpt, exec_steps=args.exec_steps)
    env = PickPlaceEnv(render=True)
    print(f"策略：{'+'.join(ens.names)}　勘察种子 {args.explore_base} 起 {args.n} 局\n")

    failures: list[dict] = []
    for i in range(args.n):
        seed = args.explore_base + i
        good = run_policy(env, ens, seed)
        if not good:
            why = classify(env)
            failures.append({"seed": seed, "why": why})
        print(f"  勘察 {i+1:3d}/{args.n}  seed={seed}  {'成功' if good else '失败：' + why}")
    rate = 1 - len(failures) / args.n
    print(f"\n勘察成功率 {args.n - len(failures)}/{args.n} = {rate*100:.0f}%，"
          f"失败 {len(failures)} 局，开始请专家补课\n")

    ds = LeRobotDataset.create(
        repo_id=f"local/{out.name}", fps=FPS, features=dataset_features(), root=out,
        robot_type="sim_arm_5dof", use_videos=True, video_backend="pyav")
    kept = 0
    for k, row in enumerate(failures, 1):
        frames, ok = record_expert(env, row["seed"])
        if ok:
            for img, st, act in frames:
                ds.add_frame({"observation.images.front": img, "observation.state": st,
                              "action": act, "task": TASK})
            ds.save_episode()
            kept += 1
        row["expert_ok"] = ok
        print(f"  补课 {k:3d}/{len(failures)}  seed={row['seed']}  {row['why']}  "
              f"{'专家做成，已录 ' + str(len(frames)) + ' 帧' if ok else '专家也没成（跳过）'}")
    ds.finalize()

    stamp = time.strftime("%m%d_%H%M")
    log = ROOT / "runs" / "logs" / f"self_improve_{stamp}.json"
    log.write_text(json.dumps({"explore_base": args.explore_base, "n": args.n,
                               "policy": ens.names, "failure_rate": 1 - rate,
                               "failures": failures}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n补课数据：{out}（{kept} 条）　失败清单：{log}")
    print("下一步：合并 human_v2 + self_v1 再训练（见 docs/换机说明.md）")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
