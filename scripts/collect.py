r"""采集示范数据：跑 N 遍专家策略，把「看到什么 + 做了什么」录成 LeRobot 数据集。

跑法：
    & "D:\envs\embodied\python.exe" scripts\collect.py                # 默认 200 条
    & "D:\envs\embodied\python.exe" scripts\collect.py --episodes 20  # 先小批量试
    & "D:\envs\embodied\python.exe" scripts\collect.py --out data\grasp_v2

数据长什么样（每条 = 一次完整的抓放）：
    每帧三样东西，一一对应：
        observation.images.front   320×240 的相机画面（策略唯一能看到的"现实"）
        observation.state          5 个关节角（编码器读数）
        action                     5 个关节目标角（接下来要下发的指令）

    注意配对方式：第 t 帧存的是「执行第 t 个动作**之前**看到的画面」。
    如果错存成执行之后，模型学到的就是"结果→原因"，训练时会一路涨分但一上手就废。

为什么方块位置要随机：
    策略看不到方块坐标，只能从画面里找。位置固定的话它只要背下一条固定轨迹
    就能拿满分，于是学不到任何视觉能力，换个位置就全废。随机化逼它真的去看。

为什么失败的轨迹默认不存：
    模仿学习是"照着做"，把做砸的示范也喂进去，等于教它做砸。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()   # 必须赶在 lerobot 建数据集之前，否则中文路径下存视频会挂

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from sim.env import (  # noqa: E402
    FPS,
    MAX_EPISODE_STEPS,
    TASK,
    PickPlaceEnv,
    ScriptedExpert,
    dataset_features,
)

# 默认写到一个新目录：别不小心覆盖掉已经采好的 grasp_v2
DEFAULT_OUT = ROOT / "data" / "grasp_v3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-failures", action="store_true", help="失败的轨迹也存下来")
    ap.add_argument("--force", action="store_true", help="输出目录已存在就删掉重来")
    args = ap.parse_args()

    out: Path = args.out
    if out.exists():
        if not args.force:
            print(f"目录已存在：{out}")
            print("换一个 --out，或者加 --force 覆盖。直接覆盖是删库操作，所以不默认做。")
            return 1
        shutil.rmtree(out)

    env = PickPlaceEnv(render=True)
    expert = ScriptedExpert(env)

    ds = LeRobotDataset.create(
        repo_id=f"local/{out.name}",
        fps=FPS,
        features=dataset_features(),
        root=out,
        robot_type="sim_arm_5dof",
        use_videos=True,
        video_backend="pyav",
    )

    print(f"输出目录 {out}")
    print(f"目标 {args.episodes} 条，每条约 {MAX_EPISODE_STEPS} 步上限，{FPS} Hz")
    print()

    t0 = time.time()
    kept = failed = 0
    lengths: list[int] = []
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        expert.reset()

        # 先攒在内存里，确认成功才落盘。边跑边写的话，中途判定失败就得回滚，
        # 而 LeRobotDataset 的 episode 缓冲区回滚很容易漏一帧。
        frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        while not expert.done and env.steps < MAX_EPISODE_STEPS:
            action = expert.act()
            frames.append((obs["image"],
                           np.asarray(obs["state"], dtype=np.float32),
                           np.asarray(action, dtype=np.float32)))
            obs = env.step(action)

        good = env.success()
        kept += good
        failed += not good
        if good or args.keep_failures:
            for img, st, act in frames:
                ds.add_frame({
                    "observation.images.front": img,
                    "observation.state": st,
                    "action": act,
                    "task": TASK,
                })
            ds.save_episode()
            lengths.append(len(frames))

        done = ep + 1
        elapsed = time.time() - t0
        eta = elapsed / done * (args.episodes - done)
        print(f"[{done:4d}/{args.episodes}] 成功 {kept:4d} 失败 {failed:3d}  "
              f"本条约 {len(frames):3d} 步  已用 {elapsed:5.1f}s  预计还需 {eta:5.1f}s")

    ds.finalize()

    print()
    print(f"完成：{kept} 条成功轨迹，{failed} 条失败被丢弃")
    if lengths:
        print(f"每条平均 {np.mean(lengths):.1f} 步（{np.mean(lengths)/FPS:.1f} 秒），"
              f"总计 {sum(lengths)} 帧")
    print(f"用时 {time.time()-t0:.1f}s，数据在 {out}")
    if kept:
        print()
        print("下一步训练：")
        print(f'  & "D:\\envs\\embodied\\python.exe" scripts\\train_act.py --dataset {out}')
    env.close()
    return 0 if kept else 2


if __name__ == "__main__":
    sys.exit(main())
