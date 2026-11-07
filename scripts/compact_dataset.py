r"""把已有数据集里的「连续重复动作」压掉。

为什么需要：
    遥操作只在按键时才改目标，两次按键之间动作完全一样。实测人类示范里
    83% 的帧是这种"保持不动"，模型会学成"什么都别做"（真发生过，闭环 0/20）。
    机器示范（解析逆解连续移动）只有 30% 重复，所以没这个问题。

    已经采好的数据不用重录，跑这个脚本压一遍就行。

跑法：
    & "D:\envs\embodied\python.exe" scripts\compact_dataset.py
    & "...python.exe" scripts\compact_dataset.py --src data\human_v1 --dst data\human_v1c

注意：压缩会改变时间轴——原本"按住 200 ms"，压缩后变成一步。
      对模仿位置目标来说是好事（去掉空转），但不要把压缩前后的数据混在一起训。
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
from sim.kinematics import JOINT_HI, JOINT_LO  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "human_v1")
    ap.add_argument("--dst", type=Path, default=ROOT / "data" / "human_v1c")
    ap.add_argument("--hold", type=int, default=1,
                    help="同一动作最多连续保留几帧。1=压缩掉全部停顿（默认）；"
                         "3=保留短停顿，夹爪夹紧那几帧不会被压没")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.hold < 1:
        print("--hold 至少是 1")
        return 1

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src: Path = args.src
    dst: Path = args.dst
    if not (src / "meta" / "info.json").exists():
        print(f"源数据集不存在：{src}")
        return 1
    if dst.exists():
        if not args.force:
            print(f"目标已存在：{dst}（加 --force 覆盖）")
            return 1
        shutil.rmtree(dst)

    old = LeRobotDataset(repo_id=f"local/{src.name}", root=src)
    n = len(old)
    eps = np.array([int(old[i]["episode_index"]) for i in range(n)])
    print(f"源：{src}  {len(np.unique(eps))} 条 / {n} 帧")

    new = LeRobotDataset.create(
        repo_id=f"local/{dst.name}", fps=old.meta.fps, features=dataset_features(),
        root=dst, robot_type="sim_arm_5dof", use_videos=True, video_backend="pyav",
    )

    kept_total = raw_total = 0
    kept_lens = []
    for e in np.unique(eps):
        idx = np.where(eps == e)[0]
        prev_act = None
        dup = 0
        kept = 0
        for i in idx:
            item = old[int(i)]
            act = item["action"].numpy()
            raw_total += 1
            if prev_act is not None and np.allclose(act, prev_act, atol=1e-6):
                dup += 1
                if dup >= args.hold:
                    continue
            else:
                dup = 0
            # 数据集读出来是 (3,H,W) 的 float [0,1]；写回去要 (H,W,3) 的 uint8
            img = (item["observation.images.front"].permute(1, 2, 0).numpy() * 255.0
                   ).round().clip(0, 255).astype(np.uint8)
            new.add_frame({
                "observation.images.front": img,
                "observation.state": item["observation.state"],
                "action": item["action"],
                "task": TASK,
            })
            prev_act = act
            kept += 1
        if kept:
            new.save_episode()
            kept_total += kept
            kept_lens.append(kept)

    new.finalize()
    print(f"目标：{dst}  {len(kept_lens)} 条 / {kept_total} 帧")
    print(f"压缩比：{kept_total}/{raw_total} = {kept_total/raw_total*100:.1f}%（去掉了 "
          f"{(1-kept_total/raw_total)*100:.1f}% 的空转帧，同一动作最多留 {args.hold} 帧）")
    if kept_lens:
        print(f"压缩后每条约 {np.mean(kept_lens):.0f} 帧"
              f"（{np.mean(kept_lens)/old.meta.fps:.1f} 秒），"
              f"最短 {min(kept_lens)} 最长 {max(kept_lens)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
