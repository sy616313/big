r"""示范修正：把人类示范里"抓偏了"的那一段平移回方块中心，再重训。

要解决的问题（量过的）：
    人类示范里同一个方块，抓取点的散布有 44 mm（机器示范只有 22.8 mm）。
    模型学到的是这些示范的"平均动作"，误差约 14 mm，而任务容差只有 10 mm——
    单次成功率卡在 65% 的根因就在这里。

做法：
    1. 认出每条示范是在哪个场景录的（靠比对首帧画面，实测 MSE≈8，认错的都在 50 以上）
    2. 回放到"合爪"那一刻，量出夹爪中心到方块中心的横向偏差 d
    3. 把合爪前后一小段的目标点平移 -d，用逆解重算关节角；迭代 2~3 轮收敛到几毫米
    4. 改完从头跑一遍：抓要更准，整条轨迹也必须还能做成，否则退回原样
    5. 全部示范都写进新数据集（改成的用新动作，没改成/改坏的保留原样）——
       这样数据集大小不变，只改"标签精度"这一个变量，后面才好对照

为什么不能靠"合爪时离方块最近"来认种子：
    试过，会认错——有的场景方块碰巧就在夹爪附近，但那条轨迹在那局里其实做不成。
    画面才是可靠指纹。

跑法：
    & "...python.exe" scripts\fix_demos.py --src data\human_v2 --out data\human_fix
输出：
    data\human_fix + 控制台一份"改前 / 改后"的偏差报告
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
from sim.env import PickPlaceEnv, dataset_features  # noqa: E402
from sim.kinematics import Unreachable, ik  # noqa: E402

CLOSE_GRIP = 0.010          # 夹爪指令低于这个值就算"开始合爪"
SEED_BASE = 20_000          # 遥操作录制用的种子起点（scripts/teleop.py）
MSE_OK = 30.0               # 首帧 MSE 低于这个值才认这个种子
SANITY_TOL = 0.050          # 回放/修正后"整条还能做成"的宽松闸门（米）
# 注意：这里用 50 mm 而不是正式判定的 37 mm。它只是"这条轨迹有没有被我们改坏"的体检，
# 不是成绩口径。人类示范里本来就有一批落在 40~50 mm（人类自己也没放正），
# 拿 37 mm 卡它们会把大量本来可用的示范筛掉。


def episode_actions(ds: LeRobotDataset, i: int) -> np.ndarray:
    ep = ds.meta.episodes[i]
    a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    return np.stack([np.asarray(ds[k]["action"], dtype=np.float64) for k in range(a, b)])


def first_close(actions: np.ndarray) -> int | None:
    idx = np.where(actions[:, 4] < CLOSE_GRIP)[0]
    return int(idx[0]) if len(idx) else None


def to_uint8_hwc(t) -> np.ndarray:
    """数据集里取出来的图是 (C,H,W) 的 float 张量；写回去要 (H,W,C) 的 uint8。"""
    if hasattr(t, "permute"):
        t = t.permute(1, 2, 0)
    a = np.asarray(t)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8) \
            if a.max() <= 1.5 else a.astype(np.uint8)
    return a


def first_frame(ds: LeRobotDataset, i: int) -> np.ndarray:
    start = int(ds.meta.episodes[i]["dataset_from_index"])
    return to_uint8_hwc(ds[start]["observation.images.front"]).astype(np.float32)


def find_seed(env: PickPlaceEnv, ds: LeRobotDataset, i: int, search: int) -> tuple[int, float]:
    """靠"第一帧画面"把场景种子认出来：同一个种子渲出来的首帧几乎逐像素相同。

    搜索范围是**整个种子池**（20000 起 search 个），不是"20000+i 附近"。
    踩过这个坑：一次全量跑下来 154 条里有 115 条认不出——因为遥操作分了几个场次，
    每场次的种子又从低位重新开始用，按"第几条 = 第几个种子"去找根本找不着。
    多渲几百帧（几分钟）换 100% 的识别率，值得。
    """
    ref = first_frame(ds, i)
    best_seed, best_mse = -1, 1e18
    for off in range(-5, search):
        seed = SEED_BASE + off
        try:
            env.reset(seed=seed)
            img = env.image().astype(np.float32)
        except Exception:      # noqa: BLE001 - 个别种子可能让渲染/物理出异常
            continue
        mse = float(np.mean((img - ref) ** 2))
        if mse < best_mse:
            best_seed, best_mse = seed, mse
    return best_seed, best_mse


def ramp(k: int, lo: int, hi: int, taper: int) -> float:
    """窗口内为 1、两端线性衰减到 0，避免动作突变把方块撞飞。"""
    if k < lo or k > hi:
        return 0.0
    if k - lo < taper:
        return (k - lo + 1) / (taper + 1)
    if hi - k < taper:
        return (hi - k + 1) / (taper + 1)
    return 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "human_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "human_fix")
    ap.add_argument("--search", type=int, default=200,
                    help="每个种子往前搜多少个（遥操作每次尝试都会消耗种子，间隔可能拉得很大）")
    ap.add_argument("--back", type=int, default=18, help="合爪前改多少帧")
    ap.add_argument("--fwd", type=int, default=6, help="合爪后改多少帧")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0 = 全部），先试跑用")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        if not args.force:
            print(f"目录已存在：{args.out}（加 --force 覆盖）")
            return 1
        shutil.rmtree(args.out)

    ds = LeRobotDataset(f"local/{args.src.name}", root=args.src)
    env = PickPlaceEnv(render=True)   # 认种子要靠渲染比对首帧
    out = LeRobotDataset.create(
        repo_id=f"local/{args.out.name}", fps=ds.fps, features=dataset_features(),
        root=args.out, robot_type="sim_arm_5dof", use_videos=True, video_backend="pyav",
    )
    print(f"{args.src.name}：{ds.num_episodes} 条 → {args.out}")
    print(f"每条往前搜 {args.search} 个种子，改合爪前 {args.back} 帧 / 后 {args.fwd} 帧\n")

    fixed = skipped = ik_fail = 0
    before: list[float] = []
    after: list[float] = []
    n_ep = ds.num_episodes if not args.limit else min(args.limit, ds.num_episodes)

    for i in range(n_ep):
        actions = episode_actions(ds, i)
        start_idx = int(ds.meta.episodes[i]["dataset_from_index"])
        new_actions = actions.copy()
        note = ""

        k_close = first_close(actions)
        if k_close is None:
            skipped += 1
            note = "没有合爪动作，保留原样"
        else:
            seed, mse = find_seed(env, ds, i, args.search)
            if seed < 0 or mse > MSE_OK:
                skipped += 1
                note = f"认不出场景种子（MSE={mse:.0f}），保留原样"
            else:
                env.reset(seed=seed)
                for k in range(len(actions)):
                    env.step(actions[k])
                c0, p0 = env.cube_xyz(), env.plate_xyz()
                d_end0 = float(np.hypot(c0[0]-p0[0], c0[1]-p0[1]))
                if d_end0 > SANITY_TOL:
                    skipped += 1
                    note = f"原轨迹回放对不上（离盘心 {d_end0*1000:.0f} mm），保留原样"
                else:
                    env.reset(seed=seed)
                    for k in range(k_close + 1):
                        env.step(actions[k])
                    cube = env.cube_xyz()
                    d0 = float(np.hypot(cube[0] - env.tip()[0], cube[1] - env.tip()[1]))
                    lo = max(0, k_close - args.back)
                    hi = min(k_close + args.fwd, len(actions) - 1)

                    for _it in range(3):
                        env.reset(seed=seed)
                        tips = []
                        for k in range(hi + 1):
                            env.step(new_actions[k])
                            tips.append(env.tip().copy())
                        dx = cube[0] - tips[k_close][0]
                        dy = cube[1] - tips[k_close][1]
                        if np.hypot(dx, dy) < 0.004:
                            break
                        for k in range(lo, hi + 1):
                            w = ramp(k, lo, hi, taper=6)
                            if w <= 0:
                                continue
                            target = tips[k] + np.array([dx * w, dy * w, 0.0])
                            try:
                                q = ik(float(target[0]), float(target[1]), float(target[2]))
                            except Unreachable:
                                ik_fail += 1
                                continue
                            new_actions[k, :4] = q

                    env.reset(seed=seed)
                    tips_final = []
                    for k in range(len(new_actions)):
                        env.step(new_actions[k])
                        tips_final.append(env.tip().copy())
                    d1 = float(np.hypot(cube[0] - tips_final[k_close][0],
                                        cube[1] - tips_final[k_close][1]))
                    c_end, p_end = env.cube_xyz(), env.plate_xyz()
                    d_end = float(np.hypot(c_end[0]-p_end[0], c_end[1]-p_end[1]))
                    if d_end <= SANITY_TOL:
                        before.append(d0)
                        after.append(d1)
                        fixed += 1
                        note = f"seed={seed} 合爪偏差 {d0*1000:5.1f} → {d1*1000:5.1f} mm"
                    else:
                        new_actions = actions.copy()
                        skipped += 1
                        note = f"修正后做不成（离盘心 {d_end*1000:.0f} mm），保留原样"

        for k in range(len(new_actions)):
            row = ds[start_idx + k]
            out.add_frame({
                "observation.images.front": to_uint8_hwc(row["observation.images.front"]),
                "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                "action": new_actions[k].astype(np.float32),
                "task": row["task"],
            })
        out.save_episode()
        print(f"  ep{i:3d} {note}")

    out.finalize()
    print(f"\n写成 {out.num_episodes} 条 / {out.num_frames} 帧")
    print(f"修正 {fixed} 条，保留原样 {skipped} 条，逆解失败 {ik_fail} 次")
    if before:
        print(f"合爪偏差：改前中位 {np.median(before)*1000:.1f} mm"
              f" → 改后中位 {np.median(after)*1000:.1f} mm")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
