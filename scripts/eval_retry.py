r"""按顺序尝试多个策略，任一成功即算成功：量"允许重试"时的任务成功率。

为什么要有这个脚本：
    单次成功率是"一次机会"的数字。真实系统不会一次失败就放弃——先看结果，不成再换一个
    策略重来。这个脚本直接驱动 src/agent.py 里的 Executor.grasp_and_place，
    **和网页演示跑的是同一段代码**：先上 A+B 集成，没成就平滑收回手臂、换 A，再换 B。

    注意两次尝试之间不再重置场景：手臂是"收回去再抓"，方块留在原地（被碰偏了就偏了）。
    这比"整局重来"更接近真机，也更诚实。

跑法：
    & "...python.exe" scripts\eval_retry.py --ckpt A_pre B_pre --n-episodes 40 --seed-base 90000
输出：成功率 + 尝试次数分布。用在汇报里必须同时给出"单次成功率"。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

from sim.ensemble import EnsemblePolicy  # noqa: E402
from sim.env import PickPlaceEnv  # noqa: E402
from src.agent import Executor  # noqa: E402
from eval_act import EVAL_SEED_BASE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True,
                    help="第一次尝试用的模型（给多个就是集成）")
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--exec-steps", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--single", action="store_true",
                    help="只跑第一次尝试（不重试），用来对照")
    args = ap.parse_args()

    ens = EnsemblePolicy.from_paths(args.ckpt, exec_steps=args.exec_steps or None)
    alts = [] if args.single else [EnsemblePolicy([m]) for m in ens.members]
    print(f"第一次：{'+'.join(ens.names)}；"
          f"备用：{'、'.join('+'.join(a.names) for a in alts) or '无'}")
    print(f"{args.n_episodes} 局，种子 {args.seed_base} 起\n")

    env = PickPlaceEnv(render=True)
    ok = 0
    att = Counter()
    for ep in range(args.n_episodes):
        seed = args.seed_base + ep
        env.reset(seed=seed)
        ex = Executor(env, ens, None, None, ens.device, seed=seed, alternates=alts)
        r = ex.grasp_and_place()
        ok += int(r["ok"])
        att[r.get("attempts", 1)] += 1
        print(f"ep{ep:3d} seed={seed}  {'成功' if r['ok'] else '失败'}"
              f"  第 {r.get('attempts', 1)} 次尝试  离盘心 {r.get('dist_mm')} mm")

    rate = ok / args.n_episodes
    head = "单次" if args.single else f"最多 {1 + len(alts)} 次"
    print(f"\n{head}：{ok}/{args.n_episodes} = {rate*100:.1f}%")
    print("尝试次数分布：" + "，".join(f"{k} 次 {v} 局" for k, v in sorted(att.items())))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
