r"""命令行跑一遍「说一句话 → 大模型拆计划 → 执行」的全流程。

跑法：
    & "<环境>\python.exe" scripts\agent_demo.py "把盘子往左挪 5 厘米，再把方块放到盘子上"
    & "...python.exe" scripts\agent_demo.py "把方块放到盘子上，如果没抓起来就重试一次" --render

不传参数时跑一组内置例子。--render 会把每步画面拼成图存到 runs/agent/。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

from sim.env import PickPlaceEnv  # noqa: E402

EXAMPLES = [
    "把方块放到盘子上",
    "把盘子往左挪 5 厘米，再把方块放到盘子上",
    "把方块放到盘子上，如果没抓起来就重试一次",
    "桌上现在有什么",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*", help="要执行的指令；不传就跑内置例子")
    ap.add_argument("--ckpt", default="runs/train/act_human_74_c40/checkpoints/020000/pretrained_model")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--render", action="store_true", help="把每步画面存成图")
    args = ap.parse_args()

    texts = args.texts or EXAMPLES

    from src.agent import Executor, make_plan
    from eval_act import load_policy          # 复用评估脚本的加载逻辑（act/diffusion 都认）

    env = PickPlaceEnv(render=True)
    ckpt = ROOT / args.ckpt
    cfg, policy, pre, post = load_policy(ckpt)
    device = next(policy.parameters()).device
    ex = Executor(env, policy, pre, post, device)

    out_dir = ROOT / "runs" / "agent"
    if args.render:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, text in enumerate(texts):
        print(f"\n=== 【{text}】")
        try:
            plan = make_plan(text)
        except Exception as e:  # noqa: BLE001
            print(f"  规划失败：{str(e)[:200]}")
            continue
        print(f"  计划（{plan.provider}）：{plan.explain}")
        for s in plan.steps:
            print(f"    - {s.tool} {s.args}")
        print(f"  失败策略：{plan.on_failure}，最多重试 {plan.max_retries} 次")

        env.reset(seed=args.seed + i)
        result = ex.run(plan)
        print(f"  结果：{'成功' if result['ok'] else '失败 @ ' + result.get('failed_at', '?')}")
        if args.render:
            env.save_frame(out_dir / f"case{i}.png") if hasattr(env, "save_frame") else None
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
