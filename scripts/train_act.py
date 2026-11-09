r"""ACT 训练入口：在 LeRobot 官方训练脚本外面套一层。

为什么要套这一层（三个都是 Windows 上真实踩到的）：
    1. 中文路径：LeRobot 存视频时用系统默认编码写 ffconcat 清单，中文路径会乱码。
       这里先把进程切到 UTF-8 模式。
    2. 符号链接：训练时它要把 checkpoints/last 建成指向最新存档的符号链接，
       而 Windows 建符号链接需要管理员权限或开发者模式，否则 WinError 1314，
       整个训练直接崩在第一次存档。这里改成退路：先试符号链接，不行就用
       目录联接（junction），再不行就跳过并提示。
    3. 参数太多：默认值写死在这，队友不用记那一长串。

跑法：
    & "D:\envs\embodied\python.exe" scripts\train_act.py --smoke      # 30 步检查链路
    & "D:\envs\embodied\python.exe" scripts\train_act.py              # 正式训练
    & "D:\envs\embodied\python.exe" scripts\train_act.py --steps 5000 --batch-size 16
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()   # 必须在任何 lerobot 导入之前

DEFAULT_DATASET = ROOT / "data" / "grasp_v2"
DEFAULT_OUT = ROOT / "runs" / "train" / "act_grasp_v3"


def parse_episodes(spec: str) -> list[int]:
    """把 "0-19" 或 "0,3,5" 解析成 episode 下标列表。

    用于「只换数据条数」的对照实验：同一套超参，只把训练集从 300 条抽到 20 条。
    """
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# ------------------------------------------------------------------ 符号链接补丁
def _patch_symlink() -> None:
    """把 lerobot 的 update_last_checkpoint 换成 Windows 上不会崩的版本。"""
    try:
        from lerobot.common import train_utils
    except ImportError:
        return

    def update_last_checkpoint(checkpoint_dir: Path) -> Path:
        last = checkpoint_dir.parent / "last"

        # 清掉旧的链接。注意不能用 shutil.rmtree：如果它是目录联接，
        # rmtree 会钻进去把真正的存档删掉。os.rmdir 只摘链接本身。
        if last.is_symlink():
            last.unlink()
        elif last.exists():
            try:
                os.rmdir(last)
            except OSError:
                shutil.rmtree(last, ignore_errors=True)

        target = checkpoint_dir.relative_to(checkpoint_dir.parent)
        try:
            last.symlink_to(target)
        except OSError:
            # 没有符号链接特权 → 用目录联接（junction），这个不需要管理员
            r = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(last), str(checkpoint_dir.resolve())],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"[警告] 无法创建 checkpoints/last（{r.stderr.strip()[:80]}）。"
                      f"训练继续，但 --resume 用不了，评估时请直接指定步数目录。",
                      file=sys.stderr)
        return last

    train_utils.update_last_checkpoint = update_last_checkpoint
    # save_checkpoint 内部是通过模块全局名调用它的，所以换掉属性就生效


# ------------------------------------------------------------------ 主流程
def build_argv(a: argparse.Namespace) -> list[str]:
    dataset: Path = a.dataset.resolve()
    out: Path = a.output.resolve()
    argv = [
        f"--dataset.repo_id=local/{dataset.name}",
        f"--dataset.root={dataset}",
        "--dataset.video_backend=pyav",      # 没装 torchcodec，显式指定免得它反复警告
        f"--policy.type={a.policy}",
        "--policy.push_to_hub=false",        # 不写成会被要求给 repo_id
        f"--policy.device={a.device}",
        f"--output_dir={out}",
        f"--job_name={out.name}",
        f"--batch_size={a.batch_size}",
        f"--num_workers={a.workers}",
        f"--steps={a.steps}",
        f"--save_freq={a.save_freq}",
        f"--log_freq={a.log_freq}",
        f"--seed={a.seed}",
        "--wandb.enable=false",              # 本地跑，不联外网
    ]
    if a.episodes:
        eps = parse_episodes(a.episodes)
        argv.append(f"--dataset.episodes=[{','.join(map(str, eps))}]")
    if a.amp:
        argv.append("--policy.use_amp=true")
    if a.resume:
        argv.append("--resume=true")
    if a.augment:
        # 图像增强：随机色彩抖动 + 轻微平移。示范少的时候（比如 20 条人类示范）
        # 有用，能压一点过拟合；示范充足（300 条）时反而可能拖慢收敛。
        argv.append("--dataset.image_transforms.enable=true")
    if a.chunk_size:
        # 一次预测多少步。默认 100 步 @20 Hz = 5 秒，比 ALOHA 的 100 步 @50 Hz（2 秒）长得多；
        # 而且我们每条轨迹只有约 100 步，等于一条轨迹一整块，模型从没"走到一半再被问一次"。
        # 调小之后一条轨迹会被切成多块，闭环重规划才有可能落在训练分布里。
        argv.append(f"--policy.chunk_size={a.chunk_size}")
    if a.n_action_steps:
        argv.append(f"--policy.n_action_steps={a.n_action_steps}")
    if a.reset_after is not None:
        argv.append(f"--policy.drop_n_last_frames={a.reset_after}")
    if a.policy_path:
        # 从已有权重接着训（微调）。注意路径必须是 .../pretrained_model 目录。
        # LeRobot 规定 --policy.path 和 --policy.type 不能同时给，所以先把 type 摘掉
        argv = [x for x in argv if not x.startswith("--policy.type=")]
        argv.insert(4, f"--policy.path={a.policy_path.resolve()}")
    if a.n_obs_steps:
        argv.append(f"--policy.n_obs_steps={a.n_obs_steps}")
    return argv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--episodes", default=None,
                    help="只用这些 episode 训练，如 0-19 或 0,3,5；默认用全部（对照实验用）")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--policy", default="act",
                    help="策略类型：act（默认）或 diffusion。人类示范多变时看 diffusion 是否更稳")
    ap.add_argument("--policy-path", type=Path, default=None,
                    help="从已有权重接着训（LeRobot 的 --policy.path）。"
                         "典型用法：先用 300 条机器示范训出精确抓取，再用人类示范微调")
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="一次预测多少步（默认 100）。调小能让一条轨迹切成多块，闭环才有意义")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="每执行多少步重新预测（默认等于 chunk_size）")
    ap.add_argument("--reset-after", type=int, default=None,
                    help="对应 LeRobot 的 drop_n_last_frames，一般不用动")
    ap.add_argument("--n-obs-steps", type=int, default=None,
                    help="策略看几帧历史（默认 1 = 只看当前帧）。"
                         "RoboMimic 那篇的结论是「人类数据上历史帧很关键」，值得开 2~3 试")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2, help="数据加载进程数，0 = 主进程里加载")
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--log-freq", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true", help="混合精度，显存吃紧时开")
    ap.add_argument("--augment", action="store_true",
                    help="开图像增强（色彩抖动 + 平移），示范条数少时压过拟合")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="只跑 30 步，检查链路通不通")
    args = ap.parse_args()

    if not (args.dataset / "meta" / "info.json").exists():
        print(f"数据集不存在或格式不对：{args.dataset}")
        print("先跑：python scripts\\collect.py --episodes 300")
        return 1

    if args.smoke:
        args.steps = 30
        args.batch_size = min(args.batch_size, 8)
        args.save_freq = 10 ** 9
        args.output = args.output.parent / (args.output.name + "_smoke")
        if args.output.exists():
            shutil.rmtree(args.output)

    _patch_symlink()

    from lerobot.scripts.lerobot_train import main as train_main  # noqa: PLC0415

    argv = build_argv(args)
    print("训练参数：")
    for a in argv:
        print("   ", a)
    print()

    sys.argv = [sys.argv[0], *argv]
    train_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
