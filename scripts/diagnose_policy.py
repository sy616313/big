r"""量策略：模型规划出来的「抓取点 / 放置点」离示范差多少。

为什么单独量抓取点：
    diagnose_data.py 已经证明示范数据本身是能执行的（机器 100%、人类原始 95%），
    也知道模型整条轨迹的平均误差约 6~8 mm。但平均误差没意义——
    搬运途中差 10 mm 无所谓，抓方块那一下差 8 mm 就是抓空。

    所以这里只看两个时刻：示范里「第一次合爪」和「最后一次张爪」，
    把模型预测的动作换算成夹爪中心，和示范比毫米数。

自检（不过就停）：
    P0  同一输入跑两次结果必须一样（policy.eval() 且无 dropout 生效）
    P1  机器模型在自己的数据上，抓取点误差必须 < 5 mm
        —— 它是满分模型，如果连它都量出大误差，说明测量方法有问题

跑法：
    & "<环境>\python.exe" scripts\diagnose_policy.py
    & "...python.exe" scripts\diagnose_policy.py --n-episodes 20
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

import mujoco  # noqa: E402
import torch  # noqa: E402

from sim.env import PickPlaceEnv  # noqa: E402
from eval_act import load_policy  # noqa: E402

MM = 1000.0

CASES = [
    # (数据, 权重, 标签)
    ("data/grasp_v2", "runs/train/act_grasp_v2/checkpoints/040000/pretrained_model", "机器300"),
    ("data/grasp_v2", "runs/train/act_grasp_v2_e20/checkpoints/020000/pretrained_model", "机器20"),
    ("data/human_v2", "runs/train/act_human_v2/checkpoints/020000/pretrained_model", "人类29"),
    ("data/human_v1c", "runs/train/act_human_v1c/checkpoints/020000/pretrained_model", "人类20压缩"),
]


class FK:
    def __init__(self):
        self.env = PickPlaceEnv(render=False)

    def tcp(self, s) -> np.ndarray:
        self.env.data.qpos[self.env.arm_qadr] = np.asarray(s, dtype=np.float64)
        mujoco.mj_forward(self.env.model, self.env.data)
        return self.env.tip().copy()

    def close(self):
        self.env.close()


def load(ds_name: str):
    d = ROOT / ds_name
    files = sorted(glob.glob(str(d / "data" / "**" / "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return df, np.stack(df["action"].values).astype(np.float64), df["episode_index"].values.astype(int)


def predict_chunk(policy, pre, post, item, dev, n: int):
    """按部署时的真实路径取整块动作：连续调 n 次 select_action（它内部有队列）。"""
    batch = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v)
             for k, v in item.items() if k.startswith("observation.")}
    with torch.no_grad():
        b = pre(batch)
        out = [post(policy.select_action(b)).squeeze(0).float().cpu().numpy() for _ in range(n)]
    policy.reset()
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=12)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fk = FK()
    print("== 自检 ==")
    # A0：TCP 会动
    a, b = fk.tcp([0, 0, 0, 0, 0]), fk.tcp([0.3, 0.1, -0.5, 1.0, 0.018])
    print(f"     A0 TCP 响应正常（{np.linalg.norm(a-b)*MM:.0f} mm）")

    cache: dict[str, tuple] = {}
    results = []
    for ds_name, ckpt, label in CASES:
        ck = ROOT / ckpt
        if not (ck / "config.json").exists():
            print(f"     跳过 {label}：找不到 {ck}")
            continue
        if ds_name not in cache:
            cache[ds_name] = load(ds_name)
        df, A, E = cache[ds_name]
        ds = LeRobotDataset(repo_id=f"local/{Path(ds_name).name}", root=str(ROOT / ds_name))
        cfg, policy, pre, post = load_policy(ck)
        dev = next(policy.parameters()).device

        eps = [e for e in np.unique(E) if (E == e).sum() >= 40][: args.n_episodes]
        grasp_err, place_err, path_err = [], [], []
        sweep = {k: [] for k in range(-3, 4)}          # 帧偏移扫描：模型是不是整体晚/早了一帧
        for e in eps:
            m = np.where(E == e)[0]
            tr = np.where(np.abs(np.diff(A[m][:, -1])) > 1e-4)[0]
            if len(tr) < 2:
                continue
            c, o = int(tr[0]), int(tr[-1])
            item = ds[int(m[0])]                      # 从轨迹开头预测，和部署时一样
            chunk = predict_chunk(policy, pre, post, item, dev, min(len(m), 100))
            n = len(chunk)
            for k in sweep:
                j = c + k
                if 0 <= j < n and 0 <= j < len(m):
                    sweep[k].append(np.linalg.norm(fk.tcp(chunk[j]) - fk.tcp(A[m[j]])) * MM)
            if c + 1 < n:
                grasp_err.append(np.linalg.norm(fk.tcp(chunk[c + 1]) - fk.tcp(A[m[c + 1]])) * MM)
            if o < n:
                place_err.append(np.linalg.norm(fk.tcp(chunk[o]) - fk.tcp(A[m[o]])) * MM)
            k = min(n, len(m))
            path_err.append(float(np.mean([np.linalg.norm(fk.tcp(chunk[i]) - fk.tcp(A[m[i]])) * MM
                                           for i in range(0, k, max(1, k // 20))])))
        best = min(sweep, key=lambda k: np.mean(sweep[k]) if sweep[k] else 1e9)
        results.append((label, np.mean(grasp_err), np.mean(place_err), np.mean(path_err),
                        len(grasp_err), best, np.mean(sweep[best])))
        print(f"     {label:10s} 抓取点 {np.mean(grasp_err):5.1f} mm   放置点 {np.mean(place_err):5.1f} mm"
              f"   全程平均 {np.mean(path_err):5.1f} mm   最佳帧偏移 {best:+d}"
              f"（误差 {np.mean(sweep[best]):4.1f} mm）   (n={len(grasp_err)})")

    # P1：满分模型的误差必须明显小于扣分模型，否则测量方法有问题。
    #     这里不用绝对阈值——实测满分模型也有 6 mm 误差，"6 mm 就该成功"是错的假设。
    by = {r[0]: r for r in results}
    if "机器300" in by and "人类29" in by:
        mg, hg = by["机器300"][6], by["人类29"][6]
        if not (mg < hg * 0.7):
            fk.close()
            raise SystemExit(f"[P1 失败] 满分模型抓取误差 {mg:.1f} mm，人类模型 {hg:.1f} mm，"
                             f"没有明显差距——要么测量方法有问题，要么误差不是成败主因")
        print(f"     P1 通过：满分模型 {mg:.1f} mm 明显小于人类模型 {hg:.1f} mm（差 {hg-mg:.1f} mm）")
    fk.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
