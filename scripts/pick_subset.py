r"""按「典型程度」给示范排序，产出两个等量、但一致性截然不同的子集。

用途：验证「示范之间的一致性」是不是成功率的主因。
    两组条数相同、场景覆盖相同，唯一差别是轨迹彼此像不像：
        - 典型组：离其他轨迹最近的那 N 条
        - 另类组：离其他轨迹最远的那 N 条
    两边条数一样，所以「数据量」这个变量被控制住了。

跑法：
    & "<环境>\python.exe" scripts\pick_subset.py --dataset data\human_v2 --n 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sim.runtime import ensure_utf8  # noqa: E402

ensure_utf8()

from diagnose_data import FK, load  # noqa: E402


def episode_paths(S, A, E, fk, n_grid: int = 60):
    """每条轨迹换到「方块原点、指向盘子」的坐标系，按归一化时间重采样。"""
    out = {}
    for e in np.unique(E):
        m = np.where(E == e)[0]
        q, a = S[m], A[m]
        tr = np.where(np.abs(np.diff(a[:, -1])) > 1e-4)[0]
        if len(tr) < 2 or len(m) < 10:
            continue
        xy = np.stack([fk.tcp(s) for s in q])
        cube = fk.tcp(q[min(tr[0] + 1, len(q) - 1)])
        plate = fk.tcp(q[tr[-1]])
        d = plate[:2] - cube[:2]
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        ex = d / n
        ey = np.array([-ex[1], ex[0]])
        rel = xy[:, :2] - cube[None, :2]
        idx = np.linspace(0, len(m) - 1, n_grid).round().astype(int)
        out[int(e)] = np.stack([rel @ ex, rel @ ey, xy[:, 2]])[:, idx].T
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/human_v2")
    ap.add_argument("--n", type=int, default=15, help="每组取几条")
    args = ap.parse_args()

    d, S, A, E, info, n_files = load(args.dataset)
    fk = FK()
    paths = episode_paths(S, A, E, fk)
    fk.close()

    eps = sorted(paths)
    if len(eps) < 2 * args.n:
        print(f"只有 {len(eps)} 条轨迹，不够分成两组各 {args.n} 条")
        return 1
    D = np.zeros((len(eps), len(eps)))
    for i, a in enumerate(eps):
        for j, b in enumerate(eps):
            D[i, j] = np.linalg.norm(paths[a] - paths[b], axis=1).mean() if i != j else 0.0
    score = D.sum(axis=1) / max(len(eps) - 1, 1)     # 越大 = 越另类

    order = np.argsort(score)
    typical = [eps[i] for i in order[: args.n]]
    atypical = [eps[i] for i in order[-args.n:]]

    print(f"数据 {d.name}：{len(eps)} 条轨迹")
    print(f"  典型组（{args.n} 条，平均离群 {score[order[:args.n]].mean()*1000:.1f} mm）：{typical}")
    print(f"  另类组（{args.n} 条，平均离群 {score[order[-args.n:]].mean()*1000:.1f} mm）：{atypical}")
    print()
    print("两组的条数相同，唯一差别是彼此像不像。训练命令：")
    print(f"  --dataset {args.dataset} --episodes {','.join(map(str, typical))} --output runs/train/act_typ")
    print(f"  --dataset {args.dataset} --episodes {','.join(map(str, atypical))} --output runs/train/act_aty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
