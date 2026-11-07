"""诊断：人类示范训出来的模型到底是「学得不够」还是「学歪了」。

方法：教师强制。把数据集里的真实观测喂给模型，比较它预测的动作和专家记录的动作。
     同时算一个"笨基线"——不管看到什么都输出数据集里的平均动作。
     如果模型的误差和笨基线差不多，说明它根本没学到东西。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\大挑战者杯校赛")

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CKPT = sys.argv[1] if len(sys.argv) > 1 else r"D:\大挑战者杯校赛\runs\train\act_human_v1\checkpoints\020000\pretrained_model"
DATA = sys.argv[2] if len(sys.argv) > 2 else r"D:\大挑战者杯校赛\data\human_v1"
NAMES = ["yaw", "shoulder", "elbow", "wrist", "grip"]
print(f"模型 {CKPT}\n数据 {DATA}\n")

cfg = PreTrainedConfig.from_pretrained(CKPT)
pol = ACTPolicy.from_pretrained(CKPT, config=cfg)
pol.eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=CKPT)
dev = next(pol.parameters()).device

ds = LeRobotDataset(repo_id="local/human_v1", root=DATA)
n = len(ds)
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(n, size=min(300, n), replace=False))

preds, gts = [], []
for i in idx:
    item = ds[int(i)]
    batch = {
        "observation.images.front": item["observation.images.front"].unsqueeze(0).to(dev),
        "observation.state": item["observation.state"].unsqueeze(0).to(dev),
    }
    with torch.no_grad():
        a = post(pol.select_action(pre(batch)))
    preds.append(a.squeeze(0).cpu().numpy())
    gts.append(item["action"].numpy())
    if len(preds) % 4 == 0:
        pol.reset()          # 每 4 帧重置一次动作队列，避免预测的是旧 chunk

P = np.array(preds)
G = np.array(gts)

print(f"抽样 {len(P)} 帧（每 4 帧重置动作队列）\n")
print(f"{'关节':<10}{'模型误差':>10}{'笨基线误差':>12}{'记录值范围':>22}")
mean_a = G.mean(axis=0)
better = 0
for k, name in enumerate(NAMES):
    e_model = np.abs(P[:, k] - G[:, k]).mean()
    e_base = np.abs(mean_a[k] - G[:, k]).mean()
    rng_s = f"[{G[:, k].min():.3f}, {G[:, k].max():.3f}]"
    print(f"{name:<10}{e_model:>10.4f}{e_base:>12.4f}{rng_s:>22}")
    if e_model < e_base * 0.7:
        better += 1

print(f"\n模型明显优于笨基线的关节数：{better}/5")

# 看模型输出是不是"几乎不动"
print(f"\n模型输出标准差: {np.round(P.std(axis=0), 4).tolist()}")
print(f"记录动作标准差: {np.round(G.std(axis=0), 4).tolist()}")
print("\n如果模型标准差远小于记录值，说明它退化成了'输出恒定动作'。")
