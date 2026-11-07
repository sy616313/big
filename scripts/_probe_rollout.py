"""看模型闭环时机械臂到底怎么动：打印夹爪轨迹和它离方块/盘子的距离。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\大挑战者杯校赛")

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from sim.env import PickPlaceEnv, IMG_W, IMG_H


def to_tensor(img):
    t = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0)
    return t.permute(2, 0, 1).unsqueeze(0)


CKPT = sys.argv[1] if len(sys.argv) > 1 else r"D:\大挑战者杯校赛\runs\train\act_human_v1\checkpoints\020000\pretrained_model"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40

cfg = PreTrainedConfig.from_pretrained(CKPT)
pol = ACTPolicy.from_pretrained(CKPT, config=cfg)
pol.eval(); pol.reset()
pre, post = make_pre_post_processors(cfg, pretrained_path=CKPT)
dev = next(pol.parameters()).device

env = PickPlaceEnv(render=True)
obs = env.reset(seed=10000)
cube = env.cube_xyz().copy()
plate = env.plate_xyz().copy()
print(f"模型 {CKPT}")
print(f"方块 {np.round(cube[:2],3).tolist()}   盘子 {np.round(plate[:2],3).tolist()}")
print(f"起点 爪 {np.round(env.tip(),3).tolist()}\n")
print(f"{'步':>4} {'爪 x':>7} {'爪 y':>7} {'爪 z':>7} {'离方块':>7} {'离盘子':>7} {'夹爪':>7}")
for k in range(STEPS):
    b = {"observation.images.front": to_tensor(obs["image"]).to(dev),
         "observation.state": torch.from_numpy(obs["state"]).float().unsqueeze(0).to(dev)}
    with torch.no_grad():
        a = post(pol.select_action(pre(b))).squeeze(0).float().cpu().numpy()
    tip = env.tip()
    if k % 4 == 0:
        print(f"{k:>4} {tip[0]:>7.3f} {tip[1]:>7.3f} {tip[2]:>7.3f} "
              f"{np.hypot(*(cube[:2]-tip[:2]))*1000:>6.0f}mm "
              f"{np.hypot(*(plate[:2]-tip[:2]))*1000:>6.0f}mm {a[4]:>7.4f}")
    obs = env.step(a)

print(f"\n最终 {STEPS} 步：方块 z={env.cube_xyz()[2]:.3f}  success={env.success()}")
env.close()
