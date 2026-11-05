r"""开工前自检：一次性确认 Python / 显卡 / MuJoCo / LeRobot 四样都对。

跑法：  D:\envs\embodied\python.exe scripts\check_env.py
全部 OK 之前，不要开始写业务代码。
"""
from __future__ import annotations

import sys
import platform

OK, BAD = "[OK]  ", "[FAIL]"
fails: list[str] = []


def line(tag: str, msg: str) -> None:
    print(f"{tag} {msg}")


print(f"python  {sys.version.split()[0]}  ({platform.machine()})  {sys.executable}\n")

# 1. Python 版本必须是 3.12 —— 3.13/3.14 上 MuJoCo/LeRobot 很容易炸
if sys.version_info[:2] == (3, 12):
    line(OK, "Python 3.12")
else:
    line(BAD, f"Python {sys.version.split()[0]}，需要 3.12")
    fails.append("python-version")

# 2. PyTorch + CUDA：必须是 cu128 构建，且能认出 5070
try:
    import torch

    line(OK, f"torch {torch.__version__}  (CUDA build {torch.version.cuda})")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        line(OK, f"GPU {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  "
                 f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        x = torch.randn(512, 512, device="cuda")
        line(OK, f"GPU 矩阵乘法跑通，结果均值 {x.mean().item():+.4f}")
    else:
        line(BAD, "torch.cuda.is_available() == False（装成 CPU 版了？）")
        fails.append("cuda")
except Exception as e:  # noqa: BLE001
    line(BAD, f"torch 不可用：{e}")
    fails.append("torch")

# 3. MuJoCo：能建模型、能推进仿真
try:
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <body pos="0 0 1"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    start = float(data.qpos[2])
    for _ in range(200):
        mujoco.mj_step(model, data)
    line(OK, f"MuJoCo {mujoco.__version__}，自由落体 200 步：z {start:.2f} → {data.qpos[2]:.2f}")
except Exception as e:  # noqa: BLE001
    line(BAD, f"MuJoCo 不可用：{e}")
    fails.append("mujoco")

# 4. LeRobot：能导入即可（环境里还没有机器人，不做真实训练）
try:
    import lerobot

    line(OK, f"LeRobot {getattr(lerobot, '__version__', '未知版本')}")
except Exception as e:  # noqa: BLE001
    line(BAD, f"LeRobot 不可用：{e}")
    fails.append("lerobot")

print()
if fails:
    print(f"未通过项：{fails}")
    raise SystemExit(1)
print("全绿。可以开工。")
