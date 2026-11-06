r"""开工前自检：把「跑起来会炸的坑」一次性全查一遍。

跑法：
    & "D:\envs\embodied\python.exe" scripts\check_env.py

全部 [OK] 再往下走。带 [!!] 的是警告：不致命，但有代价（比如训练时会慢、
或者要靠代码自动兜底）。真正致命的标 [FAIL]，最后会汇总。

这七项不是随便列的，是这轮开发里真炸过的：
    1  Python 版本        3.13/3.14 上 MuJoCo/LeRobot 的预编译包经常没有
    2  CUDA 构建          5070 是 sm_120，cu121 的轮子会报 "no kernel image"
    3  MuJoCo             基本盘
    4  LeRobot            版本对不上，接口全变
    5  数据集依赖         lerobot 只装骨架时，import 数据集会直接 ImportError
    6  ACT 可导入+权重    resnet18 预训练权重要下载，断网会退回随机初始化（掉点很多）
    7  中文路径的两个坑   编码 + 符号链接，都发生在「第 2 条轨迹」和「第一次存档」
"""
from __future__ import annotations

import locale
import platform
import sys
import tempfile
from pathlib import Path

OK, WARN, BAD = "[OK]  ", "[!!]  ", "[FAIL]"

fails: list[str] = []
warns: list[str] = []


def line(tag: str, msg: str) -> None:
    print(f"{tag} {msg}")


print(f"python  {sys.version.split()[0]}  ({platform.machine()})")
print(f"路径    {sys.executable}\n")

# ---------------------------------------------------------------- 1
if sys.version_info[:2] == (3, 12):
    line(OK, "1/7 Python 3.12")
else:
    line(BAD, f"1/7 Python {sys.version.split()[0]}，需要 3.12")
    fails.append("python-version")

# ---------------------------------------------------------------- 2
try:
    import torch

    line(OK, f"2/7 torch {torch.__version__}（CUDA {torch.version.cuda}）")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        line(OK, f"      GPU {props.name}  sm_{cap[0]}{cap[1]}  "
                 f"{props.total_memory / 1024**3:.1f} GB")
        _ = torch.randn(512, 512, device="cuda") @ torch.randn(512, 512, device="cuda")
        line(OK, "      GPU 矩阵乘法跑通")
        if cap[0] == 12 and "+cu12" not in torch.__version__:
            line(WARN, "      sm_120 显卡建议用 cu128 构建的 torch")
            warns.append("torch-build")
    else:
        line(BAD, "2/7 torch.cuda.is_available() == False（装成 CPU 版了？）")
        fails.append("cuda")
except Exception as e:  # noqa: BLE001
    line(BAD, f"2/7 torch 不可用：{e}")
    fails.append("torch")

# ---------------------------------------------------------------- 3
try:
    import mujoco

    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body pos="0 0 1"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    z0 = float(data.qpos[2])
    for _ in range(200):
        mujoco.mj_step(model, data)
    line(OK, f"3/7 MuJoCo {mujoco.__version__}，自由落体 z {z0:.2f} → {data.qpos[2]:.2f}")
except Exception as e:  # noqa: BLE001
    line(BAD, f"3/7 MuJoCo 不可用：{e}")
    fails.append("mujoco")

# ---------------------------------------------------------------- 4
try:
    import lerobot

    ver = getattr(lerobot, "__version__", "未知")
    line(OK, f"4/7 LeRobot {ver}")
    if ver != "0.6.1":
        line(WARN, f"      本项目在 0.6.1 上验证过，当前 {ver}，接口可能有出入")
        warns.append("lerobot-version")
except Exception as e:  # noqa: BLE001
    line(BAD, f"4/7 LeRobot 不可用：{e}")
    fails.append("lerobot")

# ---------------------------------------------------------------- 5
missing = []
for mod in ("datasets", "av", "pandas", "pyarrow", "accelerate", "einops"):
    try:
        __import__(mod)
    except Exception:  # noqa: BLE001
        missing.append(mod)
if missing:
    line(BAD, f"5/7 数据集/训练依赖缺失：{missing}")
    line(BAD, "      pip install 'lerobot[dataset,training]'")
    fails.append("dataset-deps")
else:
    line(OK, "5/7 数据集依赖齐（datasets / av / pandas / pyarrow / accelerate / einops）")

# ---------------------------------------------------------------- 6
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: F401
    from lerobot.policies.factory import make_pre_post_processors  # noqa: F401

    line(OK, "6/7 LeRobotDataset / ACTPolicy / 归一化处理链 均可导入")
except Exception as e:  # noqa: BLE001
    line(BAD, f"6/7 LeRobot 数据集或 ACT 不可用：{str(e)[:90]}")
    fails.append("act")

cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
w = list(cache.glob("resnet18-*.pth"))
if w:
    line(OK, f"      resnet18 预训练权重已缓存（{w[0].stat().st_size / 1e6:.0f} MB）")
else:
    line(WARN, "      resnet18 预训练权重未缓存，首次训练会联网下载 45 MB；"
               "下不动就退回随机初始化，效果会明显变差")
    warns.append("resnet-weights")

# ---------------------------------------------------------------- 7
enc = locale.getpreferredencoding(False)
if sys.flags.utf8_mode or enc.lower().replace("-", "") in ("utf8", "ascii"):
    line(OK, f"7/7 进程编码 {enc}（安全）")
else:
    line(WARN, f"7/7 当前编码 {enc}，中文路径下存视频会失败；"
               "采集/训练脚本会自动用 -X utf8 重启，手动跑 lerobot 命令则要自己加")
    warns.append("encoding")

try:
    with tempfile.TemporaryDirectory() as d:
        tgt = Path(d) / "real"
        tgt.mkdir()
        link = Path(d) / "link"
        link.symlink_to(tgt, target_is_directory=True)
        link.unlink()
    line(OK, "      可以创建符号链接（训练存档的 last 指针没问题）")
except OSError:
    line(WARN, "      不能创建符号链接（Windows 需管理员/开发者模式）；"
               "训练脚本会自动改用目录联接（junction），不用管")
    warns.append("symlink")

# ---------------------------------------------------------------- 汇总
print()
if fails:
    print(f"未通过：{fails}")
    raise SystemExit(1)
if warns:
    print(f"可以开工，但有 {len(warns)} 项警告：{warns}")
else:
    print("全绿，可以开工。")
