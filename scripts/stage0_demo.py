r"""第 0 步：让机械臂出现在屏幕上。

用法：
    python scripts\stage0_demo.py                # 渲染若干帧到 runs\stage0\，生成 png
    python scripts\stage0_demo.py --viewer       # 打开实时窗口（关掉窗口才退出）
    python scripts\stage0_demo.py --frames 60    # 自定义渲染帧数
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "sim" / "arm_scene.xml"
OUTDIR = ROOT / "runs" / "stage0"


def camera() -> "mujoco.MjvCamera":
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.16, 0.0, 0.13]
    cam.distance = 1.05
    cam.azimuth = 130.0
    cam.elevation = -18.0
    return cam


def scripted_ctrl(t: float) -> np.ndarray:
    """一段演示动作：抬臂 -> 转向方块 -> 下探 -> 合爪 -> 抬起 -> 转向盘子 -> 松开。

    t 是秒。返回 7 个执行器的目标值。
    """
    keys = [
        # 时间   yaw   shoulder elbow wrist roll  grip开合
        (0.0,   0.00, -0.30,  0.90, -0.60, 0.0, 0.030),
        (0.9,   0.00, -0.30,  0.90, -0.60, 0.0, 0.030),
        (1.8,   0.28, -0.95,  1.35, -0.40, 0.0, 0.030),
        (2.8,   0.28, -0.92,  1.40, -0.45, 0.0, 0.030),
        (3.3,   0.28, -0.92,  1.40, -0.45, 0.0, 0.004),
        (4.2,   0.28, -0.55,  1.10, -0.55, 0.0, 0.004),
        (5.4,  -0.55, -0.55,  1.10, -0.55, 0.0, 0.004),
        (6.4,  -0.55, -0.95,  1.42, -0.45, 0.0, 0.004),
        (6.9,  -0.55, -0.95,  1.42, -0.45, 0.0, 0.004),
        (7.4,  -0.55, -0.95,  1.42, -0.45, 0.0, 0.030),
        (8.2,  -0.55, -0.45,  1.00, -0.55, 0.0, 0.030),
        (9.0,   0.00, -0.30,  0.90, -0.60, 0.0, 0.030),
    ]
    ts = [k[0] for k in keys]
    out = np.zeros(7)
    for i in range(6):
        out[i] = np.interp(t, ts, [k[i + 1] for k in keys])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="打开实时窗口")
    ap.add_argument("--frames", type=int, default=8, help="渲染并保存多少张图")
    ap.add_argument("--duration", type=float, default=9.0, help="演示时长（秒）")
    args = ap.parse_args()

    # 注意：项目路径含中文，MuJoCo 的 C 层按 ANSI 打开文件会失败，
    # 所以先读成字符串再交给 from_xml_string（场景里没有外部 mesh，安全）。
    model = mujoco.MjModel.from_xml_string(SCENE.read_text(encoding="utf-8"))
    data = mujoco.MjData(model)
    print(f"场景加载成功：{model.nq} 个自由度，{model.nu} 个执行器，{model.nbody} 个刚体")

    if args.viewer:
        import mujoco.viewer as mjviewer

        data.ctrl[:] = scripted_ctrl(0.0)
        print("打开窗口…… 关掉窗口即退出")
        with mjviewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.16, 0.0, 0.13]
            viewer.cam.distance = 1.05
            viewer.cam.azimuth = 130.0
            viewer.cam.elevation = -18.0
            steps = int(args.duration / model.opt.timestep)
            while viewer.is_running():
                for _ in range(16):          # 每帧推进 16 个物理步，约 1/125 秒
                    t = data.time
                    data.ctrl[:] = scripted_ctrl(t)
                    mujoco.mj_step(model, data)
                    if t > args.duration:
                        data.qpos[:] = model.qpos0  # 循环播放
                        mujoco.mj_forward(model, data)
                viewer.sync()
        return 0

    OUTDIR.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=720, width=960)
    cam = camera()

    steps = int(args.duration / model.opt.timestep)
    save_at = set(np.linspace(0, steps - 1, args.frames).astype(int).tolist())
    saved = []
    from PIL import Image

    for step in range(steps):
        data.ctrl[:] = scripted_ctrl(data.time)
        mujoco.mj_step(model, data)
        if step in save_at:
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            p = OUTDIR / f"frame_{len(saved):02d}.png"
            Image.fromarray(img).save(p)
            saved.append(p)

    # 拼一张 2x2 的对比图，方便一眼看全
    imgs = [np.asarray(Image.open(p)) for p in saved[:4]]
    if len(imgs) == 4:
        top = np.hstack(imgs[0:2])
        bot = np.hstack(imgs[2:4])
        sheet = np.vstack([top, bot])
        sheet_path = OUTDIR / "contact_sheet.png"
        Image.fromarray(sheet).save(sheet_path)
        print(f"拼图已保存：{sheet_path}")

    print(f"共保存 {len(saved)} 帧到：{OUTDIR}")
    print(f"方块最终位置：{data.body('cube').xpos.round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
