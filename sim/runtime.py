r"""运行环境的小修小补。

目前的唯一内容：把进程切到 UTF-8 模式。

为什么需要：
    LeRobot 存视频时，会把「这一段视频由哪几个分片拼成」写进一个 .ffconcat 清单，
    而它用的 `tempfile.NamedTemporaryFile(...)` **没有指定 encoding**，于是走系统
    默认编码。中文版 Windows 是 GBK(cp936)，ffmpeg 却按 UTF-8 去读，清单里的中文
    路径就变成乱码，拼接随之失败。

    表现很有迷惑性：第 1 条轨迹存得进去（那时只做文件移动，不拼接），从第 2 条开始
    必然报 `av.error.ValueError: [Errno 22] Invalid argument`。看起来像是数据集损坏，
    实际只是编码。

    项目放在 `D:\大挑战者杯校赛` 这种中文路径下就一定会踩到。

处理方式：
    检测到当前不是 UTF-8 编码，就用 `-X utf8` 重新启动自己。子进程里
    `sys.flags.utf8_mode == 1`，直接返回，不会无限递归。
"""
from __future__ import annotations

import locale
import os
import sys

_MARK = "DSH_UTF8_REEXEC"


def in_utf8_mode() -> bool:
    return bool(sys.flags.utf8_mode) or os.environ.get("PYTHONUTF8") == "1"


def ensure_utf8(verbose: bool = False) -> None:
    """确保当前进程是 UTF-8 编码；不是就原地重启自己。"""
    if in_utf8_mode():
        return

    enc = locale.getpreferredencoding(False).lower().replace("-", "")
    if enc in ("utf8", "ascii", "ansix3.41968"):
        return

    if os.environ.get(_MARK) == "1":
        print(f"警告：已尝试切到 UTF-8 但仍失败（当前 {enc}），中文路径可能出问题",
              file=sys.stderr)
        return

    if verbose:
        print(f"当前编码 {enc}，用 -X utf8 重启一次……", file=sys.stderr)

    env = dict(os.environ, PYTHONUTF8="1")
    env[_MARK] = "1"
    argv = [sys.executable, "-X", "utf8", *sys.argv]
    # 注意：这里不能用 os.execve。Windows 上它会原地替换进程映像，
    # 本机上实测直接 0xC0000005（访问冲突）崩掉。老老实实起子进程再退出。
    import subprocess

    try:
        rc = subprocess.call(argv, env=env)
    except OSError as e:  # noqa: BLE001
        print(f"警告：无法重启到 UTF-8 模式（{e}），中文路径可能出问题", file=sys.stderr)
        return
    sys.exit(rc)
