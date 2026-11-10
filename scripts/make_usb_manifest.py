r"""给 U 盘生成交接清单：目录树、条数、大小、SHA256。

为什么要有它：
    换机拷回去以后得能验完整性。数据里 human_v1 是不可再生的，模型是几十分钟训出来的，
    拷坏了不能靠肉眼发现。所以每个大文件都记一行 SHA256。

跑法：
    & "...python.exe" scripts\make_usb_manifest.py --root E:\实验数据_2026-11-10
输出：<root>\清单.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SKIP = {"System Volume Information"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root: Path = args.root

    out: list[str] = []
    add = out.append
    add(f"# 交接清单 · {root.name}")

    data_dir = root / "数据"
    rows = []
    if data_dir.exists():
        for ds in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            info = ds / "meta" / "info.json"
            ep = fr = "-"
            if info.exists():
                j = json.loads(info.read_text(encoding="utf-8"))
                ep, fr = j.get("total_episodes"), f"{j.get('total_frames'):,}"
            size = sum(f.stat().st_size for f in ds.rglob("*") if f.is_file())
            rows.append((ds.name, ep, fr, size))
    add("")
    add("## 数据")
    add("")
    add("| 目录 | 条数 | 帧数 | 大小 |")
    add("|---|---|---|---|")
    for name, ep, fr, size in rows:
        add(f"| `{name}` | {ep} | {fr} | {human(size)} |")

    model_rows = []
    model_dir = root / "模型"
    if model_dir.exists():
        for m in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            w = m / "model.safetensors"
            if not w.exists():
                continue
            model_rows.append((m.name, w.stat().st_size, sha256(w)))
    add("")
    add("## 模型（SHA256 用来验完整性）")
    add("")
    add("| 目录 | 大小 | SHA256 |")
    add("|---|---|---|")
    for name, size, digest in model_rows:
        add(f"| `{name}` | {human(size)} | `{digest}` |")

    add("")
    add("## 数据集文件校验")
    add("")
    add("| 文件 | 大小 | SHA256 |")
    add("|---|---|---|")
    if data_dir.exists():
        for ds in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            for f in sorted(ds.rglob("*")):
                if not f.is_file() or f.suffix.lower() not in (".parquet", ".mp4"):
                    continue
                add(f"| `{f.relative_to(root)}` | {human(f.stat().st_size)} | `{sha256(f)}` |")

    add("")
    add("## 根目录文件")
    add("")
    for f in sorted(root.iterdir()):
        if f.is_file() and f.name != "清单.md":
            add(f"- `{f.name}`：{human(f.stat().st_size)}")

    docs = root / "文档"
    if docs.exists():
        add("")
        add("## 文档")
        add("")
        for f in sorted(docs.iterdir()):
            if f.is_file():
                add(f"- `{f.name}`")

    (root / "清单.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"已写 {root / '清单.md'}：{len(out)} 行，"
          f"{len(model_rows)} 个模型，{len(rows)} 个数据集")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
