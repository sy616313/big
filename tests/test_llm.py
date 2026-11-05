"""离线自测：不需要任何 API key，不需要联网。

跑法：  python tests\test_llm.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.providers import OpenAICompatProvider, ProviderError  # noqa: E402
from src.llm.router import InstructionParser, ParseError  # noqa: E402

SCENE = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["scene"]
TMP = Path(tempfile.mkdtemp(prefix="llm_test_"))
CFG = TMP / "config.json"
CFG.write_text(json.dumps({
    "scene": SCENE,
    "llm": {
        "providers": [{"type": "rule", "name": "rule_fallback"}],
        "cache_path": str(TMP / "cache.jsonl"),
        "max_retries": 0,
    },
}, ensure_ascii=False), encoding="utf-8")

parser = InstructionParser(CFG)
fails = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  <- {extra}" if not cond and extra else ""))
    if not cond:
        fails.append(name)


# 1. 三种说法 -> 同一个目标
for text, want_target, want_place in [
    ("把红色方块放到盘子上", "red_cube", "plate"),
    ("红色那个放盘子上", "red_cube", "plate"),
    ("拿蓝的方块放到盘子", "blue_cube", "plate"),
]:
    plan = parser.parse(text)
    check(f"规则解析「{text}」", plan.target == want_target and plan.place == want_place,
          f"got {plan.target}/{plan.place}")

# 2. 坐标由场景表填，不由模型编
plan = parser.parse("把红色方块放到盘子上")
check("坐标来自场景表", plan.target_xy == SCENE[plan.target]["xy"] and plan.place_xy == SCENE[plan.place]["xy"],
      f"{plan.target_xy} {plan.place_xy}")

# 3. 缓存生效（第二次不再调用 provider）
check("缓存命中", plan.source == "cache", f"source={plan.source}")

# 4. 解析不了要报错，不能瞎填
try:
    parser.parse("把月亮放到盘子上")
    check("无法解析时报错", False, "没有抛异常")
except ParseError:
    check("无法解析时报错", True)

# 5. 没配 key 时要给人话，不是堆栈
try:
    OpenAICompatProvider("deepseek", "https://api.deepseek.com/v1",
                         "deepseek-chat", api_key_env="NOT_SET_KEY_XYZ").complete("s", "u")
    check("缺 key 报可读错误", False, "没有抛异常")
except ProviderError as e:
    check("缺 key 报可读错误", "API Key" in str(e), str(e))

print()
if fails:
    print(f"{len(fails)} 项失败：{fails}")
    raise SystemExit(1)
print("全部通过（无需联网、无需 API key）")
