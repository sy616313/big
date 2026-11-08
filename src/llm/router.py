"""指令解析路由：按配置顺序调用 provider，失败自动降级，结果落盘缓存。

对外只暴露一个函数：
    parse("把红色方块放到盘子上") -> Plan
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from .providers import BaseProvider, ProviderError, build_provider

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.json"

SYSTEM_PROMPT = """你是一个具身智能系统的指令解析器。用户会用中文说一句操作指令，你要解析成结构化 JSON。

场景中存在的物体（只能从这里选，不要发明新的键名）：
{scene}

输出要求：
- 只输出 JSON，不要解释，不要 Markdown 代码块
- 字段：target（要拿的物体）、place（放到哪里）、confidence（0 到 1，你的把握）
- target 和 place 必须是上面物体列表里的键名
- 不要输出任何坐标，坐标由下游程序自己提供

示例：
输入：把红色方块放到盘子上
输出：{{"target": "red_cube", "place": "plate", "confidence": 0.95}}"""


@dataclass
class Plan:
    target: str
    place: str
    target_xy: list
    place_xy: list
    confidence: float
    provider: str
    source: str  # llm / cache / rule

    def to_dict(self) -> dict:
        return asdict(self)


class ParseError(RuntimeError):
    pass


def _strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class InstructionParser:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.scene: dict = self.config["scene"]
        self.llm_cfg: dict = self.config["llm"]
        self.cache_path = ROOT / self.llm_cfg.get("cache_path", "runs/llm_cache.jsonl")
        self.providers: list[BaseProvider] = [
            build_provider(spec, scene=self.scene) for spec in self.llm_cfg["providers"]
        ]
        self._cache: dict[str, dict] = {}
        self._load_cache()

    # ---------------- 缓存：指令 -> 解析结果，避免重复花钱 ----------------
    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                self._cache[row["key"]] = row["plan"]
            except (json.JSONDecodeError, KeyError):
                continue

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]

    def _save_cache(self, key: str, text: str, plan: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "text": text, "plan": plan}, ensure_ascii=False) + "\n")

    # ---------------- 校验：任何 provider 的输出都要过这一关 ----------------
    def _validate(self, raw: str) -> dict:
        try:
            data = json.loads(_strip_fence(raw))
        except json.JSONDecodeError as e:
            raise ParseError(f"不是合法 JSON：{raw[:120]}") from e
        target, place = data.get("target"), data.get("place")
        if target not in self.scene:
            raise ParseError(f"target「{target}」不在场景物体表里：{list(self.scene)}")
        if place not in self.scene:
            raise ParseError(f"place「{place}」不在场景物体表里：{list(self.scene)}")
        if target == place:
            raise ParseError("target 和 place 相同")
        try:
            conf = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return {"target": target, "place": place, "confidence": max(0.0, min(1.0, conf))}

    def _to_plan(self, parsed: dict, provider: str, source: str) -> Plan:
        return Plan(
            target=parsed["target"],
            place=parsed["place"],
            target_xy=self.scene[parsed["target"]]["xy"],
            place_xy=self.scene[parsed["place"]]["xy"],
            confidence=parsed["confidence"],
            provider=provider,
            source=source,
        )

    # ---------------- 主入口 ----------------
    def parse(self, text: str, use_cache: bool = True) -> Plan:
        text = text.strip()
        if not text:
            raise ParseError("指令为空")

        key = self._key(text)
        if use_cache and key in self._cache:
            return self._to_plan(self._cache[key], provider="cache", source="cache")

        scene_desc = "\n".join(
            f"- {name}：颜色={obj.get('color') or '无'}，形状={obj.get('shape')}"
            for name, obj in self.scene.items()
        )
        system = SYSTEM_PROMPT.format(scene=scene_desc)

        errors: list[str] = []
        retries = int(self.llm_cfg.get("max_retries", 1))
        for provider in self.providers:
            for attempt in range(retries + 1):
                try:
                    parsed = self._validate(provider.complete(system, text))
                except (ProviderError, ParseError) as e:
                    errors.append(f"[{provider.name} #{attempt + 1}] {e}")
                    continue
                self._save_cache(key, text, parsed)
                self._cache[key] = parsed
                return self._to_plan(parsed, provider=provider.name, source=provider.kind)
        raise ParseError("所有 provider 都失败了：\n  " + "\n  ".join(errors))


    # ---------------- 通用问答：给 agent 用，不校验成 target/place ----------------
    def ask(self, system: str, user: str, use_cache: bool = True) -> tuple[str, str]:
        """按同一套 provider 链发一次通用请求，返回 (模型原文, provider 名)。

        和 parse() 的区别：不套场景表校验，返回什么就是什么——agent 要的是
        计划 JSON，字段比 target/place 复杂得多，让调用方自己校验。
        缓存单独放一个文件，免得和指令解析的缓存混在一起。
        """
        key = hashlib.sha1((system + "\x00" + user).encode("utf-8")).hexdigest()[:16]
        cache_path = ROOT / "runs" / "agent_cache.jsonl"
        if use_cache and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("key") == key:
                    return row["out"], "cache"

        errors: list[str] = []
        retries = int(self.llm_cfg.get("max_retries", 1))
        for provider in self.providers:
            for attempt in range(retries + 1):
                try:
                    out = provider.complete(system, user)
                except ProviderError as e:
                    errors.append(f"[{provider.name} #{attempt + 1}] {e}")
                    continue
                if not out.strip():
                    errors.append(f"[{provider.name} #{attempt + 1}] 返回空")
                    continue
                if use_cache:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": key, "out": out}, ensure_ascii=False) + "\n")
                return out, provider.name
        raise ParseError("所有 provider 都失败了：\n  " + "\n  ".join(errors))


def parse(text: str) -> Plan:
    return InstructionParser().parse(text)


def _main() -> int:
    args = sys.argv[1:]
    if not args:
        args = ["把红色方块放到盘子上"]
    for text in args:
        try:
            plan = parse(text)
        except ParseError as e:
            print(f"FAIL  {text}\n  {e}")
            continue
        print(f"OK    {text}")
        print("      " + json.dumps(plan.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
