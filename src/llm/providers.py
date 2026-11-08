"""大模型调用适配层。

设计目标：换供应商 = 改 config.json，不改调用方代码。
所有 provider 都实现 complete(system, user) -> str（返回模型原始文本）。
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> int:
    """把项目根目录的 .env 读进 os.environ（缺这个依赖，所以自己写十行）。

    约定：只认 `KEY=VALUE`，`#` 开头跳过，已存在的环境变量不覆盖。
    返回读进来的条目数。返回值放在仓库里方便自检查「到底加载了没有」。
    """
    p = path or (ROOT / ".env")
    if not p.exists():
        return 0
    n = 0
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            n += 1
    return n


load_dotenv()


class ProviderError(RuntimeError):
    """单个 provider 调用失败。router 会捕获它并降级到下一个。"""


class BaseProvider:
    name = "base"
    kind = "base"

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


class OpenAICompatProvider(BaseProvider):
    """任何 OpenAI 兼容的 /chat/completions 接口。

    DeepSeek、通义千问（兼容模式）、本地 Ollama、各类中转站都是这一个形状，
    所以换供应商真的只是换 base_url + model + key。
    """

    kind = "openai_compat"

    def __init__(self, name, base_url, model, api_key_env=None, api_key=None,
                 timeout=30, temperature=0.0, json_mode=True):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self._api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.json_mode = json_mode

    def _key(self) -> str:
        key = self._api_key or os.environ.get(self.api_key_env or "", "")
        if not key:
            raise ProviderError(f"{self.name}: 未配置 API Key（环境变量 {self.api_key_env}）")
        return key

    def _post(self, payload: dict) -> str:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key()}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:300]
            raise ProviderError(f"{self.name}: HTTP {e.code} {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(f"{self.name}: 网络不可达（{e}）") from e
        except json.JSONDecodeError as e:
            raise ProviderError(f"{self.name}: 返回的不是 JSON") from e
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"{self.name}: 返回结构异常 {str(body)[:200]}") from e

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
            try:
                return self._post(payload)
            except ProviderError as e:
                # 某些中转/本地模型不支持 response_format，去掉再试一次
                if "400" not in str(e) and "response_format" not in str(e):
                    raise
        return self._post({k: v for k, v in payload.items() if k != "response_format"})


COLOR_WORDS = {
    "红色": "red", "红": "red", "蓝色": "blue", "蓝": "blue",
    "黄色": "yellow", "黄": "yellow", "绿色": "green", "绿": "green",
}
SHAPE_WORDS = {
    "方块": "cube", "方形的": "cube", "方形": "cube", "块": "cube",
    "盘子": "plate", "盘": "plate",
    "杯子": "cup", "杯": "cup",
}


class RuleProvider(BaseProvider):
    """离线兜底：不联网、不花钱，靠关键词匹配。

    精度不高，但保证 Demo 现场断网、API 欠费时整条链路还能跑完。
    """

    kind = "rule"

    def __init__(self, name="rule_fallback", scene=None):
        self.name = name
        self.scene = scene or {}

    @staticmethod
    def _hits(text: str, table: dict) -> list[str]:
        return [v for k, v in sorted(table.items(), key=lambda kv: -len(kv[0])) if k in text]

    def complete(self, system: str, user: str) -> str:
        head, sep, tail = user.partition("放")
        if not sep:
            head, tail = user, ""

        def pick(text: str) -> str | None:
            colors = self._hits(text, COLOR_WORDS)
            shapes = self._hits(text, SHAPE_WORDS)
            for shape in shapes:
                for name, obj in self.scene.items():
                    if obj.get("shape") == shape and (not colors or obj.get("color") in colors):
                        return name
            for color in colors:
                for name, obj in self.scene.items():
                    if obj.get("color") == color:
                        return name
            return None

        target = pick(head) or pick(user)
        place = pick(tail) or pick(user.replace(head, "", 1))
        if not target or not place:
            raise ProviderError(f"{self.name}: 规则无法解析「{user}」（target={target}, place={place}）")
        return json.dumps({"target": target, "place": place, "confidence": 0.55},
                          ensure_ascii=False)


def build_provider(spec: dict, scene: dict | None = None) -> BaseProvider:
    kind = spec.get("type", "openai_compat")
    if kind == "openai_compat":
        return OpenAICompatProvider(
            name=spec.get("name", "openai_compat"),
            base_url=spec["base_url"],
            model=spec["model"],
            api_key_env=spec.get("api_key_env"),
            api_key=spec.get("api_key"),
            timeout=spec.get("timeout", 30),
            temperature=spec.get("temperature", 0.0),
            json_mode=spec.get("json_mode", True),
        )
    if kind == "rule":
        return RuleProvider(name=spec.get("name", "rule_fallback"), scene=scene or {})
    raise ValueError(f"未知 provider 类型：{kind}")
