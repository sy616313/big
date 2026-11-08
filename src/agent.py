r"""智能体层：大模型负责「想」，策略负责「做」。

和 src/llm 的分工：
    src/llm.router.parse()  只做一件事——把一句话翻成 target/place 两个字段。
                            这种活规则也能干，体现不出大模型的价值。
    这里（agent）让它干规则干不了的活：
        - 拆多步：先挪盘子，再放方块
        - 空间换算："盘子左边" → 具体的 dx
        - 看结果再决定：抓空了自己决定要不要重试

工具是固定的几个，模型只输出「用哪个工具、参数多少」的 JSON。
这样它再胡说也只会输出不存在的工具名，被校验挡掉，不会直接乱动机器人。

跑法（无界面，直接驱动仿真）：
    & "<环境>\python.exe" scripts\agent_demo.py "把盘子往左挪 5 厘米，再把方块放到盘子上"
    & "...python.exe" scripts\agent_demo.py "把方块放到盘子上，如果没抓起来就重试一次" --show
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- 工具表
TOOLS: dict[str, dict] = {
    "look": {
        "desc": "看一眼场景，返回方块位置、盘子位置、夹爪是否夹着东西",
        "args": {},
    },
    "move_plate": {
        "desc": "把盘子相对挪动，单位米。dx 正数往 +x（远离机器人），dy 正数往 +y",
        "args": {"dx": "米", "dy": "米"},
    },
    "grasp_and_place": {
        "desc": "让训练好的策略执行一次「抓方块 → 放到盘子」，返回成功与否",
        "args": {},
    },
    "expert_grasp_and_place": {
        "desc": "备用方案：不用学习策略，直接用逆解走一条确定的轨迹。"
                "当 grasp_and_place 失败、或者要求必须完成时用这个",
        "args": {},
    },
    "home": {
        "desc": "机械臂回到待机位",
        "args": {},
    },
}

POLICY_EXEC = 10          # 策略每执行多少步重规划一次（实测最优）
MAX_POLICY_STEPS = 400


def _tool_doc() -> str:
    lines = []
    for name, spec in TOOLS.items():
        args = "、".join(f"{k}（{v}）" for k, v in spec["args"].items()) or "无"
        lines.append(f"- {name}：{spec['desc']}。参数：{args}")
    return "\n".join(lines)


SYSTEM_PROMPT = """你是仿真机械臂的调度大脑。用户用中文说一句话，你要拆成几个工具调用。

可用工具：
{tools}

输出要求：**只输出一个 JSON 对象**，不要解释、不要 Markdown 代码块。格式：
{{
  "steps": [{{"tool": "工具名", "args": {{...}}}}, ...],
  "explain": "一句话说明你打算怎么做",
  "on_failure": "retry" 或 "stop",
  "max_retries": 0 到 3 的整数
}}

规则：
1. 只用上面列出的工具，不要编造。
2. 一次执行只有一个「抓方块放到盘子」的动作（grasp_and_place 只能出现一次）。
3. 用户说"左边/右边/前面/后面"这类方位，换算成 move_plate 的 dx/dy：
   +x 是远离机器人的方向，+y 是机器人左手边。1 厘米 = 0.01。
4. 用户说"如果失败就重试"，就把 on_failure 设成 retry 并给 max_retries；没说就 stop、0。
5. 如果用户只是问场景（"桌上有什么"），只用 look。
6. 参数必须是数字，不要带单位、不要写字符串形式的数字。
"""


@dataclass
class Step:
    tool: str
    args: dict[str, float] = field(default_factory=dict)


@dataclass
class AgentPlan:
    steps: list[Step]
    explain: str = ""
    on_failure: str = "stop"
    max_retries: int = 0
    provider: str = ""

    def to_dict(self) -> dict:
        return {
            "steps": [{"tool": s.tool, "args": s.args} for s in self.steps],
            "explain": self.explain,
            "on_failure": self.on_failure,
            "max_retries": self.max_retries,
            "provider": self.provider,
        }


class PlanError(RuntimeError):
    pass


def _strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 有些模型会在 JSON 前后加话，抓第一个花括号块
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            text = text[i:j + 1]
    return text


def validate_plan(raw: str) -> AgentPlan:
    """模型说的话不能直接信：工具名、参数、次数都要过一遍。"""
    try:
        data = json.loads(_strip_fence(raw))
    except json.JSONDecodeError as e:
        raise PlanError(f"不是合法 JSON：{raw[:160]}") from e
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise PlanError("缺少 steps 数组")

    steps: list[Step] = []
    for item in data["steps"]:
        if not isinstance(item, dict):
            raise PlanError("steps 里有一项不是对象")
        tool = item.get("tool")
        if tool not in TOOLS:
            raise PlanError(f"工具「{tool}」不存在，可用：{list(TOOLS)}")
        raw_args = item.get("args") or {}
        if not isinstance(raw_args, dict):
            raise PlanError(f"{tool} 的 args 不是对象")
        args: dict[str, float] = {}
        for k, v in raw_args.items():
            if k not in TOOLS[tool]["args"]:
                raise PlanError(f"{tool} 没有参数 {k}")
            try:
                args[k] = float(v)
            except (TypeError, ValueError) as e:
                raise PlanError(f"{tool}.{k} 不是数字：{v!r}") from e
        steps.append(Step(tool, args))
    if not steps:
        raise PlanError("steps 是空的")

    n_grasp = sum(1 for s in steps if s.tool == "grasp_and_place")
    if n_grasp > 1:
        raise PlanError("grasp_and_place 只能出现一次")

    on_failure = str(data.get("on_failure", "stop")).lower()
    if on_failure not in ("retry", "stop"):
        on_failure = "stop"
    try:
        max_retries = int(data.get("max_retries", 0))
    except (TypeError, ValueError):
        max_retries = 0
    max_retries = max(0, min(3, max_retries))
    return AgentPlan(steps=steps, explain=str(data.get("explain", ""))[:120],
                     on_failure=on_failure, max_retries=max_retries)


def make_plan(text: str, feedback: str = "") -> AgentPlan:
    """让大模型出计划。feedback 非空时，把它看到的上一步结果一起发过去，
    相当于「出事了，你重新想一下」——这是智能体和一次性解析器的分界线。"""
    from src.llm.router import InstructionParser

    parser = InstructionParser()
    system = SYSTEM_PROMPT.format(tools=_tool_doc())
    user = text if not feedback else f"{text}\n\n【执行情况】\n{feedback}\n请根据实际结果重新给一份计划。"
    raw, provider = parser.ask(system, user)
    plan = validate_plan(raw)
    plan.provider = provider
    return plan


# ---------------------------------------------------------------- 执行器
class Executor:
    """把工具调用落到仿真上。不依赖网页，命令行和网页共用同一套。"""

    def __init__(self, env, policy=None, pre=None, post=None, device=None,
                 seed: int | None = None) -> None:
        self.env = env
        self.policy = policy
        self.pre, self.post, self.device = pre, post, device
        self.seed = seed          # 复位时用同一个种子，保证"重来一局"还是同一个场景
        self.trace: list[dict] = []

    # ---- 工具实现 ----
    def look(self) -> dict:
        c = self.env.cube_xyz()
        p = self.env.plate_xyz()
        grip = float(self.env.state()[4])
        return {"cube": [round(float(v), 3) for v in c],
                "plate": [round(float(v), 3) for v in p],
                "gripper": round(grip, 4),
                "holding": bool(c[2] > 0.05 and grip < 0.010)}

    def move_plate(self, dx: float = 0.0, dy: float = 0.0) -> dict:
        p = self.env.plate_xyz()
        # mocap：盘子每局真实位置不同，必须读实际的再加偏移
        x, y = float(p[0]) + dx, float(p[1]) + dy
        self.env.data.mocap_pos[self.env.plate_mid] = [x, y, 0.008]
        for _ in range(60):                       # 让物理稳定一下
            self.env.step(self.env.state())
        now = self.env.plate_xyz()
        return {"plate": [round(float(v), 3) for v in now],
                "moved": [round(float(now[0] - p[0]), 3), round(float(now[1] - p[1]), 3)]}

    def home(self) -> dict:
        # 用同一个种子复位：等于"这一局重来"，场景不变，否则每次重试都在比不同难度的题
        self.env.reset(seed=self.seed) if self.seed is not None else self.env.reset()
        return {"home": True}

    def expert_grasp_and_place(self) -> dict:
        """备用方案：解析专家直接算轨迹，不依赖学习策略。

        进这个工具前先把场景和机械臂复位：策略失败时手臂停在乱七八糟的姿态上，
        方块也常被碰偏，直接在原地接专家轨迹一定会算错。
        复位等于"这一局重来一遍"，演示时要说明白。

        用法是「学习策略先上，失败了再切它」——工业上常见的兜底结构，
        不是作弊，但必须说清楚哪一段是学出来的、哪一段是备用。
        """
        from sim.env import ScriptedExpert

        self.home()
        expert = ScriptedExpert(self.env)
        expert.reset()                      # 注意：构造函数不会自己 reset，忘了这步 plan 是空的
        for _ in range(MAX_POLICY_STEPS):
            if expert.done:
                break
            self.env.step(expert.act())
        good = self.env.success()
        c = np.array(self.env.cube_xyz())
        p = np.array(self.env.plate_xyz())
        return {"ok": bool(good), "fallback": "expert",
                "cube": [round(float(v), 3) for v in c],
                "dist_mm": round(float(np.hypot(c[0] - p[0], c[1] - p[1])) * 1000, 1)}

    def grasp_and_place(self) -> dict:
        if self.policy is None:
            raise PlanError("这个执行器没加载策略，不能抓放")
        import torch

        self.policy.reset()
        obs = self.env.obs()
        steps = 0
        queue: list[np.ndarray] = []
        while steps < MAX_POLICY_STEPS:
            if not queue:
                batch = {
                    "observation.images.front": torch.from_numpy(
                        np.ascontiguousarray(obs["image"])).float().div_(255.0)
                        .permute(2, 0, 1).unsqueeze(0).to(self.device),
                    "observation.state": torch.from_numpy(obs["state"]).float()
                        .unsqueeze(0).to(self.device),
                }
                b = self.pre(batch)
                with torch.no_grad():
                    self.policy.reset()
                    queue = [self.post(self.policy.select_action(b)).squeeze(0)
                             .float().cpu().numpy() for _ in range(POLICY_EXEC)]
            act = np.asarray(queue.pop(0), dtype=np.float64)
            obs = self.env.step(act)
            steps += 1
            if self.env.success():
                break
        good = self.env.success()
        c = np.array(self.env.cube_xyz())
        p = np.array(self.env.plate_xyz())
        return {"ok": bool(good), "steps": steps,
                "cube": [round(float(v), 3) for v in c],
                "dist_mm": round(float(np.hypot(c[0] - p[0], c[1] - p[1])) * 1000, 1)}

    # ---- 执行计划 ----
    def run(self, plan: AgentPlan, verbose: bool = True) -> dict:
        self.trace = []
        for step in plan.steps:
            attempts = 0
            while True:
                try:
                    result = getattr(self, step.tool)(**step.args)
                except (PlanError, TypeError) as e:
                    result = {"error": str(e)}
                self.trace.append({"tool": step.tool, "args": step.args,
                                   "result": result, "attempt": attempts + 1})
                if verbose:
                    print(f"  → {step.tool}({step.args}) = {json.dumps(result, ensure_ascii=False)}")
                failed = result.get("error") or result.get("ok") is False
                if not failed:
                    break
                if plan.on_failure != "retry" or attempts >= plan.max_retries:
                    return {"ok": False, "failed_at": step.tool, "trace": self.trace}
                attempts += 1
                if verbose:
                    print(f"     失败，第 {attempts} 次重试")
                if step.tool == "grasp_and_place":
                    self.home()          # 重试前先退回去，否则原地再抓一次没有意义
        return {"ok": True, "trace": self.trace}

    # ---- 闭环：失败之后让大模型重新出计划 ----
    def run_until_done(self, text: str, max_turns: int = 3, verbose: bool = True) -> dict:
        """跑一轮计划；失败就把实际结果回传给大模型，让它换个办法再试。

        这是智能体和"一次性解析"最关键的区别：它能看到自己的失败。
        """
        history: list[dict] = []
        feedback = ""
        for turn in range(max_turns):
            try:
                plan = make_plan(text, feedback)
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": f"规划失败：{e}", "history": history}
            if verbose:
                tag = "首次计划" if turn == 0 else f"第 {turn + 1} 次改方案"
                print(f"  {tag}（{plan.provider}）：{plan.explain}")
                for s in plan.steps:
                    print(f"    - {s.tool} {s.args}")
            result = self.run(plan, verbose=verbose)
            history.append({"turn": turn + 1, "plan": plan.to_dict(), "result": result})
            if result["ok"]:
                return {"ok": True, "turns": turn + 1, "history": history}
            if plan.on_failure == "stop" and turn == 0:
                # 用户没要求重试：仍给模型一次改方案的机会，这正是它该发挥作用的地方
                pass
            feedback = "\n".join(
                f"- 第 {i + 1} 步 {row['tool']}({row['args']}) 返回 {json.dumps(row['result'], ensure_ascii=False)}"
                for i, row in enumerate(self.trace) if row["result"].get("error") or row["result"].get("ok") is False
            ) or "上一步失败但没拿到细节"
            if verbose:
                print(f"  ↻ 把失败结果回传大模型：{feedback[:120]}")
        return {"ok": False, "error": "改了几次方案都没成", "history": history}
