r"""多模型集成：N 个策略各自看同一帧，动作逐维平均。

为什么需要它：
    评估脚本（scripts/eval_ensemble.py）和网页演示（sim/server.py）必须用同一份
    集成代码。否则「网页跑的是 76% 的单模型，报告写的是 84% 的集成」——这种不一致
    被评委问一句就露馅。

为什么集成有用、以及为什么不是模型越多越好：
    每个模型的误差里有一份是「随机的」（初始化、数据顺序、增强抽样），平均能压掉；
    但系统性偏差压不掉。再加上一个明显更差的成员，平均动作会被它往错的方向拽——
    实测：两个成员（76% / 75%）→ 84%；加进第三个 68.75% 的成员 → 掉到 74%。
    所以这里只做「等权平均」，谁的分数不够就不进这个名单，不搞自动加权。

用法：
    from sim.ensemble import EnsemblePolicy
    ens = EnsemblePolicy.from_paths([ckpt_a, ckpt_b], exec_steps=10)
    ens.reset()
    acts = ens.act(raw_batch, n=10)      # 返回 10 个 5 维动作（已反归一化）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["EnsemblePolicy", "load_one"]


def load_one(path: Path):
    """按存档里记的 policy.type 选类，act / diffusion 都能加载。"""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(path))
    if cfg.type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as Cls
    elif cfg.type == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as Cls
    else:
        raise SystemExit(f"还不认识 policy.type={cfg.type}")
    policy = Cls.from_pretrained(str(path), config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path))
    return cfg, policy, pre, post


class EnsemblePolicy:
    """把 N 个 (policy, pre, post) 包成一个「像单个策略」的对象。

    对外只暴露演示/评估真正用到的三样：config（读 n_action_steps）、reset()、act()。
    刻意不实现 select_action——每个成员的 preprocessor 归一化统计不同，必须先各自
    pre 再各自推理，把这一步藏在 act() 里，调用方就不可能用错。
    """

    def __init__(self, members: list[dict], grip_mode: str = "mean",
                 weights: list[float] | None = None) -> None:
        if not members:
            raise ValueError("集成至少需要一个模型")
        if grip_mode not in ("mean", "median", "median_all"):
            raise ValueError(f"grip_mode 只支持 mean / median / median_all，收到 {grip_mode}")
        if weights is not None:
            if len(weights) != len(members):
                raise ValueError(f"权重个数 {len(weights)} 和模型个数 {len(members)} 不一致")
            w = np.asarray(weights, dtype=np.float64)
            if np.any(w < 0) or w.sum() <= 0:
                raise ValueError("权重必须非负且和大于 0")
            weights = (w / w.sum()).tolist()
        self.weights = weights
        self.members = members
        self.grip_mode = grip_mode
        self.config = members[0]["policy"].config
        self.device = members[0]["dev"]
        # 所有成员的动作维度必须一致，否则没法逐维平均
        dims = {int(getattr(m["policy"].config, "action_dim", 0) or 0) for m in members}
        if len(dims) > 1:
            raise ValueError(f"成员的 action_dim 不一致：{sorted(dims)}")

    @classmethod
    def from_paths(cls, paths, exec_steps: int | None = None,
                   grip_mode: str = "mean", weights=None) -> "EnsemblePolicy":
        members = []
        for p in paths:
            p = Path(p).expanduser().resolve()   # 相对路径带反斜杠会被当成 hf repo id
            if not (p / "config.json").exists():
                raise SystemExit(f"找不到模型：{p}")
            cfg, policy, pre, post = load_one(p)
            if exec_steps:
                policy.config.n_action_steps = exec_steps
            members.append({"name": p.parents[2].name, "policy": policy,
                            "pre": pre, "post": post,
                            "dev": next(policy.parameters()).device})
        # 全部丢到同一个设备，省得 act() 里每步判断
        dev = members[0]["dev"]
        for m in members:
            if m["dev"] != dev:
                m["policy"].to(dev)
                m["dev"] = dev
        return cls(members, grip_mode=grip_mode, weights=weights)

    @property
    def names(self) -> list[str]:
        return [m["name"] for m in self.members]

    def reset(self) -> None:
        for m in self.members:
            m["policy"].reset()

    def act(self, raw_batch: dict, n: int) -> list[np.ndarray]:
        """对同一帧画面出 n 步动作，返回平均后的 n 个 5 维向量（已反归一化）。

        raw_batch 必须是「没经过 preprocessor」的原始 batch：每个成员有自己的
        归一化统计，在这里各自动手。
        """
        import torch

        per_model: list[list[np.ndarray]] = []
        for m in self.members:
            b = m["pre"](raw_batch)
            with torch.no_grad():
                m["policy"].reset()
                per_model.append([
                    np.asarray(m["post"](m["policy"].select_action(b)).squeeze(0)
                               .float().cpu().numpy(), dtype=np.float64)
                    for _ in range(n)])
        if len(per_model) == 1:
            return per_model[0]
        if self.grip_mode == "median_all":
            # 逐维中位数：对"某个模型跑偏"最稳健，条件是成员数 ≥ 3
            return [np.median([o[i] for o in per_model], axis=0) for i in range(n)]
        # 先各自转 float64 再平均：float32 平均会让个别步差一点，
        # 混沌系统里这点差别能让一条轨迹多走一步，复现旧数字时对不上
        out = [np.average([o[i] for o in per_model], axis=0, weights=self.weights)
               for i in range(n)]
        if self.grip_mode == "median":
            # 夹爪那一维不要取平均：两个模型一个要合爪、一个还张着，平均出来半开——
            # 抓取时刻这种"不干脆"的指令最容易变成"没夹起来"。改成取中位数（多数派）。
            for i in range(n):
                out[i][4] = float(np.median([o[i][4] for o in per_model]))
        return out
