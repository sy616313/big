"""指令解析层。

用法：
    from src.llm import parse
    plan = parse("把红色方块放到盘子上")
"""
__all__ = ["InstructionParser", "ParseError", "Plan", "parse"]


def __getattr__(name):
    if name in __all__:
        from . import router
        return getattr(router, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
