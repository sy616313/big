# 轻量具身智能体 · 代码仓库

说一句话，虚拟机械臂在仿真里自己把东西抓到指定位置。纯仿真，零硬件成本。

## 先搞清楚：这个项目里有两个模型，别混

| | 大模型（LLM） | 策略模型（ACT） |
|---|---|---|
| 干什么 | 把"那个圆的、装水的"翻译成 `target=红方块, place=盘子` | 看画面，输出机械臂下一步怎么动 |
| 谁训练的 | **我们不训练**，只调用 API | **我们自己训练**，权重是我们的 |
| 存在哪 | 服务商的服务器上，你只有一个 key | `runs/` 目录下的文件，在你硬盘上 |
| API 过期会怎样 | 换个服务商，改 `config.json` 一行 | **完全不受影响，权重还在** |
| 断网会怎样 | 解析不了指令 | 照常跑（本地推理） |

**所以：买 token 只是买"翻译服务"，不是买"模型所有权"。API 到期、余额用完、换供应商，都不会让你的 ACT 权重消失。**
唯一会丢权重的情况是：在云 GPU 上训练完**没把 `runs/` 下载回来**，实例一释放就没了 —— 这是纪律问题，不是技术问题。

## 目录

```
config.json          场景物体、LLM 供应商链、缓存路径
src/llm/             指令解析层（换 API 只改配置）
  providers.py        DeepSeek / 通义 / 任意 OpenAI 兼容接口 / 本地 Ollama / 规则兜底
  router.py           按顺序试、失败自动降级、结果缓存到磁盘
tests/test_llm.py    离线测试（不需要任何 API key）
scripts/check_env.py 环境自检，六项全绿才算装好
docs/prep.md         开工准备清单、环境决策、踩坑记录
docs/环境安装.md      给队友的配置说明（整篇丢给 AI，让它按机器自己适配）
data/                演示数据集（不进 git）
runs/                训练产物、checkpoint、缓存（不进 git）
```

## 给队友

环境**不能复制**（几千个二进制文件，跟显卡驱动绑定），得各自装一份。

不用给你写脚本：把 [docs/环境安装.md](docs/环境安装.md) 整篇丢给你的 AI，让它先探测你的机器（系统、显卡算力、磁盘、conda），再按文档里的版本约束装。装完跑 `scripts/check_env.py`，六行结果发群里。

**唯一必须一致的是版本**（Python 3.12 / torch 2.11.0 / torchvision 0.26.0 / mujoco 3.17.0 / lerobot 0.6.1）；显卡、路径、有没有 N 卡都可以不一样。

## 跑起来

环境（已建好，在 D 盘）：`D:\envs\embodied\python.exe`（Python 3.12 + PyTorch 2.11.0+cu128 + MuJoCo + LeRobot）

```powershell
# 不需要任何 API key，走规则兜底，验证链路通不通
& "D:\envs\embodied\python.exe" tests\test_llm.py

# 单条指令解析
& "D:\envs\embodied\python.exe" -m src.llm.router "把红色方块放到盘子上"
```

要用真 API 时，复制 `.env.example` 为 `.env`（或用系统环境变量）填 key，`config.json` 里 providers 数组顺序就是优先级，第一个失败自动试下一个。

## 两条设计原则

1. **大模型只输出语义，不输出坐标。** `target` 和 `place` 必须是场景物体表里的键，坐标由下游从仿真/检测里取。这样换模型、换说法，坐标不会飘。
2. **永远留一个不需要网络的兜底。** `rule` provider 用关键词匹配，Demo 现场断网也能跑完全程。
