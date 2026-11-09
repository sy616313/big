r"""生成给老师看的进度页：单文件 HTML，图片内嵌，双击就能开。

为什么用脚本生成而不是手写 HTML：
    - 图片要 base64 内嵌（单独发文件也不会丢图）
    - 图表用内联 SVG，数字直接从下面的 DATA 里来，改数字不用改排版
    - 所有结论和数字都在这一个文件里，和仓库里的实测记录对得上

跑法：
    & "...python.exe" scripts\make_report_html.py
输出：
    进度汇报.html（项目根目录）
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "进度汇报.html"

# ------------------------------------------------------------------ 数据（全部实测）
# 同一批 80 个新场景（种子 40000 / 70000 各 40 局），chunk 40，每 10 步重规划
MAIN_ROWS = [
    ("人类 74 条", 56, "chunk 40，80 局"),
    ("人类 104 条", 65, "chunk 40，80 局"),
    ("人类 104 条 + 图像增强", 75, "80 局"),
    ("人类 154 条 + 图像增强", 76, "80 局"),
    ("两个模型集成", 84, "80 局（82 / 85）"),
    ("机器示范 300 条", 100, "50 局，对照基线"),
]

STRUCT_ROWS = [               # 动作块长度：同一份数据，只改一次预测多长
    ("chunk 100（5 秒）", 59, "不重规划时最好；每 10 步重规划掉到 15%"),
    ("chunk 40（2 秒）", 76, "每 10 步重规划最优；第一步 0%"),
]

KILLED = [
    ("Diffusion Policy", "41% / 45%", "比 ACT 的 55% 还差，模型类别不是瓶颈"),
    ("时序集成（ACT 论文 +3.3%）", "38% / 59%", "不如「每 10 步重规划」的 76%"),
    ("只保留「典型」示范", "17% vs 24%", "挑一致性好的示范没有帮助"),
    ("删掉带重试的示范（23 条）", "35%", "比 29 条全量的 50% 更差，删数据提纯无效"),
    ("机器数据预训练 → 人类微调", "42% / 45%", "灾难性遗忘，底子被冲掉"),
]

FAIL_SPLIT = [("没夹起来", 15), ("放偏了", 9), ("没松手", 4)]   # 104 模型 80 局

FACTS = [
    ("仿真场景", "137 个几何体 / 19 种材质 / 0 贴图 0 外部模型，全部用代码画"),
    ("逆解", "闭式解，精度 0.00 µm，够不着时显式报错"),
    ("动作空间", "4 关节 + 1 夹爪 = 5 维；相机 320×240；控制 20 Hz"),
    ("数据规模", "机器 300 条 / 23,277 帧；人类 154 条 / 15,299 帧"),
    ("代码", "sim 5 个文件 / scripts 12 个脚本 / 三个带自检的诊断工具"),
    ("复现", "全部命令写进 docs/，一条命令可复现任一实验"),
]

TIMELINE = [
    ("11/09", "人类数据线做到 76%，两模型集成 84%", "done"),
    ("11/10", "网页智能体演示打磨 + 录屏", "next"),
    ("11/11–13", "PPT 主体（22 页上限）", ""),
    ("11/14–15", "视频 + 讲稿 + 排练", ""),
    ("11/16–17", "留白修问题", ""),
    ("11/18", "提交", ""),
]


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def bar_chart(rows, width=760, row_h=34, label_w=250, max_v=100) -> str:
    """横向条形图：左边标签，右边条 + 数值。纯内联 SVG，不依赖任何库。"""
    height = row_h * len(rows) + 10
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">']
    for i, (name, val, note) in enumerate(rows):
        y = i * row_h + 6
        w = max(2.0, (width - label_w - 70) * val / max_v)
        color = "#2563eb" if val >= 75 else ("#f59e0b" if val >= 55 else "#94a3b8")
        parts.append(f'<text x="0" y="{y + 18}" font-size="13" fill="#334155">{html.escape(name)}</text>')
        parts.append(f'<rect x="{label_w}" y="{y + 4}" width="{w:.1f}" height="20" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{label_w + w + 8:.1f}" y="{y + 19}" font-size="13" font-weight="600"'
                     f' fill="#0f172a">{val}%</text>')
    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    hero = ROOT / "docs" / "images" / "hero.jpg"
    montage = ROOT / "docs" / "images" / "policy.jpg"
    web = ROOT / "docs" / "images" / "网页效果_2026-11-08.jpg"
    for p in (hero, montage, web):
        if not p.exists():
            print(f"缺图：{p}")
            return 1

    fail_svg = bar_chart([(k, int(v / 28 * 100), "") for k, v in FAIL_SPLIT],
                         row_h=30, label_w=140, max_v=60)
    main_svg = bar_chart(MAIN_ROWS)
    struct_svg = bar_chart(STRUCT_ROWS, row_h=32, label_w=200)

    killed_rows = "".join(
        f"<tr><td>{html.escape(a)}</td><td class='num'>{html.escape(b)}</td>"
        f"<td>{html.escape(c)}</td></tr>" for a, b, c in KILLED)
    fact_rows = "".join(
        f"<tr><td class='k'>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in FACTS)
    time_rows = "".join(
        f"<tr class='{cls}'><td class='d'>{html.escape(d)}</td><td>{html.escape(t)}</td></tr>"
        for d, t, cls in TIMELINE)

    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>轻量具身智能体 · 进度</title>
<style>
  :root {{
    --ink:#0f172a; --sub:#475569; --line:#e2e8f0; --bg:#f8fafc;
    --blue:#2563eb; --green:#16a34a; --amber:#f59e0b;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
         line-height:1.65; -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:44px 28px 80px; }}
  h1 {{ font-size:30px; margin:0 0 8px; letter-spacing:-.5px; }}
  .lead {{ color:var(--sub); font-size:15px; margin:0 0 6px; }}
  .meta {{ color:#94a3b8; font-size:13px; }}
  h2 {{ font-size:19px; margin:52px 0 14px; padding-bottom:8px;
        border-bottom:2px solid var(--ink); }}
  h3 {{ font-size:15px; margin:26px 0 10px; color:var(--sub); font-weight:600; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:26px 0 8px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:18px 16px; }}
  .card .n {{ font-size:30px; font-weight:700; letter-spacing:-1px; line-height:1.1; }}
  .card .n.blue {{ color:var(--blue); }}
  .card .n.green {{ color:var(--green); }}
  .card .t {{ font-size:13px; color:var(--sub); margin-top:6px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff;
           border:1px solid var(--line); border-radius:12px; overflow:hidden; font-size:14px; }}
  th,td {{ padding:11px 14px; text-align:left; border-bottom:1px solid var(--line); }}
  th {{ background:#f1f5f9; font-weight:600; font-size:13px; color:var(--sub); }}
  tr:last-child td {{ border-bottom:none; }}
  td.num {{ font-variant-numeric:tabular-nums; font-weight:600; }}
  td.k {{ width:130px; color:var(--sub); }}
  td.d {{ width:120px; color:var(--sub); font-variant-numeric:tabular-nums; }}
  .panel {{ background:#fff; border:1px solid var(--line); border-radius:12px;
            padding:18px 20px; margin:14px 0; }}
  .shot {{ width:100%; max-width:820px; margin:0 auto; border-radius:12px;
            border:1px solid var(--line); display:block; }}
  .cap {{ font-size:13px; color:var(--sub); margin-top:8px; }}
  .flow {{ display:flex; align-items:stretch; gap:10px; flex-wrap:wrap; margin:16px 0 6px; }}
  .step {{ flex:1 1 180px; background:#fff; border:1px solid var(--line);
           border-radius:12px; padding:14px 16px; }}
  .step b {{ display:block; font-size:14px; margin-bottom:4px; }}
  .step span {{ font-size:12.5px; color:var(--sub); }}
  .arrow {{ align-self:center; color:#cbd5e1; font-size:22px; }}
  .tag {{ display:inline-block; font-size:12px; padding:2px 8px; border-radius:999px;
          background:#dbeafe; color:#1d4ed8; margin-right:6px; }}
  .tag.g {{ background:#dcfce7; color:#15803d; }}
  .tag.a {{ background:#fef3c7; color:#b45309; }}
  tr.done td {{ color:#15803d; }}
  tr.next td {{ font-weight:600; }}
  .note {{ font-size:13px; color:var(--sub); background:#fff; border-left:3px solid var(--blue);
           padding:10px 14px; border-radius:0 8px 8px 0; margin:14px 0; }}
  code {{ background:#f1f5f9; padding:1px 6px; border-radius:5px; font-size:13px; }}
  @page {{ margin:14mm; }}
  @media print {{
    body {{ background:#fff; }}
    .wrap {{ padding:0; max-width:none; }}
    .card, .panel, table {{ break-inside:avoid; }}
    h2 {{ break-after:avoid; }}
    .shot {{ break-inside:avoid; }}
  }}
  @media (max-width:640px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} }}
</style></head><body><div class="wrap">

<h1>轻量具身智能体 · 当前进度</h1>
<p class="lead">一句话控制机械臂：说「把方块放到盘子上」，大模型拆解任务，模仿学习策略自己看画面完成抓放。</p>
<p class="meta">2026-11-09 · 版本 v1.5 · 仓库 github.com/sy616313/big</p>

<div class="cards">
  <div class="card"><div class="n green">100%</div><div class="t">机器示范策略<br>50 局对照基线</div></div>
  <div class="card"><div class="n blue">84%</div><div class="t">人类示范策略<br>80 局全新场景</div></div>
  <div class="card"><div class="n green">100%</div><div class="t">策略 + 大模型兜底<br>100 局端到端</div></div>
  <div class="card"><div class="n">约 8 秒</div><div class="t">网页里从说一句话<br>到方块放进盘子</div></div>
</div>

<h2>一、系统怎么跑</h2>
<div class="flow">
  <div class="step"><b>① 说一句话</b><span>「把盘子往左挪 5 厘米，再把方块放到盘子上」</span></div>
  <div class="arrow">→</div>
  <div class="step"><b>② 大模型拆计划</b><span>输出 JSON：调用哪个工具、参数多少，严格校验</span></div>
  <div class="arrow">→</div>
  <div class="step"><b>③ 策略执行</b><span>只看相机画面 + 关节角，每 10 步重看一次画面重新规划</span></div>
  <div class="arrow">→</div>
  <div class="step"><b>④ 失败回传</b><span>真实结果给回大模型，换方案再试（实测救回 36/36）</span></div>
</div>
<div class="note">策略的输入只有 <b>相机画面 + 关节角</b>，不知道方块和盘子的坐标——和真实机器人拿到的东西一样。
作为对照，一个能直接读精确坐标的解析专家在同样任务上 100%，但把它的坐标误差加到 20 mm（真实相机量级），它掉到 48%。</div>

<h2>二、看得见的东西</h2>
<img class="shot" src="data:image/jpeg;base64,{b64(montage)}" alt="模型完整抓放过程">
<p class="cap">训练出来的策略跑完一次完整抓放（每帧间隔约 2 秒，实际 20 Hz）：接近 → 下降 → 合爪 → 抬升 → 平移 → 放下。</p>

<img class="shot" style="margin-top:18px" src="data:image/jpeg;base64,{b64(web)}" alt="网页演示">
<p class="cap">网页演示：浏览器里打字下指令，后台跑仿真，画面 MJPEG 实时回传；页面上同时显示大模型给的执行计划、每一步的真实结果、失败后换的方案。</p>

<h2>三、成绩：同一批 80 个全新场景</h2>
<p class="cap">所有数字都是<b>训练时没见过</b>的场景（种子 40000 / 70000 各 40 局）。圈内常见做法是只报 20 局最好的一组，我们按 80 局报。</p>
{main_svg}
<table style="margin-top:10px">
  <tr><th>配置</th><th style="width:90px">成功率</th><th style="width:230px">说明</th></tr>
  {''.join(f"<tr><td>{html.escape(a)}</td><td class='num'>{b}%</td><td>{html.escape(c)}</td></tr>" for a, b, c in MAIN_ROWS)}
</table>

<h3>关键发现：一次预测多长，比数据量更重要</h3>
{struct_svg}
<p class="cap">原来一次预测 100 步（5 秒），而人类每条轨迹只有约 100 步——模型只学过“从起点一口气规划到终点”，部署时任何重规划都是没见过的输入。
改成 40 步（2 秒，和 ALOHA 原论文按秒对齐）后，“重规划越勤越好”这个方向直接反过来了。</p>

<h2>四、走过的弯路（都已实测否掉）</h2>
<table>
  <tr><th style="width:280px">方案</th><th style="width:120px">结果</th><th>结论</th></tr>
  {killed_rows}
</table>

<h2>五、现在的能力边界</h2>
<div class="panel">
  <h3 style="margin-top:0">失败卡在哪（104 条模型，80 局里 28 次失败）</h3>
  {fail_svg}
  <p class="cap">按“卡在哪一步”分类统计，占比是该类占全部失败的比例。抓取是主因。</p>
  <p class="cap">根因量过：模型抓取点误差约 <b>14 mm</b>，而任务容差只有 <b>10 mm</b>（把抓取点整体偏移 10 mm，成功率从 90% 掉到 58%）。两者同量级，所以剩余误差只能靠更稳的策略压，而不是靠继续加数据——实测从 104 条加到 154 条只涨 1 个点。</p>
</div>

<h2>六、工程细节</h2>
<table>{fact_rows}</table>

<h2>七、接下来的计划</h2>
<table>{time_rows}</table>

<h2>八、怎么验证（任何人可复现）</h2>
<div class="panel">
<p class="cap" style="margin-top:0">环境 <code>D:\\envs\\embodied\\python.exe</code>，全部命令在仓库 <code>docs/</code> 里。核心三条：</p>
<p><code>python scripts\\check_env.py</code>　环境自检（7 项）</p>
<p><code>python scripts\\diagnose_data.py</code>　数据体检（自带三道自检，机器示范回放必须 100% 才继续）</p>
<p><code>python scripts\\eval_act.py --ckpt ... --n-episodes 40 --seed-base 40000 --exec-steps 10</code>　在指定场景集上评估</p>
<p><code>双击 打开网页.cmd</code>　启动网页演示（自动加载最优模型 + 智能体模式）</p>
</div>

<img class="shot" style="margin-top:26px" src="data:image/jpeg;base64,{b64(hero)}" alt="仿真场景">
<p class="cap">仿真场景：机械臂、方块、盘子全部由代码用几何体绘制，无任何贴图或外部模型文件。</p>

</div></body></html>
"""
    OUT.write_text(doc, encoding="utf-8")
    size = OUT.stat().st_size / 1024
    print(f"已生成 {OUT.name}：{size:.0f} KB（图片内嵌，单文件可独立发送）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
