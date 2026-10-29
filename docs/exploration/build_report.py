"""Generates strategy_report.html (static inline-SVG diagrams, no scripts)."""
import os
from html import escape

def lines(t): return t.split("\n")

class Fig:
    def __init__(s, w, h): s.w, s.h, s.p = w, h, []
    def node(s, x, y, w, h, text, kind="", sub=None):
        s.p.append(f'<rect class="nd {kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="5"/>')
        L = lines(text); y0 = y + h / 2 - (len(L) - 1) * 8 + 4
        t = "".join(f'<tspan x="{x + w/2}" y="{y0 + i*16}">{escape(l)}</tspan>' for i, l in enumerate(L))
        s.p.append(f'<text class="tx" text-anchor="middle">{t}</text>')
    def arrow(s, x1, y1, x2, y2, label=None, lx=None, ly=None, dash=False):
        s.p.append(f'<line class="ar{" dash" if dash else ""}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#a)"/>')
        if label: s.label((x1 + x2) / 2 if lx is None else lx, (y1 + y2) / 2 - 6 if ly is None else ly, label)
    def path(s, d, label=None, lx=0, ly=0, dash=False):
        s.p.append(f'<path class="ar{" dash" if dash else ""}" d="{d}" fill="none" marker-end="url(#a)"/>')
        if label: s.label(lx, ly, label)
    def label(s, x, y, text, anchor="middle", cls="lb"):
        s.p.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{escape(text)}</text>')
    def region(s, x, y, w, h, title, kind=""):
        s.p.append(f'<rect class="rg {kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>')
        s.label(x + 12, y + 18, title, "start", "rt")
    def svg(s, cap, alt):
        defs = '<defs><marker id="a{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ah"/></marker></defs>'
        return f'<figure><div class="fw"><svg viewBox="0 0 {s.w} {s.h}" role="img" aria-label="{escape(alt)}">{defs}{"".join(s.p)}</svg></div><figcaption>{cap}</figcaption></figure>'

N = [0]
def render(f, cap, alt):
    N[0] += 1; n = N[0]
    return f.svg(cap, alt).replace('id="a{n}"', f'id="a{n}"').replace('url(#a)', f'url(#a{n})').replace("{n}", str(n))

FIGS = {}
# ---------- ours: random ----------
f = Fig(760, 200)
f.node(20, 30, 150, 52, "深度图", "ours"); f.node(230, 30, 150, 52, "前方通道检查\n横向 < 0.35 m", "ours"); f.node(440, 30, 150, 52, "按概率选动作", "ours")
f.arrow(170, 56, 230, 56); f.arrow(380, 56, 440, 56)
for cx, t in ((320, "大步前进 0.9 m\n概率 80%"), (480, "小步前进 0.15 m\n概率 50%"), (640, "原地转 ±15°\n其余情况")):
    f.node(cx - 70, 130, 140, 48, t, "ours"); f.arrow(515, 82, cx, 130)
f.label(380, 112, "空间足够", "middle")
FIGS["random"] = render(f, "random：只靠深度图判断前方是否可飞，没有地图和记忆。", "random 策略流程图")

# ---------- ours: nbv ----------
f = Fig(760, 230)
xs = [20, 170, 320, 470, 620]
txt = ["当前位姿\n+ 体素地图", "原地试看\n7 个偏航方向", "每个方向渲染\n一张深度图", "统计未见过\n的体素数", "选增益最大的\n方向并前进"]
for i, (x, t) in enumerate(zip(xs, txt)): f.node(x, 30, 120, 56, t, "priv" if i == 2 else "ours")
for i in range(4): f.arrow(xs[i] + 120, 58, xs[i + 1], 58)
f.label(380, 128, "得分 = 新体素数 + 30·[前方可走] − 5·[需转向]", "middle", "mono")
f.label(380, 168, "虚线框：向仿真器请求假想视角的渲染，真机做不到，因此不能部署。", "middle")
FIGS["nbv"] = render(f, "nbv：每步试看 7 个方向，选能看到最多新体素的方向。", "nbv 策略流程图")

# ---------- ours: frontier ----------
f = Fig(760, 270)
x4 = [20, 200, 380, 560]
t1 = ["深度图 + 位姿", "3D 占据栅格\n未知 / 空闲 / 占据", "找 frontier 格\n水平邻接未知，\n周围未知体积足够", "BFS 找最近的\n≥1 m 的 frontier"]
for x, t in zip(x4, t1): f.node(x, 24, 150, 62, t, "ours")
for i in range(3): f.arrow(x4[i] + 150, 55, x4[i + 1], 55)
f.path("M635 86 L635 112 L95 112 L95 140", None)
t2 = ["沿路径点走\n先转向对齐，\n再大步 / 小步前进", "到达后朝未知方向\n转过去看一眼，\n标记为已访问", "重新规划"]
for x, t in zip(x4[:3], t2): f.node(x, 140, 150, 62, t, "ours")
f.arrow(170, 171, 200, 171); f.arrow(350, 171, 380, 171)
f.node(560, 140, 150, 62, "没有 frontier：\n原地扫描 24×15°，\n再退回随机", "ours")
f.arrow(530, 171, 560, 171, None, dash=True)
f.path("M455 202 L455 232 L95 232 L95 205", "重规划（每 25 步或路径用完）", 275, 226, dash=True)
FIGS["frontier"] = render(f, "frontier：用占据栅格找已知与未知的交界，BFS 选最近的去探索。", "frontier 策略流程图")

# ---------- ours: sensor-only ----------
f = Fig(760, 330)
f.region(10, 10, 290, 310, "环境（特权，agent 看不到）", "priv")
f.region(330, 10, 420, 310, "Agent（只用传感器和自己的指令）", "ours")
f.node(30, 44, 250, 70, "按真实动力学执行动作\n前进 +3% 噪声\n偏航 ±0.5° 噪声", "priv")
f.node(30, 134, 250, 62, "撞墙则动作被挡住\n计一次碰撞，不告诉 agent", "priv")
f.node(30, 216, 250, 62, "真值体素覆盖率\n只用于评测", "priv")
f.node(350, 44, 180, 70, "深度来源\nRGB-D：深度 +1% 噪声\nRGB-only：单目深度估计", "ours")
f.node(560, 44, 170, 70, "位姿：航位推算\n只积分自己发出的\n指令，会漂移", "ours")
f.node(350, 164, 180, 70, "估计位姿系下的\n3D 栅格 + frontier\n规划", "ours")
f.node(560, 164, 170, 70, "深度通道避障\n选出动作", "ours")
f.arrow(530, 79, 560, 79); f.arrow(440, 114, 440, 164); f.arrow(645, 114, 500, 164); f.arrow(530, 199, 560, 199)
f.arrow(280, 70, 350, 70, "RGB / 深度", 315, 60)
f.path("M645 234 L645 292 L290 292 L290 262", "动作：前进 / 偏航 / 上下", 470, 286)
FIGS["sensor"] = render(f, "sensor-only agent：特权信息只留在环境一侧，agent 看到的只有图像和深度。", "sensor-only agent 架构图")

# ---------- literature ----------
f = Fig(760, 230)
x5 = [20, 171, 322, 473, 624]
t = ["RGB-D +\n带噪声里程计", "Neural SLAM\n学习的建图\n+ 位姿估计", "全局策略 (RL)\n在地图上选\n长期目标", "解析路径规划\n得到短期目标", "局部策略\n(模仿学习)\n输出动作"]
for x, tt in zip(x5, t): f.node(x, 30, 125, 66, tt, "")
for i in range(4): f.arrow(x5[i] + 125, 63, x5[i + 1], 63)
f.path("M686 96 L686 140 L82 140 L82 99", "动作改变观测，所有模块一起训练", 380, 134)
f.label(380, 185, "分层把长时程探索压缩成少量长期目标；覆盖率在 Gibson 0.948，最强端到端 RL 基线 0.789。", "middle")
FIGS["ans"] = render(f, "ANS：学习的建图和位姿估计，加分层策略。", "ANS 流程图")

f = Fig(760, 270)
f.node(20, 30, 190, 62, "新的深度 /\n占据地图", ""); f.node(250, 30, 190, 62, "增量更新\nfrontier 信息结构\n簇 + 视点 + 代价", ""); f.node(480, 30, 250, 62, "全局覆盖路径\n对 frontier 簇排访问顺序", "")
f.arrow(210, 61, 250, 61); f.arrow(440, 61, 480, 61)
f.node(480, 150, 250, 62, "局部视点精修\n为下一个簇选最优视点", ""); f.node(250, 150, 190, 62, "最小时间轨迹\n优化并执行", "")
f.arrow(605, 92, 605, 150); f.arrow(480, 181, 440, 181)
f.path("M345 212 L345 245 L115 245 L115 95", "新观测", 230, 239)
FIGS["fuel"] = render(f, "FUEL：增量维护 frontier 结构，三层规划，从全局到轨迹。", "FUEL 流程图")

f = Fig(760, 250)
f.node(20, 44, 150, 56, "朝选定的\nfrontier 飞行", ""); f.node(230, 44, 150, 56, "到达分支路口？", ""); f.node(440, 44, 150, 56, "原地偏航 360°\n刷新周围地图", "")
f.arrow(170, 72, 230, 72); f.arrow(380, 72, 440, 72, "是", 410, 64)
f.path("M305 44 L305 16 L95 16 L95 44", "否：继续飞", 200, 11)
f.node(440, 150, 150, 56, "在半径 r 内\n选 frontier", ""); f.arrow(515, 100, 515, 150)
f.node(230, 150, 150, 56, "选择器变体\n几何打分 / LLM / VLM", "")
f.arrow(440, 178, 380, 178, None, dash=True)
f.path("M515 206 L515 232 L95 232 L95 100", None)
FIGS["opal"] = render(f, "OPAL：只在路口花时间转一圈，不做重型全局规划。", "OPAL 流程图")

f = Fig(760, 210)
x4b = [20, 200, 380, 560]
t = ["地图 + 候选 frontier\n及其图像", "组成多模态提示", "VLM 选出最有\n前景的 frontier", "传统规划与控制栈\n飞到该 frontier"]
for x, tt in zip(x4b, t): f.node(x, 40, 150, 62, tt, "")
for i in range(3): f.arrow(x4b[i] + 150, 71, x4b[i + 1], 71)
f.path("M635 102 L635 150 L95 150 L95 105", "到达后重复", 365, 144)
f.label(380, 190, "替换的只是“选哪个 frontier”这一步，免训练；OPAL 的消融里 LLM / VLM 变体与无模型版本相近。", "middle")
FIGS["vlm"] = render(f, "VLM 引导 frontier：VLM 只做高层选择，底层仍是传统方法。", "VLM 引导 frontier 流程图")

f = Fig(760, 210)
f.node(20, 30, 160, 52, "带位姿的 RGB 序列", ""); f.node(20, 110, 160, 52, "单目深度先验", "")
f.node(240, 56, 190, 66, "FrontierNet\n预测 frontier 位置\n与信息增益", ""); f.node(480, 56, 130, 66, "按增益和距离\n选 frontier", ""); f.node(650, 56, 100, 66, "飞行\n并更新", "")
f.arrow(180, 56, 240, 80); f.arrow(180, 136, 240, 100); f.arrow(430, 89, 480, 89); f.arrow(610, 89, 650, 89)
f.label(380, 190, "只用 RGB 加深度先验，不需要深度传感器；报告早期探索效率提升 15%。", "middle")
FIGS["frontiernet"] = render(f, "FrontierNet：学习式 frontier 与信息增益，与我们的 RGB-only 设定最近。", "FrontierNet 流程图")

f = Fig(760, 210)
for x, tt in zip(x4b, ["RGB 图像\n无深度传感器", "自监督学习\n占据场", "预测候选视点\n的覆盖增益", "选 NBV\n飞过去"]): f.node(x, 40, 150, 62, tt, "")
for i in range(3): f.arrow(x4b[i] + 150, 71, x4b[i + 1], 71)
f.path("M635 102 L635 150 L95 150 L95 105", "新图像，在线继续学习", 365, 144)
FIGS["macarons"] = render(f, "MACARONS：只用彩色图像在线自监督，学占据场并选下一视角。", "MACARONS 流程图")

CSS = """
:root{--bg:#f7f8fa;--fg:#17212b;--muted:#5d6b7a;--line:#c9d1da;--node:#ffffff;--accent:#1f5fbf;--accent-soft:#e7effb;--warn:#b3412b;--code:#eef1f5;--head:#0f1a24}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10161d;--fg:#e4e9ee;--muted:#98a6b5;--line:#33404e;--node:#18212b;--accent:#6fa3ec;--accent-soft:#1a2a40;--warn:#e07a62;--code:#1a232d;--head:#f2f5f8;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#10161d;--fg:#e4e9ee;--muted:#98a6b5;--line:#33404e;--node:#18212b;--accent:#6fa3ec;--accent-soft:#1a2a40;--warn:#e07a62;--code:#1a232d;--head:#f2f5f8;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:"IBM Plex Sans","PingFang SC","Noto Sans SC",system-ui,sans-serif;font-size:15px;line-height:1.7;padding-block:32px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:860px;margin:0 auto}
h1{font-size:28px;line-height:1.25;color:var(--head);margin:0 0 6px;text-wrap:balance}
h2{font-size:20px;color:var(--head);margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--line);text-wrap:balance}
h3{font-size:16px;color:var(--head);margin:28px 0 4px}
p{margin:8px 0;max-width:68ch}ul{padding-left:20px;margin:8px 0}li{margin:4px 0;max-width:68ch}
.meta{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--muted);letter-spacing:.02em}
.lead{font-size:16px}
.tag{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);margin-right:8px}
.tag.ext{color:var(--muted)}
figure{margin:12px 0 4px}.fw{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--node)}
svg{display:block;width:100%;min-width:640px;height:auto}
figcaption{font-size:13px;color:var(--muted);margin-top:6px}
.nd{fill:var(--node);stroke:var(--line);stroke-width:1.2}.nd.ours{fill:var(--accent-soft);stroke:var(--accent)}.nd.priv{fill:var(--node);stroke:var(--warn);stroke-dasharray:5 3}
.rg{fill:none;stroke:var(--line);stroke-width:1.2;stroke-dasharray:3 3}.rg.priv{stroke:var(--warn)}.rg.ours{stroke:var(--accent)}
.tx{fill:var(--fg);font-size:12.5px;font-family:"IBM Plex Sans","PingFang SC","Noto Sans SC",system-ui,sans-serif}
.lb{fill:var(--muted);font-size:12px;font-family:"IBM Plex Sans","PingFang SC","Noto Sans SC",system-ui,sans-serif}.lb.mono{font-family:"IBM Plex Mono",monospace;fill:var(--fg)}.rt{fill:var(--muted);font-size:12px;font-weight:600;font-family:"IBM Plex Sans","PingFang SC","Noto Sans SC",system-ui,sans-serif}
.ar{stroke:var(--muted);stroke-width:1.4}.ar.dash{stroke-dasharray:4 3}.ah{fill:var(--muted)}
.tw{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;min-width:640px;font-size:13.5px}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600;font-size:12px;letter-spacing:.03em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.note{border-left:3px solid var(--accent);background:var(--accent-soft);padding:10px 14px;margin:14px 0;max-width:68ch;border-radius:0 4px 4px 0}
.warn{border-left-color:var(--warn)}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;background:var(--code);padding:1px 5px;border-radius:3px}
a{color:var(--accent)}
"""

def sec(tag, kind, title, body, fig):
    return f'<h3><span class="tag {kind}">{tag}</span>{title}</h3>{fig}{body}'

def src(url, name): return f'<a href="{url}">{name}</a>'

H = []
H.append(f'<!-- generated by build_report.py -->\n<title>探索策略对照报告</title><meta name="claude:viewport-fit-sides" content="cover">\n<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"><style>{CSS}</style>')
H.append('<main>')
H.append('<div class="meta">ACWM-UAV · 探索 agent · 2026-10-25</div><h1>探索策略对照报告</h1>')
H.append('<p class="lead">把我们实现的 4 个探索策略，和调研到的 6 条文献路线放在一起，每个策略一张流程图，最后列出差异。目标是给世界模型采数据，所以除了覆盖率，也要看采到的数据是否多样。</p>')
H.append('<div class="note"><b>三点结论</b><ul><li>我们的策略都是朴素版本：地图加最近 frontier，没有分层规划，也没有信息增益打分。文献里的强方法都有这两样。</li><li>评测口径和文献差一个量级：我们 500 步只飞 10–60 m，OPAL 在 HM3D 上要飞 138–320 m 才完全覆盖。我们的数字不能直接放进它们的表。</li><li>sensor-only 版本已经做到不用特权信息，但 RGB-only 下 frontier 明显变差（0.231 对 0.303），碰撞和位姿漂移变多。</li></ul></div>')

H.append('<h2>一、我们实现的策略</h2>')
H.append(sec("ours", "", "random：随机游走加深度避障", FIGS["random"], '<p><b>原理</b>：只看前方通道的深度，空间够就以一定概率前进，否则随机转向。没有记忆。<b>输入</b>：深度图。<b>用途</b>：下限基线，也是 sensor-only 对照。</p>'))
H.append(sec("ours", "", "nbv：贪心的下一最佳视角", FIGS["nbv"], '<p><b>原理</b>：每步枚举 7 个偏航方向，看哪个方向能看到最多新体素。<b>局限</b>：只看这一眼，没有全局规划，容易原地打转；每步多渲染 7 次，约慢 4 倍。因为要向仿真器要假想视角，<b>不能迁移到真机</b>，所以 sensor-only 版本里去掉了它。</p>'))
H.append(sec("ours", "", "frontier：栅格加 BFS 的前沿探索", FIGS["frontier"], '<p><b>原理</b>：深度图累积成 0.25 m 的 3D 栅格，找空闲格子和未知格子的交界，BFS 选最近且至少 1 m 远的去，到达后朝未知方向看一眼。<b>局限</b>：只选最近，不比较信息量；在小房间容易被反复卡住，方差大。</p>'))
H.append(sec("ours", "", "sensor-only agent：把特权信息隔离出去", FIGS["sensor"], '<p><b>设计</b>：agent 只拿到 RGB 或 RGB-D 和自己发出的指令，位姿靠航位推算，避障只看深度通道。环境一侧负责真实运动、碰撞和评分，这些都不会传给 agent。<b>RGB-only</b> 用 Depth-Anything-V2 Metric-Indoor-Small，输出乘一个全局标量 0.41（因为我们相机视场角 90°，和模型训练时差很远）。</p>'))

H.append('<h3>我们的结果（8 个场景 × 3 种子 × 500 步）</h3><div class="tw"><table><thead><tr><th>设定</th><th>方案</th><th class="n">覆盖率</th><th class="n">碰撞</th><th class="n">位姿误差 (m)</th></tr></thead><tbody>'
 '<tr><td>含特权信息</td><td>random</td><td class="n">0.288</td><td class="n">2.4</td><td class="n">0</td></tr>'
 '<tr><td>含特权信息</td><td>nbv</td><td class="n">0.259</td><td class="n">45.7</td><td class="n">0</td></tr>'
 '<tr><td>含特权信息</td><td>frontier</td><td class="n">0.427</td><td class="n">19.7</td><td class="n">0</td></tr>'
 '<tr><td>sensor-only RGB-D</td><td>random</td><td class="n">0.290</td><td class="n">0.0</td><td class="n">1.01</td></tr>'
 '<tr><td>sensor-only RGB-D</td><td>frontier，真值位姿</td><td class="n">0.333</td><td class="n">0.0</td><td class="n">0</td></tr>'
 '<tr><td>sensor-only RGB-D</td><td>frontier，航位推算</td><td class="n">0.303</td><td class="n">1.7</td><td class="n">0.56</td></tr>'
 '<tr><td>sensor-only RGB-only</td><td>random</td><td class="n">0.311</td><td class="n">42.7</td><td class="n">15.37</td></tr>'
 '<tr><td>sensor-only RGB-only</td><td>frontier，航位推算</td><td class="n">0.231</td><td class="n">9.6</td><td class="n">3.40</td></tr>'
 '</tbody></table></div><p>覆盖率是 0.25 m 表面体素被看到的比例。RGB-only 的 random 覆盖率看着高，但碰撞 42.7 次、位姿误差 15 m，是撞墙后继续乱走蹭出来的，不能算真探索。</p>')

H.append('<h2>二、文献中的策略</h2>')
H.append(sec("论文", "ext", src("https://arxiv.org/pdf/2004.05155", "Active Neural SLAM (ANS)"), FIGS["ans"], '<p><b>要点</b>：学习的建图和位姿估计，全局策略用强化学习选长期目标，局部策略用模仿学习。每回合 1000 步，覆盖率定义为 3.2 m 内已知可通行面积的比例；传感器带噪声里程计，和我们的 sensor-only 思路一致。</p>'))
H.append(sec("论文", "ext", src("https://arxiv.org/pdf/2010.11561", "FUEL"), FIGS["fuel"], '<p><b>要点</b>：增量维护 frontier 信息结构，只更新受新观测影响的部分；三层规划（全局覆盖路径、局部视点优化、最小时间轨迹）。论文称比此前方法快 3–8 倍。我只看了摘要级信息，具体数字未核实。</p>'))
H.append(sec("论文", "ext", src("https://arxiv.org/pdf/2605.25423", "OPAL"), FIGS["opal"], '<p><b>要点</b>：在分支路口原地转 360°，再在邻域半径内选 frontier；评测用 6 个 HM3D 场景、25 个起点、50 次运行，指标是总飞行距离和“覆盖率–距离曲线下面积（AUC）”，传感器是 5 m 量程深度。消融里语言模型选择器和无模型版本差不多，收益主要来自结构。</p>'))
H.append(sec("论文", "ext", src("https://arxiv.org/pdf/2605.23165", "VLM 引导的 frontier 探索"), FIGS["vlm"], '<p><b>要点</b>：VLM 在决策点从候选 frontier 里选一个，免训练。摘要称在 6 个室内场景里覆盖率最多提升 24%。实验细节我没读到。</p>'))
H.append(sec("论文", "ext", src("https://arxiv.org/pdf/2501.04597", "FrontierNet"), FIGS["frontiernet"], '<p><b>要点</b>：只用带位姿的 RGB 加单目深度先验，学习 frontier 和信息增益，是我们 RGB-only 方向最近的参照。</p>'))
H.append(sec("论文", "ext", src("https://openaccess.thecvf.com/content/CVPR2023/html/Guedon_MACARONS_Mapping_and_Coverage_Anticipation_With_RGB_Online_Self-Supervision_CVPR_2023_paper.html", "MACARONS"), FIGS["macarons"], '<p><b>要点</b>：自监督学占据场并选下一视角，面向 3D 重建的主动视角规划，不是室内导航探索，只作为“无深度传感器”路线的参考。</p>'))

H.append('<h2>三、和我们架构的差异</h2><div class="tw"><table><thead><tr><th>维度</th><th>我们</th><th>文献里的强方法</th></tr></thead><tbody>'
 '<tr><td>决策层次</td><td>单层：地图加最近 frontier</td><td>ANS、FUEL、OPAL 都是分层：先全局选目标或排序，再局部执行</td></tr>'
 '<tr><td>frontier 选择</td><td>最近优先</td><td>FUEL 排访问顺序；OPAL 在路口刷新后按邻域选；FrontierNet 学信息增益</td></tr>'
 '<tr><td>建图</td><td>深度反投影的 3D 栅格，位姿航位推算</td><td>ANS 学习位姿估计；真机系统一般有视觉惯性里程计和回环</td></tr>'
 '<tr><td>RGB-only</td><td>现成单目深度加一个全局标量</td><td>FrontierNet 直接学 frontier 和增益；MACARONS 自监督学占据场</td></tr>'
 '<tr><td>评测口径</td><td>500 步固定预算，3D 表面体素覆盖率</td><td>ANS 1000 步、可通行面积；OPAL 跑到完全覆盖，覆盖–距离 AUC，同起点多种子</td></tr>'
 '<tr><td>目标</td><td>给世界模型采多样数据</td><td>以探索效率为主：覆盖快、路径短</td></tr>'
 '</tbody></table></div>')

H.append('<h2>四、建议的下一步</h2><ul><li><b>先对齐评测</b>：加覆盖–距离曲线和 AUC，预算加长到覆盖饱和，起点和种子固定，这样才能和 OPAL 的口径比较。</li><li><b>升级 frontier</b>：加入路口原地转一圈刷新，用信息增益和距离一起打分，不只选最近。OPAL 的消融说明结构收益大于换选择器。</li><li><b>补位姿修正</b>：RGB-only 的主要问题是撞墙后位姿漂移，需要碰撞感知或视觉里程计。</li><li><b>补多样性指标</b>：世界模型要的是多样的数据，覆盖率之外还要看轨迹分布和视角多样性，否则会采到大量重复数据。</li></ul>')
H.append('<div class="note warn"><b>可信度说明</b>：ANS、OPAL 的评测部分我读了论文全文；FUEL、FrontierNet、MACARONS、VLM 引导 frontier 只看了摘要或搜索摘要，流程图是按摘要画的，细节可能有出入。我们的结果是单机 Mac 上的小规模实验，每组 24 次运行，方差较大。</div>')
H.append('</main>')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_report.html"), "w").write("\n".join(H))
print("ok", sum(len(x) for x in H))
