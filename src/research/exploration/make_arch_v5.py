"""Draws docs/exploration/v5/architecture.png (v5 agent block diagram)."""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../docs/exploration/v5/architecture.png")
fig, ax = plt.subplots(figsize=(13, 6.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.4); ax.axis("off")


def box(x, y, w, h, title, body, ec="#1f5fbf", fc="#e7effb", tag=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", ec=ec, fc=fc, lw=1.4))
    ax.text(x + 0.12, y + h - 0.22, title, fontsize=10.5, weight="bold", va="top", color="#17212b")
    ax.text(x + 0.12, y + h - 0.55, body, fontsize=8.6, va="top", color="#33404e", linespacing=1.35)
    if tag: ax.text(x + w - 0.1, y + h - 0.2, tag, fontsize=8, va="top", ha="right", color=ec, weight="bold")


def arrow(x0, y0, x1, y1, txt=None, dash=False):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#5d6b7a", lw=1.3, ls="--" if dash else "-"))
    if txt: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.08, txt, fontsize=8, ha="center", color="#5d6b7a")


box(0.2, 0.3, 2.6, 5.8, "仿真器（特权，仅执行与评分）", "按带噪声的真实动力学执行动作\n前进 ±3%，偏航 ±0.5°\n\n撞到物体 / 出地理围栏：\n动作被挡，不告诉 agent\n\n真值表面覆盖与\n(体素, 方向) 统计\n只用于评测\n\n训练 3a/3b 时渲染候选\n视点得到标签（仅训练）", ec="#b3412b", fc="#fbeeea")
box(3.3, 4.3, 2.4, 1.8, "传感器", "RGB-D 256²，90° 视场\n（RGB-only：单目度量深度）\n+ 自己发出的指令\n+ 操作员给的飞行区域框", tag="输入")
box(6.1, 4.3, 3.0, 1.8, "深度 ICP 里程计（方案 1）", "4 自由度点到面 ICP，对关键帧\n指令运动作弱先验\n“被挡”假设 + 可观测性门限\n重访旧关键帧只允许小幅校正", tag="P1")
box(9.5, 4.3, 3.3, 1.8, "log-odds 体素地图（方案 1）", "0.25 m，钳位 log-odds，可擦除漂移造成的幽灵墙\n无回波像素视为未知，不雕刻空闲\n里程计判定的碰撞点永久保留\n围栏外永久占据；视角位掩码", tag="P1")
box(6.1, 1.9, 3.6, 2.0, "分层 frontier 规划（方案 2）", "frontier 簇（2 m 分块）→ 每簇一个视点\n（可见 frontier 最多）→ 精确网格最短路\n→ 开放 TSP 排序 → 走到第一个视点\n分级净空代价、膨胀失败回退、目标滞回", tag="P2")
box(10.0, 1.9, 2.8, 2.0, "目标打分插件", "启发式：可见 frontier 数\n3a：学习的增益回归\n3b：模仿特权神谕的排序\n神谕：只作上界", tag="P3")
box(6.1, 0.3, 3.6, 1.3, "控制与安全", "转向对齐 / 前进 0.9 或 0.15 m / 升降 / 侧移\n深度通道检查；被拒时按深度侧向微移", tag="")
box(10.0, 0.3, 2.8, 1.3, "多样性模式（方案 4）", "无 frontier 时：去看得到最多\n未见 (体素, 方向) 的位姿\n× 访问档案 × 模型不确定性", tag="P4")
box(3.3, 1.2, 2.4, 2.6, "连续飞行会话（4c）", "一次电池飞行切成多条轨迹\n地图、关键帧、档案延续\n新会话 = 新起点 + 空地图\n不需要重定位", tag="P4c", ec="#5d6b7a", fc="#f2f4f7")
arrow(5.7, 5.2, 6.1, 5.2); arrow(9.1, 5.2, 9.5, 5.2, "位姿"); arrow(11.1, 4.3, 8.3, 3.9, "地图"); arrow(10.0, 2.9, 9.7, 2.9); arrow(7.9, 1.9, 7.9, 1.6, "路径")
arrow(9.7, 0.95, 10.0, 0.95, dash=True); arrow(6.1, 0.95, 2.8, 0.95, "动作"); arrow(2.8, 5.2, 3.3, 5.2, "图像")
plt.tight_layout(); plt.savefig(OUT, dpi=140); print(OUT)
