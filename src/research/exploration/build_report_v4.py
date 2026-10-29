"""Builds docs/exploration/v4_report.html from results_v4 (figures embedded as data URIs; no scripts)."""
import base64, json, os, collections, numpy as np
import analyze_v4 as A
from html import escape
OUT = A.OUT; R = A.R
def img(fn, alt):
    p = f"{OUT}/{fn}"
    if not os.path.exists(p): return ""
    return f'<figure><img alt="{escape(alt)}" src="data:image/png;base64,{base64.b64encode(open(p, "rb").read()).decode()}"><figcaption>{escape(alt)}</figcaption></figure>'
T = A.stats("rgbd"); P = A.paired("rgbd", A.PAIRS)
has_rgb = any(r["sensor"] == "rgb" for r in R); has_fn = any(r["policy"].startswith("frontiernet") for r in R)
def f3(x): return f"{x:.3f}"
def tr(cells, th=False): return "<tr>" + "".join(f"<{'th' if th else 'td'}{' class=n' if i else ''}>{c}</{'th' if th else 'td'}>" for i, c in enumerate(cells)) + "</tr>"
rows = [tr(["方案", "运行数", "最终覆盖率", "覆盖–距离 AUC", "碰撞", "位姿误差 (m)", "飞行距离 (m)", "转圈次数"], True)]
for k, v in T.items(): rows.append(tr([k, v["n"], f"{v['cov']:.3f} ± {v['cov_sd']:.2f}", f3(v["auc"]), f"{v['coll']:.1f}", f"{v['drift']:.2f}", f"{v['dist']:.0f}", f"{v['pans']:.1f}"]))
tab_main = "<div class=tw><table>" + "".join(rows) + "</table></div>"
def ci(v): return f"{v[0]:+.3f} <span class=ci>[{v[1]:+.3f}, {v[2]:+.3f}]</span>"
rows = [tr(["对比", "覆盖率差", "AUC 差", "位姿误差差 (m)", "碰撞差", "可通行区域访问差"], True)]
for k, v in P.items(): rows.append(tr([k, ci(v["coverage"]), ci(v["auc"]), ci(v["final_drift_m"]), ci(v["collisions"]), ci(v["nav_cov"])]))
tab_pair = "<div class=tw><table>" + "".join(rows) + "</table></div>"
rows = [tr(["方案", "可通行区域访问", "回头率 (低=少冗余)", "偏航熵 (归一)", "不同视角 /100 步", "画面新颖度"], True)]
for k, v in T.items(): rows.append(tr([k, f3(v["nav_cov"]), f3(v["revisit"]), f3(v["yaw_ent"]), f"{v['views']:.1f}", f"{v['frame_nn']:.3f}"]))
tab_div = "<div class=tw><table>" + "".join(rows) + "</table></div>"
# detector quality
d = {k: v["det"] for k, v in T.items() if sum(v["det"].values())}
rows = [tr(["方案", "真撞墙被检出", "漏检", "误报", "精确率", "召回率"], True)]
for k, v in d.items():
    tp, fp, fn = v["tp"], v["fp"], v["fn"]; rows.append(tr([k, tp, fn, fp, f"{tp/max(1,tp+fp):.2f}", f"{tp/max(1,tp+fn):.2f}"]))
tab_det = "<div class=tw><table>" + "".join(rows) + "</table></div>"
# sensor comparison on matched (scene, seed)
sec_rgb = ""
if has_rgb:
    scenes = sorted({r["scene"] for r in R if r["sensor"] == "rgb"})
    names = ["frontier v1", "frontier v2", "frontier v2 + pose-corr"]
    rows = [tr(["方案", "场景数", "RGB-D 覆盖率", "RGB-only 覆盖率", "RGB-D 位姿误差", "RGB-only 位姿误差", "RGB-D 碰撞", "RGB-only 碰撞", "RGB-only ms/步"], True)]
    for nme in names:
        a = {(r["scene"], r["seed"]): r for r in R if r["sensor"] == "rgbd" and A.label(r) == nme and r["scene"] in scenes}
        b = {(r["scene"], r["seed"]): r for r in R if r["sensor"] == "rgb" and A.label(r) == nme}
        ks = [k for k in b if k in a]
        if not ks: continue
        m = lambda S, key: np.mean([S[k][key] for k in ks])
        rows.append(tr([nme, f"{len({k[0] for k in ks})}（{len(ks)} 次）", f3(m(a, "coverage")), f3(m(b, "coverage")), f"{m(a, 'final_drift_m'):.2f}", f"{m(b, 'final_drift_m'):.2f}", f"{m(a, 'collisions'):.1f}", f"{m(b, 'collisions'):.1f}", f"{m(b, 'ms_per_step'):.0f}"]))
    ab = [r["depth_absrel"] for r in R if r["sensor"] == "rgb" and r.get("depth_absrel")]
    sec_rgb = f"<h2>五、RGB-only 对比 RGB-D</h2><p>RGB-only 的深度来自单目深度估计（Depth-Anything-V2 Metric-Indoor-Small，输出乘全局标量 0.41），深度中位相对误差约 {np.mean(ab):.2f}。只比较两种传感器都跑过的场景和种子。</p><div class=tw><table>" + "".join(rows) + "</table></div>" + img("curves_rgb.png", "RGB-only：覆盖率随动作数和飞行距离的变化") + img("summary_rgb.png", "RGB-only：覆盖率、AUC、碰撞和位姿误差")
sec_fn = ""
if has_fn:
    fr = [r for r in R if r["policy"].startswith("frontiernet")]; scenes = sorted({r["scene"] for r in fr})
    rows = [tr(["方案", "场景数", "覆盖率", "覆盖–距离 AUC", "碰撞", "飞行距离 (m)", "每 1000 步 FrontierNet 耗时 (s)"], True)]
    def sub(name):
        return [r for r in R if r["sensor"] == "rgbd" and A.label(r) == name and r["scene"] in scenes and r["seed"] < 2]
    for nme in ("random", "frontier v1", "frontier v2 + pose-corr", "FrontierNet (orig. code) + corr"):
        rs = sub(nme) if nme != "FrontierNet (orig. code) + corr" else fr
        if not rs: continue
        aucs = A.auc_dist({s: [r for r in (sub("random") + sub("frontier v1") + sub("frontier v2 + pose-corr") + fr) if r["scene"] == s] for s in scenes})
        rows.append(tr([nme, len({r["scene"] for r in rs}), f3(np.mean([r["coverage"] for r in rs])), f3(np.mean(aucs[nme])), f"{np.mean([r['collisions'] for r in rs]):.1f}", f"{np.mean([r['dist'] for r in rs]):.0f}",
                        f"{np.mean([r.get('fn_s', 0) for r in rs]):.0f}" if nme.startswith("Frontier") else "-"]))
    sec_fn = "<div class=tw><table>" + "".join(rows) + "</table></div>"
CSS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../docs/exploration/build_report.py")).read().split('CSS = """')[1].split('"""')[0]
CSS += "img{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:6px;background:#fff}.ci{color:var(--muted);font-size:12px}.good{color:var(--accent)}.bad{color:var(--warn)}"
H = f'''<title>探索 agent 二轮评测</title><meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"><style>{CSS}</style>
<main><div class="meta">ACWM-UAV · 探索 agent · 二轮评测</div><h1>探索 agent 二轮评测</h1>
<p class="lead">升级 frontier（信息增益打分、路口转圈）、补位姿修正、补多样性指标，8 个场景重新评测，并把 FrontierNet 官方代码拉下来接进同一个传感器闭环。</p>
@@SUMMARY@@
<h2>一、这一轮改了什么</h2><ul>
<li><b>frontier v2</b>：对所有可达 frontier 按“未知体积增益 − 0.04 × 路径长度”打分，不再只选最近的；在分支路口（4 m 内有至少 2 个候选）原地转 360°（24 次 15° 转向）刷新地图，两次转圈至少相隔 3 m。</li>
<li><b>位姿修正</b>：每次大步前进后比较前后两帧中心区域的深度。深度没缩短、且前方 0.9 m 内本来就有障碍，就判定被挡住，把位姿退回去，并把前方格子标成障碍。只用深度，不需要碰撞标志。</li>
<li><b>多样性指标</b>：可通行区域访问比例、回头率、偏航熵、不同视角数、画面新颖度。</li>
<li><b>评测口径</b>：预算改成“执行的动作数”1000（转圈也算），新增覆盖–距离 AUC（按 OPAL 的做法，终止后覆盖率保持不变），起点和种子在各方法间固定。</li>
<li><b>修了一个会让评测失真的问题</b>：智能体被深度安全检查反复拒绝时会原地不动，旧口径里一次运行可能只执行几十个动作。现在连续 4 次没动作就转向并重新规划，所有方法都用满 1000 个动作；避障通道从 0.35 m 放宽到 0.30 m。</li></ul>
<h2>二、评测设置</h2><p>8 个场景（gibson 2、replica 2、mp3d 2、hm3d 2）× 3 个种子 × 1000 个动作；agent 只看 RGB-D 和自己发出的指令，位姿靠航位推算（带 3% 前进噪声、0.5° 偏航噪声），避障只看深度。覆盖率是 0.25 m 表面体素被看到的比例。置信区间是对 24 组配对结果做 2000 次自助重采样得到的 95% 区间，样本小，区间宽。</p>
<h2>三、RGB-D 定量对比</h2>{tab_main}{img("summary_rgbd.png", "最终覆盖率、覆盖–距离 AUC、碰撞、位姿误差（8 场景 × 3 种子平均）")}{img("curves_rgbd.png", "覆盖率随动作数（左）和飞行距离（右）的变化")}
<h3>配对对比：每项改动单独的效果</h3><p>每行是同一场景、同一起点、同一种子下两个方案之差，括号内是 95% 区间；区间跨 0 表示没有证据说明有差别。</p>{tab_pair}
<h3>位姿修正的检测质量</h3>{tab_det}
<h2>四、多样性</h2>{tab_div}{img("diversity_rgbd.png", "多样性指标对比")}
<p>多样性指标和覆盖率不同步：random 飞得最远，可通行区域访问和视角数最高，但覆盖率最低，因为它在同一片区域来回晃；frontier 走得更深，覆盖更高，但路线更窄。要给世界模型采数据，后面需要把“新颖度”或“视角多样性”直接放进目标里，而不是只看覆盖率。</p>
<h3>轨迹可视化（种子 0，颜色表示时间，星号是起点）</h3>{img("traj_replica_apartment_0.png", "replica_apartment_0")}{img("traj_mp3d_17DRP5sb8fy.png", "mp3d_17DRP5sb8fy")}{img("traj_hm3d_12e6joG2B2T.png", "hm3d_12e6joG2B2T")}{img("traj_gibson_Anaheim.png", "gibson_Anaheim")}
{sec_rgb}
<h2>{"六" if has_rgb else "五"}、FrontierNet 官方代码作为基线</h2><p>FrontierNet 开源了代码和权重（<a href="https://github.com/cvg/FrontierNet">cvg/FrontierNet</a>），输入只有带位姿的 RGB 加深度，符合不用特权信息的要求，所以拉下来接进了同一个闭环：用官方的 frontier 检测网络、frontier 管理和 OMPL 规划，替换掉它的 Open3D 渲染和 Metric3D 深度。</p>
<ul><li><b>环境隔离</b>：FrontierNet 跑在独立的 conda 环境（Python 3.11），通过本机 socket 与 habitat 环境通信，agent 一侧只传 RGB、深度和估计位姿。</li>
<li><b>替换了建图后端</b>：官方用 wavemap（C++），在这台机器上编译不过（先是新版编译器报错，修了之后又卡在 glog 版本不匹配），换成了接口相同的 numpy 体素栅格。检测网络、frontier 管理和规划器是原代码。</li>
<li><b>相机和规划参数</b>：相机视场角改成它训练用的 77.3°；规划器离障碍的最小距离从 0.2 m 调到 0.4 m，和我们的避障通道一致；加了“14 步没挪动就转开并重规划”的脱困。</li>
<li><b>深度来源</b>：这里用传感器深度。它自带的 Metric3D 需要额外的 mmcv 依赖，没有接。</li></ul>{sec_fn if has_fn else "<p>FrontierNet 的批量评测还没跑完。</p>"}
<div class="note warn"><b>可信度说明</b>：FrontierNet 的结果反映的是“官方代码经我们改造后”的表现，不是它在论文设定下的水平。它原本在 Open3D 渲染场景里用 wavemap 建图、时间不限，我们换了建图、相机、动作空间和安全边距，参数没有针对我们的场景调过。每个方案每个场景只有 2–3 个种子，24 次运行的区间都很宽。</div>
<h2>{"七" if has_rgb else "六"}、结论和下一步</h2>@@NEXT@@
</main>'''
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../docs/exploration/v4_report_template.html"), "w").write(H)
print("template written", len(H))
