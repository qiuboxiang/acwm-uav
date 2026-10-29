# sensor-only 探索 agent：现状诊断与改良方案（2026-10-27）

目标：给世界模型采数据的探索 agent。约束：不用任何特权信息（网格真值、真值位姿、碰撞标志、假想视角渲染），
且每个改动在真机上也能实现（只依赖机载 RGB/RGB-D、机载里程计/VIO、自己的地图）。

## 一、现在的架构

代码：[explore3.py](../../src/research/exploration/explore3.py)（Env/SensorAgent）、[explore4.py](../../src/research/exploration/explore4.py)（Frontier2/3、位姿修正）、
[collect_data.py](../../src/research/exploration/collect_data.py)（采集用的 frontier3 + 游走 + 换高度）。

```
RGB-D 256², 90° HFOV ──┐
                       ├─ 航位推算位姿（只积分指令；大步前进后用前后帧中心深度判"被挡"并回退）
自己发出的指令 ────────┘
        │
        ▼
0.25 m 三值体素栅格（未知/空闲/占据），固定 96×12×96 m
  · 占据一旦写入永不清除；空闲只能写到"未知"上
        │
        ▼
frontier 规划（单层、贪心）
  · v1：3D 栅格里最近的 frontier（BFS）
  · v2：未知体积增益 − λ·路程 − μ·转角；路口 360° 扫一圈
  · v3：相机高度 ±0.25 m 的 2D 切片做经典 frontier，按簇大小 − λ·路程
  · 失败阶梯：放宽阈值 → 乐观穿越未知 → 原地扫 24×15° → 随机游走
        │
        ▼
沿 BFS 路径点走：转向对齐 → 深度通道检查（横向/纵向 0.30 m）→ 0.9/0.15 m 前进；
到达后朝未知方向看一眼；卡住 4 次就转向重规划；bad 集合记失败目标
```

特点：没有分层（没有全局访问顺序）、没有概率建图、没有学习模块、没有跨回合记忆、目标函数只有覆盖率。
nbv 因为要向仿真器要假想视角已被弃用，这是对的。

## 二、覆盖率现状（RGB-D，8 场景）

| 口径 | 结果 | 出处 |
|---|---|---|
| 1000 动作 | random 0.336，frontier v1 0.478，v2+修正 0.432，v2 真值位姿 0.514，FrontierNet 改造版 0.320 | v4 报告 |
| 60 m 飞行预算 | 所有变体 0.44–0.51，λ、转圈、转角代价两两差异都在置信区间内 | sweep |
| 300 m / 15000 动作，停在 97% | random 0.60，v1 0.66，v2 无转圈 0.71（但 189 次碰撞、40 m 漂移），v2 转圈 0.55；只有 10–25% 的运行达到 97% | results_long |
| 100 m 压力测试 | v1 与 v2 各有 10–12/24 次运行在最后 1000 动作里覆盖率不再增长；frontier3 有 76% 的迭代"找不到 frontier" | stress |
| 10 h 采集（696 条轨迹） | 每条中位覆盖 0.34、飞 44 m；规划器状态：52% 找不到 frontier（退回游走）、37% 侧移、2% 沿规划路径前进 | collected_10h |
| RGB-only | 覆盖 0.33–0.39，碰撞 100–200 次，位姿漂移 37–63 m | v4 报告 |

和文献的差距：OPAL/FUEL/FALCON 类方法在 HM3D/室内跑到接近完全覆盖，飞 138–320 m；我们 300 m 平均 0.6–0.7，且四分之一到一半的运行会死锁。
**现在的瓶颈不是"选哪个 frontier"（λ/转圈扫描都没有显著差异），而是死锁和地图质量。**

## 三、根因判断（按代码读出来的）

1. **占据永不清除 + 位姿漂移 = 幽灵墙。** `observe()` 里占据格只增不减，航位推算漂移后同一面墙会被写到多个位置，门洞被堵死，可达 frontier 消失，于是进入"找不到 frontier"分支。压力测试里 no_frontier 占 18–76% 正是这个表现。
2. **frontier 定义脆弱。** v3 只看相机高度 ±1 格的切片，换高度后切片大半是未知，frontier 为空；v1/v2 的 3D 未知体积阈值对散斑敏感。
3. **位姿只有航位推算。** 被挡的小步、侧移、后退都观测不到，RGB-only 下漂移几十米。真机上这一层本来就有 VIO，仿真里应该模拟一个"VIO 级"的位姿源而不是纯积分。
4. **单层贪心没有全局顺序。** 每 25 步重规划、bad 集合、到达后转头看一眼，这些让 agent 在几个 frontier 之间来回。
5. **目标函数不是数据多样性。** random 的可通行区域访问、视角数最高，覆盖率最低；frontier 反之。采集轨迹里 52% 的时间是游走，不是设计出来的多样性。

## 四、改良方案（按优先级）

每条都注明：参考文献、为什么真机可用、预期影响、代价。

### 方案 1：概率建图 + 传感器位姿源（修死锁，最先做）

- **做法**：(a) 体素改 log-odds（OctoMap 式，钳位上下限），射线穿过的格子做"空闲更新"，可以把过期的占据擦掉；(b) 位姿改为"视觉里程计 + 指令先验"：仿真里先用相邻帧深度 ICP / 特征点做帧间位姿，或者至少把现有的"深度一致性判被挡"推广到所有平移动作；(c) 建图时按里程计不确定性收缩安全区（drift-aware mapping）。
- **参考**：FUEL / FALCON / EDEN 真机系统全部用 log-odds 占据栅格；RayFronts 的 log-odds 钳位实践；Partsey et al. (CVPR 2022) 在 Habitat 里训练的视觉里程计零样本迁移到真机 LoCoBot；Michigan 的 drift-aware 安全区收缩。
- **真机**：真机本来就是 VIO + 概率栅格，这一步是让仿真 agent 和真机架构对齐。
- **预期**：直接针对 10–12/24 的死锁和 no_frontier 比例；v2 真值位姿版本比估计位姿高 0.08，说明位姿每改善一点覆盖就跟着涨。
- **代价**：建图 1–2 天；深度 ICP 里程计 2–3 天。

### 方案 2：分层 frontier 规划（FUEL / FALCON 的结构）

- **做法**：frontier 聚类 → 每簇采样视点（位置 + 偏航，按可见未知体积打分）→ 全局层对簇求访问顺序（小规模 ATSP 或贪心 + 2-opt）→ 局部层只执行下一两个视点；frontier 结构增量更新（只重算受新观测影响的簇）；用"地图变化"而不是固定 25 步触发重规划。
- **参考**：FUEL（Zhou et al. 2021）、FALCON（Zhang et al. 2024, T-RO，真机全机载，比 SOTA 快 14–30%）、EDEN（2025，真机 1.79 m/s）。
- **真机**：三篇都是机载深度相机 + VIO 的真机结果。
- **预期**：减少来回折返，提高覆盖–距离 AUC；比 v2 的"路口转圈"便宜（转圈每次花 24 个动作，sweep 里没有显著收益）。
- **代价**：3–5 天；FALCON 开源，可以直接对照实现。

### 方案 3：学习式 frontier/目标打分（训练时用特权信息，测试时只用自己的地图）

- **做法**：输入是 agent 自己的局部占据图（和真机同分布），输出是每个 frontier 的价值。两条路线：
  (a) **监督学**：Occupancy Anticipation / PONI / SEER 的思路，用仿真真值地图监督"未观测区域的占据"或"每个位置能解锁的面积"，测试时只看自己的地图。SEER 已在真机 MAV 上验证（路径短 24%）。
  (b) **RL 学**：ARiADNE 在 frontier 图上用注意力网络学非短视的目标选择，在带真实传感器模型的 ROS 仿真里验证；DARE 用扩散策略直接生成探索路径，报告真机结果。
- **真机**：输入只有自己的栅格图，和 sensor-only 约束一致；Habitat 8 个场景 + HM3D 可以提供训练地图。
- **预期**：解决"剩下的房间在哪"这类几何启发式答不了的问题，是覆盖率上限的来源。
- **代价**：(a) 1–2 周；(b) 2–4 周。建议先 (a)。

### 方案 4：把"数据多样性"写进目标（世界模型真正要的）

- **做法**：frontier 打分里加一项内在奖励，候选来源：
  (a) **模型不确定性**：用正在训练的世界模型的 ensemble 分歧（Plan2Explore；自动驾驶里的 InDRiVE 用 DreamerV3 ensemble 分歧做无奖励预训练）或 RND 给候选视点打分，模型越不会预测的地方越值得去。
  (b) **视角覆盖**：现有的 VIEWSET（体素 × 24 个观察方向）已经在 dry-run 里统计，可以直接变成打分项，鼓励从不同方向、不同高度看同一面。
  (c) **跨回合记忆（Go-Explore / 游戏自动测试）**：在自己的地图里维护格子访问计数的档案，每回合优先"回到访问最少但有前景的格子再探索"。微软的 Go-Explore 3D 游戏可达性测试、EA 的好奇心驱动 playtesting 都是这个思路，覆盖率都优于纯 RL 好奇心。
- **真机**：(a)(b) 只要自己的地图和模型；(c) 的"回去"是在已知空闲空间里导航，一栋楼内多次飞行可以保留地图并重定位。
- **预期**：random 的 nav_cov 0.224 / views 21.9 与 frontier 的覆盖 0.48 合在一起；10 h 采集的并集覆盖率上升。
- **代价**：(b)(c) 各 1–2 天；(a) 取决于世界模型训练流程何时可用。

### 方案 5：RGB-only 路线（真机只有单目相机时）

- **做法**：(a) 位姿用视觉里程计而不是航位推算（37–63 m 漂移主要来自没检出的碰撞）；(b) 深度不单帧使用，做 TSDF/概率时间融合，安全通道余量按深度不确定性放大（MonoNav 用保守换来 4 倍碰撞下降）；(c) 直接上 FrontierNet 式模型：带位姿 RGB + 单目深度先验，学 frontier 和信息增益，不依赖 3D 地图（论文报告真机实验，早期探索效率 +15%）。
- **真机**：MonoNav 在 37 g Crazyflie 上跑通，FrontierNet 有真机实验。
- **代价**：(a)(b) 一周；(c) 需要在 Linux GPU 上重跑官方代码（当前 Mac 改造版不代表其水平）。

### 方案 6（长期）：学习式局部策略与世界模型闭环

- NoMaD / ViNT：一个扩散策略同时做目标导航和无目标探索，真机训练数据来自多机器人数据集，高层仍是 frontier。
- Navigation World Models（CVPR 2025）：用世界模型模拟候选轨迹并打分。这和本项目的目标一致：世界模型训好后，可以用它给候选动作/视点排序，形成"探索采数据 → 训 WM → WM 指导探索"的闭环。

## 五、一周内可做的小修（不改架构）

- frontier 切片跟随当前高度，或用带竖直邻居的 3D frontier（`vfront=True` 已有开关，但没系统评测）。
- 换高度不要清空规划（`alt` 执行完 `F.path=[]` 会触发重规划）。
- 地图有显著更新时清 `bad` 集合，而不是只在 idle≥12 时清。
- 采集时的动作配比：后退 15% 偏多，优先让分层规划器用平移到达视点，而不是靠游走。
- 评测统一用 OPAL 口径：覆盖–距离 AUC、跑到饱和、固定起点；另报"死锁率"和"并集覆盖 / 小时"。

## 六、建议的顺序

方案 1 → 方案 5(a)(b)（同一层代码）→ 方案 2 → 方案 4(b)(c) → 方案 3(a) → 方案 4(a) → 方案 3(b)/6。
方案 1 和 2 做完再评估，预计 300 m 下 0.6–0.7 的覆盖率和 40% 左右的死锁率会有明显变化；方案 3、4 才是追求上限和多样性。

## 参考文献

- OPAL: Omnidirectional Path-efficient Aerial 3D expLoration — https://arxiv.org/abs/2605.25423
- FrontierNet: Learning Visual Cues to Explore — https://arxiv.org/abs/2501.04597
- FALCON: Fast Autonomous Aerial Exploration using Coverage Path Guidance (T-RO) — https://arxiv.org/abs/2407.00577
- EDEN: Efficient Dual-Layer Exploration Planning for Fast UAV Autonomous Exploration — https://arxiv.org/abs/2506.05106
- SEER: Safe Efficient Exploration for Aerial Robots using Learning to Predict Information Gain — https://arxiv.org/abs/2209.11034
- SAGE: Semantic-Aware Guided Drone Exploration (FALCON 之上加 CLIP，真机 Starling 2) — https://arxiv.org/abs/2605.23160
- ARiADNE: attention-based DRL exploration — https://arxiv.org/abs/2301.11575
- DARE: Diffusion Policy for Autonomous Robot Exploration — https://arxiv.org/abs/2410.16687
- Occupancy Anticipation (ECCV 2020) — https://arxiv.org/abs/2008.09285 ；Active Neural SLAM (ICLR 2020) — https://arxiv.org/abs/2004.05155
- PONI (CVPR 2022) — https://arxiv.org/abs/2201.10029
- Is Mapping Necessary for Realistic PointGoal Navigation? (VO 零样本迁移真机) — https://arxiv.org/abs/2206.00997
- MonoNav: MAV Navigation via Monocular Depth Estimation — https://arxiv.org/abs/2311.14100
- NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration — https://arxiv.org/abs/2310.07896
- Navigation World Models (CVPR 2025) — https://arxiv.org/abs/2412.03572
- InDRiVE: Reward-Free World-Model Pretraining for Autonomous Driving via Latent Disagreement — https://arxiv.org/abs/2512.18850
- Plan2Explore — https://arxiv.org/abs/2005.05960
- Improving Playtesting Coverage via Curiosity Driven RL Agents (EA SEED) — https://arxiv.org/abs/2103.13798
- Go-Explore Complex 3D Game Environments for Automated Reachability Testing (Microsoft) — https://www.microsoft.com/en-us/research/?p=936006
- RayFronts (log-odds 钳位实践) — https://arxiv.org/abs/2504.06994
- Certifiably-Correct Mapping for Safe Navigation (drift-aware 安全区) — https://arxiv.org/abs/2504.18713
