# 数据采集脚本与首批数据

脚本：[collect_data.py](../../src/research/exploration/collect_data.py)。策略是 sensor-only agent（RGB-D 加航位推算位姿，不用任何特权信息）：2D 前沿探索（frontier3），找不到前沿时退回“游走”策略，并在飞行中随机改变高度。动作包括前进、后退、左右平移、上下、原地转向；靠近目标点时优先平移而不是转身，找不到前沿时在前进受阻后先尝试侧移或后退，最后才转向。

## 运行

```bash
cd src/research/exploration
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/indooruav/bin/python collect_data.py \
  --scene gibson_Adrian --out ../../../data/collected --episodes 16 --max_frames 700 --seed0 100
```

主要参数：`--max_frames`（每条轨迹的帧数上限）、`--dist_budget`（飞行距离上限，米）、`--p_alt_step`（每步开始换高度的概率）、`--max_turn_run`（连续纯转向超过该次数就强制尝试平移，默认 4）、`--translate`（1 打开侧移和后退，默认开）、`--min_flight`（飞行距离低于该值的轨迹丢弃并换起点重采，默认 15 m）、`--save_depth`（另存 16 位深度 PNG，体积大，默认不存）。

## 输出格式（对齐 IndoorUAV）

每条轨迹一个目录 `data/collected/<group>/<scene>/traj_<k>/`：

| 文件 | 内容 |
|---|---|
| `screenshots/<i>.jpg` | 720×1280、视场角 90°、10 fps，`i = 0..N`，第 0 帧是初始位姿 |
| `saved_transformations/<i>.txt` | JSON，`{"action": [x, y, z, yaw_rad]}`，官方 `load_position` 可直接读 |
| `posture.txt` | JSON，每帧 `[x, y, z, yaw_deg]`，官方 `load_posture` 可直接读 |
| `real_action.json` | 第 `i` 帧是由哪个动作产生的，如 `{"fly_forward": 0.15}`，单位米或度 |
| `annotation.json` | 标准动作标签对应的帧号：`fly_forward_small/large`、`turn_left/right_small/large`、`fly_up`、`fly_down`、`fly_left`、`fly_right`、`fly_backward`（官方抽样里没有这一类，是我们新增的）、`stop` |
| `episode.npz` | `states (N+1,4)`、`actions_abs (N,4)`（下一帧位姿）、`action_class`、`blocked`、`classes` |
| `meta.json` | 场景、种子、相机参数、策略配置、飞行距离、覆盖率、碰撞数 |

坐标：x 是 habitat 的 x，y 是 habitat 的 z，z 是机体底座高度（相机比它高 1.5 m），偏航角 0–360°、向左转为正。每个离散动作被拆成逐帧子步：前进、后退和上下每帧最多 0.45 m、侧移每帧最多 0.3 m、转向每帧最多 6°，和官方每帧位移量级一致；一个动作的标签打在它的最后一帧上。

## 首批数据：gibson_Adrian

16 条轨迹，共 10758 帧，约 1.2 GB（每条 576 到 703 帧）。官方加载函数 `load_position` 和 `load_posture` 都能读；对每一类动作，相邻帧位姿差和 `real_action` 记录的量吻合（最大误差 0.0001）。画面示例见 [collected_contact_sheet.jpg](collected_contact_sheet.jpg)。每条轨迹飞行 21.9 到 80.1 m（中位数 54.6 m），覆盖率 0.16 到 0.46。

| 动作标签占比 | 加平移之前 | 现在 | 官方（抽样 299 条） |
|---|---|---|---|
| 转向（左右、大小） | 80.4% | 41.0% | 44.8% |
| 前进（小步+大步） | 14.6% | 26.5% | 39.7% |
| 上升+下降 | 5.0% | 5.7% | 10.5% |
| 左右平移 | 0% | 11.7% | 4.0% |
| 后退 | 0% | 15.3% | 0%（官方抽样中没有） |

按帧算：转向约 65%（之前约 90%），前进约 16%，后退约 8%，侧移约 7%，上下约 3%。官方转向帧约 54%、前进帧约 34%、上下帧约 8.7%。

注意：同一份 agent 在丢弃低飞行距离轨迹时共重采了 2 次起点（陷在小空间里的起点）。

## 已知问题

- **后退偏多**（占动作标签约 15%）。官方抽样里没有后退，侧移也比官方（约 4%）多。如果要和官方数据混合预训练，需要再压低这两类，或者训练时按动作类别重新加权。
- **转向仍然占大头**，前进偏少（26.5%，官方约 40%），上下移动偏少（5.7%，官方约 10.5%）。
- 约一半的迭代仍花在“找不到前沿后的游走”上，前沿规划的失败原因还没查清。
- 侧移和后退看不到画面，只靠 agent 自己的占据地图判断是否安全；这批数据里碰撞很少（16 条轨迹共 15 次，已被环境挡住），但地图不全的新区域里这类动作被拒绝得更多。
- 没有指令文本（`instruction*.json`、`key_frames.json`），因为探索数据没有任务描述。

## v5 采集（第三轮改良后的 agent）

脚本：[collect_data5.py](../../src/research/exploration/collect_data5.py)，输出格式与上面完全相同（同一套 `Recorder` 和标签规则），可以和 v4 数据直接混用。agent 仍然只用 RGB-D 和自己发出的指令，改动见 [improvement_proposals.md](improvement_proposals.md) 和 v5 报告：

- **建图**：钳位 log-odds 体素图，空闲观测可以擦掉漂移造成的幽灵墙；没有回波的深度像素视为未知；里程计判定的碰撞点永久保留。
- **位姿**：4 自由度深度 ICP 里程计（对关键帧配准，指令运动作先验，带“被挡”假设），并有进度看门狗：一次平移命令实际移动不到 30% 就当作被挡。
- **规划**：分层 frontier（簇 → 视点 → 开放 TSP 排序 → 路径），带分级净空代价、窄通道记忆和卡住后的游走恢复。
- **多样性模式**：没有 frontier 时，去能看到最多“未见过的 (体素, 观察方向)”的位置。
- **地理围栏**：agent 拿到一个飞行区域框（场景边界内缩 0.3 m），框外视为障碍；仿真器也会挡住出框的运动。真机上对应操作员设定的室内飞行区域。

### 运行

单条命令（干跑只记位姿和标签，不渲染 720p 画面，适合调参）：

```bash
cd src/research/exploration
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/anaconda3/envs/indooruav/bin/python collect_data5.py \
  --scene gibson_Adrian --out ../../../data/collected_v5 --episodes 16 --seed0 100 --video
```

10 小时单场景重采（4 个进程 × 170 条，种子与 v4 的 `run_collect_10h.sh` 相同，输出到 `data/collected_10h_v5/`）：

```bash
src/research/exploration/run_collect_10h_v5.sh gibson_Adrian 170
```

主要参数（默认值即本轮选定的搭配）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--div` | 1 | 打开多样性模式 |
| `--session_len` | 1 | 每条轨迹独立起飞；大于 1 时连续若干条轨迹组成一次飞行，地图和档案延续 |
| `--sel` | `tsp` | 选下一个视点的规则：`tsp`、`ratio`（启发式信息速率）、`learned`（方案 3a，需 `--value_model`）、`rank`（方案 3b，需 `--rank_model`） |
| `--wm` | 0 | 方案 4a：用前向模型集成的分歧作新颖度（本轮干跑中没有收益，默认关闭） |
| `--align_tol` / `--arrive_tol` | 20 / 40 | 航向误差小于该值（度）就直接前进 / 到达后不再转向目标 |
| `--p_alt_step` / `--p_alt` | 0.02 / 0.3 | 飞行中 / 到达视点后随机换高度的概率 |
| `--stats` | 无 | 把每条轨迹后的累计真值覆盖和多视角统计写入 jsonl（仅评测） |
| `--door` | 0 | 穿门模式：常规安全检查拒绝前进时，按机身大小的通道检查前进，并在深度图里找开口居中。0 关闭；1 每次被拒都启用；2 只在之前被拒过的通道上启用 |
| `--narrow_ttl` | 400 | 前进被拒后，该通道被标记为"太窄"的有效观测次数 |
| `--narrow_reset` | 0 | 1 表示地图明显增长时清除"太窄"标记，并在规划失败时清掉标记重试一次 |

`--door`、`--narrow_ttl`、`--narrow_reset` 的默认值保持原来的行为，所以已有数据可以逐帧复现。在探索评测脚本里，对应的配置写法是 `door=1,narrow_ttl=60,narrow_reset=1`。

每条轨迹的 `meta.json` 比 v4 多记录：所属会话、frontier / 多样性目标数、规划失败数、看门狗触发数、里程计统计和终点位姿误差。

### 评测与对比

```bash
cd src/research/exploration
./run_replay.sh ../../../data/collected_10h_v5 v5 3          # 按记录的位姿回放，计算真值覆盖与多视角统计
/opt/anaconda3/envs/indooruav/bin/python analyze_collection.py summary --data ../../../data/collected_10h_v5 \
  --replay ../exploration_v5/replay/v5_0.npz ../exploration_v5/replay/v5_1.npz ../exploration_v5/replay/v5_2.npz \
  --out ../exploration_v5/analysis/collect_v5_summary.json
```

结果和图见 [v5_report.html](v5_report.html)。

### v5 轨迹视频库

`build_gallery_v5.py` 把每条轨迹的第一视角、俯视路径、高度条和逐帧动作合成一段 10 fps 视频，覆盖全部 680 条轨迹，并生成可筛选、可按任一指标正序或倒序排序的网页（卡片视图和表格视图）。因为网页附带文件最多 511 个、256MB，视频按 8 条一组打包成 85 个文件，上传到网页的素材存储（1GB），页面只播放每条轨迹对应的时间段；封面拼成精灵图。输出放在已忽略的 `src/research/exploration_v5/gallery_all/`，不进 Git。上传后把各包的素材 ID 写进 `packs.json`，再用 `--page_only` 重建页面。

```bash
cd src/research/exploration
/opt/anaconda3/envs/indooruav/bin/python build_gallery_v5.py --data ../../../data/collected_10h_v5 --out ../exploration_v5/gallery_all --procs 4
/opt/anaconda3/envs/indooruav/bin/python build_gallery_v5.py --data ../../../data/collected_10h_v5 --out ../exploration_v5/gallery_all --page_only
```

页面顶部的三维热力图来自 `heatmap_v5.py`。它汇总全部轨迹的位姿回放结果，统计每个表面体素被多少条轨迹、从几个方向看到，以及相机在每个 0.25 米格子里停留的帧数。这是评测侧数据，用了仿真器位姿和真值几何。

每个采集版本都要存一份热力图。脚本按 `--tag` 写到 `docs/exploration/heatmaps/<版本>/`，里面有三维数据、俯视总览图和记录来源与校验和的 `meta.json`。已存在的版本不会被覆盖，新一轮采集请用新的版本号。目前已存 v4 和 v5。先跑热力图，再用 `--page_only` 重建页面：

```bash
cd src/research/exploration
KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/indooruav/bin/python heatmap_v5.py --tag v5 --data ../../../data/collected_10h_v5 --replay ../exploration_v5/replay/v5_*.npz
/opt/anaconda3/envs/indooruav/bin/python build_gallery_v5.py --data ../../../data/collected_10h_v5 --out ../exploration_v5/gallery_all --page_only
```
