# UAV 探索 agent 与数据采集工作流

适用代码：`src/research/exploration/`。agent 一律 sensor-only：只用相机图像、自己的动作指令、自己建的地图和操作员给定的飞行区域框；网格几何、真值位姿、碰撞标志只给模拟器侧（执行、评测）或训练标签使用。

## 前置条件

- habitat 环境：conda 环境 `indooruav`（py3.9，habitat-sim 0.3.3）。运行时需 `KMP_DUPLICATE_LIB_OK=TRUE`，并为每个进程设 `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`，否则 torch（RGB-only 深度网络）会线程过订。
- 场景放在 `scenes/`（不入库），也可用 `UAV_SCENES_ROOT` 指定；获取方式见 [local-assets.md](local-assets.md)。
- 仓库自身的校验和测试用项目 `.venv`（py3.12）：`uv run python -m pytest`。
- `run_*.sh` 里的解释器路径是作者本机的 conda 路径，换机器时需要自行改成自己的 `indooruav` 环境。
- 同时跑的 habitat 进程以 12 个左右为上限，再多会因显存和交换而变慢。

## 版本与入口

| 版本 | 入口 | 说明 |
|---|---|---|
| baseline | `explore2.py`、`explore3.py` | 含特权信息 / sensor-only（RGB-D + 航位推算） |
| v4 | `explore4.py`、`run_v4.sh`、`run_all_v4.sh` | 信息增益 + 距离的 frontier 打分、路口 360° 扫描、深度一致性碰撞校正 |
| v5 | `explore5.py`、`mapping5.py` | 日志几率地图、4-DoF 深度 ICP 里程计、分层 frontier 规划，可选学习式 frontier 价值 |
| FrontierNet 对照 | `explore_fn.py`、`fn_server.py`、`run_fn.sh` | FrontierNet 在独立 conda 环境（py3.11）里通过 RPC 运行 |
| 数据采集 | `collect_data.py`（v4）、`collect_data5.py`（v5） | 输出 IndoorUAV 格式，见 [data_collection.md](../exploration/data_collection.md) |

`mapping5.py` 不依赖模拟器，可单独测试：`uv run python -m pytest tests/test_exploration_mapping5.py`。

## 运行

```bash
cd src/research/exploration
# v5 消融（脚本内含场景、种子、配置字符串）
bash run_v5_ablation.sh
# 单场景采集；--dry 只做试跑
KMP_DUPLICATE_LIB_OK=TRUE python collect_data5.py --scene gibson_Adrian --out ../../../data/collected_v5 --max_frames 600
```

预期输出：结果与日志写到 `src/research/exploration_v5/`（`results*/`、`logs*/`、`exploration_v5/` 均被 `.gitignore` 忽略，不要提交）；采集数据写到 `data/`（不入库）。

## 分析与报告

所有数值来自分析脚本产出的结果文件；报告和图是生成物，改脚本再重新生成，不要手改 HTML。

```bash
python analyze_v5.py --res ../exploration_v5/ablation --out ../../../docs/exploration/v5 --tag ablation
python build_report_v5.py      # 生成 docs/exploration/v5_report.html
```

- 文字内容在 `report_v5_text.json`，图嵌入为 data URI，不含脚本。
- `validate_collect_replay.py` 做配对重放：默认参数下应与已记录的数据完全一致，可用来检验改动是否改变行为。

## 每个采集版本都要存档 3D 热力图

每次新的采集和位姿重放结束后运行：

```bash
python heatmap_v5.py --tag <版本名> --data ../../../data/<采集目录> --replay ../exploration_v5/replay/<版本名>_*.npz
```

结果写入 `docs/exploration/heatmaps/<版本名>/`（`heatmap3d.json`、`heatmap_topdown.png`、`meta.json`）。已存在的 tag 不会被覆盖；需要有意重建时才加 `--force`，并说明原因。目前已存档 v4、v5。

## 结论（截至 v5）

- 分层 frontier 规划是主要收益（相对 v4 f3 覆盖率约 +0.12）；日志几率地图和深度 ICP 主要降低漂移和卡死。
- 学习式 frontier 价值、oracle 模仿，甚至特权 oracle 选择器，在闭环里都没有显著增益：目标选择不是瓶颈。
- 剩余问题是动作中转向占比偏高（约 63%，官方约 45%）和飞行较短。低覆盖轨迹多起于 Adrian 的低层或上层，执行器的走廊检查拒绝进入约 0.8 m 的门洞。详见 [trap_analysis.md](../exploration/trap_analysis.md)。
- 门洞穿越的三个开关（`door`、`narrow_ttl`、`narrow_reset`）已实现，默认保持旧行为；试验结果没有显著覆盖率增益。后续方向：楼梯穿越、出口优先的全局规划，以及执行层（减少原地转向、增加高度变化）和里程计鲁棒性。

## 出错时

- 进程被杀或磁盘写满：删掉对应 `exploration_v5/` 子目录后重跑；不要手动拼接部分结果。
- 等待后台任务时用 `pgrep -f '[a]blation4'` 这类带方括号的模式，避免匹配到等待循环自己的命令行。
- 不要在没有 `tests/conftest.py` 本地资产排除的情况下跑全量 pytest：`data/` 可能有十几 GB，会被复制进每个测试的临时目录。
