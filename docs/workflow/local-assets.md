# 本地资产的获取方式

下列目录都被 `.gitignore` 忽略，不会随仓库一起提交。克隆仓库后需要按本文自行准备。路径一律相对仓库根目录；不要把绝对路径写进脚本或文档。

| 目录 | 大小（作者机器） | 来源 | 是否必需 |
|---|---|---|---|
| `scenes/scene_datasets/` | 约 1 GB | IndoorUAV 数据集里的 `scene_datasets.zip` 解压而来 | 跑任何探索或采集都需要 |
| `IndoorUAV-Agent/` | 约 18 MB | GitHub 克隆 | 仅对照官方数据格式、评测时需要 |
| `data/indooruav_meta/` | 约 13 MB | 同一个 IndoorUAV 数据集里的说明、划分 CSV 和 `vla_ins.zip` | 仅对照官方数据分布时需要 |
| `data/collected*/` | 约 16 GB | 用本仓库脚本重新采集 | 不需要；是生成物 |
| `third_party/FrontierNet/` | 含权重 | GitHub 克隆 + 下载权重 | 仅 FrontierNet 对照 |
| `third_party/wavemap/` | 约 860 MB（含历史） | GitHub 克隆 | 可选，目前未被使用 |
| `src/research/exploration_v5/` | 实验输出 | 重新运行脚本 | 不需要；是生成物 |

## 1. 场景（必需）

`src/research/exploration/scenes.py` 的 `SCENES` 表固定了用到的 8 个场景，每个都要有网格文件和同名的 `.navmesh`，放成下面的结构；也可以用环境变量 `UAV_SCENES_ROOT` 指向别处的 `scene_datasets` 目录。

```
scenes/scene_datasets/
├── gibson/    Adrian.glb  Adrian.navmesh  Anaheim.glb  Anaheim.navmesh
├── replica/   apartment_0/habitat/mesh_semantic.{ply,navmesh}
│              frl_apartment_0/habitat/mesh_semantic.{ply,navmesh}
├── mp3d/      17DRP5sb8fy/17DRP5sb8fy.{glb,navmesh}
│              1LXtFkjw3qL/1LXtFkjw3qL.{glb,navmesh}
└── hm3d/      00083-16tymPtM7uS/16tymPtM7uS.basis.{glb,navmesh}
               00542-12e6joG2B2T/12e6joG2B2T.basis.{glb,navmesh}
```

来源：IndoorUAV 官方数据集（ModelScope：`valyentine/Indoor_UAV`，https://www.modelscope.cn/datasets/valyentine/Indoor_UAV）里的 `scene_datasets.zip`，内含 gibson、mp3d、hm3d、replica 四类场景的 habitat-sim 版本。整个数据集是 TB 级，不要整体下载，只取这一个压缩包，解压后把 `scene_datasets` 放到 `scenes/` 下。官方 README 也是同样的做法（把场景路径指向自己的 `scene_datasets` 目录）。

自查：

```bash
cd src/research/exploration
python -c "import scenes,os; [print(n, all(os.path.exists(p) for p in scenes.paths(n))) for n in scenes.SCENES]"
```

八个都应输出 `True`。

## 2. Python 环境

- 探索与采集：conda 环境 `indooruav`，py3.9，`habitat-sim 0.3.3`，另需 `trimesh`、`scipy`、`numpy`；RGB-only 版本还需 `torch`（单目度量深度网络）。
- FrontierNet 对照：独立的 conda 环境 `frontiernet`，py3.11，依赖见其仓库的 `requirements.txt`。
- 仓库自身的校验和测试：项目 `.venv`（py3.12），`researchctl` 和 `pytest` 在这里运行。

habitat-sim 的安装方式以其官方文档为准。

## 3. IndoorUAV 官方代码与元数据（对照用）

```bash
git clone https://github.com/valyentinee/IndoorUAV-Agent.git IndoorUAV-Agent   # 作者机器上的提交：4626bbe
```

官方数据集在 ModelScope：`valyentine/Indoor_UAV`（TB 级，不要整体下载）。其中与本仓库相关的小文件：

- `scene_datasets.zip`：场景，见上一节。
- `train.csv`、`test_seen.csv`、`test_unseen.csv`：训练集、已见测试集、未见测试集的轨迹划分。
- `vla_ins.zip`：VLA 指令。
- `without_screenshots.zip`：不含截图的 VLN 数据集（截图可用 `posture.txt` 在 habitat-sim 里重新渲染）。
- `depth_images`：VLN 轨迹的深度图（官方 2026 年 2 月补充上传）。

`data/indooruav_meta/` 里只放了说明、三份划分 CSV 和 `vla_ins.zip`，没有轨迹图像本身。

## 4. 采集数据（可重新生成）

`data/collected*/` 全部由脚本生成，格式见 [data_collection.md](../exploration/data_collection.md)：

```bash
cd src/research/exploration
bash run_collect_10h_v5.sh gibson_Adrian      # v5 的 10 小时单场景采集；v4 对应 run_collect_10h.sh
```

作者机器上的目录：`collected`（早期小批量）、`collected_10h`（v4 基线）、`collected_10h_v5`（v5 重跑）。完整采集耗时长且占磁盘（`collected_10h_v5` 约 8 GB），先用 `--dry` 或少量 episode 确认流程。

## 5. FrontierNet 对照（可选）

```bash
git clone https://github.com/cvg/FrontierNet.git third_party/FrontierNet   # 作者机器上的提交：eaa4492
cd third_party/FrontierNet && bash download_weights.sh                      # 下载 rgbd_11cls.pth 到 model_weights/（脚本用 gdown）
```

`fn_server.py` 默认读取 `third_party/FrontierNet`，也可用 `FRONTIERNET_DIR` 指向别处。官方建图后端 wavemap 在作者机器上编译不过，仓库用 `fn_mapper.py`（numpy 版，接口相同）替代，因此对照结果是“官方代码经我们改造后”的表现，详见 v4 报告。

## 6. wavemap（可选，目前未被使用）

```bash
git clone https://github.com/ethz-asl/wavemap.git third_party/wavemap      # 作者机器上的提交：db4f2ff
```

只有在想让官方 `pywavemap` 后端跑起来时才需要；没有它，FrontierNet 对照会自动退回 numpy 版建图。

## 维护约定

本页和 [exploration.md](exploration.md) 是复现的依据，由 agent 和作者共同维护：新增或修改脚本、依赖、场景、权重、随机种子、默认参数或输出位置时，要在同一次改动里更新这两页，使别人克隆仓库后只靠文档就能重建代码和数据。新增外部资产时写明来源、使用的版本或提交、放置位置和检查命令；文档里的命令要实际运行过，没验证过的要如实写出。

## 出错时

- 报找不到场景文件：先运行上面的自查命令，看是哪一个场景缺哪个文件，再核对 `UAV_SCENES_ROOT`。
- 磁盘不够：优先不要拉 `data/collected_10h*`；它们能由脚本重建，不是必需输入。
- 不要为了“方便复现”把以上任何目录加入 git；需要登记数据集版本时用 `researchctl` 记录 ID 和哈希，数据本体仍留在仓库之外。
