# ACWM-UAV：无人机室内导航世界模拟器

用世界模型（WM）作为室内无人机 VLN 的 simulator：自动探索 agent 采数据 → 两阶段训练 WM → reward model 评测 / 数据回流。
基于 IndoorUAV（Habitat）。本仓库由 `research_template` 生成，模板说明见 [README.template.md](README.template.md)。

## 当前进度
- 探索 agent baseline（random / nbv / frontier，含 sensor-only 版本）：[src/research/exploration](src/research/exploration)
- sensor-only 探索 agent v4 / v5（日志几率地图、深度 ICP 里程计、分层 frontier 规划）与数据采集：见 [docs/workflow/exploration.md](docs/workflow/exploration.md)
- 探索报告：[v4](docs/exploration/v4_report.html)、[v5](docs/exploration/v5_report.html)；结论摘要与后续方向见该文档“结论”一节
- 环境：conda 环境 `indooruav`（habitat-sim 0.3.3，py3.9）；场景放 `scenes/`（不入库），可用 `UAV_SCENES_ROOT` 指定
- 场景、第三方代码、权重等本地资产的获取步骤：[docs/workflow/local-assets.md](docs/workflow/local-assets.md)
- IndoorUAV 数据在 ModelScope `valyentine/Indoor_UAV`（TB 级，不入库），官方代码 https://github.com/valyentinee/IndoorUAV-Agent

## 运行探索 baseline
```bash
cd src/research/exploration
python explore2.py --scene replica_apartment_0 --steps 500 --seeds 3   # 含特权信息的 baseline
python explore3.py --scene replica_apartment_0 --steps 500 --seeds 3   # sensor-only（RGB-D + 航位推算）
```
