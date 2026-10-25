# ACWM-UAV：无人机室内导航世界模拟器

用世界模型（WM）作为室内无人机 VLN 的 simulator：自动探索 agent 采数据 → 两阶段训练 WM → reward model 评测 / 数据回流。
基于 IndoorUAV（Habitat）。本仓库由 `research_template` 生成，模板说明见 [README.template.md](README.template.md)。

## 当前进度
- 探索 agent baseline（random / nbv / frontier，含 sensor-only 版本）：[src/research/exploration](src/research/exploration)
- 环境：conda 环境 `indooruav`（habitat-sim 0.3.3，py3.9）；场景放 `scenes/`（不入库），可用 `UAV_SCENES_ROOT` 指定
- IndoorUAV 数据在 ModelScope `valyentine/Indoor_UAV`（TB 级，不入库），官方代码 https://github.com/valyentinee/IndoorUAV-Agent

## 运行探索 baseline
```bash
cd src/research/exploration
python explore2.py --scene replica_apartment_0 --steps 500 --seeds 3   # 含特权信息的 baseline
python explore3.py --scene replica_apartment_0 --steps 500 --seeds 3   # sensor-only（RGB-D + 航位推算）
```
