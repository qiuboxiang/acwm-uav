# Sensor-only 探索 agent：RGB-D vs RGB-only（8 场景 × 3 种子 × 500 步）

| 方案 | 覆盖率 | 碰撞 | 终点位姿误差(m) | ms/步 |
|---|---|---|---|---|
| random + 深度避障（RGB-D） | 0.290 | 0.0 | 1.01 | 7 |
| random + 深度避障（RGB-only） | 0.311 | 42.7 | 15.37 | 143 |
| frontier（真值位姿，RGB-D） | 0.333 | 0.0 | 0.00 | 11 |
| frontier（航位推算，RGB-D） | 0.303 | 1.7 | 0.56 | 9 |
| frontier（航位推算，RGB-only） | 0.231 | 9.6 | 3.40 | 137 |

RGB-only 用 Depth-Anything-V2 Metric-Indoor-Small（MPS），输出乘一个全局标量 0.41（相机 90° HFOV 与模型训练 FOV 不一致，4 个场景实测原始比例 2.4–2.9）。
航位推算位姿不知道碰撞，撞墙后位姿会漂移，是 RGB-only 误差大的主因之一。
