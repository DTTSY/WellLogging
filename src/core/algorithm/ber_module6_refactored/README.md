# ber_module6（Route B：6大模块重构版）

## 目录结构
- `ber_module6/data_io.py`：模块1 数据读取 + 缓存（含全局配置）
- `ber_module6/preprocess.py`：模块2 数据预处理（清洗/特征/标签/数据集/Loader）
- `ber_module6/models.py`：模块3 模型网络（结构/损失/诊断）
- `ber_module6/train_engine.py`：模块4 训练（训练循环/阈值搜索/OPC训练/PCA拟合）
- `ber_module6/inference.py`：模块5 推理（推理/融合/评估/主流程 main）
- `ber_module6/viz.py`：模块6 可视化（出图/保存）

## 运行方式（与原脚本 main 等价）
```bash
# 方式1：作为模块运行（推荐）
python -m ber_module6.inference

# 方式2：python 直接运行
python ber_module6/inference.py
```

## 说明
- 所有“读取 Excel/TXT”相关逻辑已集中在 `data_io.load_dynamic_log_excels / load_formation_xlsx / load_cal_file`。
- `build_train_well_dfs()` 中的“seg_plots 出图”仍保留，但通过函数内局部 import 调用 `viz.plot_ber_segments_for_well`，避免循环依赖。
- 顶层条目（Assign/Function/Class）已全部映射，详见 `MODULE_MAP.md`。
