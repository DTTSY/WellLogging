import os
import math
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import joblib
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F


# ============================
# 1) 数据读取 + 缓存（含全局配置）
# ============================


def load_dynamic_log_excels(log_cfg, well_name: str = None, add_src_file: bool = True, verbose: bool = True):
    """读取动态录井 Excel：支持单文件/多文件列表，并统一生成/校验 WELLDATETIME。"""
    # 兼容两种写法：str 或 list/tuple
    if isinstance(log_cfg, str):
        log_paths = [log_cfg]
    else:
        log_paths = list(log_cfg)

    valid_paths = []
    for p in log_paths:
        exists = os.path.exists(p)
        if verbose:
            print(f"[TRAIN-LOG] {well_name}: {p}  存在={exists}")
        if exists:
            valid_paths.append(p)
        else:
            if verbose:
                print(f"[WARN] 录井路径不存在，已跳过: {p}")

    if not valid_paths:
        raise FileNotFoundError(f"井 {well_name} 没有任何可用录井文件: {log_paths}")

    df_list = []
    for p in valid_paths:
        df_tmp = pd.read_excel(p)
        if add_src_file:
            df_tmp["__SRC_FILE"] = os.path.basename(p)
        df_list.append(df_tmp)

    df_log_raw = pd.concat(df_list, axis=0, ignore_index=True)

    # 统一时间列
    df_log_raw = _ensure_time_column(df_log_raw, well_name=well_name)
    df_log_raw = df_log_raw.sort_values("WELLDATETIME").reset_index(drop=True)

    if well_name is not None and "WELL" not in df_log_raw.columns:
        df_log_raw["WELL"] = well_name

    return df_log_raw

NUM_WORKERS = 8  # 可以根据你 CPU 核心数调，大概 4~8 比较合适

PIN_MEMORY = torch.cuda.is_available()

RUN_CFG = {
    # 总输出根目录（你自己改成常用路径）
    "OUTDIR_ROOT": r"F:/pycharm/pip.venv/BER_timeseries",
    # 本次运行的名字（可以手动写，也可以留空自动用测试井名）
    "RUN_NAME": "X3_train_X7_test_v28",
    # 是否保存模型与工具
    "SAVE_MODEL": True,
    "SAVE_SCALERS": True,
    "SAVE_OPC": True,
    "SAVE_PCA": True,
    "PLOT_SEG_FIG": True,
    "USE_DF_CACHE": True,  # True：优先用缓存的 train_dfs / test_base
    "SEG_CLS_DO_DIAG": True,  # 诊断函数开关
    "VIS_SAVE_CLS_DEBUG": True,

}

CLEAN_CFG = {
    # 时间列 & 深度列
    "TIME_COL": "WELLDATETIME",
    "DEP_COL_CANDIDATES": ["DEP", "BITDEP"],
    "DEP_PHYS_LIMIT": (0.0, None),
    # 认为是“时间断点”的阈值（秒）
    "MAX_DT_SEC": 5 * 60,
    # 物理上下限（按需改）
    "PHYS_LIMITS": {
        "WOB": (0, 300),
        "RPM": (0, 300),
        "TOR": (0, 120),
        "SPP": (0, 60),
        "FLOWIN": (0, 60),
        "FLOWOUT": (0, 60),
        "HKLD": (0, 2000),
    },
    # IQR 统计裁剪
    "IQR_COLS": ["WOB", "RPM", "TOR", "SPP", "FLOWIN", "FLOWOUT"],
    "IQR_K": 3.0,
    # 权重缩放
    "OUTLIER_WEIGHT_FACTOR": 0.2,
    "IQR_WEIGHT_FACTOR": 0.5,
    "FLOW_ANOM_WEIGHT_FACTOR": 0.3,
    # 流量失衡阈值
    "FLOW_BAL_THRESH": 0.15,
    # 插值允许的最大连续缺失点数
    "INTERP_LIMIT": 10,
}

MERGE_CFG = {
    # 地层深度列
    "FORMATION_DEPTH_COL": "Depth",
    # CAL 文本深度列
    "CAL_DEPTH_COL": "DEPTH",
    # CAL 中的 BER 列（目标值）
    "CAL_BER_COL": "BER",
    # 可选：对 BER 做裁剪，防止极端值
    "BER_CLIP": (0.0, 50.0),
}

SEG_CFG = {
    # 沿深度平滑 BER 的窗口长度（点数，建议奇数）
    "SMOOTH_WIN": 5,
    # 扩径段起始 / 终止阈值（基于平滑后的 BER，单位：%）
    "START_THRESH": 5.0,
    "END_THRESH": 3.0,
    # 扩径段最小长度（米），小于该长度视为噪声
    "MIN_LEN_M": 1.0,
    # 扩径段等级划分阈值（基于平滑后的 BER，单位：%），依次对应 0/1/2/3 四档：
    # <5 正常；5–10 轻微；10–15 中等；≥15 严重
    "SEG_LEVEL_THRESHOLDS": [5.0, 10.0, 15.0],
    # 时间域 pre-alert：在扩径段开始时间之前，往前看的时间长度（分钟）
    # 例如 60 表示：扩径开始前 60 分钟内的点，如果尚未进入扩径段，则标记为 BER_PRE_ALERT=1
    "PRE_ALERT_LEN_MIN": 20.0,
    # B 方案：用于“未来扩径风险”二分类的前看步数（基于采样点数）
    # 通常建议与 MODEL_CFG["LOOKAHEAD_STEPS"] 保持一致，例如 60
    "SEG_FUTURE_STEPS": "auto",
    # 扩径段内样本的额外权重倍率（>1 表示放大扩径段权重）
    "WEIGHT_IN_SEG": 1.50,
    # 高 BER 峰值在 sample_weight 上允许的最大放大倍数
    "MAX_BER_WEIGHT": 5.0,
    "VIS_MAX_POINTS": 6000,   # 太长就下采样到这个点数
    "VIS_SMOOTH_WIN": 7,      # 概率/分数的中值平滑窗口

    # === 扩径等级（四档）相关 ===
    # 阈值定义：
    #   level 0: maxBER < 5%
    #   level 1: 5% ≤ maxBER < 10%
    #   level 2: 10% ≤ maxBER < 15%
    #   level 3: maxBER ≥ 15%
    "LEVEL4_THRESHOLDS": [5.0, 10.0, 15.0],
    # 每个等级在回归中的权重放大倍率（软约束用）
    # 你可以根据需要再调，例如 [1.0, 1.5, 2.0, 3.0]
    "LEVEL4_WEIGHTS": [1.0, 2.0, 4.0, 8.0],

    # 训练时对“扩径窗口”做过采样的倍数：
    # 1.0 = 不额外过采样；3.0 = 扩径窗口被采样概率约提升 3 倍
    "OVERSAMPLE_FACTOR": 3.0,
}

MODEL_CFG = {
    # LSTM 相关
    "SEQ_LEN": 1800,
    "BATCH_SIZE": 128,
    "EPOCHS": 30,
    "LR": 5e-5,
    "HIDDEN_SIZE": 64,
    "NUM_LAYERS": 2,
    "BIDIRECTIONAL": False,
    "VAL_SPLIT": 0.2,
    "USE_SAMPLE_WEIGHT": True,
    "DROPOUT": 0.5,  # ★ 新增：注意力后的 dropout 比例
    # === CNN 前端相关（回归主模型用）===
    # 是否启用 CNN+LSTM 结构；如果你想先对比纯 LSTM，可以改成 False
    "CNN_ENABLE": True,
    # CNN 输出通道数（卷积后每个时间步的特征维度）
    "CNN_OUT_CHANNELS": 64,
    # 卷积核大小（沿时间维度），3 表示看前后各 1 个点的局部模式
    "CNN_KERNEL_SIZE": 3,
    # 卷积层数：2 代表 Conv-ReLU-Conv-ReLU（越多越复杂，也越容易过拟合）
    "CNN_NUM_LAYERS": 2,
    # === 多尺度 LSTM 窗口长度（单位：采样点数）===
    # 短窗：最近 20 个点，主要看高频操作扰动 / 仪器噪声
    "MS_SHORT_LEN": 60,
    # 中窗：最近 60 个点，重点覆盖一个完整的扩径过程
    "MS_MID_LEN": 600,
    # 长窗：最近 120 个点，看更长时间尺度的地层 / 井眼整体趋势
    "MS_LONG_LEN": 1800,

    # 这里切换为“未来扩径风险标签”，与 SEG_FUTURE_LABEL 保持一致
    "SEG_LABEL_COL": "SEG_FUTURE_LABEL",
    # 红黄灯阈值
    "BER_THRESH_YELLOW": 10.0,
    "BER_THRESH_RED": 15.0,
    "OPC_THRESH_YELLOW": 0.3,
    "OPC_THRESH_RED": 0.6,
    # 钻井液水化时间效应（仅作为 OPC 的附加特征）
    # 启用开关
    "HYDRATION_ENABLE_FOR_OPC": True,
    # 认为“钻头几乎不动”的深度差阈值（米），小于这个就视为停在同一深度被水化
    "HYDRATION_STOP_DDEP_THRESH": 0.5,
    # 暴露时间累积的时间常数 tau_acc（分钟），
    # 暴露时间 ~ tau_acc 时，风险基本趋近饱和
    "HYDRATION_ACC_TAU_MIN": 60.0,
    # 恢复钻进后风险衰减的时间常数 tau_decay（分钟），
    # 越小衰减越快
    "HYDRATION_DECAY_TAU_MIN": 30.0,
    # 暴露时间上限（分钟），防止超长停钻导致数值过大
    "HYDRATION_MAX_MIN": 360.0,
    # 联合灯权重
    "FUSE_W_MAIN": 0.7,
    "FUSE_W_OPC": 0.3,
    "FUSE_THRESH_Y": 0.5,
    "FUSE_THRESH_R": 1.2,
    # ================== 严格二阶段门控（Two-Stage Strict Gating） ==================
    # 第一阶段：仅判“未来是否会扩径”（二分类 gate）
    # 第二阶段：只有 gate==1  四分类结果给“严重度主灯”
    # 使用方式：把 MAIN_LIGHT_MODE 设为 "gated"
    "MAIN_LIGHT_MODE": "gated",

    # --- Stage1: gate 的概率列与阈值 ---
    # 这里要填“第一阶段二分类模型输出概率列”（比如你预测 SEG_FUTURE_LABEL 的概率）
    "GATE_STAGE1_PROB_COL": "SEG_PROB_NOW",
    # gate 阈值：>=该值 → 判定“会扩径”
    "GATE_STAGE1_THR": 0.6,

    # 可选：gate 去抖（0 表示不去抖，>0 表示 rolling max 窗口，避免1-2点抖动导致闪烁）
    "GATE_SMOOTH_WIN": 0,
    # ===== Stage2(seg4) 段级融合（按 gate 预测段）=====
    # gate 段之间允许的小空隙（0 段）合并：<=该值则把中间空隙填成 1，避免断续
    "GATE_MERGE_GAP_STEPS": 3,

    # 预测扩径段最短长度（点数）。过短的 gate 段直接当作噪声抛掉
    "GATE_MIN_LEN_STEPS": 10,

    # seg4 段级严重度聚合方式：
    #   "max"      -> 段内取最大严重度（最保守）
    #   "majority" -> 段内众数（更平滑）
    "SEG4_SEG_FUSE_METHOD": "max",

    # seg4 置信度阈值（有 conf 列才生效；低于阈值的点不参与段级投票）
    "SEG4_CONF_MIN": 0.0,

    # 段级融合输出列名（会在 df 里新增）
    "SEG4_FUSED_COL": "SEG_LEVEL4_PRED_FUSED",

    # --- Stage2: 严重度来源 ---

    # "seg4" : 用四分类输出列给主灯（但只在 gate==1）
    "GATE_STAGE2_MODE": "seg4",
    # 如果用 seg4：四分类预测列名（你需要在推理阶段生成该列）
    "GATE_STAGE2_SEG4_COL": "SEG_LEVEL4_PRED",

    # gate==0 时，是否强制主灯为绿灯（严格门控就是 True）
    "GATE_FORCE_GREEN_WHEN_NOEXPAND": True,

    # 当 MAIN_LIGHT_MODE 使用分类时，从哪一列读分类概率：
    # 你后续在推理阶段把分类模型的预测概率写到这个列，比如
    #   df["SEG_PROB_NOW"] = seg_prob
    "MAIN_CLS_PROB_COL": "SEG_PROB_NOW",

    # 分类概率对应的红黄灯阈值：
    #   p < MAIN_CLS_THRESH_Y      -> 绿灯
    #   MAIN_CLS_THRESH_Y <= p < MAIN_CLS_THRESH_R -> 黄灯
    #   p >= MAIN_CLS_THRESH_R     -> 红灯
    "MAIN_CLS_THRESH_Y": 0.5,
    "MAIN_CLS_THRESH_R": 0.8,

    # 段级红黄灯平滑：在一个扩径段内 R/Y 占比达到多少时，将整段提升为 R/Y
    "SEG_FUSED_Y_FRAC": 0.2,  # >=20% 点为 Y/R，就整段 Y
    "SEG_FUSED_R_FRAC": 0.5,  # >=50% 点为 R，就整段 R
    # 单峰平滑参数：前置/后置平滑窗口 & 允许的最大坡度
    "FOLD_UNIMODAL_PREWIN": 5,  # 段内预平滑窗口（越大越圆润）
    "FOLD_UNIMODAL_POSTWIN": 3,  # 单峰约束后再平滑一次
    "FOLD_UNIMODAL_SLOPE_SCALE": 3.0,  # 每步最大允许斜率 = 总振幅/(len-1)*scale
    # PCA 维度
    "PCA_N_COMPONENTS": 5,
    # Seg-Now 二分类标签构造相关：只在扩径段“核心区域”标 1
    # 两侧边界各去掉多少比例（0.2 表示两端各去掉 20%，中间 60% 为核心）
    "SEG_NOW_BORDER_FRAC": 0.05,
    # 如果段太短，至少保留多少个点作为核心区
    "SEG_NOW_MIN_CORE_POINTS": 10,

    # === 扩径段二分类训练相关（Seg-Now） ===
    # 设为 True 时，会在主回归模型之前额外训练一个“当前是否处于扩径段”的分类器
    "SEG_CLS_ENABLE": True,
    "SEG_CLS_EPOCHS": 20,
    # BCEWithLogitsLoss 的正样本权重，类别极不平衡时可以适当调大
    "SEG_CLS_POS_WEIGHT": 3.0,
    # ---- Focal Loss 配置（用于处理类别不平衡）----
    # 是否启用 Focal Loss 来代替 BCEWithLogitsLoss
    "SEG_CLS_USE_FOCAL": True,
    # Focal Loss 的聚焦因子 gamma，一般 1~5，常用 2.0
    "SEG_CLS_FOCAL_GAMMA": 2.0,
    # Focal Loss 的 alpha（正类权重），0~1 之间，典型 0.25 / 0.5
    # 为 None 时不使用 alpha 加权，仅使用 (1-p_t)^gamma
    "SEG_CLS_FOCAL_ALPHA": 0.25,
    "SEG_RECALL_FLOOR": 0.8,  # ★ 扩径段分类的召回率安全底线
    # === 扩径严重度 4 分类相关 ===
    # 是否启用 0/1/2/3 四档严重度分类模型（只在扩径核心区样本上）
    "SEG_LEVEL4_ENABLE": True,
    "SEG_LEVEL4_EPOCHS": 30,
    "SEG_LEVEL4_LR": 5e-5,
    "SEG_LEVEL4_SEQ_LEN": 300,
    "SEG_LEVEL4_BATCH_SIZE": 128,
    # 类别权重: [w0, w1, w2, w3]，也可以先设为 None
    "SEG_LEVEL4_CLASS_WEIGHTS": [1.0, 1.0, 4.0, 8.0],
}

AUG_CFG = {
    # 是否启用扩径段扰动增强
    "ENABLE_SEG_PERTURB": True,
    # 对“窗口最后一点处于扩径段(BER_IS_SEG=1)”的样本，加到标准化特征上的高斯噪声 σ
    # 注意：特征已经做过 StandardScaler 标准化，大多数特征方差 ~1，这里设 0.05 属于“很轻微抖动”
    "FEATURE_NOISE_STD": 0.05,
    # 一般不用动标签，如果你想做 label smoothing 可以开一点点
    "LABEL_NOISE_STD": 0.0,
}

TRAIN_WELLS = {
     "X3": {

         "log": [
            #r"F:/1/X3/4.xlsx",
            r"F:/1/X3/202001-1.xlsx",
            r"F:/1/X3/202002-1.xlsx",

        ],
         "form": r"F:/1/地层/X3.xlsx",
         "cal":  r"F:/1/双鱼石001-X3.txt",
     },
     "H2": {

         "log": [

            r"F:/1/H2/201910-1.xlsx",
            r"F:/1/H2/201911-1.xlsx",

        ],
         "form": r"F:/1/地层/H2.xlsx",
         "cal":  r"F:/1/双鱼石001-H2.txt",
     },
     "ST101": {

         "log": [

            r"F:/1/ST101/201804-1.xlsx",
            r"F:/1/ST101/201805-1.xlsx",

        ],
         "form": r"F:/1/地层/101.xlsx",
         "cal":  r"F:/1/双探101.txt",
     },
     "ST102": {

         #"log":
         "log": [

            r"F:/1/ST102/201904-1.xlsx",
            r"F:/1/ST102/201905-1.xlsx",
            r"F:/1/ST102/201906-1.xlsx",

        ],
         "form": r"F:/1/地层/102.xlsx",
         "cal":  r"F:/1/双探102.txt",
     },
    "H6": {
         #"log":
         "log": [

            r"F:/1/H6/202105-1.xlsx",
            r"F:/1/H6/202106-1.xlsx",

        ],
         "form": r"F:/1/地层/H6.xlsx",
         "cal":  r"F:/1/双鱼石001-H6.txt",
     },
     "ST106": {
        "log": [
            r"F:/1/ST106/202104-1.xlsx",
        ],
        "form": r"F:/1/地层/106.xlsx",
        "cal":  r"F:/1/双探106.txt",
     },
    "ST108": {
         #"log":
         "log": [

            r"F:/1/ST108/202002-1.xlsx",
            r"F:/1/ST108/202003-1.xlsx",

        ],
         "form": r"F:/1/地层/108.xlsx",
         "cal":  r"F:/1/双探108.txt",
     },

}

TEST_WELLS = {
    "X7": {
        "log": [
            r"F:/1/X7/202106-1.xlsx",
            # r"F:/1/X7/1.xlsx",
        ],
        "form": r"F:/1/X7/stress.xlsx",
        "cal":  r"F:/1/双鱼石001-X7.txt",
    },


}

CAL_TXT_READ_CFG = {
    "sep": r"\s+",
    "names": ["DEPTH", "CAL", "BER", "BIT"],
    "engine": "python",
}

def load_formation_xlsx(path: str, well_name: str = None) -> pd.DataFrame:
    """
    读取地层 / 应力等 .xlsx 文件，并确保有统一的深度列 'Depth'，
    方便后续按 MERGE_CFG["FORMATION_DEPTH_COL"] 对齐。
    """
    if path is None:
        raise ValueError("formation xlsx 路径为 None")

    if not os.path.exists(path):
        raise FileNotFoundError(f"[FORMATION] 文件不存在: {path}")

    print(f"[FORMATION] 打开地层文件: {path}  well={well_name}")
    df_form = pd.read_excel(path)
    print(f"[FORMATION] 读取完成: {path}  行数={len(df_form)}  列数={df_form.shape[1]}")

    # 确保有 Depth 列
    if "Depth" not in df_form.columns:
        # 尝试常见候选
        for cand in ["DEPTH", "MD", "TVD", "TVDSS", "深度"]:
            if cand in df_form.columns:
                df_form = df_form.rename(columns={cand: "Depth"})
                print(f"[FORMATION] 将列 {cand} 重命名为 Depth 用于对齐")
                break

    if "Depth" not in df_form.columns:
        raise ValueError(
            f"[FORMATION] 在 {path} 中找不到 Depth 列，也没有常见候选(DEPT/MD/TVD等)，"
            f"无法与录井对齐，请检查地层表字段。"
        )

    # 深度转为数值
    df_form["Depth"] = pd.to_numeric(df_form["Depth"], errors="coerce")
    df_form = df_form.dropna(subset=["Depth"]).reset_index(drop=True)

    return df_form

def load_cal_file(path: str, well_name: str = None) -> pd.DataFrame:
    """
    读取 CAL 文本（1=DEPTH, 2=CAL, 3=BER, 4=BIT），
    列名按 CAL_TXT_READ_CFG 中的设置。
    """
    if path is None:
        raise ValueError("CAL 路径为 None")

    if not os.path.exists(path):
        raise FileNotFoundError(f"[CAL] 文件不存在: {path}")

    print(f"[CAL] 打开CAL: {path}  well={well_name}")
    df_cal = pd.read_csv(path, **CAL_TXT_READ_CFG)
    print(f"[CAL] 读取CAL完成: {path}  行数={len(df_cal)}  列数={df_cal.shape[1]}")

    # 基础清洗：深度数值化
    if "DEPTH" in df_cal.columns:
        df_cal["DEPTH"] = pd.to_numeric(df_cal["DEPTH"], errors="coerce")
        df_cal = df_cal.dropna(subset=["DEPTH"]).reset_index(drop=True)

    return df_cal

def _ensure_time_column(df: pd.DataFrame, time_col: str = "WELLDATETIME") -> pd.DataFrame:
    """确保有统一时间列"""
    df = df.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        if "WELLDATE" in df.columns and "WELLTIME" in df.columns:
            df[time_col] = pd.to_datetime(
                df["WELLDATE"].astype(str) + " " + df["WELLTIME"].astype(str),
                errors="coerce",
            )
        else:
            raise ValueError(
                f"找不到 {time_col}，也没有 WELLDATE+WELLTIME 无法构造时间，请检查录井表字段。"
            )
    df = df[~df[time_col].isna()].copy()
    return df

def cache_train_dfs(train_dfs, outdir):
    """
    将每口训练井的 df 单独缓存到 outdir/cache_train/train_df_井名.pkl
    下次跑脚本时可以直接从缓存恢复，避免重新读 Excel + 特征工程。
    """
    cache_dir = os.path.join(outdir, "cache_train")
    os.makedirs(cache_dir, exist_ok=True)

    for df in train_dfs:
        if df is None or len(df) == 0:
            continue
        if "WELL" in df.columns:
            well_name = str(df["WELL"].iloc[0])
        else:
            # 理论上不会走到这里，防御一下
            well_name = f"well_{len(os.listdir(cache_dir))}"

        path = os.path.join(cache_dir, f"train_df_{well_name}.pkl")
        df.to_pickle(path)
        print(f"[CACHE] 已缓存训练井 df[{well_name}] -> {path}")

def load_cached_train_dfs(outdir):
    """
    从 outdir/cache_train/ 中恢复训练井 df 列表。
    若不存在或读取失败，返回 None。
    """
    cache_dir = os.path.join(outdir, "cache_train")
    if not os.path.isdir(cache_dir):
        return None

    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pkl"))
    if not files:
        return None

    dfs = []
    for fn in files:
        path = os.path.join(cache_dir, fn)
        try:
            df = pd.read_pickle(path)
            dfs.append(df)
        except Exception as e:
            print(f"[CACHE][WARN] 读取缓存 {path} 失败: {e}")

    if not dfs:
        return None

    print(f"[CACHE] 共加载 {len(dfs)} 个训练井 df 缓存")
    return dfs

def cache_test_base_df(df_base, outdir, test_name: str):
    """
    缓存“测试井基础特征 df”：
      - 已做完清洗 + 衍生特征 + 地层+CAL 合并 + 交互 + 趋势
      - 尚未做标准化和 LSTM 预测
    """
    cache_dir = os.path.join(outdir, "cache_test")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"test_base_{test_name}.pkl")
    df_base.to_pickle(path)
    print(f"[CACHE] 已缓存测试井 {test_name} 基础特征 df -> {path}")

def load_cached_test_base_df(outdir, test_name: str):
    """
    从缓存恢复测试井基础特征 df，若不存在则返回 None。
    """
    cache_dir = os.path.join(outdir, "cache_test")
    path = os.path.join(cache_dir, f"test_base_{test_name}.pkl")
    if os.path.isfile(path):
        try:
            df_base = pd.read_pickle(path)
            print(f"[CACHE] 从缓存读取测试井 {test_name} 基础特征 df -> {path}")
            return df_base
        except Exception as e:
            print(f"[CACHE][WARN] 读取测试井 {test_name} 缓存失败: {e}")
    return None
