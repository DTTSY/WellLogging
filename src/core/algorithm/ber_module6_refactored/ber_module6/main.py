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

# ===== 全局随机控制（保证多次运行结果尽量一致） =====
NUM_WORKERS = 8  # 可以根据你 CPU 核心数调，大概 4~8 比较合适
PIN_MEMORY = torch.cuda.is_available()

def set_global_seed(seed: int = 42):
    """固定所有常见随机源，尽量保证训练可复现。"""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN 选择确定性算法
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id: int):
    """给 DataLoader 的每个 worker 单独设 seed（用于 num_workers>0 的情况）。"""
    import random
    base_seed = 42
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
# ===== 全局随机控制 END =====
# ===== Matplotlib 中文字体配置 =====
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# =========================
# 配置：清洗 / 对齐 / 模型
# =========================
RUN_CFG = {
    # 总输出根目录（你自己改成常用路径）
    "OUTDIR_ROOT": r"output/ber_module6",
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
# ========= 扩径段相关配置 =========

AUG_CFG = {
    # 是否启用扩径段扰动增强
    "ENABLE_SEG_PERTURB": True,
    # 对“窗口最后一点处于扩径段(BER_IS_SEG=1)”的样本，加到标准化特征上的高斯噪声 σ
    # 注意：特征已经做过 StandardScaler 标准化，大多数特征方差 ~1，这里设 0.05 属于“很轻微抖动”
    "FEATURE_NOISE_STD": 0.05,
    # 一般不用动标签，如果你想做 label smoothing 可以开一点点
    "LABEL_NOISE_STD": 0.0,
}
# =========================
# 这里改成你的井路径
# =========================

TRAIN_WELLS = {
     "X3": {

         "log": [
            r"data/4.xlsx",
            # r"data/X3/1-1.xlsx",
            # r"data/X3/2-1.xlsx",

        ],
         "form": r"data/X3.xlsx",
         "cal":  r"data/双鱼石001-X3.txt",
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
            # r"data/202106-1.xlsx",
            r"data/1.xlsx",
        ],
        "form": r"data/stress.xlsx",
        "cal":  r"data/双鱼石001-X7.txt",
    },
}




# 读取 CAL 文本：1=DEPTH, 2=CAL, 3=BER(目标), 4=BIT
CAL_TXT_READ_CFG = {
    "sep": r"\s+",
    "names": ["DEPTH", "CAL", "BER", "BIT"],
    "engine": "python",
}


# =========================
# 地层 / CAL 读取函数
# =========================

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


# =========================
# 基础工具函数
# =========================

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


def reweight_by_ber(df: pd.DataFrame,
                    ber_col: str = "BER",
                    base_thresh: float = 5.0,
                    max_factor: float = 5.0) -> pd.DataFrame:
    """
    根据 BER 大小对 sample_weight 做放大：
      - BER <= base_thresh: 基本不变
      - BER 越大，权重越高，但不超过 max_factor 倍
    """
    df = df.copy()
    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0

    ber = df[ber_col].fillna(0.0).clip(lower=0.0)
    # 比例因子：比如 BER=5 时 ~2 倍，BER>=base_thresh*5 时逼近 max_factor
    ratio = ber / (base_thresh + 1e-6)
    factor = 1.0 + ratio
    factor = factor.clip(1.0, max_factor)

    df["sample_weight"] *= factor
    return df


def _choose_depth_column(df: pd.DataFrame, candidates) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"在录井数据中找不到任何深度列候选: {candidates}")


# =========================
# 时间序列清洗 & 对齐
# =========================

def clean_time_series_per_well(df_log_raw: pd.DataFrame, well_name: str = None, cfg: dict = None) -> pd.DataFrame:
    """按时间序列清洗单井录井数据"""
    if cfg is None:
        cfg = CLEAN_CFG

    df = df_log_raw.copy()
    df.replace([-999, -999.25, -9999], np.nan, inplace=True)

    # 1) 时间排序
    time_col = cfg.get("TIME_COL", "WELLDATETIME")
    df = _ensure_time_column(df, time_col=time_col)
    df.sort_values(time_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    dep_col = _choose_depth_column(df, cfg["DEP_COL_CANDIDATES"])
    df[dep_col] = pd.to_numeric(df[dep_col], errors="coerce")

    # 去掉没有深度的数据
    df = df.dropna(subset=[dep_col])

    # 物理上下限剔除（例如忽略负深度）
    dep_lo, dep_hi = cfg.get("DEP_PHYS_LIMIT", (None, None))
    if dep_lo is not None:
        df = df[df[dep_col] >= dep_lo]
    if dep_hi is not None:
        df = df[df[dep_col] <= dep_hi]

    df.reset_index(drop=True, inplace=True)
    # 2) sample_weight
    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0

    # 3) 时间间隔 & 断点
    df["dt_sec"] = df[time_col].diff().dt.total_seconds()
    df.loc[df["dt_sec"].isna(), "dt_sec"] = 0.0
    df["FLAG_TIME_GAP"] = (df["dt_sec"] > cfg["MAX_DT_SEC"]).astype(int)

    # 4) 深度速度
    dep_col = _choose_depth_column(df, cfg["DEP_COL_CANDIDATES"])
    df["dDEP"] = df[dep_col].diff()
    dt_for_v = df["dt_sec"].copy()
    dt_for_v.replace(0, np.nan, inplace=True)
    df["v_depth"] = df["dDEP"] / dt_for_v
    df["v_depth"] = df["v_depth"].fillna(0.0)

    # 5) 物理上下限裁剪 + 权重衰减
    for col, (lo, hi) in cfg["PHYS_LIMITS"].items():
        if col not in df.columns:
            continue
        mask_low = df[col] < lo
        mask_high = df[col] > hi
        mask = mask_low | mask_high
        if mask.any():
            flag_col = f"FLAG_PHYS_{col}"
            df[flag_col] = 0
            df.loc[mask, flag_col] = 1
            df.loc[mask, "sample_weight"] *= cfg["OUTLIER_WEIGHT_FACTOR"]
            df[col] = df[col].clip(lo, hi)

    # 6) 流量平衡检查
    if "FLOWIN" in df.columns and "FLOWOUT" in df.columns:
        denom = df["FLOWIN"].abs().replace(0, np.nan)
        df["FLOW_BAL"] = (df["FLOWIN"] - df["FLOWOUT"]) / denom
        df["FLOW_BAL"] = df["FLOW_BAL"].fillna(0.0)
        mask_flow_anom = df["FLOW_BAL"].abs() > cfg["FLOW_BAL_THRESH"]
        df["FLAG_FLOW_ANOM"] = mask_flow_anom.astype(int)
        df.loc[mask_flow_anom, "sample_weight"] *= cfg["FLOW_ANOM_WEIGHT_FACTOR"]

    # 7) IQR 统计裁剪（按工况分组）
    iqr_cols = [c for c in cfg["IQR_COLS"] if c in df.columns]

    if iqr_cols:
        if "RIGSTA" in df.columns:
            groups = df.groupby("RIGSTA")
        else:
            # 只有一个整体分组
            groups = [(None, df)]

        for _, group in groups:
            idx = group.index
            for col in iqr_cols:
                series = df.loc[idx, col]
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if not np.isfinite(iqr) or iqr <= 0:
                    continue
                k = cfg["IQR_K"]
                lower = q1 - k * iqr
                upper = q3 + k * iqr
                mask = (series < lower) | (series > upper)
                if mask.any():
                    flag_col = f"FLAG_IQR_{col}"
                    if flag_col not in df.columns:
                        df[flag_col] = 0
                    df.loc[idx[mask], flag_col] = 1
                    df.loc[idx[mask], "sample_weight"] *= cfg["IQR_WEIGHT_FACTOR"]
                    df.loc[idx[mask], col] = series.clip(lower, upper)

    # 8) 数值列插值（短缺失）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # ===== [ANTI-LEAK ANCHOR] clean_time_series_per_well: forbid using future values =====
    # 只允许前向填充（因果），禁止 bfill / 双向插值
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    ffill_limit = CLEAN_CFG.get("FFILL_LIMIT", None)  # 可选：限制最多连续填充多少个点
    df[numeric_cols] = df[numeric_cols].ffill(limit=ffill_limit)

    # 序列开头仍可能有 NaN：统一置 0（或你也可以用列中位数/井内均值，但一定不能 bfill）
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    # ===== [ANTI-LEAK ANCHOR END] =====

    return df


def smooth_ber_along_depth(df,
                           depth_col: str = "DEPTH",
                           ber_col: str = "BER",
                           smooth_win: int = 5,
                           ber_clip: tuple = (0.0, 50.0)) -> pd.DataFrame:
    """
    沿井深对 BER 做滚动中值平滑，压制孤立尖刺。
    - smooth_win: 窗口大小（点数），建议奇数，如 5 / 7
    """
    if ber_col not in df.columns or depth_col not in df.columns:
        return df

    df = df.sort_values(depth_col).reset_index(drop=True)
    ber = df[ber_col].astype(float).clip(*ber_clip).fillna(0.0)

    # 滚动中值：孤立的尖刺会被邻域的正常值“淹没”
    ber_smooth = ber.rolling(window=smooth_win,
                             center=True,
                             min_periods=1).median()

    df[ber_col + "_SMOOTH"] = ber_smooth
    return df


def build_ber_segments_simple(
        df: pd.DataFrame,
        depth_col: str = "DEPTH",
        ber_col: str = "BER_SMOOTH",
        abs_thresh: float = 6.0,
        min_seg_len: float = 0.30,
) -> pd.DataFrame:
    """
    多等级扩径段检测：基于“绝对阈值 + 最小长度 + 段内最大 BER 分级”。

    逻辑：
    1) 按深度排序；
    2) 在平滑后的 BER 上，找到 BER >= base_thresh 的连续区间；
       - base_thresh = SEG_CFG["SEG_LEVEL_THRESHOLDS"][0]（例如 5%）
    3) 每个连续区间如果深度长度 >= min_seg_len，则认定为一个“扩径段”；
    4) 对每个扩径段，取该段内 BER 最大值 seg_max：
         <5    → 段级 level=0（正常）
         5–10  → level=1 轻微扩径
         10–15 → level=2 中等扩径
         ≥15   → level=3 严重扩径
       段内所有点共享同一段级 level；
    5) 输出：
       - BER_SEG_ID    : 段编号 (0 表示非扩径)
       - BER_IS_SEG    : 是否处于扩径段 (0/1)
       - BER_SEG_LEVEL : 段级 0/1/2/3
       - BER_SEG_MAX   : 段内最大 BER

    重要性质：
    - 只要平滑后 BER 始终 ≥ base_thresh，这一整段区间都会视为同一段。
      也就是：如果中间谷底仍 >5%，不会被切成两段，等价于你说的“谷底>5 要合并”。
    """
    # 基本健壮性检查：缺列则直接返回全 0
    if depth_col not in df.columns or ber_col not in df.columns:
        df = df.copy()
        df["BER_SEG_ID"] = 0
        df["BER_IS_SEG"] = 0
        df["BER_SEG_LEVEL"] = 0
        df["BER_SEG_MAX"] = 0.0
        return df

    df = df.copy()
    df = df.sort_values(by=depth_col).reset_index(drop=True)

    depth = pd.to_numeric(df[depth_col], errors="coerce").values
    ber = pd.to_numeric(df[ber_col], errors="coerce").fillna(0.0).values

    n = len(df)
    seg_id_arr = np.zeros(n, dtype=int)
    seg_level_arr = np.zeros(n, dtype=int)
    seg_max_arr = np.zeros(n, dtype=float)

    if n < 2:
        df["BER_SEG_ID"] = seg_id_arr
        df["BER_IS_SEG"] = 0
        df["BER_SEG_LEVEL"] = seg_level_arr
        df["BER_SEG_MAX"] = seg_max_arr
        return df

    # 从配置读取三档阈值
    level_thresholds = SEG_CFG.get("SEG_LEVEL_THRESHOLDS", None)
    if level_thresholds is not None and len(level_thresholds) >= 3:
        t1, t2, t3 = map(float, level_thresholds[:3])
    else:
        # 兜底：如果没配置，就以 abs_thresh 为最小阈值，往上随便拉两档
        base = float(abs_thresh)
        t1, t2, t3 = base, base + 4.0, base + 9.0

    # 用第一档阈值作为“进入扩径候选区”的起点，一般是 5%
    base_thresh = t1

    # 点级阈值 mask：>= base_thresh 视为“扩径候选区”
    mask_high = ber >= base_thresh

    cur_seg_id = 0
    i = 0
    while i < n:
        if not mask_high[i]:
            i += 1
            continue

        # 一段连续 True 区间 [start_idx, end_idx]
        start_idx = i
        while i < n and mask_high[i]:
            i += 1
        end_idx = i - 1

        if end_idx <= start_idx:
            continue

        # 计算这一段的真实深度长度
        seg_len = float(depth[end_idx] - depth[start_idx])
        if seg_len < min_seg_len:
            # 段太短，当作噪声丢弃
            continue

        # 段内最大 BER
        seg_max = float(np.nanmax(ber[start_idx:end_idx + 1]))

        # 段级 level 判定：0/1/2/3
        if seg_max < t1:
            level = 0
        elif seg_max < t2:
            level = 1
        elif seg_max < t3:
            level = 2
        else:
            level = 3  # 严重扩径

        cur_seg_id += 1
        seg_id_arr[start_idx:end_idx + 1] = cur_seg_id
        seg_level_arr[start_idx:end_idx + 1] = level
        seg_max_arr[start_idx:end_idx + 1] = seg_max

    df["BER_SEG_ID"] = seg_id_arr
    df["BER_IS_SEG"] = (seg_id_arr > 0).astype(int)
    df["BER_SEG_LEVEL"] = seg_level_arr  # 段级 0/1/2/3
    df["BER_SEG_MAX"] = seg_max_arr  # 段内最大 BER，便于诊断
    return df


def add_time_pre_alert(df: pd.DataFrame,
                       time_col: str = "WELLDATETIME",
                       seg_id_col: str = "BER_SEG_ID",
                       pre_alert_minutes: float = 0.0) -> pd.DataFrame:
    """
    基于“扩径段开始时间”构造时间域的预警区：
    - 对每个扩径段，取其起点时间 t_start
    - 在 [t_start - pre_alert_minutes, t_start) 区间内，且尚未进入扩径段的点，
      标记 BER_PRE_ALERT = 1

    注意：
    - 不改变 BER_IS_SEG / BER_SEG_ID，只额外加一列 BER_PRE_ALERT（0/1）
    - 如果 pre_alert_minutes <= 0，则不做任何预警标记（全 0）
    """
    df = df.copy()

    # 如果没有时间或没有分段信息，直接返回（全 0）
    if time_col not in df.columns or seg_id_col not in df.columns:
        df["BER_PRE_ALERT"] = 0
        return df

    try:
        t_series = pd.to_datetime(df[time_col], errors="coerce")
    except Exception:
        # 时间列异常时，直接不做预警
        df["BER_PRE_ALERT"] = 0
        return df

    seg_ids = df[seg_id_col].fillna(0).astype(int).values
    n = len(df)
    pre_flag = np.zeros(n, dtype=int)

    # 预警时间长度（分钟）
    if pre_alert_minutes is None or pre_alert_minutes <= 0:
        df["BER_PRE_ALERT"] = pre_flag
        return df

    delta = pd.Timedelta(minutes=float(pre_alert_minutes))

    # 遍历每个扩径段，基于其起点时间生成时间窗
    unique_seg_ids = np.unique(seg_ids)
    for sid in unique_seg_ids:
        if sid <= 0:
            continue
        idx_seg = np.where(seg_ids == sid)[0]
        if idx_seg.size == 0:
            continue

        seg_start_i = int(idx_seg[0])
        t_start = t_series.iloc[seg_start_i]
        if pd.isna(t_start):
            continue

        t_pre_start = t_start - delta

        # 在 [t_pre_start, t_start) 内，且当前不在任何扩径段（seg_ids == 0）的点作为 pre-alert
        mask = (
                (t_series >= t_pre_start) &
                (t_series < t_start) &
                (seg_ids == 0)
        )

        # 将这些点标记为预警区
        pre_flag[mask.to_numpy()] = 1

    # 确保扩径段内部不被标记为 pre-alert
    pre_flag[seg_ids > 0] = 0

    df["BER_PRE_ALERT"] = pre_flag
    return df


def build_future_risk_label(
        df: pd.DataFrame,
        seg_col: str = "BER_IS_SEG",
        out_col: str = "SEG_FUTURE_LABEL",
        horizon_steps: int = 60,
        time_col: str | None = None,
) -> pd.DataFrame:
    """
    B 方案：未来窗口内是否会出现 seg_col==1 的二分类标签。

    关键修复：
    - 如果提供 time_col，则先按 time_col 排序计算 future，再回填到原行顺序；
      避免“当前 df 处于 depth 排序时计算 future”，导致与 LSTM 的时间滚窗错位。
    """
    df = df.copy()
    if seg_col not in df.columns:
        df[out_col] = 0
        return df

    future_steps = int(horizon_steps) if horizon_steps is not None else 0
    future_steps = max(future_steps, 0)
    if future_steps <= 0 or len(df) == 0:
        df[out_col] = df[seg_col].fillna(0).astype(int).to_numpy()
        return df

    # --- 按时间排序计算（如果可用）---
    if time_col is not None and time_col in df.columns:
        df_sorted = df.sort_values(time_col).reset_index(drop=False)  # 保留原 index 到列 "index"
        seg_arr = df_sorted[seg_col].fillna(0).astype(int).to_numpy()
        n = len(seg_arr)

        future_flag = np.zeros(n, dtype=np.int32)
        next_one = None
        for i in range(n - 1, -1, -1):
            if seg_arr[i] == 1:
                next_one = i
                future_flag[i] = 1
            elif next_one is not None and (next_one - i) <= future_steps:
                future_flag[i] = 1
            else:
                future_flag[i] = 0

        df_sorted[out_col] = future_flag

        # 回填到原顺序
        orig_idx = df_sorted["index"].to_numpy()
        back = pd.Series(df_sorted[out_col].to_numpy(), index=orig_idx)
        df[out_col] = back.reindex(df.index).fillna(0).astype(int).to_numpy()
        return df

    # --- 没有 time_col 就按当前顺序计算（兜底）---
    seg_arr = df[seg_col].fillna(0).astype(int).to_numpy()
    n = len(seg_arr)
    future_flag = np.zeros(n, dtype=np.int32)
    next_one = None
    for i in range(n - 1, -1, -1):
        if seg_arr[i] == 1:
            next_one = i
            future_flag[i] = 1
        elif next_one is not None and (next_one - i) <= future_steps:
            future_flag[i] = 1
        else:
            future_flag[i] = 0
    df[out_col] = future_flag
    return df
def build_future_ber_threshold_labels(
        df: pd.DataFrame,
        ber_col: str = "BER",
        time_col: str = "WELLDATETIME",
        horizon_steps: int = 60,
        thr_y: float = 10.0,
        thr_r: float = 15.0,
        out_max_col: str = "BER_FUTURE_MAX",
        out_y_col: str = "SEG_FUTURE_Y_LABEL",
        out_r_col: str = "SEG_FUTURE_R_LABEL",
) -> pd.DataFrame:
    """
    未来窗口最大 BER（时间域）→ 黄/红标签：
    - BER_FUTURE_MAX: max(BER[t : t+h])
    - SEG_FUTURE_Y_LABEL: future_max >= thr_y
    - SEG_FUTURE_R_LABEL: future_max >= thr_r
    """
    df = df.copy()
    if ber_col not in df.columns:
        df[out_max_col] = np.nan
        df[out_y_col] = 0
        df[out_r_col] = 0
        return df

    h = max(int(horizon_steps) if horizon_steps is not None else 0, 0)
    win = h + 1

    if time_col in df.columns:
        df_sorted = df.sort_values(time_col).reset_index(drop=False)
    else:
        df_sorted = df.reset_index(drop=False)

    s = pd.to_numeric(df_sorted[ber_col], errors="coerce").fillna(0.0)
    future_max = s.iloc[::-1].rolling(window=win, min_periods=1).max().iloc[::-1].to_numpy()

    df_sorted[out_max_col] = future_max
    df_sorted[out_y_col] = (future_max >= float(thr_y)).astype(int)
    df_sorted[out_r_col] = (future_max >= float(thr_r)).astype(int)

    orig_idx = df_sorted["index"].to_numpy()
    df[out_max_col] = pd.Series(df_sorted[out_max_col].to_numpy(), index=orig_idx).reindex(df.index).to_numpy()
    df[out_y_col] = pd.Series(df_sorted[out_y_col].to_numpy(), index=orig_idx).reindex(df.index).fillna(0).astype(int).to_numpy()
    df[out_r_col] = pd.Series(df_sorted[out_r_col].to_numpy(), index=orig_idx).reindex(df.index).fillna(0).astype(int).to_numpy()
    return df



def build_seg_now_labels(df: pd.DataFrame,
                         seg_id_col: str = "BER_SEG_ID") -> pd.DataFrame:
    """
    基于连续扩径段 (BER_SEG_ID) 构造“当前是否处于扩径段核心区”的干净标签：
    - 新增列 SEG_NOW_LABEL ∈ {0,1}
    - 对每个扩径段，只在“中间核心部分”标记为 1，两端边界点标记为 0
      这样用于 Seg-Now 二分类训练时，正样本更干净、稳定。
    """
    if seg_id_col not in df.columns:
        # 没有扩径信息时，全部置 0
        df["SEG_NOW_LABEL"] = 0
        return df

    seg_ids = df[seg_id_col].fillna(0).astype(int).values
    n = len(df)
    seg_now = np.zeros(n, dtype=int)

    # 从全局 SEG_CFG 读取参数
    border_frac = float(SEG_CFG.get("SEG_NOW_BORDER_FRAC", 0.2))
    min_core_points = int(SEG_CFG.get("SEG_NOW_MIN_CORE_POINTS", 5))

    if border_frac < 0.0:
        border_frac = 0.0
    if border_frac > 0.45:
        # 防止两边去得太狠，至少保留中间 10%
        border_frac = 0.45

    unique_ids = np.unique(seg_ids)
    for sid in unique_ids:
        if sid <= 0:
            continue

        idx = np.where(seg_ids == sid)[0]
        L = idx.size
        if L == 0:
            continue

        # 根据段长度计算需要丢弃的边界点数
        border_points = int(round(L * border_frac))
        if border_points < 0:
            border_points = 0

        # 如果段太短，直接整段作为核心区
        if L <= 2 * border_points + min_core_points:
            seg_now[idx] = 1
            continue

        core_start_pos = border_points
        core_end_pos = L - border_points  # Python 切片右开

        core_idx = idx[core_start_pos:core_end_pos]
        if core_idx.size > 0:
            seg_now[core_idx] = 1

    df["SEG_NOW_LABEL"] = seg_now
    return df


def add_seg_level4_label(df: pd.DataFrame,
                         seg_id_col: str = "BER_SEG_ID",
                         seg_level_col: str = "BER_SEG_LEVEL",
                         core_col: str = "SEG_NOW_LABEL",
                         out_col: str = "SEG_LEVEL4_LABEL") -> pd.DataFrame:
    """
    在扩径段内新增 4 档扩径等级标签，并额外提供“核心区专用”标签列。

    依赖前置步骤:
    - build_ber_segments_simple 已经在 df 中生成:
        * {seg_id_col}: 每个扩径段的编号 (0 = 非扩径)
        * {seg_level_col}: 段级扩径等级 (0/1/2/3)，来自整段最大 BER
    - build_seg_now_labels 已经在 df 中生成:
        * {core_col}: 当前是否处于扩径段核心区 (0/1)

    输出:
    - {out_col}:
        全井 0/1/2/3 标签:
          0 = 非扩径段
          1 = 轻微扩径
          2 = 中等扩径
          3 = 严重扩径
      段内所有点共享同一等级; 段外统一为 0。
    - {out_col}_CORE_ONLY:
        仅在“扩径核心区”(core_col == 1 且 seg_id>0) 保留 0/1/2/3;
        其他位置全部置为 -1，方便 Dataset 在训练四分类时过滤掉非核心样本。
    """
    df = df.copy()

    # 缺列时兜底: 保证后续代码可以统一访问这两列
    if seg_id_col not in df.columns or seg_level_col not in df.columns:
        df[out_col] = 0
        df[out_col + "_CORE_ONLY"] = -1
        return df

    seg_ids = df[seg_id_col].fillna(0).astype(int).values
    seg_levels = df[seg_level_col].fillna(0).astype(int).values

    if core_col in df.columns:
        core = df[core_col].fillna(0).astype(int).values
    else:
        core = np.zeros_like(seg_ids, dtype=int)

    # 段级 0/1/2/3 → 点级 0/1/2/3
    seg_levels = np.clip(seg_levels, 0, 3)
    level4 = np.zeros_like(seg_ids, dtype=int)

    # 在扩径段内，直接抄段级等级；段外默认 0（正常）
    mask_in_seg = seg_ids > 0
    level4[mask_in_seg] = seg_levels[mask_in_seg]

    df[out_col] = level4

    # 仅在扩径“核心区”内保留标签，其他位置统一设成 -1
    core_only = np.where((core == 1) & mask_in_seg, level4, -1)
    df[out_col + "_CORE_ONLY"] = core_only

    return df


def merge_formation_and_cal_to_time(df_log_clean: pd.DataFrame,
                                    df_form: pd.DataFrame,
                                    df_cal: pd.DataFrame,
                                    clean_cfg: dict = None,
                                    merge_cfg: dict = None):
    """
    将地层特征 + CAL 文件中的 BER(第3列) 对齐到按时间排序的录井数据上，
    并基于 BER 构造平滑值和连续扩径段。
    """
    if clean_cfg is None:
        clean_cfg = CLEAN_CFG
    if merge_cfg is None:
        merge_cfg = MERGE_CFG

    time_col = clean_cfg.get("TIME_COL", "WELLDATETIME")
    dep_col = _choose_depth_column(df_log_clean, clean_cfg["DEP_COL_CANDIDATES"])

    df_all = df_log_clean.copy()

    # ========= 1) 地层特征按深度对齐 =========
    form_depth_col = merge_cfg["FORMATION_DEPTH_COL"]
    df_form = df_form.copy()

    # 确保深度列是数值
    df_form[form_depth_col] = pd.to_numeric(df_form[form_depth_col], errors="coerce")
    df_form = df_form.dropna(subset=[form_depth_col])
    df_form = df_form.sort_values(form_depth_col).reset_index(drop=True)

    # 先按深度 merge_asof
    df_all[dep_col] = pd.to_numeric(df_all[dep_col], errors="coerce")
    df_all = df_all.dropna(subset=[dep_col])
    df_all = df_all.sort_values(dep_col).reset_index(drop=True)

    df_all = pd.merge_asof(
        df_all,
        df_form,
        left_on=dep_col,
        right_on=form_depth_col,
        direction="nearest",
    )

    # 再按时间排回去
    df_all = df_all.sort_values(time_col).reset_index(drop=True)

    # ========= 2) CAL → 直接使用第三列 BER =========
    cal_depth_col = merge_cfg["CAL_DEPTH_COL"]
    cal_ber_col = merge_cfg["CAL_BER_COL"]
    ber_lo, ber_hi = merge_cfg["BER_CLIP"]

    if cal_depth_col not in df_cal.columns or cal_ber_col not in df_cal.columns:
        raise ValueError(
            f"CAL 文件缺少必要列: 深度列={cal_depth_col}, BER列={cal_ber_col}，"
            f"当前列为: {df_cal.columns.tolist()}"
        )

    # 替换典型无效值
    df_cal[cal_ber_col] = df_cal[cal_ber_col].replace([-999, -999.25, -9999], np.nan)

    # 深度类型转为数值，并过滤 NaN
    df_cal[cal_depth_col] = pd.to_numeric(df_cal[cal_depth_col], errors="coerce")
    df_cal = df_cal.dropna(subset=[cal_depth_col])

    # 录井深度范围
    log_min = df_all[dep_col].min()
    log_max = df_all[dep_col].max()
    cal_min = df_cal[cal_depth_col].min()
    cal_max = df_cal[cal_depth_col].max()

    print(f"[DEBUG] 录井深度范围: [{log_min:.2f}, {log_max:.2f}]")
    print(f"[DEBUG] CAL 原始深度范围: [{cal_min:.2f}, {cal_max:.2f}]")

    # 只保留 CAL 中落在录井深度范围内的行，避免整口井对齐到端点
    df_cal = df_cal[(df_cal[cal_depth_col] >= log_min) &
                    (df_cal[cal_depth_col] <= log_max)].copy()

    if df_cal.empty:
        raise ValueError(
            "CAL 深度与录井深度完全不重叠，请检查数据（或检查 DEP/BITDEP 单位是否一致）。"
        )

    # 对 BER 做简单裁剪以防极端值
    df_cal[cal_ber_col] = df_cal[cal_ber_col].clip(ber_lo, ber_hi)

    # merge_asof 之前 right key 必须排序
    df_cal = df_cal.sort_values(cal_depth_col).reset_index(drop=True)
    df_all = df_all.sort_values(dep_col).reset_index(drop=True)

    df_all = pd.merge_asof(
        df_all,
        df_cal[[cal_depth_col, cal_ber_col]],
        left_on=dep_col,
        right_on=cal_depth_col,
        direction="nearest",
    )

    # 统一标签名为 "BER"
    if "BER" in df_all.columns and cal_ber_col != "BER":
        df_all.drop(columns=["BER"], inplace=True)
    df_all.rename(columns={cal_ber_col: "BER"}, inplace=True)

    # ========= 3) 基于 BER 构造平滑值、扩径段 + 时间域 pre-alert =========
    if "BER" in df_all.columns:
        ber_clip = merge_cfg.get("BER_CLIP", (0.0, 50.0))

        # 3.1 沿深度平滑 BER，压制 2→50→2 这种尖刺
        df_all = smooth_ber_along_depth(
            df_all,
            depth_col=dep_col,
            ber_col="BER",
            smooth_win=SEG_CFG["SMOOTH_WIN"],
            ber_clip=ber_clip,
        )

        # 理论上这里一定会有 "BER_SMOOTH"，做个安全检查防止之后再 KeyError
        if "BER_SMOOTH" not in df_all.columns:
            print("[WARN] smooth_ber_along_depth 未生成 'BER_SMOOTH'，将直接使用原始 BER。")
            df_all["BER_RAW"] = df_all["BER"]
            df_all["BER_PRE_ALERT"] = 0
        else:
            # 3.2 基于平滑后的 BER 构造连续扩径段（仍然在深度域）
            df_all = build_ber_segments_simple(
                df_all,
                depth_col=dep_col,
                ber_col="BER_SMOOTH",
                abs_thresh=SEG_CFG["START_THRESH"],  # 这里就当作绝对阈值使用
                min_seg_len=SEG_CFG["MIN_LEN_M"],
            )

            # 训练/评估统一使用平滑后的 BER 作为主标签，
            # 原始值保存在 BER_RAW，方便后续诊断
            df_all["BER_RAW"] = df_all["BER"]
            df_all["BER"] = df_all["BER_SMOOTH"]

            # 3.3 先构造 Seg-Now “核心区”标签（更干净、更稀疏）
            df_all = build_seg_now_labels(df_all, seg_id_col="BER_SEG_ID")

            # 3.4 B 方案：未来扩径风险标签 —— 用核心区 SEG_NOW_LABEL 生成（不要用 BER_IS_SEG）
            lookahead_steps = int(MODEL_CFG.get("LOOKAHEAD_STEPS", 60))
            future_steps_cfg = SEG_CFG.get("SEG_FUTURE_STEPS", "auto")

            if future_steps_cfg in (None, "auto"):
                future_steps = lookahead_steps
            else:
                future_steps = int(future_steps_cfg)
                if future_steps != lookahead_steps:
                    print(
                        f"[SEG][WARN] SEG_FUTURE_STEPS={future_steps} 与 LOOKAHEAD_STEPS={lookahead_steps} 不一致，已强制对齐。")
                    future_steps = lookahead_steps

            df_all = build_future_risk_label(
                df_all,
                seg_col="SEG_NOW_LABEL",  # ★关键：用核心区，不用 BER_IS_SEG
                out_col="SEG_FUTURE_LABEL",
                horizon_steps=future_steps,
            )

            # 3.5 基于时间的 pre-alert：扩径段开始前 PRE_ALERT_LEN_MIN 分钟
            pre_min = SEG_CFG.get("PRE_ALERT_LEN_MIN", 0.0)
            if pre_min is not None and pre_min > 0:
                df_all = add_time_pre_alert(
                    df_all,
                    time_col=time_col,
                    seg_id_col="BER_SEG_ID",
                    pre_alert_minutes=pre_min,
                )
            else:
                df_all["BER_PRE_ALERT"] = 0

            # 3.6 在扩径段内新增 4 档扩径等级标签（基于整段最大 BER）
            df_all = add_seg_level4_label(
                df_all,
                seg_id_col="BER_SEG_ID",
                seg_level_col="BER_SEG_LEVEL",
                core_col="SEG_NOW_LABEL",  # 只在核心区做 4 分类可用 *_CORE_ONLY
                out_col="SEG_LEVEL4_LABEL",
            )
    else:
        # 完全没有 BER（比如没 CAL）的井，扩径段/预警都置空
        print("[WARN] merge_formation_and_cal_to_time: 未找到 'BER' 列，跳过扩径段与 pre-alert 构造。")
        df_all["BER_RAW"] = np.nan
        df_all["BER_PRE_ALERT"] = 0
        df_all["SEG_FUTURE_LABEL"] = 0
    # 再按时间整理，方便后面 LSTM
    df_all = df_all.sort_values(time_col).reset_index(drop=True)

    # Debug：看平滑后的 BER 分布 & 扩径段数量
    if "BER" in df_all.columns:
        print("[DEBUG] 合并后 BER 分布（平滑后 BER）：")
        print(df_all["BER"].describe())
    if "BER_SEG_ID" in df_all.columns:
        print("[DEBUG] 扩径段数量:", int(df_all["BER_SEG_ID"].max()))

    return df_all


def plot_ber_segments_for_well(df: pd.DataFrame,
                               well_name: str,
                               outdir: str,
                               depth_col: str = "DEPTH",
                               ber_col: str = "BER",
                               ber_smooth_col: str = "BER_SMOOTH",
                               seg_flag_col: str = "BER_IS_SEG",
                               future_label_col: str | None = "SEG_FUTURE_LABEL",
                               start_thresh: float | None = None) -> None:
    """
    为单口井画一张“BER 曲线 + 扩径/未来风险标签”的示意图，并保存到 outdir。

    - 上图：原始 BER_RAW（如果有） + 当前用于训练的 BER（平滑后）
    - 下图：BER_IS_SEG（当前是否处于扩径段） + SEG_FUTURE_LABEL（未来是否会扩径）
    """
    # 列检查
    if depth_col not in df.columns or ber_col not in df.columns:
        print(f"[PLOT][WARN] 井 {well_name} 缺少 {depth_col} 或 {ber_col}，跳过画图。")
        return

    # 按深度排序，方便看成“测井曲线”
    df_plot = df.copy()
    df_plot = df_plot.sort_values(depth_col).reset_index(drop=True)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        5, 1, figsize=(16, 13), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.05, 0.40, 1.15, 0.28], "hspace": 0.220}
    )
    fig.suptitle(f"{well_name} | 测井风格分类-only 看板", y=0.99, fontsize=13)

    # 只设置一次深度方向（避免每个子图 invert）
    axes[0].set_xlim(df_plot[depth_col].max(), df_plot[depth_col].min())


    # ===== 上图：BER 曲线 =====
    ax_top = axes[0]
    x = df_plot[depth_col].values

    # 原始 BER（如果有）
    if "BER_RAW" in df_plot.columns:
        ax_top.plot(x, df_plot["BER_RAW"].values,
                    label="BER_RAW", linewidth=0.8, alpha=0.4)

    # 当前用于训练的 BER（一般就是 BER_SMOOTH）
    ax_top.plot(x, df_plot[ber_col].values,
                label=ber_col, linewidth=1.0)

    # 单独画一下 BER_SMOOTH 列（如果跟 BER 列不一样）
    if (ber_smooth_col in df_plot.columns) and (ber_smooth_col != ber_col):
        ax_top.plot(x, df_plot[ber_smooth_col].values,
                    label=ber_smooth_col, linewidth=0.8, alpha=0.8, linestyle="--")

    # 阈值线：START_THRESH
    if start_thresh is not None:
        ax_top.axhline(start_thresh,
                       linestyle="--",
                       linewidth=0.8,
                       label=f"START_THRESH={start_thresh}")

    ax_top.set_ylabel("BER (%)")
    ax_top.set_title(f"{well_name} 扩径段检测示意")
    ax_top.grid(True, linestyle="--", alpha=0.3)
    ax_top.legend(loc="upper right", fontsize=8)

    # ===== 下图：标签（扩径段 / 未来风险） =====
    ax_bot = axes[1]

    if seg_flag_col in df_plot.columns:
        ax_bot.step(x, df_plot[seg_flag_col].values,
                    where="post", label=seg_flag_col)

    if future_label_col and (future_label_col in df_plot.columns):
        ax_bot.step(x, df_plot[future_label_col].values,
                    where="post", label=future_label_col, alpha=0.7)

    ax_bot.set_xlabel(depth_col)
    ax_bot.set_ylabel("Label")
    ax_bot.grid(True, linestyle="--", alpha=0.3)
    ax_bot.legend(loc="upper right", fontsize=8)

    fig.subplots_adjust(top=0.95, hspace=0.12)

    os.makedirs(outdir, exist_ok=True)
    fig_path = os.path.join(outdir, f"{well_name}_depth_profile.png")
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[PLOT] 扩径段检测图已保存: {fig_path}")


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在已经 merge 好地层特征 + 工程参数 + CAL 之后，
    构造一些“地层 × 工程参数”的交互特征。
    注意：不使用任何基于 BER / BER_IS_SEG / BER_SEG_ID 的信息，避免标签泄露。
    """
    df = df.copy()
    eps = 1e-3  # 防止除零

    # ========= 1) 载荷 / 岩石强度类 =========
    # WOB / UCS：井底载荷相对抗压强度
    if {"WOB", "UCS"}.issubset(df.columns):
        denom = df["UCS"].abs().clip(lower=eps)
        df["WOB_over_UCS"] = (df["WOB"] / denom).clip(-50, 50)

    # TOR / UCS：扭矩相对抗剪能力（粗略）
    if {"TOR", "UCS"}.issubset(df.columns):
        denom = df["UCS"].abs().clip(lower=eps)
        df["TOR_over_UCS"] = (df["TOR"] / denom).clip(-50, 50)

    # HKLD / Sv：摩阻相对垂向应力
    if {"HKLD", "Sv"}.issubset(df.columns):
        denom = df["Sv"].abs().clip(lower=eps)
        df["HKLD_over_Sv"] = (df["HKLD"] / denom).clip(-100, 100)

    # ========= 2) 泥浆压力窗 / 安全裕度类 =========
    # 用 MWIN × TVD / Depth 构一个近似 P_mud，只做相对比较，不追求绝对单位
    P_mud = None
    if "MWIN" in df.columns and "TVD" in df.columns:
        P_mud = df["MWIN"] * df["TVD"]
    elif "MWIN" in df.columns and "Depth" in df.columns:
        P_mud = df["MWIN"] * df["Depth"]

    if P_mud is not None:
        # 超平衡 / 欠平衡：相对孔隙压力
        if "Pp1" in df.columns:
            df["delta_p_pore"] = P_mud - df["Pp1"]

        # 距破裂压力的裕度
        if "Pf" in df.columns:
            df["delta_p_frac"] = df["Pf"] - P_mud

        # 安全窗口内相对位置：0 近坍塌侧，1 近破裂侧
        if "Pp1" in df.columns and "Pf" in df.columns:
            denom = (df["Pf"] - df["Pp1"]).abs().clip(lower=eps)
            frac_ratio = (P_mud - df["Pp1"]) / denom
            df["frac_window_ratio"] = frac_ratio.clip(-2, 2)

    # ========= 3) 水平应力各向异性 × 井斜 =========
    if {"SHH", "Sh"}.issubset(df.columns):
        df["stress_aniso"] = df["SHH"] - df["Sh"]  # 水平主应力差
        if "DEV" in df.columns:
            df["dev_times_aniso"] = df["DEV"] * df["stress_aniso"]

    # ========= 4) 井身方位编码（方便模型利用角度信息） =========
    if "DAZ" in df.columns:
        rad = np.deg2rad(df["DAZ"])
        df["cos_DAZ"] = np.cos(rad)
        df["sin_DAZ"] = np.sin(rad)

    # ========= 5) 清洗能力 × 岩屑产出 =========
    # 机械能 proxy：WOB × RPM（破岩强度）
    mech_energy = None
    if "WOB" in df.columns and "RPM" in df.columns:
        mech_energy = df["WOB"] * df["RPM"]
        df["mech_energy_proxy"] = mech_energy
    # 清洗能力 proxy：FLOWIN × MWIN（有 MWIN 就乘一乘）
    cleaning_cap = None
    if "FLOWIN" in df.columns:
        if "MWIN" in df.columns:
            cleaning_cap = df["FLOWIN"] * df["MWIN"]
        else:
            cleaning_cap = df["FLOWIN"]
        df["cleaning_cap_proxy"] = cleaning_cap

    if mech_energy is not None and cleaning_cap is not None:
        denom = cleaning_cap.abs().clip(lower=eps)
        df["cuttings_over_cleaning"] = (mech_energy / denom).clip(0, 1000)

    return df


# =========================
# 特征工程（差分 + 滚动）
# =========================

def add_derived_features(df: pd.DataFrame, dynamic_cols: List[str]) -> pd.DataFrame:
    """一阶差分 + 短/中期滚动统计"""
    df = df.copy()
    if "dt_sec" not in df.columns:
        raise ValueError("dt_sec 列不存在，请先调用 clean_time_series_per_well。")

    # 一阶时间差分
    for col in dynamic_cols:
        if col not in df.columns:
            continue
        dcol = f"d{col}_dt"
        df[dcol] = df[col].diff() / df["dt_sec"].replace(0, np.nan)
        df[dcol] = df[dcol].replace([np.inf, -np.inf], 0.0)
        df[dcol] = df[dcol].fillna(0.0)

    # 滚动统计（短 / 中）
    roll_windows = [15, 60]
    for col in dynamic_cols:
        if col not in df.columns:
            continue
        for w in roll_windows:
            df[f"{col}_mean_w{w}"] = df[col].rolling(window=w, min_periods=1).mean()
            df[f"{col}_std_w{w}"] = df[col].rolling(window=w, min_periods=1).std().fillna(0.0)

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于关键工程参数 / 交互特征构造“趋势特征”，
    让分类 / 回归模型能看到扩径前的一段持续变化。
    """
    df = df.copy()

    # 1) 选取可能与扩径相关的基础列
    dyn_candidates = ["WOB", "RPM", "TOR", "SPP", "FLOWIN", "FLOWOUT", "HKLD"]
    # 这些是典型地层参数名称，如果你的表里叫 Pp / Pp1 等，只要重名就会自动命中
    form_candidates = ["Pp1", "Pc", "Pf", "UCS", "Sv", "Sh", "SHH"]
    # 如果你之前已经加过交互特征（如 delta_p_pore 等），这里也一并利用
    interact_candidates = [
        "delta_p_pore",
        "delta_p_frac",
        "WOB_over_UCS",
        "HKLD_over_Sv",
        "cuttings_over_cleaning",
    ]

    base_cols: List[str] = []
    for c in dyn_candidates + form_candidates + interact_candidates:
        if c in df.columns:
            base_cols.append(c)

    if not base_cols:
        # 没有可用列就直接返回
        return df

    # 2) 定义短/中期窗口长度（按采样点数，不是绝对时间）
    #    这里可以理解为“短期 20 点 / 中期 60 点 的趋势”
    windows = [20, 60]

    for col in base_cols:
        s = df[col].astype(float)

        for w in windows:
            # a) w 步差分：反映在 w 步内的总体抬升幅度
            diff = s - s.shift(w)
            df[f"{col}_diff_w{w}"] = diff

            # b) 相对滚动均值偏移：当前值相对过去 w 步平均值的偏高程度
            ma = s.rolling(window=w, min_periods=1).mean()
            df[f"{col}_trend_vs_mean_w{w}"] = s - ma

    # NaN / inf 会在标准化前统一做 ffill + 填 0，这里不用重复处理
    return df


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """自动选择数值特征列（排除标签/权重/FLAG/原始深度）"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_prefixes = ["FLAG_"]

    # ★ 把 DEP / BITDEP 从特征里排除，避免被标准化覆盖掉真实深度
    dep_cols = CLEAN_CFG.get("DEP_COL_CANDIDATES", [])
    exclude_exact = {
                        "BER", "BER_SMOOTH", "BER_RAW",
                        "BER_SEG_ID", "BER_IS_SEG", "BER_PRE_ALERT",
                        "SEG_NOW_LABEL", "SEG_FUTURE_LABEL",
                        "SEG_LEVEL4_LABEL", "SEG_LEVEL4_LABEL_CORE_ONLY",
                        "OPC_LABEL", "OPC_PROB",
                        "sample_weight", "LONGTIME", "CW",
                    } | set(dep_cols)

    # 这种列名里一般也是各种标签、灯光派生结果
    exclude_substr = [
        "_label",  # 各种 *_LABEL
        "label_",  # 以 label_ 开头的
        "light_",  # e.g. LIGHT_MAIN / LIGHT_FUSED
        "seg_",  # e.g. SEG_NOW_LABEL / SEG_FUTURE_LABEL / SEG_XXX
        "opc_",  # OPC_* 标签 / 打分
        "risk_",  # 风险相关派生结果
    ]

    feats = []
    for c in num_cols:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if any(s in c for s in exclude_substr):
            continue
        feats.append(c)
    suspicious = [c for c in feats if "LABEL" in c.upper()]
    if suspicious:
        print("[WARN] feature_cols 中出现疑似标签列:", suspicious)
    return feats


def analyze_seg_future_labels(
        train_dfs,
        outdir: str,
        depth_col: str = "DEPTH",
        time_col: str = "WELLDATETIME",
        future_label_col: str = "SEG_FUTURE_LABEL",
        seg_col: str = "BER_IS_SEG",
) -> None:
    """
    分析各口井的未来扩径风险标签 (B 方案) 分布情况，主要关注：
    - 每口井的正样本数量与比例
    - 正样本在深度 / 时间上的分布范围
    - 连续正样本段（run）的数量与长度统计

    结果会输出到 outdir / "future_label_analysis.csv"
    并在日志中打印每口井的简要统计。
    """
    import os
    import numpy as np
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)
    rows = []

    print("[ANALYSIS] 开始分析各口井的未来扩径风险标签 (SEG_FUTURE_LABEL)...")

    for df in train_dfs:
        if future_label_col not in df.columns:
            # 这口井没有未来风险标签，跳过或按 0 处理
            well_name = str(df.get("WELL", "UNKNOWN"))
            print(f"[ANALYSIS][WARN] 井 {well_name} 缺少列 '{future_label_col}'，跳过。")
            continue

        well_name = str(df.get("WELL", df.get("WELL_NAME", "UNKNOWN")))
        n = len(df)
        if n == 0:
            continue

        label = df[future_label_col].fillna(0).astype(int).to_numpy()

        n_pos = int(label.sum())
        pos_ratio = n_pos / float(n)

        # 深度 / 时间分布（只考虑正样本）
        if n_pos > 0:
            pos_idx = np.where(label == 1)[0]

            if depth_col in df.columns:
                depth_vals = pd.to_numeric(df[depth_col], errors="coerce").to_numpy()
                depth_pos = depth_vals[pos_idx]
                min_depth_pos = float(np.nanmin(depth_pos))
                max_depth_pos = float(np.nanmax(depth_pos))
            else:
                min_depth_pos = np.nan
                max_depth_pos = np.nan

            if time_col in df.columns:
                time_vals = pd.to_datetime(df[time_col], errors="coerce")
                time_pos = time_vals.iloc[pos_idx]
                min_time_pos = time_pos.min()
                max_time_pos = time_pos.max()
            else:
                min_time_pos = pd.NaT
                max_time_pos = pd.NaT

            # 统计连续 1 段（run-length）
            diffs = np.diff(pos_idx)
            # run 的起始位置：差分 != 1 的地方
            run_starts = np.concatenate(([0], np.where(diffs > 1)[0] + 1))
            n_runs = len(run_starts)

            run_lengths = []
            for i_run in range(n_runs):
                start = run_starts[i_run]
                if i_run + 1 < n_runs:
                    end = run_starts[i_run + 1]
                else:
                    end = len(pos_idx)
                run_len = end - start
                run_lengths.append(run_len)

            run_lengths = np.array(run_lengths, dtype=int)
            avg_run_len = float(run_lengths.mean())
            median_run_len = float(np.median(run_lengths))
            max_run_len = int(run_lengths.max())
        else:
            min_depth_pos = np.nan
            max_depth_pos = np.nan
            min_time_pos = pd.NaT
            max_time_pos = pd.NaT
            n_runs = 0
            avg_run_len = 0.0
            median_run_len = 0.0
            max_run_len = 0

        # 可选：查看“标签=1 时真正处于扩径段的比例”，衡量预警 vs 段内占比
        if seg_col in df.columns and n_pos > 0:
            seg_flag = df[seg_col].fillna(0).astype(int).to_numpy()
            n_pos_in_seg = int(seg_flag[label == 1].sum())
            frac_pos_in_seg = n_pos_in_seg / float(n_pos)
        else:
            n_pos_in_seg = np.nan
            frac_pos_in_seg = np.nan

        # 将统计结果存一行
        rows.append({
            "well": well_name,
            "n_samples": n,
            "n_pos": n_pos,
            "pos_ratio": pos_ratio,
            "n_runs": n_runs,
            "avg_run_len": avg_run_len,
            "median_run_len": median_run_len,
            "max_run_len": max_run_len,
            "min_depth_pos": min_depth_pos,
            "max_depth_pos": max_depth_pos,
            "min_time_pos": min_time_pos,
            "max_time_pos": max_time_pos,
            "n_pos_in_seg": n_pos_in_seg,
            "frac_pos_in_seg": frac_pos_in_seg,
        })

        # 在日志里打印一个简要 summary
        print(
            f"[ANALYSIS][{well_name}] n={n}, pos={n_pos} ({pos_ratio:.4f}), "
            f"runs={n_runs}, avg_run_len={avg_run_len:.1f}, "
            f"median_run_len={median_run_len:.1f}, max_run_len={max_run_len}, "
            f"depth_pos≈[{min_depth_pos:.1f}, {max_depth_pos:.1f}], "
            f"pos_in_seg={n_pos_in_seg}, frac_pos_in_seg={frac_pos_in_seg:.3f}"
        )

    if len(rows) == 0:
        print("[ANALYSIS] 未生成任何 future_label 分析结果（可能缺少 SEG_FUTURE_LABEL 列）。")
        return

    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "future_label_analysis.csv")
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[ANALYSIS] 未来扩径风险标签统计已保存到: {csv_path}")


def fit_feature_scaler(train_dfs: List[pd.DataFrame], feature_cols: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    X_list = []
    for df in train_dfs:
        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        # ★ 只允许向前填充，不回填未来值
        X = X.ffill()
        # 对开头整列都是 NaN 的极端情况，统一兜底成 0（表示“无信息”）
        X = X.fillna(0.0)
        X_list.append(X.values)
    X_all = np.vstack(X_list)
    scaler.fit(X_all)
    return scaler


def apply_feature_scaler(df: pd.DataFrame, feature_cols: List[str], scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    # ★ 同样只做向前填充
    X = X.ffill()
    X = X.fillna(0.0)
    df[feature_cols] = scaler.transform(X.values)
    return df


# =========================
# 数据集 & LSTM+Attn 模型
# =========================

from torch.utils.data import Dataset


class TimeSeriesBERDataset(Dataset):
    """
    支持时间前看的数据集：
    - 每个样本 = [seq_len, feature_dim] 的窗口
    - 标签 y = 窗口最后一点向前看 LOOKAHEAD_STEPS 个点的 BER（向量）
    - 如果 LOOKAHEAD_ONLY_IN_SEG=True，只选择处于扩径段或前视窗口内的样本；
      若在该模式下没有构造出任何样本，则自动放宽为“全时段均可取样”。
    """

    def __init__(self, dfs, feature_cols, seq_len,
                 target_col: str = "BER",
                 lookahead_steps: int = 0,
                 lookahead_only_in_seg: bool = False,
                 lookahead_margin: int = 0,
                 allow_relax: bool = True,
                 seg_label_col: str = "BER_IS_SEG"):
        self.dfs = dfs
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.target_col = target_col
        self.lookahead_steps = lookahead_steps
        self.lookahead_only_in_seg = lookahead_only_in_seg
        self.lookahead_margin = lookahead_margin
        self.allow_relax = allow_relax
        self.seg_label_col = seg_label_col

        # 先按当前配置（可能是 only_in_seg=True）严选一遍
        (samples,
         seg_flags,
         lookahead_labels,
         lookahead_masks) = self._build_samples(only_in_seg=lookahead_only_in_seg)

        # 如果严格模式下没有任何样本，且允许放宽，就退回到“全时段均可取样”
        if len(samples) == 0 and lookahead_only_in_seg and allow_relax:
            print(
                "[WARN] TimeSeriesBERDataset: 在 "
                "LOOKAHEAD_ONLY_IN_SEG=True 模式下没有构造出任何样本，"
                "将自动放宽为 LOOKAHEAD_ONLY_IN_SEG=False 重新构造。"
            )
            (samples,
             seg_flags,
             lookahead_labels,
             lookahead_masks) = self._build_samples(only_in_seg=False)

        # 如果仍然没有样本，才真正报错
        if len(samples) == 0:
            raise ValueError("TimeSeriesBERDataset: 没有构造出任何样本，请检查数据与配置。")

        self.samples = samples
        self.seg_flags = seg_flags
        self.lookahead_labels = np.array(lookahead_labels)
        self.lookahead_masks = np.array(lookahead_masks)

        print(f"[INFO] 数据集构造完成: {len(self.samples)} 个样本")
        print(f"[INFO] 前看步数: {self.lookahead_steps}, 标签形状: {self.lookahead_labels.shape}")

    # -------- 内部函数：真正构造样本的逻辑 --------
    def _build_samples(self, only_in_seg: bool):
        samples = []  # (well_index, end_row_index)
        seg_flags = []  # 当前点是否在扩径段内
        lookahead_labels = []  # 前看标签序列
        lookahead_masks = []  # 标签掩码

        for wi, df in enumerate(self.dfs):
            n = len(df)
            if n < self.seq_len + self.lookahead_steps:
                continue

            flag_gap = df["FLAG_TIME_GAP"].values if "FLAG_TIME_GAP" in df.columns else np.zeros(n)

            # 选择用于 seg_flags 的列：
            # 优先使用 self.seg_label_col；
            # 如果该列不存在，则退回优先 SEG_NOW_LABEL，再退回 BER_IS_SEG。
            seg_label_col = self.seg_label_col

            if seg_label_col not in df.columns:
                # ★ 当 lookahead_steps == 0 时，我们基本上是在做“纯分类”DataLoader，
                #   此时如果指定的 seg_label_col 不存在，就直接报错，而不是悄悄换其他列。
                if self.lookahead_steps == 0 and self.seg_label_col is not None:
                    raise ValueError(
                        f"TimeSeriesBERDataset: 期望使用 seg_label_col='{self.seg_label_col}' "
                        f"作为分类标签，但在 df.columns 中找不到，请检查是否已构造该列。"
                    )

                # 只有在“带前看的回归模式”下才允许 fallback
                if "SEG_NOW_LABEL" in df.columns:
                    seg_label_col = "SEG_NOW_LABEL"
                elif "BER_IS_SEG" in df.columns:
                    seg_label_col = "BER_IS_SEG"
                else:
                    seg_label_col = None

            if seg_label_col is not None:
                has_seg = True
                seg_arr = df[seg_label_col].values.astype(int)
            else:
                has_seg = False
                seg_arr = np.zeros(n, dtype=int)

            ber_arr = df[self.target_col].values if self.target_col in df.columns else np.zeros(n)

            for end in range(self.seq_len - 1, n - self.lookahead_steps):
                start = end - self.seq_len + 1

                # 时间断点检查：窗口 + 未来 lookahead 步中不能跨大时间缝
                if "FLAG_TIME_GAP" in df.columns:
                    if flag_gap[start + 1: end + self.lookahead_steps + 1].sum() > 0:
                        continue

                window = df.iloc[start: end + 1]
                future_window = df.iloc[end + 1: end + self.lookahead_steps + 1] if self.lookahead_steps > 0 else None

                # 特征 / 标签 NaN 检查
                if window[self.feature_cols].isna().any().any():
                    continue
                if future_window is not None and future_window[self.target_col].isna().any():
                    continue
                # 如果是“无前看”的分类模式，并且 target_col 使用 -1 表示无效标签，则跳过
                if self.lookahead_steps == 0 and self.target_col in df.columns:
                    last_label = window[self.target_col].iloc[-1]
                    try:
                        last_label_val = float(last_label)
                    except Exception:
                        last_label_val = None

                    # 约定：SEG_LEVEL4_LABEL_CORE_ONLY 用 -1 表示“不参与训练”
                    if last_label_val is not None and last_label_val < 0:
                        continue

                # 只在扩径/预警区取样的条件（可开关）
                if only_in_seg and has_seg:
                    current_in_seg = seg_arr[end] == 1
                    future_in_seg = seg_arr[end + 1: end + self.lookahead_steps + 1].sum() > 0
                    lookback_in_seg = seg_arr[max(0, end - self.lookahead_margin): end + 1].sum() > 0

                    if not (current_in_seg or future_in_seg or lookback_in_seg):
                        continue

                # 构造前看标签
                if self.lookahead_steps > 0:
                    future_labels = ber_arr[end + 1: end + self.lookahead_steps + 1].astype(np.float32)
                    future_mask = np.ones_like(future_labels, dtype=np.float32)
                    lookahead_labels.append(future_labels)
                    lookahead_masks.append(future_mask)
                else:
                    # 兼容无前看的情况
                    future_labels = np.array(
                        [float(window[self.target_col].iloc[-1])],
                        dtype=np.float32
                    )
                    future_mask = np.array([1.0], dtype=np.float32)
                    lookahead_labels.append(future_labels)
                    lookahead_masks.append(future_mask)

                samples.append((wi, end))
                is_seg = int(seg_arr[end]) if has_seg else 0
                seg_flags.append(is_seg)

        return samples, seg_flags, lookahead_labels, lookahead_masks

    # -------- DataLoader 需要的接口 --------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wi, end = self.samples[idx]
        df = self.dfs[wi]
        start = end - self.seq_len + 1
        window = df.iloc[start: end + 1]

        x = window[self.feature_cols].values.astype(np.float32)
        y = self.lookahead_labels[idx]
        mask = self.lookahead_masks[idx]
        w = float(window["sample_weight"].iloc[-1]) if "sample_weight" in df.columns else 1.0
        seg_flag = float(self.seg_flags[idx])

        return (
            torch.from_numpy(x),  # [T, D]
            torch.tensor(y, dtype=torch.float32),  # [lookahead_steps]
            torch.tensor(mask, dtype=torch.float32),  # [lookahead_steps]
            torch.tensor(w, dtype=torch.float32),  # []
            torch.tensor(seg_flag, dtype=torch.float32),  # []
        )





class EnhancedSegClassifier(nn.Module):
    """
    增强版扩径段二分类器（加入残差学习）：
    - 前面用 1D 卷积提短/中尺度局部模式
    - 后面用 BiLSTM 捕获长时依赖
    - 再用注意力聚合成一个全序列表示
    - 最后用 “基线线性层 + 残差 MLP” 输出 logit：
        logit = base_logit(context) + delta_logit(context)
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            bidirectional: bool = True,
            dropout: float = 0.3,
    ):
        super().__init__()

        # 1) 卷积特征提取（相当于不同时间尺度的局部滤波）
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

        # 2) BiLSTM 捕获时序依赖
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.num_directions = 2 if bidirectional else 1
        enc_dim = hidden_dim * self.num_directions

        # 3) 注意力：对时间维做加权平均
        self.attention = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # 4) 残差分类头：
        #    - base_fc: 直接对 context 做一层线性，给出“基线” logit
        #    - delta_mlp: 多层 MLP 学习残差校正
        self.dropout = nn.Dropout(dropout)
        self.base_fc = nn.Linear(enc_dim, 1)

        self.delta_mlp = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape

        # ---- 1) Conv 提特征 ----
        x_conv = x.transpose(1, 2)  # [B, D, T]
        x_conv = F.relu(self.bn1(self.conv1(x_conv)))  # [B, 32, T]
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))  # [B, 64, T]
        x_conv = x_conv.transpose(1, 2)  # [B, T, 64]

        # ---- 2) LSTM 时序编码 ----
        lstm_out, _ = self.lstm(x_conv)  # [B, T, enc_dim]

        # ---- 3) 注意力聚合 ----
        attn_score = self.attention(lstm_out).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(attn_score, dim=1)  # [B, T]
        context = torch.sum(
            lstm_out * attn_weights.unsqueeze(-1), dim=1
        )  # [B, enc_dim]

        context = self.dropout(context)

        # ---- 4) 残差 logit = 基线 + 残差 ----
        base_logit = self.base_fc(context).squeeze(-1)  # [B]
        delta_logit = self.delta_mlp(context).squeeze(-1)  # [B]
        logits = base_logit + delta_logit  # [B]

        return logits


class ExpansionLevelClassifier(nn.Module):
    """
    扩径严重度四分类模型：
    - 前端结构基本沿用 EnhancedSegClassifier：
        Conv1D → Conv1D → BiLSTM → Attention 聚合
    - 差别：最后输出 4 维 logits，对应 0/1/2/3 四个等级。
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            bidirectional: bool = True,
            dropout: float = 0.3,
            num_classes: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ---- 1) 两层 1D 卷积提局部模式 ----
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        # ---- 2) BiLSTM 提时序依赖 ----
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        enc_dim = hidden_dim * (2 if bidirectional else 1)

        # ---- 3) 注意力聚合 ----
        self.attention = nn.Sequential(
            nn.Linear(enc_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.dropout = nn.Dropout(p=dropout)

        # ---- 4) 残差形式 logits = base + delta ----
        self.base_fc = nn.Linear(enc_dim, num_classes)
        self.delta_mlp = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        return: logits [B, num_classes]
        """
        # Conv 期望输入 [B, D, T]
        x_conv = x.transpose(1, 2)  # [B, D, T]
        x_conv = F.relu(self.bn1(self.conv1(x_conv)))  # [B, 32, T]
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))  # [B, 64, T]
        x_conv = x_conv.transpose(1, 2)  # [B, T, 64]

        # LSTM
        lstm_out, _ = self.lstm(x_conv)  # [B, T, enc_dim]

        # Attention
        attn_score = self.attention(lstm_out).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(attn_score, dim=1)  # [B, T]
        context = torch.sum(
            lstm_out * attn_weights.unsqueeze(-1), dim=1
        )  # [B, enc_dim]

        context = self.dropout(context)

        base_logits = self.base_fc(context)  # [B, C]
        delta_logits = self.delta_mlp(context)  # [B, C]
        logits = base_logits + delta_logits  # [B, C]

        return logits


class FocalLossWithLogits(nn.Module):
    """
    二分类用的 Focal Loss（输入为 logits）：
    - input: [B] 的 logits
    - target: [B] 的 0/1 标签（float 或 long 都可）
    - alpha: 正类权重（0~1），为 None 时不使用 alpha
    - gamma: 聚焦因子，gamma 越大越聚焦难样本
    """

    def __init__(self,
                 alpha: float = None,
                 gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 保证 target 为 float 型
        target = target.float()
        # 先算普通 BCE loss（逐样本）
        bce_loss = F.binary_cross_entropy_with_logits(
            input, target, reduction="none"
        )  # [B]

        # p = sigmoid(logit)，p_t = p (y=1) 或 1-p (y=0)
        p = torch.sigmoid(input)
        p_t = p * target + (1 - p) * (1 - target)

        # Focal 重点：对易分样本 (p_t 大) 降权，对难样本 (p_t 小) 放大
        focal_weight = (1 - p_t).pow(self.gamma)

        # alpha 加权：正负类不同权重
        if self.alpha is not None:
            # 正类权重 alpha，负类权重 1-alpha
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce_loss  # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ModelDiagnosticSystem:
    def __init__(self, name="classification"):
        self.name = name
        self.logs = []

    def log(self, level: str, tag: str, message: str):
        """
        统一的日志接口
        level: "info" / "warning" / "error"
        tag:   模块名，比如 "CLASSIFICATION_VAL"
        """
        lvl = level.upper()
        tag = str(tag)
        msg = f"[DIAG][{self.name}][{tag}][{lvl}] {message}"
        print(msg)
        self.logs.append({"level": lvl, "tag": tag, "message": message})

    def to_dataframe(self):
        import pandas as pd
        if not self.logs:
            return pd.DataFrame(columns=["level", "tag", "message"])
        return pd.DataFrame(self.logs)

    def save_logs(self, outdir: str, filename: str = "classification_diag_logs.csv"):
        import os
        os.makedirs(outdir, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(os.path.join(outdir, filename), index=False, encoding="utf-8-sig")


# =========================
# 主模型训练 / 评估
# =========================


# ====== 扩径段二分类训练（Seg-Now 分类器） ======
def _multiclass_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        num_classes: int = 4):
    """
    简单多分类指标:
    - overall accuracy
    - macro F1
    - 各类别 recall 列表
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t < 0 or t >= num_classes:
            continue
        if p < 0 or p >= num_classes:
            continue
        cm[t, p] += 1

    per_recall = []
    per_prec = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - tp)
        fp = float(cm[:, c].sum() - tp)

        rec = tp / (tp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        per_recall.append(rec)
        per_prec.append(prec)

    macro_recall = float(np.mean(per_recall)) if per_recall else 0.0
    macro_prec = float(np.mean(per_prec)) if per_prec else 0.0
    macro_f1 = 2.0 * macro_prec * macro_recall / (macro_prec + macro_recall + 1e-6)

    acc = float(np.trace(cm)) / (len(y_true) + 1e-6)

    return acc, macro_f1, per_recall


def train_one_epoch_seg_level4(model,
                               loader,
                               optimizer,
                               device,
                               num_classes: int = 4,
                               class_weight: torch.Tensor = None):
    """
    扩径严重度 4 分类训练：
    - 使用 TimeSeriesBERDataset, LOOKAHEAD_STEPS=0, target_col='SEG_LEVEL4_LABEL_CORE_ONLY'
    - y: [B, 1] → squeeze 成 [B] 的整数标签 0/1/2/3
    """
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        # x: [B,T,D], y: [B,1], mask: [B,1], w: [B], seg_flag: [B]
        x, y, mask, w, seg_flag = batch
        x = x.to(device)
        # y 是 float，形状 [B,1]，先 squeeze 再转 long
        y_cls = y[:, 0].long().to(device)

        optimizer.zero_grad()
        logits = model(x)  # [B, num_classes]
        loss = criterion(logits, y_cls)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        t = y_cls.detach().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(t)

    if n_samples == 0:
        return 0.0, 0.0, 0.0, []

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc, macro_f1, rec_list = _multiclass_metrics(
        all_targets, all_preds, num_classes=num_classes
    )
    avg_loss = total_loss / n_samples
    return avg_loss, acc, macro_f1, rec_list


def eval_one_epoch_seg_level4(model,
                              loader,
                              device,
                              num_classes: int = 4,
                              class_weight: torch.Tensor = None):
    """
    扩径严重度 4 分类验证：
    返回: (loss, acc, macro_f1, per_class_recall_list)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x, y, mask, w, seg_flag = batch
            x = x.to(device)
            y_cls = y[:, 0].long().to(device)

            logits = model(x)
            loss = criterion(logits, y_cls)

            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            t = y_cls.detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(t)

    if n_samples == 0:
        return 0.0, 0.0, 0.0, []

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc, macro_f1, rec_list = _multiclass_metrics(
        all_targets, all_preds, num_classes=num_classes
    )
    avg_loss = total_loss / n_samples
    return avg_loss, acc, macro_f1, rec_list


def train_one_epoch_seg_cls(model,
                            loader,
                            optimizer,
                            device,
                            pos_weight: torch.Tensor = None,
                            use_focal: bool = False,
                            focal_gamma: float = 2.0,
                            focal_alpha: float = None):
    """
    扩径段二分类训练：
    - 输入: (x, y_reg, mask, w, seg_flag)
    - 只使用 seg_flag 作为标签，忽略其他
    - 当 use_focal=True 时，使用 Focal Loss 处理类别不平衡；
      否则回退到 BCEWithLogitsLoss（可带 pos_weight）。
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    # 选择损失函数
    if use_focal:
        criterion = FocalLossWithLogits(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction="mean",
        )
    else:
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction="mean",
            )
        else:
            criterion = nn.BCEWithLogitsLoss(reduction="mean")

    for batch in loader:
        # 现在 batch 包含 5 个元素
        if len(batch) == 5:
            x, _, _, _, seg_flag = batch  # 解包所有5个元素
        elif len(batch) == 4:
            # 老版本兼容
            x, _, _, seg_flag = batch
        else:
            # 理论上不会到这里，但做保护
            print(f"[WARN] 意外的batch长度: {len(batch)}")
            continue

        x = x.to(device)  # [B, T, D]
        labels = seg_flag.to(device).view(-1)  # [B]

        optimizer.zero_grad()
        logits = model(x).view_as(labels)  # [B]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss


def find_best_threshold_with_recall_floor(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        recall_floor: float = 0.9,
):
    """
    在一系列阈值上扫描，找到:
    - 若存在 recall >= recall_floor 的阈值:
        在这些阈值中选 F1 最高的那个;
    - 否则:
        在所有阈值中选 recall 最大的那个(退而求其次),
        并标记 status="below_floor"。

    返回:
        {
            "thr":      最终选择的阈值,
            "precision":该阈值下的 precision,
            "recall":   该阈值下的 recall,
            "f1":       该阈值下的 F1,
            "status":   "ok" 或 "below_floor"
        }
    """
    thr_list = np.linspace(0.05, 0.95, 19)

    best_under_floor = {
        "thr": thr_list[0],
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    candidates = []  # 满足 recall >= floor 的阈值集合

    for thr in thr_list:
        y_pred = (y_prob >= thr).astype(int)

        tp = float(((y_true == 1) & (y_pred == 1)).sum())
        fp = float(((y_true == 0) & (y_pred == 1)).sum())
        fn = float(((y_true == 1) & (y_pred == 0)).sum())

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-6)

        # 记录“全局最高 recall”的点(用于 floor 达不到时退而求其次)
        if recall > best_under_floor["recall"] + 1e-8:
            best_under_floor = {
                "thr": thr,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        # 记录所有满足 recall >= floor 的候选
        if recall >= recall_floor:
            candidates.append((thr, precision, recall, f1))

    if candidates:
        # ✅ 有阈值能达到安全线: 在候选中按 F1 最高选
        thr, precision, recall, f1 = max(candidates, key=lambda x: x[3])
        status = "ok"
    else:
        # ⚠ 无阈值达到安全线: 选“全局最高 recall”的那个阈值
        thr = best_under_floor["thr"]
        precision = best_under_floor["precision"]
        recall = best_under_floor["recall"]
        f1 = best_under_floor["f1"]
        status = "below_floor"

    return {
        "thr": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "status": status,
    }


def eval_one_epoch_seg_cls(model,
                           loader,
                           device,
                           pos_weight: torch.Tensor = None,
                           use_focal: bool = False,
                           focal_gamma: float = 2.0,
                           focal_alpha: float = None,
                           recall_floor: float = 0.9):
    """
    扩径段二分类验证:
    - 计算 threshold=0.5 下的 acc / recall / F1 / AUC (主指标返回值)
    - 额外:
        * 扫一遍阈值, 找出 best_F1 (不加约束)
        * 使用 recall_floor 约束, 找出在 recall>=floor 前提下 F1 最优的阈值;
          若没有阈值能达到 recall_floor, 则退而求其次选"全局 recall 最高"的阈值。

    返回:
        avg_loss, acc@0.5, recall@0.5, f1@0.5, auc
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    # 损失函数
    if use_focal:
        criterion = FocalLossWithLogits(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction="mean",
        )
    else:
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
                reduction="mean",
            )
        else:
            criterion = nn.BCEWithLogitsLoss(reduction="mean")

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                x, _, _, _, seg_flag = batch
            elif len(batch) == 4:
                x, _, _, seg_flag = batch
            else:
                continue

            x = x.to(device)
            labels = seg_flag.to(device).view(-1)

            logits = model(x).view_as(labels)
            loss = criterion(logits, labels)

            bs = x.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.5

    avg_loss = total_loss / n_samples
    y_true = np.concatenate(all_labels).astype(int)
    y_prob = np.concatenate(all_probs)

    unique, counts = np.unique(y_true, return_counts=True)
    print(f"[SEG][DEBUG] val y_true 分布: {dict(zip(unique, counts))}")

    # 防止极端数值
    y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)

    # === (1) 固定阈值 0.5 下的指标(作为返回值使用) ===
    thr_default = 0.5
    y_pred = (y_prob >= thr_default).astype(int)
    acc = float((y_pred == y_true).mean())
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)

    # === (2) 无约束 best_F1(仅供参考) ===
    best_f1 = -1.0
    best_thr = thr_default
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred_t = (y_prob >= thr).astype(int)
        tp_t = float(((y_true == 1) & (y_pred_t == 1)).sum())
        fp_t = float(((y_true == 0) & (y_pred_t == 1)).sum())
        fn_t = float(((y_true == 1) & (y_pred_t == 0)).sum())

        prec_t = tp_t / (tp_t + fp_t + 1e-6)
        rec_t = tp_t / (tp_t + fn_t + 1e-6)
        f1_t = 2.0 * prec_t * rec_t / (prec_t + rec_t + 1e-6)

        if f1_t > best_f1:
            best_f1 = f1_t
            best_thr = thr

    # === (3) 带召回率安全线的 best_F1 阈值 ===
    best_floor = find_best_threshold_with_recall_floor(
        y_true=y_true,
        y_prob=y_prob,
        recall_floor=recall_floor,
    )

    print(
        f"[SEG][DEBUG] F1@0.50={f1:.4f}, Recall@0.50={recall:.4f}, "
        f"best_F1={best_f1:.4f} @thr={best_thr:.2f}; "
        f"recall_floor={recall_floor:.2f}, "
        f"thr_floor={best_floor['thr']:.2f}, "
        f"recall_floor_best={best_floor['recall']:.4f}, "
        f"F1_floor_best={best_floor['f1']:.4f}, "
        f"status={best_floor['status']}"
    )

    # === (4) AUC ===
    unique = np.unique(y_true)
    if len(unique) < 2:
        auc = 0.5
    else:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            print(f"[SEG][WARN] roc_auc_score 计算失败: {e}")
            auc = float("nan")

    # ★ 返回的仍然是 0.5 阈值下的各种指标
    return avg_loss, acc, recall, f1, auc


def classification_diagnostic_analysis(model,
                                       train_loader,
                                       val_loader,
                                       device,
                                       outdir=None,
                                       diagnostic_system=None,
                                       recall_floor: float = 0.8,
                                       n_bins: int = 10):
    """
    扩径二分类模型的全面诊断分析（适配当前 TimeSeriesBERDataset 的 batch 结构）

    - 收集 train / val 上的预测概率和真实标签
    - 计算多阈值下的 Accuracy / Precision / Recall / F1
    - 在验证集上做精细阈值扫描，找：
        * F1 最优阈值
        * Recall 最优阈值
        * 在 recall >= recall_floor 前提下 F1 最优阈值
    - 做概率校准分箱，评估模型置信度是否偏高/偏低
    - 统计正负样本的概率分布与分离度
    - 基于业务阈值(默认 0.5)输出混淆矩阵和 FP/FN 误差统计
    - 可选：把报告和日志保存到 outdir/seg_diag/ 下

    返回:
        diagnostic_report (dict), diagnostic_system (ModelDiagnosticSystem)
    """
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )
    import json
    import os

    model.eval()

    if diagnostic_system is None:
        diagnostic_system = ModelDiagnosticSystem("seg_classification")

    diagnostic_system.log("info", "CLASSIFICATION", "开始分类模型诊断")

    # ===== 1. 收集预测和标签（适配 4/5 元组 batch）=====
    def _collect_probs_and_labels(_loader):
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in _loader:
                # 适配 5 元组 / 4 元组两种返回形式
                if len(batch) == 5:
                    x, _, _, _, seg_flag = batch
                elif len(batch) == 4:
                    x, _, _, seg_flag = batch
                else:
                    # 未知结构，跳过
                    continue

                x = x.to(device)
                labels = seg_flag.to(device).view(-1)
                logits = model(x).view_as(labels)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if not all_labels:
            return None, None

        y_prob = np.concatenate(all_probs).astype(float)
        y_true = np.concatenate(all_labels).astype(int)
        # 稍微剪裁，防止数值太极端
        y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        return y_prob, y_true

    y_train_prob, y_train_true = _collect_probs_and_labels(train_loader)
    y_val_prob, y_val_true = _collect_probs_and_labels(val_loader)

    if y_train_true is None or y_val_true is None:
        diagnostic_system.log("warning", "CLASSIFICATION", "DataLoader 中没有有效样本，无法进行诊断")
        return None, diagnostic_system

    diagnostic_report = {
        "overall_metrics": {},
        "threshold_analysis": {},
        "calibration_analysis": {},
        "class_metrics": {},
        "confusion_analysis": {},
        "error_analysis": {},
    }

    # ===== 2. 多层次整体指标（train / val 各一份）=====
    thresholds = [0.3, 0.5, 0.7]

    for split_name, y_true, y_prob in [
        ("train", y_train_true, y_train_prob),
        ("val", y_val_true, y_val_prob),
    ]:
        split_metrics = {}
        n = len(y_true)
        pos_rate = float(y_true.mean())
        diagnostic_system.log(
            "info",
            f"CLASS_DISTR_{split_name.upper()}",
            f"{split_name} 样本数={n}, 正样本比例={pos_rate:.4f}",
        )

        # 固定几个典型阈值
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)

            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "pos_rate": float(y_pred.mean()),
            }
            split_metrics[f"threshold_{thr}"] = metrics

        # AUC / PR-AUC
        if len(np.unique(y_true)) > 1:
            try:
                split_metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                diagnostic_system.log(
                    "warning",
                    f"CLASS_METRIC_{split_name.upper()}",
                    f"计算 AUC 失败: {e}",
                )
                split_metrics["auc_roc"] = float("nan")

            try:
                split_metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
            except Exception as e:
                diagnostic_system.log(
                    "warning",
                    f"CLASS_METRIC_{split_name.upper()}",
                    f"计算 PR-AUC 失败: {e}",
                )
                split_metrics["auc_pr"] = float("nan")

        diagnostic_report["overall_metrics"][split_name] = split_metrics

        # 默认业务阈值 0.5 的指标作为基准
        base_thr = 0.5
        base_rec = split_metrics[f"threshold_{base_thr}"]["recall"]
        base_f1 = split_metrics[f"threshold_{base_thr}"]["f1"]
        diagnostic_system.log(
            "info",
            f"CLASS_BASE_{split_name.upper()}",
            f"thr={base_thr:.2f}, Recall={base_rec:.3f}, F1={base_f1:.3f}",
        )

    # ===== 3. 验证集上的精细阈值扫描 =====
    y_true = y_val_true
    y_prob = y_val_prob

    fine_thresholds = np.linspace(0.05, 0.95, 19)
    threshold_metrics = []

    best_by_f1 = None
    best_by_recall = None
    best_floor = None

    for thr in fine_thresholds:
        y_pred = (y_prob >= thr).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

        row = {
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        threshold_metrics.append(row)

        if (best_by_f1 is None) or (f1 > best_by_f1["f1"]):
            best_by_f1 = row
        if (best_by_recall is None) or (recall > best_by_recall["recall"]):
            best_by_recall = row
        if recall >= recall_floor:
            if (best_floor is None) or (f1 > best_floor["f1"]):
                best_floor = row

    diagnostic_report["threshold_analysis"] = {
        "thresholds": threshold_metrics,
        "best_by_f1": best_by_f1,
        "best_by_recall": best_by_recall,
        "best_with_recall_floor": best_floor,
        "recall_floor": float(recall_floor),
    }

    if best_floor is not None:
        diagnostic_system.log(
            "info",
            "CLASS_THRESH",
            f"在 recall >= {recall_floor:.2f} 条件下, 最佳 F1={best_floor['f1']:.3f} @thr={best_floor['threshold']:.2f}",
        )
    else:
        diagnostic_system.log(
            "warning",
            "CLASS_THRESH",
            f"没有任何阈值能够达到 recall_floor={recall_floor:.2f}",
        )

    # ===== 4. 概率校准分析 =====
    prob_bins = np.digitize(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
    calibration_data = []
    for bin_idx in range(1, n_bins + 1):
        mask = prob_bins == bin_idx
        if mask.sum() == 0:
            continue
        bin_true = y_true[mask]
        bin_prob = y_prob[mask]
        true_pos_rate = float(bin_true.mean())
        pred_prob_mean = float(bin_prob.mean())
        calibration_data.append({
            "bin": int(bin_idx),
            "true_positive_rate": true_pos_rate,
            "predicted_prob": pred_prob_mean,
            "calibration_error": float(pred_prob_mean - true_pos_rate),
            "n_samples": int(mask.sum()),
        })

    if calibration_data:
        ece = float(np.mean([abs(b["calibration_error"]) for b in calibration_data]))
    else:
        ece = float("nan")

    diagnostic_report["calibration_analysis"] = {
        "bins": calibration_data,
        "expected_calibration_error": ece,
    }

    diagnostic_system.log(
        "info",
        "CLASS_CALIB",
        f"预估校准误差(ECE)={ece:.4f} (越接近 0 越好)",
    )

    # ===== 5. 正负样本概率分布 & 分离度 =====
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]

    if len(pos_probs) == 0 or len(neg_probs) == 0:
        separation_score = float("nan")
    else:
        separation_score = float(
            abs(pos_probs.mean() - neg_probs.mean())
            / (pos_probs.std() + neg_probs.std() + 1e-8)
        )

    class_stats = {
        "positive_class": {
            "mean_prob": float(pos_probs.mean()) if len(pos_probs) > 0 else float("nan"),
            "std_prob": float(pos_probs.std()) if len(pos_probs) > 0 else float("nan"),
            "median_prob": float(np.median(pos_probs)) if len(pos_probs) > 0 else float("nan"),
            "q95_prob": float(np.percentile(pos_probs, 95)) if len(pos_probs) > 0 else float("nan"),
            "count": int(len(pos_probs)),
        },
        "negative_class": {
            "mean_prob": float(neg_probs.mean()) if len(neg_probs) > 0 else float("nan"),
            "std_prob": float(neg_probs.std()) if len(neg_probs) > 0 else float("nan"),
            "median_prob": float(np.median(neg_probs)) if len(neg_probs) > 0 else float("nan"),
            "q95_prob": float(np.percentile(neg_probs, 95)) if len(neg_probs) > 0 else float("nan"),
            "count": int(len(neg_probs)),
        },
        "separation_score": separation_score,
    }

    diagnostic_report["class_metrics"] = class_stats

    diagnostic_system.log(
        "info",
        "CLASS_SEPARATION",
        f"正负样本分离度={separation_score:.3f} "
        f"(pos_mean={class_stats['positive_class']['mean_prob']:.3f}, "
        f"neg_mean={class_stats['negative_class']['mean_prob']:.3f})",
    )

    # ===== 6. 混淆矩阵（默认 thr=0.5）=====
    best_thr = 0.5
    y_pred = (y_prob >= best_thr).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    confusion = {
        "threshold": best_thr,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "rates": {
            "true_positive_rate": tp / (tp + fn + 1e-8),
            "false_positive_rate": fp / (fp + tn + 1e-8),
            "precision": tp / (tp + fp + 1e-8),
            "negative_predictive_value": tn / (tn + fn + 1e-8),
        },
        "prevalence": float(y_true.mean()),
    }
    diagnostic_report["confusion_analysis"] = confusion

    diagnostic_system.log(
        "info",
        "CLASS_CONFUSION",
        f"thr={best_thr:.2f}, TPR={confusion['rates']['true_positive_rate']:.3f}, "
        f"FPR={confusion['rates']['false_positive_rate']:.3f}, "
        f"Precision={confusion['rates']['precision']:.3f}",
    )

    # ===== 7. 误差样本分析（val 集）=====
    errors = np.abs(y_pred - y_true)
    error_indices = np.where(errors == 1)[0]

    error_analysis = {
        "total_errors": int(errors.sum()),
        "error_rate": float(errors.mean()),
        "fp_count": int(fp),
        "fn_count": int(fn),
        "sample_errors": [],
    }

    max_samples = min(20, len(error_indices))
    for idx in error_indices[:max_samples]:
        true_label = int(y_true[idx])
        pred_label = int(y_pred[idx])
        prob = float(y_prob[idx])
        err_type = "FP" if (true_label == 0 and pred_label == 1) else "FN"
        confidence = prob if pred_label == 1 else 1.0 - prob

        error_analysis["sample_errors"].append({
            "index": int(idx),
            "true_label": true_label,
            "pred_label": pred_label,
            "probability": prob,
            "error_type": err_type,
            "confidence": float(confidence),
        })

    diagnostic_report["error_analysis"] = error_analysis

    if error_analysis["fp_count"] > 0:
        fp_conf = y_prob[(y_true == 0) & (y_pred == 1)]
        diagnostic_system.log(
            "warning",
            "CLASS_ERROR_FP",
            f"误报(FP) 数量={error_analysis['fp_count']}, 平均置信度={float(fp_conf.mean()):.3f}",
        )
    if error_analysis["fn_count"] > 0:
        fn_conf = 1.0 - y_prob[(y_true == 1) & (y_pred == 0)]
        diagnostic_system.log(
            "warning",
            "CLASS_ERROR_FN",
            f"漏报(FN) 数量={error_analysis['fn_count']}, 平均置信度={float(fn_conf.mean()):.3f}",
        )

    # ===== 8. 如指定 outdir, 保存报告到磁盘 =====
    if outdir is not None:
        diag_dir = os.path.join(outdir, "seg_diag")
        os.makedirs(diag_dir, exist_ok=True)

        report_path = os.path.join(diag_dir, "seg_classification_diag_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(diagnostic_report, f, ensure_ascii=False, indent=2)
        diagnostic_system.log("info", "CLASS_SAVE", f"诊断报告已保存到: {report_path}")

        try:
            diagnostic_system.save_logs(diag_dir)
        except Exception as e:
            diagnostic_system.log("warning", "CLASS_SAVE", f"保存诊断日志失败: {e}")

    diagnostic_system.log("info", "CLASSIFICATION", "分类模型诊断完成")
    return diagnostic_report, diagnostic_system


def run_seg_cls_training(train_loader,
                         val_loader,
                         feature_cols,
                         device,
                         outdir,
                         model_cfg):
    """
    使用当前的多尺度 LSTM 结构，训练一个"当前扩径段二分类"模型
    """
    input_dim = len(feature_cols)

    # 或者使用原来的回归模型，但只取第一步输出
    seg_model = EnhancedSegClassifier(
        input_dim=input_dim,
        hidden_dim=model_cfg["HIDDEN_SIZE"],
        num_layers=model_cfg["NUM_LAYERS"],
        bidirectional=model_cfg.get("BIDIRECTIONAL", True),
        dropout=model_cfg.get("DROPOUT", 0.0),
    ).to(device)

    lr = float(model_cfg.get("LR", 1e-3))
    optimizer = torch.optim.Adam(seg_model.parameters(), lr=lr)

    # 准备 pos_weight（仅在使用 BCEWithLogitsLoss 时有效）
    pos_weight = None
    seg_pos_weight = model_cfg.get("SEG_CLS_POS_WEIGHT", None)

    # ★ auto：用训练集 seg_flags 估计 neg/pos
    if (not model_cfg.get("SEG_CLS_USE_FOCAL", False)) and seg_pos_weight in ("auto", None):
        try:
            ds = train_loader.dataset
            if isinstance(ds, torch.utils.data.ConcatDataset):
                flags = []
                for sub in ds.datasets:
                    if hasattr(sub, "seg_flags"):
                        flags.append(np.asarray(sub.seg_flags).astype(int))
                y = np.concatenate(flags) if flags else None
            else:
                y = np.asarray(ds.seg_flags).astype(int) if hasattr(ds, "seg_flags") else None

            if y is not None and y.size > 0:
                pos = float((y == 1).sum())
                neg = float((y == 0).sum())
                auto_pw = neg / max(pos, 1.0)
                auto_pw = float(np.clip(auto_pw, 1.0, 20.0))
                pos_weight = torch.tensor([auto_pw], device=device)
                print(f"[SEG][INFO] auto pos_weight={auto_pw:.3f} (neg={neg:.0f}, pos={pos:.0f})")
        except Exception as e:
            print(f"[SEG][WARN] auto pos_weight 失败: {e}")

    elif (not model_cfg.get("SEG_CLS_USE_FOCAL", False)) and seg_pos_weight is not None:
        pos_weight = torch.tensor([float(seg_pos_weight)], device=device)

    use_focal = bool(model_cfg.get("SEG_CLS_USE_FOCAL", False))
    focal_gamma = float(model_cfg.get("SEG_CLS_FOCAL_GAMMA", 2.0))
    focal_alpha = model_cfg.get("SEG_CLS_FOCAL_ALPHA", None)
    if focal_alpha is not None:
        focal_alpha = float(focal_alpha)

    # ★ 以验证集 Recall@0.5 作为“最佳模型”选择标准
    best_recall = -1.0
    best_state = None
    epochs = int(model_cfg.get("SEG_CLS_EPOCHS", model_cfg.get("EPOCHS", 20)))

    print("[INFO] 开始训练扩径二分类模型（Seg-Now）...")
    hist = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_seg_cls(
            seg_model,
            train_loader,
            optimizer,
            device,
            pos_weight=pos_weight,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        # loss, acc, recall, f1, auc
        val_loss, val_acc, val_rec, val_f1, val_auc = eval_one_epoch_seg_cls(
            seg_model,
            val_loader,
            device,
            pos_weight=pos_weight,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            recall_floor=float(model_cfg.get("SEG_RECALL_FLOOR", 0.9)),
        )

        print(
            f"[SEG][E{epoch:02d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} "
            f"recall={val_rec:.4f} F1={val_f1:.4f} AUC={val_auc:.4f}"
        )
        hist.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_recall": float(val_rec),
            "val_f1": float(val_f1),
            "val_auc": float(val_auc),
        })

        # ★ 用 Recall 作为选择最优模型的指标
        if val_rec > best_recall:
            best_recall = val_rec
            best_state = {k: v.cpu().clone() for k, v in seg_model.state_dict().items()}

    # 保存最佳模型
    if best_state is not None and RUN_CFG.get("SAVE_MODEL", True):
        seg_model.load_state_dict(best_state)
        seg_path = os.path.join(outdir, "seg_classifier.pt")
        torch.save(seg_model.state_dict(), seg_path)
        print(f"[INFO] 最佳扩径分类模型已保存到: {seg_path} (best recall={best_recall:.4f})")
    # 保存训练曲线（分类）
    try:
        dfh = pd.DataFrame(hist)
        dfh.to_csv(os.path.join(outdir, "seg_now_history.csv"), index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dfh["epoch"], dfh["val_recall"], marker="o", label="val_recall")
        ax.plot(dfh["epoch"], dfh["val_f1"], marker="o", label="val_f1")
        ax.plot(dfh["epoch"], dfh["val_auc"], marker="o", label="val_auc")
        ax.set_xlabel("Epoch");
        ax.set_ylabel("Score")
        ax.set_title("Seg-Now metrics")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.15)
        fig.savefig(os.path.join(outdir, "seg_now_metrics.png"), dpi=600)
        plt.close(fig)
    except Exception as e:
        print(f"[SEG][WARN] 保存训练曲线失败: {e}")

    return seg_model


def apply_seg_classifier_on_df(df: pd.DataFrame,
                               seg_model: nn.Module,
                               feature_cols: List[str],
                               seg_model_cfg: dict,
                               device,
                               out_col: str = "SEG_PROB_NOW") -> pd.DataFrame:
    """
    使用训练好的 Seg-Now 分类模型，在整口井上生成“未来扩径风险概率”列。

    - df: 单口井的 DataFrame（已合并地层 + CAL + 特征工程 + 标准化）
    - seg_model: run_seg_cls_training 返回的 EnhancedSegClassifier
    - seg_model_cfg: 当时训练 Seg-Now 用的 config（至少要包含 SEQ_LEN / BATCH_SIZE / SEG_LABEL_COL）
    - out_col: 输出概率列名，建议与 MODEL_CFG["MAIN_CLS_PROB_COL"] 保持一致
    """
    df = df.copy()
    if seg_model is None:
        df[out_col] = np.nan
        return df

    seq_len = seg_model_cfg.get("SEQ_LEN", MODEL_CFG["SEQ_LEN"])
    batch_size = seg_model_cfg.get("BATCH_SIZE", 128)
    seg_label_col = seg_model_cfg.get("SEG_LABEL_COL", "SEG_FUTURE_LABEL")

    try:
        ds = TimeSeriesBERDataset(
            dfs=[df],
            feature_cols=feature_cols,
            seq_len=seq_len,
            target_col="BER",  # 这里随便用一个存在的回归列即可
            lookahead_steps=0,  # 纯分类，不做前看
            lookahead_only_in_seg=False,  # 全井范围都取样
            lookahead_margin=0,
            allow_relax=False,  # 如果构不出样本就直接报错
            seg_label_col=seg_label_col,
        )
    except Exception as e:
        print(f"[SEG][WARN] apply_seg_classifier_on_df 构造数据集失败: {e}")
        df[out_col] = np.nan
        return df

    # DataLoader：按原顺序遍历，不打乱
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # TimeSeriesBERDataset 内部：
    #   samples = [(well_index, end_row_index), ...]
    # 对于单井 dfs=[df] 的情况，well_index 恒为 0，end_row_index 就是 df 的行号
    anchor_idx = np.array([end for (wi, end) in ds.samples], dtype=int)
    prob_arr = np.full(len(df), np.nan, dtype=float)

    seg_model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            # 兼容 (x, y, mask, w, seg_flag) 结构，只取第一个 x
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            logits = seg_model(x).view(-1)  # [B]
            probs = torch.sigmoid(logits).cpu().numpy().astype(float)
            all_probs.append(probs)

    if not all_probs:
        print("[SEG][WARN] apply_seg_classifier_on_df: 未得到任何预测，全部置 NaN。")
        df[out_col] = np.nan
        return df

    all_probs = np.concatenate(all_probs)
    if all_probs.shape[0] != anchor_idx.shape[0]:
        print("[SEG][WARN] 样本数与 anchor_idx 不一致，尽量对齐赋值。")
        n = min(len(all_probs), len(anchor_idx))
        prob_arr[anchor_idx[:n]] = all_probs[:n]
    else:
        prob_arr[anchor_idx] = all_probs

    df[out_col] = prob_arr
    # 为避免前几行 / 中间 NaN 导致灯光断裂，做一次前向+后向填充
    df[out_col] = df[out_col].ffill().fillna(0.0)

    return df


# ===== [STAGE2 ANCHOR] apply_seg_level4_classifier_on_df =====
def apply_seg_level4_classifier_on_df(
        df: pd.DataFrame,
        seg4_model: nn.Module,
        feature_cols: List[str],
        seg4_seq_len: int,
        seg4_batch_size: int,
        device,
        out_pred_col: str = "SEG_LEVEL4_PRED",
        out_conf_col: str = "SEG_LEVEL4_CONF",
) -> pd.DataFrame:
    """
    用训练好的 4 分类模型输出：
      - out_pred_col: 0/1/2/3
      - out_conf_col: max softmax prob
    注意：这里不做 gate，gate 在 assign_traffic_lights 里做“严格门控”。
    """
    df = df.copy()
    if seg4_model is None:
        df[out_pred_col] = np.nan
        df[out_conf_col] = np.nan
        return df

    # 用一个“永远存在”的 target_col 占位，避免 dataset 因 label 缺失而跳样本
    target_col = "BER" if "BER" in df.columns else feature_cols[0]

    ds = TimeSeriesBERDataset(
        dfs=[df],
        feature_cols=feature_cols,
        seq_len=seg4_seq_len,
        target_col=target_col,
        lookahead_steps=0,
        lookahead_only_in_seg=False,
        lookahead_margin=0,
        allow_relax=False,
        seg_label_col="SEG_FUTURE_LABEL",  # 占位，不影响输出
    )

    loader = DataLoader(ds, batch_size=seg4_batch_size, shuffle=False)
    anchor_idx = np.array([end for (wi, end) in ds.samples], dtype=int)

    pred_arr = np.full(len(df), np.nan, dtype=float)
    conf_arr = np.full(len(df), np.nan, dtype=float)

    seg4_model.eval()
    all_pred, all_conf = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            logits = seg4_model(x)  # [B,4]
            prob = torch.softmax(logits, dim=1)
            conf, pred = torch.max(prob, dim=1)
            all_pred.append(pred.cpu().numpy().astype(float))
            all_conf.append(conf.cpu().numpy().astype(float))

    if all_pred:
        all_pred = np.concatenate(all_pred)
        all_conf = np.concatenate(all_conf)

        n = min(len(anchor_idx), len(all_pred))
        pred_arr[anchor_idx[:n]] = all_pred[:n]
        conf_arr[anchor_idx[:n]] = all_conf[:n]

    # 前几行没有足够窗口会是 NaN：用 0 兜底（不影响 gate==0 时强制绿灯）
    df[out_pred_col] = pd.Series(pred_arr).fillna(0).astype(int).values
    df[out_conf_col] = pd.Series(conf_arr).fillna(0.0).values

    return df


# ===== [STAGE2 ANCHOR END] =====

def run_seg_level4_training(train_dfs,
                            feature_cols,
                            device,
                            outdir,
                            model_cfg):
    """
    使用 ExpansionLevelClassifier 训练“扩径严重度 0/1/2/3”四分类模型
    —— 标签来源：SEG_LEVEL4_LABEL_CORE_ONLY
    """
    print("[INFO] 构建扩径严重度 4 分类 DataLoader...")
    train_loader, val_loader = build_seg_level4_dataloaders(
        train_dfs, feature_cols, model_cfg
    )

    input_dim = len(feature_cols)
    num_classes = 4

    seg4_model = ExpansionLevelClassifier(
        input_dim=input_dim,
        hidden_dim=model_cfg["HIDDEN_SIZE"],
        num_layers=model_cfg["NUM_LAYERS"],
        bidirectional=model_cfg.get("BIDIRECTIONAL", True),
        dropout=model_cfg.get("DROPOUT", 0.3),
        num_classes=num_classes,
    ).to(device)

    lr = float(model_cfg.get("SEG_LEVEL4_LR", model_cfg.get("LR", 1e-3)))
    optimizer = torch.optim.Adam(seg4_model.parameters(), lr=lr)

    # 类别权重（可选），例如 [1.0, 2.0, 3.0, 4.0] 对严重扩径加大权重
    class_weights = model_cfg.get("SEG_LEVEL4_CLASS_WEIGHTS", None)
    if class_weights is not None:
        class_weights = torch.tensor(
            [float(w) for w in class_weights],
            dtype=torch.float32,
            device=device,
        )
    else:
        class_weights = None

    epochs = int(model_cfg.get("SEG_LEVEL4_EPOCHS", 10))
    best_macro_f1 = -1.0
    best_state = None

    print("[INFO] 开始训练扩径严重度 4 分类模型...")
    hist4 = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1, train_rec = train_one_epoch_seg_level4(
            seg4_model,
            train_loader,
            optimizer,
            device,
            num_classes=num_classes,
            class_weight=class_weights,
        )
        val_loss, val_acc, val_f1, val_rec = eval_one_epoch_seg_level4(
            seg4_model,
            val_loader,
            device,
            num_classes=num_classes,
            class_weight=class_weights,
        )

        print(
            f"[SEG4][E{epoch:02d}] "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} F1={train_f1:.4f} "
            f"| val_loss={val_loss:.4f} acc={val_acc:.4f} F1={val_f1:.4f} "
            f"| val_recall={['%.2f' % r for r in val_rec]}"
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in seg4_model.state_dict().items()}

    if best_state is not None and RUN_CFG.get("SAVE_MODEL", True):
        seg4_model.load_state_dict(best_state)
        seg4_path = os.path.join(outdir, "seg_level4_classifier.pt")
        torch.save(seg4_model.state_dict(), seg4_path)
        print(f"[INFO] 最佳扩径严重度 4 分类模型已保存到: {seg4_path} (best macro F1={best_macro_f1:.4f})")

    return seg4_model


from torch.utils.data import DataLoader, WeightedRandomSampler


def rebuild_segments_for_split(df: pd.DataFrame,
                               time_col: str | None = None) -> pd.DataFrame:
    """
    为避免 train/val 之间的“跨边界扩径段 / 未来窗口”带来信息泄露，
    在按时间切分之后，对每一个子 DataFrame(如 df_train / df_val) 单独：
      1) 基于 BER_SMOOTH + 深度 重新做一次扩径段检测；
      2) 在该子集内部重算 SEG_FUTURE_LABEL / BER_PRE_ALERT / SEG_NOW_LABEL；
      3) 如果已经实现了 add_seg_level4_label，则补上 4 档严重度标签。

    注意：这里不再使用整口井的 BER_SEG_ID / SEG_FUTURE_LABEL，
    每个子集的段号 / 未来标签都是“局部定义”的。
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # 时间列 & 深度列
    if time_col is None:
        time_col = CLEAN_CFG.get("TIME_COL", "WELLDATETIME")
    dep_col = _choose_depth_column(df, CLEAN_CFG["DEP_COL_CANDIDATES"])

    if "BER_SMOOTH" not in df.columns:
        # 没有 BER_SMOOTH（通常意味着没有 CAL），就不强行构造，直接返回
        return df

    # --- 1) 基于当前子集的 BER_SMOOTH 重新识别扩径段（深度域） ---
    df = build_ber_segments_simple(
        df,
        depth_col=dep_col,
        ber_col="BER_SMOOTH",
        abs_thresh=SEG_CFG.get("START_THRESH", 5.0),
        min_seg_len=SEG_CFG.get("MIN_LEN_M", 0.30),
    )

    # 统一 BER 使用平滑后
    if "BER" in df.columns:
        df["BER_RAW"] = df["BER"]
    else:
        df["BER_RAW"] = df["BER_SMOOTH"]
    df["BER"] = df["BER_SMOOTH"]

    # --- 2) 在该子集内部构造“未来扩径风险”标签 ---
    lookahead_steps = int(MODEL_CFG.get("LOOKAHEAD_STEPS", 60))
    future_steps_cfg = SEG_CFG.get("SEG_FUTURE_STEPS", "auto")

    if future_steps_cfg in (None, "auto"):
        future_steps = lookahead_steps
    else:
        future_steps = int(future_steps_cfg)
        if future_steps != lookahead_steps:
            print(
                f"[SEG][WARN] SEG_FUTURE_STEPS={future_steps} 与 LOOKAHEAD_STEPS={lookahead_steps} 不一致，已强制对齐。")
            future_steps = lookahead_steps

    df = build_future_risk_label(
        df,
        seg_col="BER_IS_SEG",
        out_col="SEG_FUTURE_LABEL",
        horizon_steps=future_steps,
    )

    # --- 3) 时间 pre-alert：扩径开始前 PRE_ALERT_LEN_MIN 分钟 ---
    pre_min = SEG_CFG.get("PRE_ALERT_LEN_MIN", 0.0)
    if pre_min is not None and pre_min > 0:
        df = add_time_pre_alert(
            df,
            time_col=time_col,
            seg_id_col="BER_SEG_ID",
            pre_alert_minutes=pre_min,
        )
    else:
        df["BER_PRE_ALERT"] = 0

    # --- 4) 当前扩径核心区标签（Seg-Now） ---
    df = build_seg_now_labels(df, seg_id_col="BER_SEG_ID")

    # --- 5) 如果你已经实现了“段级 4 档严重度标签”，在这里补上 ---
    if "add_seg_level4_label" in globals():
        try:
            df = add_seg_level4_label(
                df,
                seg_id_col="BER_SEG_ID",
                seg_level_col="BER_SEG_LEVEL",  # 你在新版扩径检测里定义的段级阈值列
                core_col="SEG_NOW_LABEL",
                out_col="SEG_LEVEL4_LABEL",
            )
        except Exception as e:
            print(f"[WARN] rebuild_segments_for_split: add_seg_level4_label 失败: {e}")

    # 最后再按时间排回时间顺序，保证后续 LSTM 按时间滚动
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    return df


def build_dataloaders(train_dfs, feature_cols, model_cfg):
    """
    构造训练 / 验证 DataLoader（支持时间前看）：
    """
    seq_len = model_cfg["SEQ_LEN"]
    val_ratio = float(model_cfg.get("VAL_SPLIT", 0.2))
    lookahead_steps = model_cfg.get("LOOKAHEAD_STEPS", 0)
    lookahead_only_in_seg = model_cfg.get("LOOKAHEAD_ONLY_IN_SEG", False)
    lookahead_margin = model_cfg.get("LOOKAHEAD_MARGIN", 0)
    seg_label_col = model_cfg.get("SEG_LABEL_COL", "BER_IS_SEG")

    train_sets = []
    val_sets = []

    for df in train_dfs:
        # 1) 保证按时间排序
        time_col = CLEAN_CFG.get("TIME_COL", "WELLDATETIME")
        if time_col not in df.columns:
            raise ValueError(f"build_dataloaders: 找不到时间列 {time_col}")

        df = df.sort_values(time_col).reset_index(drop=True)

        # 2) 预先保证 BER/特征无 NaN
        df = df.copy()
        df["BER"] = df["BER"].astype(float)
        df = df[~df["BER"].isna()]

        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        # ★ 只用 ffill，避免用未来值回填过去
        df[feature_cols] = df[feature_cols].ffill()
        # 仍然存在的 NaN（例如整段都是空）统一兜底为 0，后面的 DataSet 会再次过滤含 NaN 的窗口
        df[feature_cols] = df[feature_cols].fillna(0.0)

        # 3) 按时间切分 train/val（切分之后，再分别在子集上做一次扩径识别，防止信息泄露）
        n_rows = len(df)
        if n_rows <= (seq_len + lookahead_steps) * 2:
            # 太短的井，全部作为训练，验证靠别的井
            df_all = rebuild_segments_for_split(df, time_col=time_col)

            ds_all = TimeSeriesBERDataset(
                dfs=[df_all],
                feature_cols=feature_cols,
                seq_len=seq_len,
                target_col="BER",
                lookahead_steps=lookahead_steps,
                lookahead_only_in_seg=lookahead_only_in_seg,
                lookahead_margin=lookahead_margin,
                seg_label_col=seg_label_col,
            )

            train_sets.append(ds_all)
            continue

        split_row = int(n_rows * (1.0 - val_ratio))
        df_train = df.iloc[:split_row].reset_index(drop=True)
        df_val = df.iloc[split_row:].reset_index(drop=True)

        # ★★ 关键：在 train / val 子集上分别重算扩径段和未来风险标签，互不“看到”对方的数据
        df_train = rebuild_segments_for_split(df_train, time_col=time_col)
        df_val = rebuild_segments_for_split(df_val, time_col=time_col)

        ds_train = TimeSeriesBERDataset(
            dfs=[df_train],
            feature_cols=feature_cols,
            seq_len=seq_len,
            target_col="BER",
            lookahead_steps=lookahead_steps,
            lookahead_only_in_seg=lookahead_only_in_seg,
            lookahead_margin=lookahead_margin,
            seg_label_col=seg_label_col,
        )
        ds_val = TimeSeriesBERDataset(
            dfs=[df_val],
            feature_cols=feature_cols,
            seq_len=seq_len,
            target_col="BER",
            lookahead_steps=lookahead_steps,
            lookahead_only_in_seg=lookahead_only_in_seg,
            lookahead_margin=lookahead_margin,
            seg_label_col=seg_label_col,
        )

        train_sets.append(ds_train)
        val_sets.append(ds_val)

    # 把多口井的数据集合并成一个大 Dataset
    if len(train_sets) == 1:
        train_ds = train_sets[0]
    else:
        train_ds = torch.utils.data.ConcatDataset(train_sets)

    if len(val_sets) == 0:
        # 极端情况：所有井都太短，退化成用部分训练样本做验证
        full_ds = train_ds
        n_samples = len(full_ds)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split = int(n_samples * (1.0 - val_ratio))
        train_idx = indices[:split]
        val_idx = indices[split:]
        train_ds = torch.utils.data.Subset(full_ds, train_idx)
        val_ds = torch.utils.data.Subset(full_ds, val_idx)
    elif len(val_sets) == 1:
        val_ds = val_sets[0]
    else:
        val_ds = torch.utils.data.ConcatDataset(val_sets)

    # ====== 训练集：构造采样权重，实现“扩径窗口过采样” ======
    def _gather_seg_flags(ds):
        if isinstance(ds, TimeSeriesBERDataset):
            return np.array(ds.seg_flags, dtype=float)
        elif isinstance(ds, torch.utils.data.ConcatDataset):
            flags = []
            for sub in ds.datasets:
                flags.append(_gather_seg_flags(sub))
            return np.concatenate(flags)
        else:
            raise TypeError(f"不支持的数据集类型: {type(ds)}")

    try:
        seg_flags_train = _gather_seg_flags(train_ds)
    except Exception as e:
        print(f"[WARN] 提取 seg_flags 失败，将不使用 WeightedRandomSampler: {e}")
        seg_flags_train = None

    if seg_flags_train is not None:
        over_factor = float(model_cfg.get("OVERSAMPLE_FACTOR", SEG_CFG.get("OVERSAMPLE_FACTOR", 1.0)))

        sample_weights = np.ones_like(seg_flags_train, dtype=float)
        sample_weights[seg_flags_train > 0.5] *= over_factor

        if over_factor > 1.0:
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(seg_flags_train),
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=model_cfg["BATCH_SIZE"],
                sampler=train_sampler,
                shuffle=False,
                drop_last=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                worker_init_fn=seed_worker,  # ★ 固定每个 worker 的随机种子
            )

        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=model_cfg["BATCH_SIZE"],
                shuffle=True,
                drop_last=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                worker_init_fn=seed_worker,  # ★ 固定每个 worker 的随机种子
            )

    # ====== 验证集：保持原始分布，不做过采样 ======
    val_loader = DataLoader(
        val_ds,
        batch_size=model_cfg["BATCH_SIZE"],
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,  # ★ 固定每个 worker 的随机种子
    )

    return train_loader, val_loader


def build_seg_level4_dataloaders(train_dfs, feature_cols, model_cfg):
    """
    构造扩径严重度 4 分类的 DataLoader：
    - 只用 SEG_LEVEL4_LABEL_CORE_ONLY >= 0 的点作为样本（前面 Dataset 已经过滤）
    - LOOKAHEAD_STEPS = 0，纯分类，不做时间前看
    """
    seq_len = model_cfg.get("SEG_LEVEL4_SEQ_LEN", model_cfg["SEQ_LEN"])
    val_ratio = float(model_cfg.get("VAL_SPLIT", 0.2))

    train_sets = []
    val_sets = []

    time_col = CLEAN_CFG.get("TIME_COL", "WELLDATETIME")
    target_col = "SEG_LEVEL4_LABEL_CORE_ONLY"

    for df in train_dfs:
        if time_col not in df.columns:
            raise ValueError(f"build_seg_level4_dataloaders: 找不到时间列 {time_col}")

        df = df.sort_values(time_col).reset_index(drop=True)

        if target_col not in df.columns:
            print(f"[WARN] build_seg_level4_dataloaders: df 中缺少 {target_col}，跳过该井。")
            continue

        # 特征补 NaN
        df = df.copy()
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

        n_rows = len(df)
        if n_rows < seq_len + 1:
            continue

        split_row = int(n_rows * (1.0 - val_ratio))
        df_train = df.iloc[:split_row].reset_index(drop=True)
        df_val = df.iloc[split_row:].reset_index(drop=True)

        # 关键：切分后在各自子集内重算扩径段/未来标签/四档严重度，避免跨边界泄露
        df_train = rebuild_segments_for_split(df_train, time_col=time_col)
        df_val = rebuild_segments_for_split(df_val, time_col=time_col)

        # target_col 可能是重算后才出现：所以这里再检查
        if target_col not in df_train.columns or target_col not in df_val.columns:
            print(f"[WARN] build_seg_level4_dataloaders: 重建后仍缺少 {target_col}，跳过该井。")
            continue

        # 子集内补 NaN：只允许 ffill（过去值），禁止 bfill（未来值）
        df_train[feature_cols] = df_train[feature_cols].ffill().fillna(0.0)
        df_val[feature_cols] = df_val[feature_cols].ffill().fillna(0.0)

        ds_train = TimeSeriesBERDataset(
            dfs=[df_train],
            feature_cols=feature_cols,
            seq_len=seq_len,
            target_col=target_col,
            lookahead_steps=0,
            lookahead_only_in_seg=False,
            lookahead_margin=0,
            seg_label_col=None,  # 不用 seg_flags 做采样控制
        )
        ds_val = TimeSeriesBERDataset(
            dfs=[df_val],
            feature_cols=feature_cols,
            seq_len=seq_len,
            target_col=target_col,
            lookahead_steps=0,
            lookahead_only_in_seg=False,
            lookahead_margin=0,
            seg_label_col=None,
        )

        train_sets.append(ds_train)
        val_sets.append(ds_val)

    if not train_sets:
        raise ValueError("build_seg_level4_dataloaders: 没有任何井可用于 4 分类训练，请检查数据。")

    if len(train_sets) == 1:
        train_ds = train_sets[0]
        val_ds = val_sets[0]
    else:
        train_ds = torch.utils.data.ConcatDataset(train_sets)
        val_ds = torch.utils.data.ConcatDataset(val_sets)

    batch_size = model_cfg.get("SEG_LEVEL4_BATCH_SIZE", 128)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader





# =========================
# 水化时间衰减特征（供 OPC 使用）
# =========================

def add_hydration_time_decay_feature(df: pd.DataFrame,
                                     time_col: str,
                                     depth_col: str,
                                     clean_cfg: dict = CLEAN_CFG,
                                     model_cfg: dict = MODEL_CFG) -> pd.DataFrame:
    """
    钻井液水化时间效应：
    - 当钻头深度几乎不变（|dDEP| < HYDRATION_STOP_DDEP_THRESH）时，认为该深度附近在被水化，
      暴露时间 exposure_sec 按 dt_sec 线性累积；
    - 当深度在变化（钻进 / 起下钻）时，水化暴露按时间常数 tau_decay 指数衰减；
    - 暴露越久，风险 HYDRO_RISK = 1 - exp(-exposure_sec / tau_acc)，在 (0,1) 区间内饱和。

    输出列：
    - HYDRO_TIME_MIN：当前时刻累计的水化暴露时间（分钟）
    - HYDRO_RISK：0~1 的水化风险（用于 OPC 特征）
    """
    df = df.copy()

    # -------- 读取配置 --------
    max_dt_sec = clean_cfg.get("MAX_DT_SEC", 5 * 60)
    stop_ddep_thresh = float(model_cfg.get("HYDRATION_STOP_DDEP_THRESH", 0.5))
    tau_acc_min = float(model_cfg.get("HYDRATION_ACC_TAU_MIN", 60.0))
    tau_decay_min = float(model_cfg.get("HYDRATION_DECAY_TAU_MIN", 30.0))
    max_exposure_min = float(model_cfg.get("HYDRATION_MAX_MIN", 360.0))

    # -------- 按时间排序，保证时间单调 --------
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        raise ValueError(f"add_hydration_time_decay_feature: 缺少时间列 {time_col}")

    # -------- 时间间隔 dt_sec --------
    if "dt_sec" in df.columns:
        dt_sec = df["dt_sec"].to_numpy(dtype=float)
    else:
        dt = df[time_col].diff().dt.total_seconds()
        dt = dt.fillna(0.0)
        dt_sec = dt.to_numpy(dtype=float)

    # -------- 深度差 dDEP --------
    if depth_col not in df.columns:
        raise ValueError(f"add_hydration_time_decay_feature: 缺少深度列 {depth_col}")
    ddep = df[depth_col].astype(float).diff()
    ddep = ddep.fillna(0.0)
    ddep = ddep.to_numpy(dtype=float)

    # 视为“停在同一深度”的标志
    static_mask = np.abs(ddep) < stop_ddep_thresh

    # 极端大的 dt 当作断点，统一截断
    dt_sec = np.where((dt_sec < 0) | (dt_sec > max_dt_sec * 10),
                      max_dt_sec, dt_sec)

    tau_acc = max(tau_acc_min * 60.0, 1e-6)  # [s]
    tau_decay = max(tau_decay_min * 60.0, 1e-6)  # [s]
    max_exp_sec = max_exposure_min * 60.0  # [s]

    n = len(df)
    exposure = np.zeros(n, dtype=float)

    # 若存在 FLAG_TIME_GAP，大时间断点直接重置暴露
    if "FLAG_TIME_GAP" in df.columns:
        flag_gap = df["FLAG_TIME_GAP"].to_numpy()
    else:
        flag_gap = np.zeros(n, dtype=int)

    for i in range(1, n):
        if flag_gap[i] == 1:
            # 认为是新一段数据，水化暴露重置
            exposure[i] = 0.0
            continue

        if static_mask[i]:
            # 钻头几乎不动 → 水化暴露线性累积（上限 max_exp_sec）
            exposure[i] = min(exposure[i - 1] + dt_sec[i], max_exp_sec)
        else:
            # 在动 → 暴露按 tau_decay 指数衰减
            exposure[i] = exposure[i - 1] * math.exp(-dt_sec[i] / tau_decay)

    # 水化风险：0~1，暴露越久越接近 1
    risk = 1.0 - np.exp(-exposure / tau_acc)

    df["HYDRO_TIME_MIN"] = exposure / 60.0
    df["HYDRO_RISK"] = risk

    return df


# =========================
# OPC 分支 + PCA 解释
# =========================

def train_opc_model(train_dfs: List[pd.DataFrame],
                    feature_cols: List[str],
                    ber_col: str = "BER",
                    ber_thresh: float = 5.0,
                    val_ratio: float | None = None,
                    time_col: str | None = None) -> Tuple[RandomForestClassifier, List[str]]:
    """
    训练 OPC 风险分类器（防泄露 + 水化效应版）。

    1) 对每口井：
       - 按时间切成 train/val（同主 LSTM 使用 VAL_SPLIT）；
       - 在各自子集上调用 rebuild_segments_for_split，单独重算 BER_SEG_ID / BER_IS_SEG 等；
       - 在训练子集上调用 add_hydration_time_decay_feature 计算 HYDRO_TIME_MIN / HYDRO_RISK；
       - OPC 训练只用“训练子集”的样本。
    2) 标签优先来自 BER_IS_SEG，退化时退回点级 BER 阈值。
    3) 特征会在原来的 feature_cols 基础上自动追加 ["HYDRO_TIME_MIN","HYDRO_RISK"]
       （如果启用了水化且这两列存在）。
    """

    if val_ratio is None:
        val_ratio = float(MODEL_CFG.get("VAL_SPLIT", 0.2))
    if time_col is None:
        time_col = CLEAN_CFG.get("TIME_COL", "WELLDATETIME")
    seq_len = MODEL_CFG.get("SEQ_LEN", 200)

    opc_train_list: List[pd.DataFrame] = []
    opc_val_list: List[pd.DataFrame] = []

    # ===== 1) 按井切分 + 重算扩径段 + 水化特征 =====
    for df in train_dfs:
        if df is None or df.empty:
            continue

        df_tmp = df.copy()
        if time_col in df_tmp.columns:
            df_tmp = df_tmp.sort_values(time_col).reset_index(drop=True)

        n_rows = len(df_tmp)
        if n_rows == 0:
            continue

        # 太短的井：全部作为训练段
        if n_rows <= seq_len * 2:
            df_train = df_tmp
            df_val = df_tmp.iloc[0:0].copy()
        else:
            split_row = int(n_rows * (1.0 - val_ratio))
            df_train = df_tmp.iloc[:split_row].reset_index(drop=True)
            df_val = df_tmp.iloc[split_row:].reset_index(drop=True)

        # 在 train/val 子集上分别重算扩径段，防止段跨越 split
        try:
            df_train = rebuild_segments_for_split(df_train, time_col=time_col)
        except Exception as e:
            print(f"[WARN][OPC] rebuild_segments_for_split(train) 失败，使用原始扩径段: {e}")
        try:
            df_val = rebuild_segments_for_split(df_val, time_col=time_col)
        except Exception as e:
            print(f"[WARN][OPC] rebuild_segments_for_split(val) 失败，使用原始扩径段: {e}")

        # 在训练子集上增加水化时间衰减特征（可选）
        if MODEL_CFG.get("HYDRATION_ENABLE_FOR_OPC", True):
            try:
                dep_col_train = _choose_depth_column(df_train, CLEAN_CFG["DEP_COL_CANDIDATES"])
                df_train = add_hydration_time_decay_feature(
                    df_train,
                    time_col=time_col,
                    depth_col=dep_col_train,
                    clean_cfg=CLEAN_CFG,
                    model_cfg=MODEL_CFG,
                )
                # val 上也算一份，方便将来若要单独评估 OPC 在验证段的表现
                if len(df_val) > 0:
                    dep_col_val = _choose_depth_column(df_val, CLEAN_CFG["DEP_COL_CANDIDATES"])
                    df_val = add_hydration_time_decay_feature(
                        df_val,
                        time_col=time_col,
                        depth_col=dep_col_val,
                        clean_cfg=CLEAN_CFG,
                        model_cfg=MODEL_CFG,
                    )
            except Exception as e:
                print(f"[WARN][OPC] add_hydration_time_decay_feature 失败: {e}")

        opc_train_list.append(df_train)
        opc_val_list.append(df_val)

    if not opc_train_list:
        raise ValueError("train_opc_model: 没有任何可用的 OPC 训练样本，请检查 train_dfs / CAL / BER。")

    # 合并所有训练井的“训练段”
    df_all = pd.concat(opc_train_list, axis=0, ignore_index=True)
    df_all = df_all.copy()
    df_all = df_all[~df_all[ber_col].isna()]

    # ===== 2) 构造 OPC_LABEL =====
    if "BER_IS_SEG" in df_all.columns:
        df_all["OPC_LABEL"] = df_all["BER_IS_SEG"].astype(int)
        print("[OPC] 使用 BER_IS_SEG 作为 OPC 标签（连续扩径段，已按 train 子集单独识别）。")

        num_pos_seg = int(df_all["OPC_LABEL"].sum())
        num_all_seg = len(df_all)
        if num_pos_seg == 0 or num_pos_seg == num_all_seg:
            print("[OPC][WARN] BER_IS_SEG 标签全为同一类别，退回点级 BER 阈值。")
            df_all["OPC_LABEL"] = (df_all[ber_col] >= ber_thresh).astype(int)
            print(f"[OPC] 使用点级 BER 阈值作为 OPC 标签，ber_thresh={ber_thresh}。")
    else:
        df_all["OPC_LABEL"] = (df_all[ber_col] >= ber_thresh).astype(int)
        print(f"[OPC] 使用点级 BER 阈值作为 OPC 标签，ber_thresh={ber_thresh}。")

    # ===== 3) 打印最终正负样本数量 =====
    num_pos = int(df_all["OPC_LABEL"].sum())
    num_neg = int((df_all["OPC_LABEL"] == 0).sum())
    print(f"[OPC] 正样本数={num_pos}, 负样本数={num_neg}")
    if num_pos == 0 or num_neg == 0:
        print("[WARN][OPC] OPC_LABEL 只有一个类别，RandomForest 训练可能失败，请检查阈值或扩径段参数。")

    # ===== 4) 组装特征：原有 OPC 特征 + 水化特征 =====
    feature_cols_extended = list(feature_cols)
    if MODEL_CFG.get("HYDRATION_ENABLE_FOR_OPC", True):
        for c in ["HYDRO_TIME_MIN", "HYDRO_RISK"]:
            if c in df_all.columns and c not in feature_cols_extended:
                feature_cols_extended.append(c)
        print(
            f"[OPC] 已在 OPC 特征中加入水化特征: {[c for c in ['HYDRO_TIME_MIN', 'HYDRO_RISK'] if c in feature_cols_extended]}")

    X = df_all[feature_cols_extended].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().fillna(0.0)
    y = df_all["OPC_LABEL"].values

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X.values, y)

    # 返回“扩展后的特征列表”，后面 apply_opc_model 时要用同一套
    return clf, feature_cols_extended


def apply_opc_model(df: pd.DataFrame,
                    clf: RandomForestClassifier,
                    feature_cols: List[str]) -> pd.DataFrame:
    """
    使用已训练的 OPC 模型，对整口井计算风险概率 OPC_PROB。

    增强：
    - 若启用 HYDRATION_ENABLE_FOR_OPC，则在推理前自动为该井计算
      HYDRO_TIME_MIN / HYDRO_RISK 两个水化特征（与训练侧一致）；
    - 根据 clf.classes_ 自适应选择“正类”列；
    - 当模型只学到一个类别时，给出合理的概率（全 0 或全 1），避免 IndexError。
    """
    df = df.copy()

    # 先补水化特征（如果启用）
    if MODEL_CFG.get("HYDRATION_ENABLE_FOR_OPC", True):
        try:
            time_col = CLEAN_CFG.get("TIME_COL", "WELLDATETIME")
            dep_col = _choose_depth_column(df, CLEAN_CFG["DEP_COL_CANDIDATES"])
            df = add_hydration_time_decay_feature(
                df,
                time_col=time_col,
                depth_col=dep_col,
                clean_cfg=CLEAN_CFG,
                model_cfg=MODEL_CFG,
            )
        except Exception as e:
            print(f"[WARN] apply_opc_model: 计算水化特征失败，将不使用水化特征: {e}")

    # 按训练时返回的 feature_cols 顺序取特征
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().fillna(0.0)

    proba = clf.predict_proba(X.values)  # [N, n_classes]
    classes = getattr(clf, "classes_", None)

    # 安全性检查：决定“正类”概率
    if proba.ndim != 2 or proba.shape[1] == 0:
        # 非常异常的情况，直接给 0 风险
        prob = np.zeros(len(X), dtype=float)
    else:
        n_classes = proba.shape[1]

        if classes is not None and len(classes) == n_classes:
            classes = list(classes)

            if n_classes == 1:
                # 只学到一个类别
                if classes[0] == 1:
                    # 模型认为所有样本都是“正类”，概率恒为 1
                    prob = proba[:, 0]
                else:
                    # 只有负类，全部视为 0 风险
                    prob = np.zeros(len(X), dtype=float)
            else:
                # n_classes >= 2，优先找标签=1 作为“正类”
                if 1 in classes:
                    idx_pos = classes.index(1)
                else:
                    # 没有显式的标签 1，就默认取“类别值最大的那个”作为正类
                    idx_pos = int(np.argmax(classes))
                prob = proba[:, idx_pos]
        else:
            # 没有可靠的 classes_ 信息，只能退化处理
            if n_classes == 1:
                prob = np.zeros(len(X), dtype=float)
            else:
                # 默认取最后一列当正类
                prob = proba[:, -1]

    df["OPC_PROB"] = prob
    return df


def fit_pca_on_dynamic(train_dfs: List[pd.DataFrame],
                       dynamic_cols: List[str],
                       n_components: int = 5):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    X_list = []
    for df in train_dfs:
        X = df[dynamic_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().fillna(0.0)
        X_list.append(X.values)
    X_all = np.vstack(X_list)
    scaler.fit(X_all)
    X_scaled = scaler.transform(X_all)
    pca.fit(X_scaled)
    return scaler, pca


def explain_risk_point(df: pd.DataFrame,
                       idx: int,
                       dynamic_cols: List[str],
                       scaler: StandardScaler,
                       pca: PCA,
                       top_k: int = 5):
    """
    对某个时间点（红/黄灯）做 PCA 解释，返回主导动态参数
    """
    row = df.loc[idx, dynamic_cols].values.reshape(1, -1)
    row = np.nan_to_num(row)
    row_scaled = scaler.transform(row)
    scores = pca.transform(row_scaled)[0]  # 每个主成分得分
    loadings = pca.components_  # [PC, feature]

    contrib = np.zeros(len(dynamic_cols))
    for j in range(len(scores)):
        contrib += abs(scores[j]) * np.abs(loadings[j, :])
    contrib = contrib / (contrib.sum() + 1e-9)

    top_idx = np.argsort(contrib)[::-1][:top_k]
    results = [(dynamic_cols[i], float(contrib[i])) for i in top_idx]
    return results


def analyze_risk_segments_and_suggest_controls(
        df: pd.DataFrame,
        dynamic_cols: List[str],
        outdir: str,
        name: str,
        cfg: dict = None,
        light_col: str = "LIGHT_FUSED",
        depth_col: str = None,
        min_seg_len: int = 5,
) -> pd.DataFrame:
    """
    对红灯/黄灯区域内的 WOB/RPM/TOR/SPP/FLOWIN/FLOWOUT/HKLD 等参数进行统计分析，
    并基于“绿灯段的正常分布”给出一组参数调控建议，同时画出更直观的可视化:
      1) 多参数随深度的剖面图 + 红/黄灯背景 + 绿灯正常带阴影
      2) “风险段 × 参数”的归一化偏差热力图，一眼看出每段主要问题参数

    结果:
    -----
    - {name}_risk_param_suggestions.csv
        每一行 = 一段连续红/黄灯区间，包含建议文本和核心参数组合;
    - {name}_risk_param_profiles.png
        多行子图: 每个动态参数一条曲线, 背景带有红/黄灯区间 + 绿灯正常带阴影;
    - {name}_risk_param_dev_heatmap.png
        行 = 风险段, 列 = 参数, 颜色 = 相对于绿灯正常带的归一化偏差。
    """
    if cfg is None:
        cfg = MODEL_CFG

    df = df.copy()
    os.makedirs(outdir, exist_ok=True)

    if depth_col is None:
        depth_col = _choose_depth_column(df, CLEAN_CFG["DEP_COL_CANDIDATES"])

    if light_col not in df.columns:
        print(f"[RISK-ANALYSIS] df 中不存在灯光列 '{light_col}'，跳过红黄灯参数分析。")
        return df

    # 过滤出真正存在的动态列
    dyn_cols = [c for c in dynamic_cols if c in df.columns]
    if not dyn_cols:
        print("[RISK-ANALYSIS] 未找到任何动态参数列，无法进行红黄灯参数分析。")
        return df

    # ===== 1) 以绿灯作为“正常工况”基线，统计每个参数的分布 =====
    mask_green = df[light_col] == "G"
    df_green = df.loc[mask_green, dyn_cols].dropna(how="all")
    if df_green.empty:
        print("[RISK-ANALYSIS] 全井几乎没有绿灯点，无法构造正常工况基线。")
        return df

    baseline_stats = {}
    for col in dyn_cols:
        col_series = pd.to_numeric(df_green[col], errors="coerce").dropna()
        if col_series.empty:
            continue
        baseline_stats[col] = {
            "q10": float(col_series.quantile(0.10)),
            "q25": float(col_series.quantile(0.25)),
            "median": float(col_series.quantile(0.50)),
            "q75": float(col_series.quantile(0.75)),
            "q90": float(col_series.quantile(0.90)),
        }

    # ===== 2) 按时间顺序识别连续红/黄灯段 =====
    light = df[light_col].astype(str).values
    n = len(df)
    seg_ids = np.zeros(n, dtype=int)
    seg_meta = []

    current_seg = 0
    start_idx = None
    for i in range(n):
        if light[i] in ("Y", "R"):
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                end_idx = i - 1
                if end_idx - start_idx + 1 >= min_seg_len:
                    current_seg += 1
                    seg_ids[start_idx:end_idx + 1] = current_seg
                    seg_meta.append((current_seg, start_idx, end_idx))
                start_idx = None
    if start_idx is not None:
        end_idx = n - 1
        if end_idx - start_idx + 1 >= min_seg_len:
            current_seg += 1
            seg_ids[start_idx:end_idx + 1] = current_seg
            seg_meta.append((current_seg, start_idx, end_idx))

    df["RISK_SEG_ID"] = seg_ids

    if not seg_meta:
        print("[RISK-ANALYSIS] 未发现长度超过阈值的连续红黄灯段。")
        return df

    # ===== 3) 对每一段红黄灯区，统计参数与绿灯基线的差异，并生成调控建议（表格部分） =====
    records = []
    for seg_id, s_idx, e_idx in seg_meta:
        seg_df = df.iloc[s_idx:e_idx + 1]
        seg_lights = seg_df[light_col].values
        seg_level = "R" if np.any(seg_lights == "R") else "Y"

        depth_start = float(seg_df[depth_col].iloc[0])
        depth_end = float(seg_df[depth_col].iloc[-1])

        row = {
            "name": name,
            "seg_id": seg_id,
            "light_level": seg_level,
            "depth_start": depth_start,
            "depth_end": depth_end,
            "n_points": int(len(seg_df)),
        }

        suggestions = []

        for col in dyn_cols:
            base = baseline_stats.get(col)
            if base is None:
                continue

            base_low, base_med, base_high = base["q25"], base["median"], base["q75"]
            seg_series = pd.to_numeric(seg_df[col], errors="coerce").dropna()
            if seg_series.empty:
                continue

            seg_med = float(seg_series.quantile(0.50))

            row[f"{col}_seg_median"] = seg_med
            row[f"{col}_green_median"] = base_med
            row[f"{col}_green_q25"] = base_low
            row[f"{col}_green_q75"] = base_high

            if base_high > base_low:
                norm_dev = (seg_med - base_med) / (base_high - base_low)
            else:
                norm_dev = 0.0
            row[f"{col}_norm_dev"] = float(norm_dev)

            high_thr = base["q90"]
            low_thr = base["q10"]

            if seg_med > high_thr:
                target = base_high
                suggestions.append(
                    f"{col}: 段内中值≈{seg_med:.2f}，显著高于绿灯正常上限≈{high_thr:.2f}，"
                    f"建议适当下调到 {target:.2f} 附近。"
                )
            elif seg_med < low_thr:
                target = base_low
                suggestions.append(
                    f"{col}: 段内中值≈{seg_med:.2f}，显著低于绿灯正常下限≈{low_thr:.2f}，"
                    f"建议适当上调到 {target:.2f} 附近。"
                )

        # 取 |norm_dev| 最大的若干参数作为“核心调控组合”
        dev_items = [(col, row.get(f"{col}_norm_dev", 0.0)) for col in dyn_cols]
        dev_items = [(c, abs(v)) for c, v in dev_items if v is not None]
        dev_items.sort(key=lambda x: x[1], reverse=True)
        core_params = [c for c, v in dev_items[:3] if v > 0.2]

        row["core_params"] = ",".join(core_params) if core_params else ""
        row["suggestion_text"] = "；".join(suggestions) if suggestions else ""

        records.append(row)

    df_sug = pd.DataFrame(records)
    csv_path = os.path.join(outdir, f"{name}_risk_param_suggestions.csv")
    df_sug.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[RISK-ANALYSIS] 红黄灯区域参数分析与调控建议已保存到: {csv_path}")

    # ===== 4) 可视化一: 多参数剖面图 + 红/黄灯背景 + 绿灯正常带 =====
    df_sorted = df.sort_values(depth_col)
    depth = pd.to_numeric(df_sorted[depth_col], errors="coerce").values
    lights_sorted = df_sorted[light_col].astype(str).values

    # 预先识别红/黄灯的深度区间，用于背景色带
    spans = []
    cur_light = None
    span_start = None
    for i in range(len(df_sorted)):
        L = lights_sorted[i]
        if L in ("Y", "R"):
            if span_start is None:
                span_start = depth[i]
                cur_light = L
            elif L != cur_light:
                spans.append((span_start, depth[i - 1], cur_light))
                span_start = depth[i]
                cur_light = L
        else:
            if span_start is not None:
                spans.append((span_start, depth[i - 1], cur_light))
                span_start = None
                cur_light = None
    if span_start is not None:
        spans.append((span_start, depth[-1], cur_light))

    if baseline_stats:
        n_rows = len(dyn_cols)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.5 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        for idx_ax, col in enumerate(dyn_cols):
            ax = axes[idx_ax]
            vals = pd.to_numeric(df_sorted[col], errors="coerce").values

            ax.plot(depth, vals, linewidth=0.8)
            ax.set_ylabel(col)

            base = baseline_stats.get(col)
            if base is not None:
                q25, med, q75 = base["q25"], base["median"], base["q75"]
                ax.axhspan(q25, q75, alpha=0.15)
                ax.axhline(med, linestyle="--", linewidth=0.8, alpha=0.5)

            for (d0, d1, L) in spans:
                if L == "R":
                    ax.axvspan(d0, d1, alpha=0.12)
                elif L == "Y":
                    ax.axvspan(d0, d1, alpha=0.08)

            ax.grid(True, linestyle="--", alpha=0.4)

        axes[0].set_title(f"{name} - 动态参数剖面 (含红/黄灯区间与绿灯正常带)")
        axes[-1].set_xlabel(depth_col or "Depth")

        # 深度从左浅右深，和你其他图保持一致
        for ax in axes:
            ax.invert_xaxis()

        fig.tight_layout()
        fig_path = os.path.join(outdir, f"{name}_risk_param_profiles.png")
        fig.savefig(fig_path, dpi=600)
        plt.close(fig)
        print(f"[RISK-ANALYSIS] 红黄灯参数剖面图已保存: {fig_path}")

    # ===== 5) 可视化二: “风险段 × 参数”归一化偏差热力图 =====
    if not df_sug.empty:
        heat_params = dyn_cols
        # 行标签: 灯光等级 + 深度范围
        seg_labels = []
        for _, r in df_sug.iterrows():
            lvl = r.get("light_level", "")
            ds = r.get("depth_start", np.nan)
            de = r.get("depth_end", np.nan)
            if np.isfinite(ds) and np.isfinite(de):
                seg_labels.append(f"{lvl}[{ds:.0f}-{de:.0f}]")
            else:
                seg_labels.append(str(lvl))

        dev_matrix = []
        for _, r in df_sug.iterrows():
            row_vals = []
            for col in heat_params:
                v = r.get(f"{col}_norm_dev", 0.0)
                if pd.isna(v):
                    v = 0.0
                row_vals.append(float(v))
            dev_matrix.append(row_vals)

        dev_matrix = np.array(dev_matrix, dtype=float)

        fig2, ax2 = plt.subplots(
            figsize=(1.8 * len(heat_params), 0.5 * max(len(seg_labels), 2) + 2)
        )
        im = ax2.imshow(dev_matrix, aspect="auto")

        ax2.set_xticks(np.arange(len(heat_params)))
        ax2.set_xticklabels(heat_params, rotation=45, ha="right")
        ax2.set_yticks(np.arange(len(seg_labels)))
        ax2.set_yticklabels(seg_labels)

        ax2.set_xlabel("参数")
        ax2.set_ylabel("风险区间 (等级@深度范围)")
        ax2.set_title(f"{name} - 各风险段参数归一化偏差")

        cbar = fig2.colorbar(im, ax=ax2)
        cbar.set_label("相对于绿灯正常带的归一化偏差")

        fig2.tight_layout()
        heat_path = os.path.join(outdir, f"{name}_risk_param_dev_heatmap.png")
        fig2.savefig(heat_path, dpi=600)
        plt.close(fig2)
        print(f"[RISK-ANALYSIS] 红黄灯参数偏差热力图已保存: {heat_path}")
    # ===== 6) 可视化三：参数调控“从红/黄到绿”的控制图 =====
    if not df_sug.empty:
        # 按严重程度优先（红灯 > 黄灯），其次按段长度排序，挑最典型的几段
        max_segments = 4  # 最多画 4 段，避免图太挤

        def _severity_key(i):
            row = df_sug.iloc[i]
            # 红灯优先
            is_red = 1 if str(row.get("light_level", "")) == "R" else 0
            # 段长
            n_pts = row.get("n_points", 0)
            return (-is_red, -n_pts)

        order = sorted(range(len(df_sug)), key=_severity_key)[:max_segments]
        sel = df_sug.iloc[order].copy()

        n_seg = len(sel)
        fig, axes = plt.subplots(
            n_seg,
            1,
            figsize=(1.8 * len(dyn_cols), 2.5 * n_seg),
            sharex=True,
        )
        if n_seg == 1:
            axes = [axes]

        x = np.arange(len(dyn_cols))

        for row, ax in zip(sel.itertuples(), axes):
            seg_label = f"{row.light_level}[{row.depth_start:.0f}-{row.depth_end:.0f}]"

            current_vals = []
            target_vals = []
            q25_list = []
            q75_list = []

            for col in dyn_cols:
                seg_med = getattr(row, f"{col}_seg_median", np.nan)
                q25 = getattr(row, f"{col}_green_q25", np.nan)
                q75 = getattr(row, f"{col}_green_q75", np.nan)

                if np.isnan(seg_med) or np.isnan(q25) or np.isnan(q75):
                    current_vals.append(np.nan)
                    target_vals.append(np.nan)
                    q25_list.append(np.nan)
                    q75_list.append(np.nan)
                else:
                    current_vals.append(seg_med)
                    target_vals.append((q25 + q75) / 2.0)  # 绿灯目标：正常带中值
                    q25_list.append(q25)
                    q75_list.append(q75)

            current_vals = np.array(current_vals, dtype=float)
            target_vals = np.array(target_vals, dtype=float)
            q25_arr = np.array(q25_list, dtype=float)
            q75_arr = np.array(q75_list, dtype=float)

            # 绿灯正常带 Q25~Q75
            ax.fill_between(
                x,
                q25_arr,
                q75_arr,
                alpha=0.15,
                label="Green band (Q25–Q75)",
            )

            # 当前红/黄灯段中值
            ax.scatter(
                x,
                current_vals,
                s=20,
                label="Current seg median",
            )

            # 目标：绿灯正常带中值
            ax.scatter(
                x,
                target_vals,
                marker="x",
                s=30,
                label="Target (green median)",
            )

            # 用箭头表示从当前值调控到目标值
            for j in range(len(dyn_cols)):
                if (
                        np.isfinite(current_vals[j])
                        and np.isfinite(target_vals[j])
                        and abs(current_vals[j] - target_vals[j]) > 1e-6
                ):
                    ax.annotate(
                        "",
                        xy=(x[j], target_vals[j]),
                        xytext=(x[j], current_vals[j]),
                        arrowprops=dict(
                            arrowstyle="->",
                            linewidth=0.8,
                            alpha=0.7,
                        ),
                    )

            ax.set_ylabel(seg_label)
            ax.grid(True, linestyle="--", alpha=0.4)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(dyn_cols, rotation=45, ha="right")

        # 只在最上面的子图画图例，避免重复
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, loc="upper right", fontsize=8)

        fig.suptitle(
            f"{name} - 参数调控控制图（由红/黄灯向绿灯区间调整）",
            fontsize=11,
        )
        fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.10, hspace=0.28)

        ctrl_path = os.path.join(outdir, f"{name}_risk_param_control.png")
        fig.savefig(ctrl_path, dpi=600)
        plt.close(fig)
        print(f"[RISK-ANALYSIS] 参数调控控制图已保存: {ctrl_path}")

    return df


def plot_risk_segments_pca_radar(
        df: pd.DataFrame,
        dynamic_cols: List[str],
        scaler: StandardScaler,
        pca: PCA,
        outdir: str,
        name: str,
        light_col: str = "LIGHT_FUSED",
        depth_col: str | None = None,
        min_seg_len: int = 5,
        max_segments: int = 6,
) -> pd.DataFrame:
    """
    基于 PCA 的“红/黄灯段风险雷达图”：

    - 先按 RISK_SEG_ID（或 LIGHT_FUSED）识别连续红/黄灯段；
    - 对每一段内的所有点，计算 PCA 主导贡献向量并取平均；
    - 将每一段的参数贡献向量画在同一张雷达图上（每段一条多边形曲线）。
    """
    if scaler is None or pca is None:
        print("[PCA-RADAR] 无 PCA 标准化器或模型，跳过雷达图绘制。")
        return df

    if depth_col is None:
        depth_col = _choose_depth_column(df, CLEAN_CFG["DEP_COL_CANDIDATES"])

    if light_col not in df.columns:
        print(f"[PCA-RADAR] df 中不存在灯光列 '{light_col}'，跳过雷达图绘制。")
        return df

    dyn_cols = [c for c in dynamic_cols if c in df.columns]
    if not dyn_cols:
        print("[PCA-RADAR] 未找到任何动态参数列，无法绘制雷达图。")
        return df

    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    # ===== 1) 确定红/黄灯段（优先使用 RISK_SEG_ID，如果没有则重新基于灯光识别） =====
    if "RISK_SEG_ID" in df.columns:
        seg_ids = df["RISK_SEG_ID"].fillna(0).astype(int).values
        seg_meta = []
        for seg_id in sorted(np.unique(seg_ids)):
            if seg_id <= 0:
                continue
            idx_pos = np.where(seg_ids == seg_id)[0]
            if len(idx_pos) < min_seg_len:
                continue
            seg_meta.append((seg_id, idx_pos[0], idx_pos[-1]))
    else:
        # 如果没有 RISK_SEG_ID，就再按灯光扫一遍（与 analyze_risk_segments_and_suggest_controls 类似）
        light = df[light_col].astype(str).values
        seg_ids = np.zeros(n, dtype=int)
        seg_meta = []
        cur_seg = 0
        start_idx = None
        for i in range(n):
            if light[i] in ("Y", "R"):
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    end_idx = i - 1
                    if end_idx - start_idx + 1 >= min_seg_len:
                        cur_seg += 1
                        seg_ids[start_idx:end_idx + 1] = cur_seg
                        seg_meta.append((cur_seg, start_idx, end_idx))
                    start_idx = None
        if start_idx is not None:
            end_idx = n - 1
            if end_idx - start_idx + 1 >= min_seg_len:
                cur_seg += 1
                seg_ids[start_idx:end_idx + 1] = cur_seg
                seg_meta.append((cur_seg, start_idx, end_idx))

        df["RISK_SEG_ID"] = seg_ids

    if not seg_meta:
        print("[PCA-RADAR] 未发现连续红黄灯段，跳过雷达图绘制。")
        return df

    depth = pd.to_numeric(df[depth_col], errors="coerce").values
    lights = df[light_col].astype(str).values

    # ===== 2) 为每一段计算 PCA 贡献向量（对段内所有点取平均） =====
    loadings = pca.components_
    n_comp = loadings.shape[0]

    seg_infos = []  # (seg_id, s_idx, e_idx, severity, length)
    contrib_list = []  # 每段一个 [len(dynamic_cols),] 向量
    label_list = []

    for seg_id, s_idx, e_idx in seg_meta:
        idx_pos = np.where(df["RISK_SEG_ID"].values == seg_id)[0]
        if len(idx_pos) < min_seg_len:
            continue

        # 段内优先看红灯，如果没有红灯就全是黄灯
        seg_lights = lights[idx_pos]
        if np.any(seg_lights == "R"):
            lvl = "R"
        else:
            lvl = "Y"

        # 段内严重度：优先用 SEG_LEVEL4_LABEL 的最大值
        if "SEG_LEVEL4_LABEL" in df.columns:
            seg_sev = int(df.iloc[idx_pos]["SEG_LEVEL4_LABEL"].max())
        else:
            seg_sev = 1 if lvl == "R" else 0

        seg_len = len(idx_pos)
        seg_infos.append((seg_id, s_idx, e_idx, seg_sev, seg_len))

        # 计算该段的平均 PCA 贡献向量
        contrib_sum = np.zeros(len(dyn_cols), dtype=float)
        valid_cnt = 0

        for pos in idx_pos:
            row = df.iloc[pos][dyn_cols]
            row = pd.to_numeric(row, errors="coerce").values.reshape(1, -1)
            row = np.nan_to_num(row)

            row_scaled = scaler.transform(row)
            scores = pca.transform(row_scaled)[0]  # 长度 = n_comp

            contrib = np.zeros(len(dyn_cols), dtype=float)
            for j in range(n_comp):
                contrib += abs(scores[j]) * np.abs(loadings[j, :])

            s = contrib.sum()
            if s > 0:
                contrib /= s
            contrib_sum += contrib
            valid_cnt += 1

        if valid_cnt == 0:
            continue

        contrib_avg = contrib_sum / valid_cnt
        contrib_list.append(contrib_avg)

        d0 = float(depth[s_idx])
        d1 = float(depth[e_idx])
        label_list.append(f"{lvl}[{d0:.0f}-{d1:.0f}]")

    if not contrib_list:
        print("[PCA-RADAR] 所有红黄灯段内 PCA 计算无有效样本。")
        return df

    # 按严重度优先 + 段长次之，取前 max_segments 个段绘图
    order = sorted(
        range(len(seg_infos)),
        key=lambda i: (-seg_infos[i][3], -seg_infos[i][4])
    )
    order = order[:max_segments]

    contrib_list = [contrib_list[i] for i in order]
    label_list = [label_list[i] for i in order]

    # ===== 3) 绘制雷达图 =====
    num_vars = len(dyn_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for contrib, label in zip(contrib_list, label_list):
        values = contrib.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.2, label=label)
        ax.fill(angles, values, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dyn_cols)
    ax.set_yticklabels([])

    ax.set_title(f"{name} - 红/黄灯段 PCA 主导参数雷达图")

    # 图例放到外侧，避免遮挡
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), borderaxespad=0.0, fontsize=8)

    fig.tight_layout()
    fig_path = os.path.join(outdir, f"{name}_risk_pca_radar.png")
    fig.savefig(fig_path, dpi=600)
    plt.close(fig)
    print(f"[PCA-RADAR] 雷达风险图已保存: {fig_path}")

    return df


def _build_gate_segments(
        gate: np.ndarray,
        merge_gap_steps: int = 0,
        min_len_steps: int = 1,
):
    """
    输入 gate(0/1)，输出：
      gate_filled: 合并小空隙后的 gate
      seg_id:      gate==1 的连续段编号（0 表示非 gate 段）
    """
    g = gate.astype(int).copy()
    n = len(g)
    if n == 0:
        return g, np.zeros(0, dtype=int)

    # 1) 合并小空隙：1 ... 0(短) ... 1  -> 填成全 1
    if merge_gap_steps and merge_gap_steps > 0:
        i = 0
        while i < n:
            if g[i] == 0:
                j = i
                while j < n and g[j] == 0:
                    j += 1
                gap_len = j - i
                left1 = (i - 1 >= 0 and g[i - 1] == 1)
                right1 = (j < n and g[j] == 1)
                if left1 and right1 and gap_len <= merge_gap_steps:
                    g[i:j] = 1
                i = j
            else:
                i += 1

    # 2) 连续段编号 + 过滤短段
    seg_id = np.zeros(n, dtype=int)
    sid = 0
    i = 0
    while i < n:
        if g[i] == 1:
            j = i
            while j < n and g[j] == 1:
                j += 1
            L = j - i
            if L >= max(1, int(min_len_steps)):
                sid += 1
                seg_id[i:j] = sid
            else:
                # 太短当噪声抛掉
                g[i:j] = 0
            i = j
        else:
            i += 1

    return g, seg_id


def fuse_seg4_by_gate_segments(
        df: pd.DataFrame,
        gate_col: str = "GATE_EXPAND",
        seg4_col: str = "SEG_LEVEL4_PRED",
        conf_col: str | None = "SEG_LEVEL4_CONF",
        out_col: str = "SEG_LEVEL4_PRED_FUSED",
        merge_gap_steps: int = 0,
        min_len_steps: int = 1,
        method: str = "max",  # "max" / "majority"
        conf_min: float = 0.0,
) -> pd.DataFrame:
    """
    在“预测扩径段”（由 gate 定义）内，把点级 seg4_col 融合成段级严重度，并回填到段内每个点。
    """
    df = df.copy()
    if gate_col not in df.columns or seg4_col not in df.columns:
        df[out_col] = np.nan
        return df

    gate = pd.to_numeric(df[gate_col], errors="coerce").fillna(0).astype(int).values
    g_filled, seg_id = _build_gate_segments(
        gate,
        merge_gap_steps=merge_gap_steps,
        min_len_steps=min_len_steps,
    )
    df["GATE_EXPAND_FILLED"] = g_filled
    df["GATE_SEG_ID"] = seg_id

    lv = pd.to_numeric(df[seg4_col], errors="coerce").values  # 可能有 nan
    # 可选：置信度过滤
    if conf_col and (conf_col in df.columns) and (conf_min is not None) and (conf_min > 0):
        conf = pd.to_numeric(df[conf_col], errors="coerce").fillna(0.0).values
        lv = np.where(conf >= conf_min, lv, np.nan)

    out = np.full(len(df), np.nan, dtype=float)

    # 段级聚合
    max_sid = int(np.nanmax(seg_id)) if len(seg_id) else 0
    for sid in range(1, max_sid + 1):
        idx = np.where(seg_id == sid)[0]
        if idx.size == 0:
            continue
        vals = lv[idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            seg_lv = 0
        else:
            vals = np.clip(vals.astype(int), 0, 3)
            if method == "majority":
                # 众数
                binc = np.bincount(vals, minlength=4)
                seg_lv = int(np.argmax(binc))
            else:
                # 默认：max（最保守）
                seg_lv = int(np.max(vals))
        out[idx] = seg_lv

    df[out_col] = out
    return df


def assign_traffic_lights(df: pd.DataFrame,
                          opc_prob_col: str = "OPC_PROB",
                          cfg: dict = None) -> pd.DataFrame:
    """
    纯分类版红黄灯：
      - Stage1: gate（二分类概率 -> GATE_EXPAND / LIGHT_STAGE1）
      - Stage2: 严重度（优先用 4分类 SEG_LEVEL4_PRED，且仅 gate==1 生效）
      - OPC: 事件风险灯
      - FUSED: 主灯 + OPC 融合

    """
    if cfg is None:
        cfg = MODEL_CFG
    if df is None or df.empty:
        return df

    df = df.copy()
    n = len(df)

    # ---------- 配置 ----------
    main_mode = str(cfg.get("MAIN_LIGHT_MODE", "gated")).lower().strip()  # 允许: cls / gated
    cls_col = cfg.get("MAIN_CLS_PROB_COL", "SEG_PROB_NOW")
    cy = float(cfg.get("MAIN_CLS_THRESH_Y", 0.5))
    cr = float(cfg.get("MAIN_CLS_THRESH_R", 0.8))

    # gate（Stage1）
    gate_col = cfg.get("GATE_STAGE1_PROB_COL", cls_col)
    gate_thr = float(cfg.get("GATE_STAGE1_THR", cy))
    gate_smooth_win = int(cfg.get("GATE_SMOOTH_WIN", 0))

    # Stage2：严重度（4分类）
    stage2_mode = str(cfg.get("GATE_STAGE2_MODE", "seg4")).lower().strip()  # 你这里一般是 "seg4"
    seg4_col = cfg.get("GATE_STAGE2_SEG4_COL", "SEG_LEVEL4_PRED")
    force_green = bool(cfg.get("GATE_FORCE_GREEN_WHEN_NOEXPAND", True))

    # OPC 阈值
    oy = float(cfg.get("OPC_THRESH_YELLOW", 0.5))
    or_ = float(cfg.get("OPC_THRESH_RED", 0.8))

    w_main = float(cfg.get("FUSE_W_MAIN", 0.6))
    w_opc = float(cfg.get("FUSE_W_OPC", 0.4))
    fused_y = float(cfg.get("FUSE_THRESH_Y", 0.6))
    fused_r = float(cfg.get("FUSE_THRESH_R", 1.2))

    # ---------- 工具函数 ----------
    def _main_light_cls(p):
        if not np.isfinite(p):
            return "G"
        if p >= cr:
            return "R"
        if p >= cy:
            return "Y"
        return "G"

    def _opc_light(p):
        if not np.isfinite(p):
            return "G"
        if p >= or_:
            return "R"
        if p >= oy:
            return "Y"
        return "G"

    # ---------- Stage1: gate ----------
    if gate_col in df.columns:
        p_gate = pd.to_numeric(df[gate_col], errors="coerce").fillna(0.0).values
    elif cls_col in df.columns:
        p_gate = pd.to_numeric(df[cls_col], errors="coerce").fillna(0.0).values
    else:
        p_gate = np.ones(n, dtype=float)  # 没有概率列就默认全开门（避免全绿“假安全”）

    gate = (p_gate >= gate_thr).astype(int)

    # 去抖（避免闪烁）
    if gate_smooth_win and gate_smooth_win > 1:
        gate = pd.Series(gate).rolling(gate_smooth_win, min_periods=1).max().astype(int).values

    df["STAGE1_PROB"] = p_gate
    df["GATE_EXPAND"] = gate
    df["LIGHT_STAGE1"] = np.where(gate == 1, "Y", "G")

    # 如果你已有更“段级”的 gate（填洞/短段剔除），优先用它
    if "GATE_EXPAND_FILLED" in df.columns:
        gate = df["GATE_EXPAND_FILLED"].astype(int).values
        df["GATE_EXPAND"] = gate
        df["LIGHT_STAGE1"] = np.where(gate == 1, "Y", "G")

    # ---------- Stage2: 严重度（仅 gate==1 生效） ----------
    df["LIGHT_STAGE2"] = ["G"] * n
    df["LIGHT_MAIN"] = ["G"] * n

    if main_mode == "cls":
        # 纯 cls（不门控）：直接用 cls_prob 画主灯
        if cls_col in df.columns:
            p_cls = pd.to_numeric(df[cls_col], errors="coerce").fillna(0.0).values
            df["LIGHT_MAIN"] = pd.Series(p_cls).apply(_main_light_cls).values
        else:
            df["LIGHT_MAIN"] = ["G"] * n
    else:
        # gated：Stage2 优先用 seg4（如果有）
        if stage2_mode in ("seg4", "gated_seg4") and (seg4_col in df.columns):
            # 段级融合（可选，但推荐：减抖 + 保守）
            fuse_method = str(cfg.get("SEG4_SEG_FUSE_METHOD", "max")).lower().strip()
            merge_gap = int(cfg.get("GATE_MERGE_GAP_STEPS", 0))
            min_len = int(cfg.get("GATE_MIN_LEN_STEPS", 1))
            conf_min = float(cfg.get("SEG4_CONF_MIN", 0.0))
            fused_col = cfg.get("SEG4_FUSED_COL", "SEG_LEVEL4_PRED_FUSED")

            df = fuse_seg4_by_gate_segments(
                df,
                gate_col="GATE_EXPAND",
                seg4_col=seg4_col,
                conf_col="SEG_LEVEL4_CONF",
                out_col=fused_col,
                merge_gap_steps=merge_gap,
                min_len_steps=min_len,
                method=fuse_method,
                conf_min=conf_min,
            )

            lv4 = pd.to_numeric(df[fused_col], errors="coerce").fillna(0).astype(int).clip(0, 3).values
            df["SEV_LEVEL4_RAW"] = lv4
            df["SEV_LEVEL4_GATED"] = np.where(gate == 1, lv4, 0)

            map4 = {0: "G", 1: "Y", 2: "Y", 3: "R"}
            light2 = pd.Series(df["SEV_LEVEL4_GATED"]).map(map4).fillna("G").values
            df["LIGHT_STAGE2"] = light2
            df["LIGHT_MAIN"] = light2 if not force_green else np.where(gate == 1, light2, "G")
        else:
            # 没有 seg4：退回“门控后的 cls 主灯”
            if cls_col in df.columns:
                p_cls = pd.to_numeric(df[cls_col], errors="coerce").fillna(0.0).values
                light_cls = pd.Series(p_cls).apply(_main_light_cls).values
                df["LIGHT_STAGE2"] = light_cls
                df["LIGHT_MAIN"] = light_cls if not force_green else np.where(gate == 1, light_cls, "G")
            else:
                df["LIGHT_MAIN"] = ["G"] * n
                df["LIGHT_STAGE2"] = ["G"] * n

    # ---------- OPC 灯 ----------
    if opc_prob_col in df.columns:
        opc_prob = pd.to_numeric(df[opc_prob_col], errors="coerce").fillna(0.0).values
    else:
        opc_prob = np.full(n, np.nan, dtype=float)
    df["LIGHT_OPC"] = pd.Series(opc_prob).apply(_opc_light).values

    # ---------- 联合灯 ----------
    score_map = {"G": 0.0, "Y": 1.0, "R": 2.0}
    fused_scores = []
    for m_light, o_light in zip(df["LIGHT_MAIN"], df["LIGHT_OPC"]):
        s = w_main * score_map.get(m_light, 0.0) + w_opc * score_map.get(o_light, 0.0)
        fused_scores.append(s)
    fused_scores = np.array(fused_scores)

    df["FUSED_SCORE"] = fused_scores
    df["S_MAIN"] = df["LIGHT_MAIN"].map(score_map).astype(float)
    df["S_OPC"] = df["LIGHT_OPC"].map(score_map).astype(float)

    fused_light = []
    for s in fused_scores:
        if s >= fused_r:
            fused_light.append("R")
        elif s >= fused_y:
            fused_light.append("Y")
        else:
            fused_light.append("G")
    df["LIGHT_FUSED"] = fused_light

    return df



def smooth_lights_by_segment(df: pd.DataFrame,
                             depth_col: str,
                             seg_id_col: str = "BER_SEG_ID",
                             fused_col: str = "LIGHT_FUSED",
                             cfg: dict = None) -> pd.DataFrame:
    """
    基于连续扩径段（BER_SEG_ID）对联合红黄灯做段级平滑：
    - 每个 seg_id>0 段内，如果 R/Y 点所占比例足够大，则整段统一提为 R 或 Y，
      避免图上出现大量“点状”红黄灯。
    """
    if cfg is None:
        cfg = MODEL_CFG

    if seg_id_col not in df.columns or fused_col not in df.columns:
        # 没有扩径段信息或没有联合灯，直接返回
        return df

    df = df.copy()
    if depth_col in df.columns:
        df = df.sort_values(depth_col).reset_index(drop=True)

    seg_y_frac = cfg.get("SEG_FUSED_Y_FRAC", 0.2)
    seg_r_frac = cfg.get("SEG_FUSED_R_FRAC", 0.5)

    seg_ids = [sid for sid in np.unique(df[seg_id_col].values) if sid > 0]
    if not seg_ids:
        return df

    fused = df[fused_col].astype(str).values
    seg_arr = df[seg_id_col].values

    for sid in seg_ids:
        mask = (seg_arr == sid)
        if not np.any(mask):
            continue
        n = int(mask.sum())
        lights_seg = fused[mask]
        n_r = int(np.sum(lights_seg == "R"))
        n_y = int(np.sum(lights_seg == "Y"))
        frac_r = n_r / n
        frac_ry = (n_r + n_y) / n

        # 先判红段，再判黄段
        if frac_r >= seg_r_frac:
            fused[mask] = "R"
        elif frac_ry >= seg_y_frac:
            fused[mask] = "Y"
        # 否则保持原值（通常是 G）

    df[fused_col] = fused
    return df


def analyze_segmentation_patterns(df: pd.DataFrame, lookahead_steps: int = 60):
    """
    分析扩径段的时间模式，帮助理解隐形规律
    """
    if "BER_IS_SEG" not in df.columns:
        print("[WARN] 没有扩径段标签，无法分析模式")
        return

    seg_arr = df["BER_IS_SEG"].values
    ber_arr = df["BER"].values if "BER" in df.columns else None

    n = len(seg_arr)
    transitions = []  # 记录扩径段开始和结束的位置

    in_seg = False
    seg_start = None

    for i in range(n):
        if seg_arr[i] == 1 and not in_seg:
            # 进入扩径段
            in_seg = True
            seg_start = i
        elif seg_arr[i] == 0 and in_seg:
            # 离开扩径段
            in_seg = False
            if seg_start is not None:
                transitions.append({
                    "type": "start",
                    "index": seg_start,
                    "depth": df.iloc[seg_start][df.columns[0]] if len(df.columns) > 0 else seg_start,
                    "lookahead_pattern": seg_arr[
                        seg_start:min(seg_start + lookahead_steps, n)] if ber_arr is None else ber_arr[
                        seg_start:min(seg_start + lookahead_steps, n)]
                })
                transitions.append({
                    "type": "end",
                    "index": i - 1,
                    "depth": df.iloc[i - 1][df.columns[0]] if len(df.columns) > 0 else i - 1,
                })

    # 分析扩径段前视窗口的统计特征
    print(f"[ANALYSIS] 扩径段数量: {len([t for t in transitions if t['type'] == 'start'])}")

    # 分析扩径段开始前的模式
    pre_seg_patterns = []
    for trans in transitions:
        if trans["type"] == "start":
            start_idx = trans["index"]
            # 查看扩径段开始前的 lookahead_margin 个点
            pre_window = max(0, start_idx - MODEL_CFG.get("LOOKAHEAD_MARGIN", 20))
            if ber_arr is not None:
                pre_pattern = ber_arr[pre_window:start_idx]
                if len(pre_pattern) > 0:
                    pre_seg_patterns.append(pre_pattern)

    if pre_seg_patterns:
        avg_pre_pattern = np.mean([p.mean() for p in pre_seg_patterns])
        print(f"[ANALYSIS] 扩径段开始前{lookahead_steps}点平均BER: {avg_pre_pattern:.2f}")

    return transitions


# =========================
# 主流程 main()
# =========================
# --- 新增：简单的 DataFrame 缓存工具（训练井 + 测试井基础特征） ---
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


def build_train_well_dfs() -> List[pd.DataFrame]:
    train_dfs: List[pd.DataFrame] = []

    for well, cfg in TRAIN_WELLS.items():
        print(f"[WELL] 处理训练井 {well}")

        # ========= 录井数据：兼容单文件/多文件 =========
        log_cfg = cfg["log"]

        # 支持两种写法：
        # 1) "log": r"F:/1/X3/202001-1.xlsx"
        # 2) "log": [r"F:/1/X3/202001-1.xlsx", r"F:/1/X3/202002-1.xlsx"]
        if isinstance(log_cfg, str):
            log_paths = [log_cfg]
        else:
            # 假定是可迭代的列表/元组
            log_paths = list(log_cfg)

        valid_paths = []
        for p in log_paths:
            exists = os.path.exists(p)
            print(f"[TRAIN-LOG] {well}: {p}  存在={exists}")
            if exists:
                valid_paths.append(p)
            else:
                print(f"[WARN] 录井路径不存在，已跳过: {p}")

        if not valid_paths:
            print(f"[ERROR] 井 {well} 没有任何有效的录井文件，跳过本井")
            continue

        # 多个 Excel 拼成一口井
        df_list = []
        for p in valid_paths:
            df_tmp = pd.read_excel(p)
            df_tmp["__SRC_FILE"] = os.path.basename(p)
            df_list.append(df_tmp)

        df_log_raw = pd.concat(df_list, axis=0, ignore_index=True)
        print(f"[INFO] 井 {well} 录井合并后行数={len(df_log_raw)}")

        # 如果没有统一时间列，这里尝试构造
        if "WELLDATETIME" not in df_log_raw.columns:
            if "WELLDATE" in df_log_raw.columns and "WELLTIME" in df_log_raw.columns:
                df_log_raw["WELLDATETIME"] = pd.to_datetime(
                    df_log_raw["WELLDATE"].astype(str) + " " + df_log_raw["WELLTIME"].astype(str),
                    errors="coerce",
                )
                print(f"[INFO] 井 {well} 通过 WELLDATE+WELLTIME 构造 WELLDATETIME")
            else:
                raise ValueError(
                    f"井 {well} 缺少 WELLDATETIME 或 WELLDATE+WELLTIME，无法构造统一时间列。"
                )

        df_log_raw = df_log_raw[~df_log_raw["WELLDATETIME"].isna()].copy()
        df_log_raw = df_log_raw.sort_values("WELLDATETIME").reset_index(drop=True)

        # ========= 录井清洗 =========
        df_clean = clean_time_series_per_well(df_log_raw, well_name=well)
        # ========= 动态特征：一阶差分 + 滚动统计 =========
        dynamic_cols = [c for c in ["WOB", "RPM", "TOR", "SPP", "FLOWIN", "FLOWOUT", "HKLD"]
                        if c in df_clean.columns]
        if dynamic_cols:
            df_clean = add_derived_features(df_clean, dynamic_cols=dynamic_cols)
        # ========= 地层 & CAL =========
        form_path = cfg.get("form")
        cal_path = cfg.get("cal")

        df_form = load_formation_xlsx(form_path, well_name=well) if form_path else None
        df_cal = load_cal_file(cal_path, well_name=well) if cal_path else None

        if df_form is None or df_cal is None:
            print(f"[WARN] 井 {well} 缺少地层或 CAL 文件，无法对齐 BER，已跳过本井。")
            continue

        # ========= 深度对齐 + 构造扩径段 =========
        df_all = merge_formation_and_cal_to_time(df_clean, df_form, df_cal)
        # 标记井名，方便后续分析 / 导出
        if "WELL" not in df_all.columns:
            df_all["WELL"] = well
            # （可选）为每口训练井输出扩径段检测示意图
        if RUN_CFG.get("PLOT_SEG_FIG", True) and ("BER" in df_all.columns):
            # 复用主 outdir 结构：OUTDIR_ROOT / RUN_NAME / seg_plots
            out_root = RUN_CFG["OUTDIR_ROOT"]
            run_name = RUN_CFG.get("RUN_NAME") or "ber_timeseries_run"
            seg_plot_dir = os.path.join(out_root, run_name, "seg_plots")

            # 选择一个合适的深度列（DEP / BITDEP）
            dep_col = _choose_depth_column(df_all, CLEAN_CFG["DEP_COL_CANDIDATES"])

            plot_ber_segments_for_well(
                df_all,
                well_name=well,
                outdir=seg_plot_dir,
                depth_col=dep_col,
                ber_col="BER",  # merge_formation_and_cal_to_time 里已统一为平滑后 BER
                ber_smooth_col="BER_SMOOTH",
                seg_flag_col="BER_IS_SEG",
                future_label_col="SEG_FUTURE_LABEL",
                start_thresh=SEG_CFG.get("START_THRESH"),
            )
        df_all = add_interaction_features(df_all)
        # （可选）再加趋势特征，基于交互 + 动态量做滚动均值/方差
        df_all = add_trend_features(df_all)

        # 先根据 BER 大小做一轮基础权重放大（全井）
        df_all = reweight_by_ber(
            df_all,
            ber_col="BER",
            base_thresh=5.0,
            max_factor=5.0,
        )

        # 保证有 sample_weight 列
        if "sample_weight" not in df_all.columns:
            df_all["sample_weight"] = 1.0

        # (1) 扩径段整体加权：让扩径段样本在训练中更“值钱”
        weight_in_seg = float(SEG_CFG.get("WEIGHT_IN_SEG", 1.0))
        if "BER_IS_SEG" in df_all.columns and weight_in_seg > 1.0:
            df_all.loc[df_all["BER_IS_SEG"] == 1, "sample_weight"] *= weight_in_seg
            print(f"[INFO] 井 {well} 对扩径段样本加权 x{weight_in_seg}")

        # (1.5) 四档扩径等级软约束：基于 SEG_LEVEL4_LABEL 进一步放大权重
        level4_weights = SEG_CFG.get("LEVEL4_WEIGHTS", None)
        if level4_weights is not None and "SEG_LEVEL4_LABEL" in df_all.columns:
            level4_weights = np.array(level4_weights, dtype=float)
            # 防御：长度不对就不启用
            if level4_weights.size == 4:
                level_arr = df_all["SEG_LEVEL4_LABEL"].fillna(0).astype(int).values
                level_arr = np.clip(level_arr, 0, 3)
                factor = level4_weights[level_arr]
                df_all["sample_weight"] *= factor
                print(f"[INFO] 井 {well} 按四档扩径等级做软约束权重放大: {level4_weights.tolist()}")

        # (2) 再按 BER 大小做一次针对扩径阈值的权重放大（强调真正高 BER 段）
        df_all = reweight_by_ber(
            df_all,
            ber_col="BER",
            base_thresh=SEG_CFG["START_THRESH"],
            max_factor=SEG_CFG.get("MAX_BER_WEIGHT", 5.0),
        )

        train_dfs.append(df_all)

    return train_dfs


# ===== (NEW) Two-stage strict-gating segment evaluation =====
def _segments_from_id(df: pd.DataFrame,
                      id_col: str,
                      depth_col: str,
                      level_point_col: str = None,
                      level_reduce: str = "max"):
    """
    从 id_col(>0) 提取连续段，返回 list[dict]
    dict keys: seg_id,start_i,end_i,start_d,end_d,len_m,n_points, level(optional)
    """
    segs = []
    if df is None or df.empty or id_col not in df.columns or depth_col not in df.columns:
        return segs

    sid = pd.to_numeric(df[id_col], errors="coerce").fillna(0).astype(int).values
    dep = pd.to_numeric(df[depth_col], errors="coerce").values

    n = len(df)
    i = 0
    while i < n:
        if sid[i] <= 0:
            i += 1
            continue
        cur = sid[i]
        s = i
        while i < n and sid[i] == cur:
            i += 1
        e = i - 1
        if e <= s:
            continue

        start_d = float(dep[s]) if np.isfinite(dep[s]) else np.nan
        end_d = float(dep[e]) if np.isfinite(dep[e]) else np.nan
        seg_len = float(end_d - start_d) if np.isfinite(start_d) and np.isfinite(end_d) else np.nan

        item = {
            "seg_id": int(cur),
            "start_i": int(s),
            "end_i": int(e),
            "start_d": start_d,
            "end_d": end_d,
            "len_m": seg_len,
            "n_points": int(e - s + 1),
        }

        if level_point_col is not None and level_point_col in df.columns:
            lv = pd.to_numeric(df.iloc[s:e + 1][level_point_col], errors="coerce").fillna(0).astype(int).values
            if lv.size > 0:
                if level_reduce == "mode":
                    # 众数（更稳）
                    vals, cnts = np.unique(lv, return_counts=True)
                    item["level"] = int(vals[int(np.argmax(cnts))])
                else:
                    # 默认 max（更保守，贴合“段级取最大BER定级”的逻辑）
                    item["level"] = int(np.max(lv))
            else:
                item["level"] = 0
        segs.append(item)

    return segs


def _seg_iou_m(a: dict, b: dict) -> float:
    """按深度区间计算 IoU（单位 m）"""
    if a is None or b is None:
        return 0.0
    if not (np.isfinite(a.get("start_d", np.nan)) and np.isfinite(a.get("end_d", np.nan)) and
            np.isfinite(b.get("start_d", np.nan)) and np.isfinite(b.get("end_d", np.nan))):
        return 0.0
    a0, a1 = float(a["start_d"]), float(a["end_d"])
    b0, b1 = float(b["start_d"]), float(b["end_d"])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    uni = max(a1, b1) - min(a0, b0)
    return float(inter / uni) if uni > 0 else 0.0


def _match_segments_greedy(true_segs, pred_segs, min_iou=0.10):
    """
    贪心匹配：每个真段找 IoU 最大的预测段（且预测段只用一次）
    返回:
      matches: list[(true_seg, pred_seg_or_None, best_iou)]
      fps: unmatched pred segs
    """
    used = set()
    matches = []
    for t in true_segs:
        best_p = None
        best_iou = 0.0
        for p in pred_segs:
            if p["seg_id"] in used:
                continue
            iou = _seg_iou_m(t, p)
            if iou > best_iou:
                best_iou = iou
                best_p = p
        if best_p is not None and best_iou >= float(min_iou):
            used.add(best_p["seg_id"])
            matches.append((t, best_p, float(best_iou)))
        else:
            matches.append((t, None, 0.0))

    fps = [p for p in pred_segs if p["seg_id"] not in used]
    return matches, fps


def _plot_cm(cm: np.ndarray, labels, out_png: str, title: str):
    """简单混淆矩阵图（matplotlib）"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 数字标注
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


def evaluate_two_stage_segments(df_fold: pd.DataFrame, outdir: str, name: str, dep_col: str = None):
    """
    基于折叠后的 df_fold 做段级评估：
      - Stage1: 段检出(Recall/Precision) + 点级(Precision/Recall)
      - Stage2: matched segments 的 4分类混淆矩阵
    输出：
      {name}_seg_eval.json
      {name}_seg_matches.csv
      {name}_seg4_cm.png
    """
    os.makedirs(outdir, exist_ok=True)
    if df_fold is None or df_fold.empty:
        return

    if dep_col is None:
        dep_col = _choose_depth_column(df_fold, CLEAN_CFG["DEP_COL_CANDIDATES"])

    df_fold = df_fold.copy()
    df_fold = df_fold.sort_values(dep_col).reset_index(drop=True)

    # ---- 确保真值段列存在；如果没有，就用 build_ber_segments_simple 现算（只用于评估，不参与推理）----
    if "BER_SEG_ID" not in df_fold.columns or "BER_IS_SEG" not in df_fold.columns or "BER_SEG_LEVEL" not in df_fold.columns:
        # 用平滑后的 BER 更稳；如果你折叠后 BER 已经是平滑BER，那直接用 BER
        ber_col = "BER"
        df_fold = build_ber_segments_simple(
            df_fold,
            depth_col=dep_col,
            ber_col=ber_col,
            abs_thresh=SEG_CFG.get("START_THRESH", 5.0),
            min_seg_len=SEG_CFG.get("MIN_LEN_M", 0.30),
        )

    # ---- 预测门控列 & 门控后的严重度列 ----
    gate_col_candidates = ["GATE_EXPAND", "GATE_PRED", "SEG_PRED_NOW"]
    gate_col = next((c for c in gate_col_candidates if c in df_fold.columns), None)
    if gate_col is None:
        print("[SEG-EVAL][WARN] 找不到 gate 列(GATE_EXPAND/GATE_PRED/SEG_PRED_NOW)，跳过段级评估。")
        return

    # 如果你按我之前建议做了 GATE_SEG_ID，用它；否则这里现算一个
    if "GATE_SEG_ID" not in df_fold.columns:
        gate = pd.to_numeric(df_fold[gate_col], errors="coerce").fillna(0).astype(int).values
        seg_id = np.zeros_like(gate, dtype=int)
        cur = 0
        in_seg = False
        for i, g in enumerate(gate):
            if g == 1 and not in_seg:
                cur += 1
                in_seg = True
            if g == 0:
                in_seg = False
            seg_id[i] = cur if in_seg else 0
        df_fold["GATE_SEG_ID"] = seg_id

    pred_level_candidates = ["SEV_LEVEL4_GATED", "SEG_LEVEL4_PRED_FUSED", "SEV_LEVEL4_RAW", "SEG_LEVEL4_PRED"]
    pred_level_col = next((c for c in pred_level_candidates if c in df_fold.columns), None)

    # ---- 提取真值段 / 预测段 ----
    true_segs = _segments_from_id(df_fold, "BER_SEG_ID", dep_col, level_point_col="BER_SEG_LEVEL", level_reduce="max")
    pred_segs = _segments_from_id(df_fold, "GATE_SEG_ID", dep_col, level_point_col=pred_level_col, level_reduce="max")

    # ---- Stage1：段检出指标（IoU 匹配）----
    min_iou = float(MODEL_CFG.get("SEG_MATCH_MIN_IOU", 0.10))
    matches, fps = _match_segments_greedy(true_segs, pred_segs, min_iou=min_iou)

    n_true = len(true_segs)
    n_pred = len(pred_segs)
    n_hit = sum(1 for (_, p, _) in matches if p is not None)

    seg_recall = (n_hit / n_true) if n_true > 0 else 0.0
    seg_prec = (n_hit / n_pred) if n_pred > 0 else 0.0

    # ---- Stage1：点级（用 BER_IS_SEG vs gate）----
    y_true = pd.to_numeric(df_fold["BER_IS_SEG"], errors="coerce").fillna(0).astype(int).values
    y_pred = pd.to_numeric(df_fold[gate_col], errors="coerce").fillna(0).astype(int).values

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    pt_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pt_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # ---- Stage2：严重度 4分类（只对 matched 段算）----
    cm = np.zeros((4, 4), dtype=int)
    seg_rows = []
    for t, p, iou in matches:
        true_lv = int(t.get("level", 0))
        pred_lv = int(p.get("level", 0)) if p is not None else 0
        true_lv = int(np.clip(true_lv, 0, 3))
        pred_lv = int(np.clip(pred_lv, 0, 3))
        cm[true_lv, pred_lv] += 1

        seg_rows.append({
            "true_seg_id": t["seg_id"],
            "true_level": true_lv,
            "true_start_d": t["start_d"],
            "true_end_d": t["end_d"],
            "pred_seg_id": (p["seg_id"] if p is not None else 0),
            "pred_level": pred_lv,
            "pred_start_d": (p["start_d"] if p is not None else np.nan),
            "pred_end_d": (p["end_d"] if p is not None else np.nan),
            "iou": float(iou),
        })

    # 导出 matches 表
    out_csv = os.path.join(outdir, f"{name}_seg_matches.csv")
    pd.DataFrame(seg_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SEG-EVAL] seg matches saved: {out_csv}")

    # 画 cm
    out_png = os.path.join(outdir, f"{name}_seg4_cm.png")
    _plot_cm(cm, labels=["0", "1", "2", "3"], out_png=out_png, title=f"{name} Seg4 (matched, IoU>={min_iou})")
    print(f"[SEG-EVAL] seg4 cm saved: {out_png}")

    # 汇总 json
    out_json = os.path.join(outdir, f"{name}_seg_eval.json")
    summary = {
        "name": name,
        "gate_col": gate_col,
        "pred_level_col": pred_level_col,
        "min_iou": min_iou,
        "stage1_segment": {"n_true": n_true, "n_pred": n_pred, "n_hit": n_hit, "recall": seg_recall,
                           "precision": seg_prec},
        "stage1_point": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": pt_recall, "precision": pt_prec},
        "stage2_cm": cm.tolist(),
        "false_positive_pred_segments": len(fps),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[SEG-EVAL] summary saved: {out_json}")

    # 一个强制自检：非 gate 区域的严重度应当全为 0（如果你做的是严格门控）
    if pred_level_col is not None and pred_level_col in df_fold.columns:
        bad = df_fold.loc[pd.to_numeric(df_fold[gate_col], errors="coerce").fillna(0).astype(int) == 0, pred_level_col]
        bad_max = float(pd.to_numeric(bad, errors="coerce").fillna(0).max()) if len(bad) else 0.0
        if bad_max > 0:
            print(f"[SEG-EVAL][WARN] 非 gate 区域 {pred_level_col} 最大值={bad_max}，说明门控/派生灯仍可能没用到门控列！")


def fold_predictions_by_depth(df: pd.DataFrame, dep_col: str) -> pd.DataFrame:
    """
    将同一深度的多条时间记录折叠为一条：

      - OPC_PROB 取最大值（更保守）
      - 时间列取最早时间，方便定位现场记录
    并重新计算红黄灯。
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    time_col = CLEAN_CFG.get("TIME_COL", "WELLDATETIME")

    agg = {}
    # 时间列：取最早时间
    if time_col in df.columns:
        agg[time_col] = "min"
    # BER 真值（对照）：同一深度多个时间点取中位数
    if "BER" in df.columns:
        agg["BER"] = "median"

    # seg4：同一深度多个时间点，严重度用 max（保守不漏报），置信度用 max
    if "SEG_LEVEL4_PRED" in df.columns:
        agg["SEG_LEVEL4_PRED"] = "max"
    if "SEG_LEVEL4_CONF" in df.columns:
        agg["SEG_LEVEL4_CONF"] = "max"

    # OPC 概率：取最大值，防止漏掉高风险时间点
    if "OPC_PROB" in df.columns:
        agg["OPC_PROB"] = "max"
    # ★ 主灯分类概率：折叠后仍需保留，否则 MAIN_LIGHT_MODE='cls' 会找不到列而回退到回归灯
    # 保守策略：同一深度多个时间点取最大概率（更偏向不漏报）
    cls_col = MODEL_CFG.get("MAIN_CLS_PROB_COL", "SEG_PROB_NOW")
    if cls_col in df.columns:
        agg[cls_col] = "max"
    # 兼容：即使 MAIN_CLS_PROB_COL 被改名，也尽量保留默认列
    if "SEG_PROB_NOW" in df.columns and "SEG_PROB_NOW" not in agg:
        agg["SEG_PROB_NOW"] = "max"
    # 兼容其它概率列命名（训练/验证预测时可能用 MAIN_PROB_CLS）
    for c in ["MAIN_PROB_CLS", "PROB_CLS", "CLS_PROB", "SEG_PROB"]:
        if c in df.columns and c not in agg:
            agg[c] = "max"

    # 保留扩径段信息：只要某深度上有一个点在某个段内，就认为该深度属于该段
    if "BER_SEG_ID" in df.columns:
        agg["BER_SEG_ID"] = "max"
    if "BER_IS_SEG" in df.columns:
        agg["BER_IS_SEG"] = "max"

    if not agg:
        # 没有可聚合的列，直接按深度去重
        return df.drop_duplicates(subset=[dep_col]).sort_values(dep_col)

    # 按深度折叠
    df_fold = df.groupby(dep_col, as_index=False).agg(agg)
    df_fold = df_fold.sort_values(dep_col).reset_index(drop=True)

    df_fold = assign_traffic_lights(
        df_fold,
        opc_prob_col="OPC_PROB",
        cfg=MODEL_CFG,
    )

    # 有段ID的话再做段级平滑（保留）
    if "BER_SEG_ID" in df_fold.columns:
        df_fold = smooth_lights_by_segment(
            df_fold,
            depth_col=dep_col,
            seg_id_col="BER_SEG_ID",
            fused_col="LIGHT_FUSED",
            cfg=MODEL_CFG,
        )

    return df_fold


def _vis_pick_idx(n: int, max_points: int):
    if (max_points is None) or (max_points <= 0) or (n <= max_points):
        return None
    return np.linspace(0, n - 1, max_points).astype(int)

def _vis_smooth(y, win: int):
    if y is None:
        return None
    if win is None or win <= 1:
        return y
    return pd.Series(y).rolling(win, center=True, min_periods=1).median().values

def _lights_to_int(light_series: pd.Series):
    arr = light_series.astype(str).values
    out = np.zeros(len(arr), dtype=int)  # G=0
    out[arr == "Y"] = 1
    out[arr == "R"] = 2
    return out


# ===== 灯带可视化：细线轨道（替代 imshow 色带，避免“太粗”）=====
def _iter_light_spans(depth_arr: np.ndarray, lights: pd.Series):
    """把 G/Y/R 序列压缩成连续区间 (xmin, xmax, light) 列表。"""
    if lights is None or len(lights) == 0:
        return []
    d = np.asarray(depth_arr)
    a = pd.Series(lights).astype(str).fillna("G").values
    spans = []
    s = 0
    cur = a[0]
    for i in range(1, len(a)):
        if a[i] != cur:
            x0 = float(d[s]); x1 = float(d[i - 1])
            spans.append((min(x0, x1), max(x0, x1), cur))
            s = i
            cur = a[i]
    x0 = float(d[s]); x1 = float(d[len(a) - 1])
    spans.append((min(x0, x1), max(x0, x1), cur))
    return spans

def _plot_lights_as_lines(ax, depth_arr: np.ndarray, lights: pd.Series, y: float,
                         lw: float = 3.0,
                         color_map: dict = None):
    """在 y 位置画一条由多段颜色组成的“灯线”。"""
    if color_map is None:
        color_map = {"G": "green", "Y": "gold", "R": "red"}
    for xmin, xmax, c in _iter_light_spans(depth_arr, lights):
        ax.hlines(y, xmin, xmax, colors=color_map.get(c, "green"), linewidth=lw)




def plot_test_results(df_test: pd.DataFrame, outdir: str, test_name: str):
    """
    分类-only 可视化（单口井，按深度折叠后画剖面）：
      (1) BER 真值（对照）+ 参考阈值线
      (2) Seg-Now 二分类概率 + 阈值
      (3) 严重度 4 分类（pred + conf）
      (4) OPC 概率（左轴）+ 融合分数（右轴）
      (5) 三条红黄灯轨道：LIGHT_MAIN / LIGHT_OPC / LIGHT_FUSED
    并导出“非绿灯点”的风险点表格。
    """
    os.makedirs(outdir, exist_ok=True)

    # 选深度列
    dep_col = _choose_depth_column(df_test, CLEAN_CFG["DEP_COL_CANDIDATES"])

    # ★ 出图前按深度折叠：同一深度多个时间点 → 1 个聚合结果
    df_test = fold_predictions_by_depth(df_test, dep_col)

    # ★ 段级评估：二阶段严格门控是否真正闭环（可保留）
    try:
        evaluate_two_stage_segments(df_test, outdir, test_name, dep_col=dep_col)
    except Exception as e:
        print(f"[SEG-EVAL][WARN] evaluate_two_stage_segments failed: {e}")

    # 折叠后再取深度 / 真值 / 分类概率 / 严重度 / OPC / 融合分数 / 灯光
    depth = df_test[dep_col].values

    # 1) BER 真值（仅做对照）
    ber_true = df_test["BER"].values if "BER" in df_test.columns else None

    # 2) 二分类概率（Seg-Now）
    # 二分类概率列：自动兜底（避免折叠后列名不一致导致缺列）
    prob_candidates = [
        MODEL_CFG.get("MAIN_CLS_PROB_COL", "SEG_PROB_NOW"),
        "SEG_PROB_NOW", "MAIN_PROB_CLS", "PROB_CLS"
    ]
    cls_col = next((c for c in prob_candidates if c in df_test.columns), None)
    p_cls = pd.to_numeric(df_test[cls_col], errors="coerce").values if cls_col else None

    # 3) 严重度 4 分类（pred + conf）
    seg4_pred = pd.to_numeric(df_test["SEG_LEVEL4_PRED"], errors="coerce").values if "SEG_LEVEL4_PRED" in df_test.columns else None
    seg4_conf = pd.to_numeric(df_test["SEG_LEVEL4_CONF"], errors="coerce").values if "SEG_LEVEL4_CONF" in df_test.columns else None

    # 4) OPC 概率 & 融合分数
    opc_prob = pd.to_numeric(df_test["OPC_PROB"], errors="coerce").values if "OPC_PROB" in df_test.columns else None
    fused_score = pd.to_numeric(df_test["FUSED_SCORE"], errors="coerce").values if "FUSED_SCORE" in df_test.columns else None

    # 5) 阈值（画参考线）
    by = float(MODEL_CFG.get("BER_THRESH_YELLOW", 8.0))
    br = float(MODEL_CFG.get("BER_THRESH_RED", 15.0))
    cy = float(MODEL_CFG.get("MAIN_CLS_THRESH_Y", 0.5))
    cr = float(MODEL_CFG.get("MAIN_CLS_THRESH_R", 0.8))
    oy = float(MODEL_CFG.get("OPC_THRESH_YELLOW", 0.3))
    or_ = float(MODEL_CFG.get("OPC_THRESH_RED", 0.6))
    fy = float(MODEL_CFG.get("FUSE_THRESH_Y", 0.6))
    fr = float(MODEL_CFG.get("FUSE_THRESH_R", 1.2))
    # ===== 不同来源的红黄灯阈值颜色，避免在同一子图里混淆 =====
    # 主模型 Seg-Now 阈值（第二行子图）
    col_cls_thr_y = MODEL_CFG.get("VIS_COL_CLS_THR_Y", "gold")
    col_cls_thr_r = MODEL_CFG.get("VIS_COL_CLS_THR_R", "red")

    # OPC 阈值（第三行左轴）
    col_opc_thr_y = MODEL_CFG.get("VIS_COL_OPC_THR_Y", "tab:green")   # OPC_Y
    col_opc_thr_r = MODEL_CFG.get("VIS_COL_OPC_THR_R", "darkgreen")   # OPC_R

    # FUSED 阈值（第三行右轴）
    col_fuse_thr_y = MODEL_CFG.get("VIS_COL_FUSE_THR_Y", "orange")    # FUSE_Y
    col_fuse_thr_r = MODEL_CFG.get("VIS_COL_FUSE_THR_R", "darkred")   # FUSE_R

    # 三种灯光：主灯（分类派生）、OPC 派生、融合后
    lights_main = (
        df_test["LIGHT_MAIN"].fillna("G")
        if "LIGHT_MAIN" in df_test.columns
        else pd.Series(["G"] * len(df_test), index=df_test.index)
    )
    lights_opc = (
        df_test["LIGHT_OPC"].fillna("G")
        if "LIGHT_OPC" in df_test.columns
        else pd.Series(["G"] * len(df_test), index=df_test.index)
    )
    lights_fused = (
        df_test["LIGHT_FUSED"].fillna("G")
        if "LIGHT_FUSED" in df_test.columns
        else pd.Series(["G"] * len(df_test), index=df_test.index)
    )

    # ==================== 画图：Dashboard（更直观） ====================
    max_pts = int(MODEL_CFG.get("VIS_MAX_POINTS", 6000))
    smooth_win = int(MODEL_CFG.get("VIS_SMOOTH_WIN", 7))

    # 可选下采样（保证图不糊、不慢）
    idx = _vis_pick_idx(len(depth), max_pts)
    if idx is not None:
        depth = depth[idx]
        if ber_true is not None: ber_true = ber_true[idx]
        if p_cls is not None: p_cls = p_cls[idx]
        if seg4_pred is not None: seg4_pred = seg4_pred[idx]
        if seg4_conf is not None: seg4_conf = seg4_conf[idx]
        if opc_prob is not None: opc_prob = opc_prob[idx]
        if fused_score is not None: fused_score = fused_score[idx]
        lights_main = lights_main.iloc[idx].reset_index(drop=True)
        lights_opc = lights_opc.iloc[idx].reset_index(drop=True)
        lights_fused = lights_fused.iloc[idx].reset_index(drop=True)

    # 平滑（只平滑概率/分数，别平滑灯）
    p_cls_s = _vis_smooth(p_cls, smooth_win) if p_cls is not None else None
    opc_s = _vis_smooth(opc_prob, smooth_win) if opc_prob is not None else None
    fused_s = _vis_smooth(fused_score, smooth_win) if fused_score is not None else None

    fig, axes = plt.subplots(
        5, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 1.05, 1.05, 0.55], "hspace": 0.12}
    )

    fig.suptitle(f"{test_name} | 分类-only 决策看板（Depth Dashboard）", y=0.99, fontsize=13)

    # 统一 depth 方向：只设置一次 xlim（避免每个子图 invert_xaxis）
    axes[0].set_xlim(depth.max(), depth.min())
    # ===== 可视化配色：每条曲线用不同颜色，避免全是蓝色 =====
    col_ber = "black"         # BER 真值
    col_cls = "tab:blue"      # Seg-Now 概率
    col_seg4 = "tab:purple"   # 4 类严重度
    col_opc = "tab:green"     # OPC 概率
    col_fused = "tab:orange"  # FUSED_SCORE
    col_thr_y = "gold"        # 黄灯阈值
    col_thr_r = "red"         # 红灯阈值

    # --- 1) BER 真值（对照） ---
    ax = axes[0]
    if ber_true is not None:
        # BER 真值：黑色实线
        ax.plot(depth, ber_true, linewidth=1.5, color=col_ber, label="BER_true")
        # BER 阈值：黄/红虚线
        ax.axhline(by, linestyle="--", linewidth=1.2, color=col_thr_y, alpha=0.8, label=f"BER_Y({by:g})")
        ax.axhline(br, linestyle="--", linewidth=1.2, color=col_thr_r, alpha=0.8, label=f"BER_R({br:g})")
    else:
        ax.text(0.02, 0.5, "缺少列: BER", transform=ax.transAxes)
    ax.set_ylabel("BER (%)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    # --- 2) Seg 概率 + 严重度（右轴） ---
    ax = axes[1]
    if p_cls_s is not None:
        # Seg-Now 概率：蓝色
        ax.plot(depth, p_cls_s, linewidth=1.5, color=col_cls, label=f"{cls_col}(smooth)")
        ax.axhline(cy, linestyle="--", linewidth=1, color=col_cls_thr_y, alpha=0.8, label=f"CLS_Y({cy:g})")
        ax.axhline(cr, linestyle="--", linewidth=1, color=col_cls_thr_r, alpha=0.8, label=f"CLS_R({cr:g})")

        ax.set_ylim(0, 1.0)
    else:
        ax.text(0.02, 0.5, f"缺少列: {cls_col}", transform=ax.transAxes)
    ax.set_ylabel("Seg Prob")
    ax.grid(True, linestyle="--", alpha=0.25)

    # 严重度（如果有）叠到右轴，避免单独占一整行
    if seg4_pred is not None:
        axr = ax.twinx()
        # 4 类严重度：紫色阶梯线
        axr.step(depth, seg4_pred, where="post", linewidth=1.5, color=col_seg4, alpha=0.9, label="SEG_LEVEL4_PRED")
        axr.set_yticks([0, 1, 2, 3])
        axr.set_ylim(-0.5, 3.5)
        axr.set_ylabel("Level4")
    # 合并图例（左右轴）
    handles, labels = ax.get_legend_handles_labels()
    if seg4_pred is not None:
        h2, l2 = axr.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels, fontsize=8, loc="upper right")

    # --- 3) OPC 概率（单独一行） ---
    ax = axes[2]
    if opc_s is not None:
        ax.plot(depth, opc_s, linewidth=1.3, color=col_opc, label="OPC_PROB(smooth)")
        ax.axhline(oy, linestyle="--", linewidth=1, color=col_opc_thr_y, alpha=0.9, label=f"OPC_Y({oy:g})")
        ax.axhline(or_, linestyle="--", linewidth=1, color=col_opc_thr_r, alpha=0.9, label=f"OPC_R({or_:g})")
        ax.set_ylim(0, 1.0)
    else:
        ax.text(0.02, 0.65, "缺少列: OPC_PROB", transform=ax.transAxes)

    ax.set_ylabel("OPC Prob")
    ax.set_title("OPC 概率 + 阈值")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    # --- 4) FUSED_SCORE（单独一行） ---
    ax = axes[3]
    if fused_s is not None:
        ax.plot(depth, fused_s, linewidth=1.3, color=col_fused, alpha=0.95, label="FUSED_SCORE(smooth)")
        ax.axhline(fy, linestyle=":", linewidth=1, color=col_fuse_thr_y, alpha=0.9, label=f"FUSE_Y({fy:g})")
        ax.axhline(fr, linestyle="-.", linewidth=1, color=col_fuse_thr_r, alpha=0.9, label=f"FUSE_R({fr:g})")

        # 让 y 轴别太挤：至少覆盖阈值线
        try:
            ymax = float(np.nanmax([np.nanpercentile(fused_s, 99), fy, fr]))
            if np.isfinite(ymax):
                ax.set_ylim(0, max(1.0, ymax * 1.05))
        except Exception:
            pass
    else:
        ax.text(0.02, 0.65, "缺少列: FUSED_SCORE", transform=ax.transAxes)

    ax.set_ylabel("Fused")
    ax.set_title("融合分数 + 阈值")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    # --- 5) 灯带（Main/OPC/Fused）：细线轨道 ---
    ax = axes[4]
    from matplotlib.patches import Patch
    lw = float(MODEL_CFG.get("VIS_LIGHT_LW", 3.0))
    _plot_lights_as_lines(ax, depth, lights_main, y=0, lw=lw)
    _plot_lights_as_lines(ax, depth, lights_opc, y=1, lw=lw)
    _plot_lights_as_lines(ax, depth, lights_fused, y=2, lw=lw)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Main", "OPC", "Fused"])
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Lights")
    ax.grid(False)

    # 灯颜色图例
    ax.legend(
        handles=[Patch(color="green", label="G"), Patch(color="gold", label="Y"), Patch(color="red", label="R")],
        fontsize=8, loc="upper right"
    )

    # 图右上角加一个小摘要（更好读）
    def _cnt(s):
        vc = s.astype(str).value_counts()
        return f"G/Y/R={int(vc.get('G', 0))}/{int(vc.get('Y', 0))}/{int(vc.get('R', 0))}"

    ax.text(
        0.01, 0.95,
        f"Main {_cnt(lights_main)} | OPC {_cnt(lights_opc)} | Fused {_cnt(lights_fused)}",
        transform=ax.transAxes, va="top", fontsize=9
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.08, hspace=0.22)
    fig_path = os.path.join(outdir, f"{test_name}_depth_profile.png")
    fig.savefig(fig_path, dpi=600)
    plt.close(fig)
    print(f"[INFO] 测试井深度剖面图已保存(新Dashboard): {fig_path}")
    # 额外保存一张“分类-only 全链路诊断图”
    if MODEL_CFG.get("VIS_SAVE_CLS_DEBUG", True):
        try:
            plot_classification_inference_debug(
                df_fold=df_test,
                outdir=outdir,
                name=test_name,
                dep_col=dep_col,
                cfg=MODEL_CFG,
            )
        except Exception as e:
            print(f"[VIS][WARN] plot_classification_inference_debug 失败: {e}")


def plot_classification_inference_debug(df_fold: pd.DataFrame,
                                        outdir: str,
                                        name: str,
                                        dep_col: str = None,
                                        cfg: dict = None):
    """
    分类-only 全链路诊断图（同屏确认“为什么触发黄/红”）：
      (1) BER_true + 参考阈值线（对照）
      (2) Seg-Now 概率 + 阈值（门控/预警）
      (3) 严重度 4 分类 pred + conf（趋势）
      (4) OPC_PROB + 阈值（工况风险）
      (5) FUSED_SCORE + 阈值（联合决策）
      (6) 灯轨道：LIGHT_MAIN / LIGHT_OPC / LIGHT_FUSED
    """
    if cfg is None:
        cfg = MODEL_CFG

    os.makedirs(outdir, exist_ok=True)

    # 深度列
    if dep_col is None:
        dep_col = _choose_depth_column(df_fold, CLEAN_CFG["DEP_COL_CANDIDATES"])
    depth = df_fold[dep_col].values

    # 关键列
    ber_true = df_fold["BER"].values if "BER" in df_fold.columns else None

    cls_col = cfg.get("MAIN_CLS_PROB_COL", "SEG_PROB_NOW")
    p_cls = None
    if cls_col in df_fold.columns:
        p_cls = pd.to_numeric(df_fold[cls_col], errors="coerce").values
    elif "MAIN_PROB_CLS" in df_fold.columns:
        p_cls = pd.to_numeric(df_fold["MAIN_PROB_CLS"], errors="coerce").values

    seg4_pred = pd.to_numeric(df_fold["SEG_LEVEL4_PRED"], errors="coerce").values if "SEG_LEVEL4_PRED" in df_fold.columns else None
    seg4_conf = pd.to_numeric(df_fold["SEG_LEVEL4_CONF"], errors="coerce").values if "SEG_LEVEL4_CONF" in df_fold.columns else None

    opc_prob = pd.to_numeric(df_fold["OPC_PROB"], errors="coerce").values if "OPC_PROB" in df_fold.columns else None
    fused_score = pd.to_numeric(df_fold["FUSED_SCORE"], errors="coerce").values if "FUSED_SCORE" in df_fold.columns else None

    # 灯轨道
    lights_main = df_fold["LIGHT_MAIN"].fillna("G") if "LIGHT_MAIN" in df_fold.columns else pd.Series(["G"] * len(df_fold))
    lights_opc = df_fold["LIGHT_OPC"].fillna("G") if "LIGHT_OPC" in df_fold.columns else pd.Series(["G"] * len(df_fold))
    lights_fused = df_fold["LIGHT_FUSED"].fillna("G") if "LIGHT_FUSED" in df_fold.columns else pd.Series(["G"] * len(df_fold))

    # 阈值
    by = float(cfg.get("BER_THRESH_YELLOW", 8.0))
    br = float(cfg.get("BER_THRESH_RED", 15.0))
    cy = float(cfg.get("MAIN_CLS_THRESH_Y", 0.5))
    cr = float(cfg.get("MAIN_CLS_THRESH_R", 0.8))
    oy = float(cfg.get("OPC_THRESH_YELLOW", 0.3))
    or_ = float(cfg.get("OPC_THRESH_RED", 0.6))
    fy = float(cfg.get("FUSE_THRESH_Y", 0.6))
    fr = float(cfg.get("FUSE_THRESH_R", 1.2))

    # ===== 统计摘要（便于快速定位“为什么全黄/全红”）=====
    def _count_lights(s: pd.Series):
        vc = s.astype(str).value_counts()
        return {"G": int(vc.get("G", 0)), "Y": int(vc.get("Y", 0)), "R": int(vc.get("R", 0))}

    summary = {
        "n_points": int(len(df_fold)),
        "cls_col": cls_col if cls_col in df_fold.columns else ("MAIN_PROB_CLS" if "MAIN_PROB_CLS" in df_fold.columns else "NA"),
        "main_mode": str(cfg.get("MAIN_LIGHT_MODE", "cls")),
        "main_light": _count_lights(lights_main),
        "opc_light": _count_lights(lights_opc),
        "fused_light": _count_lights(lights_fused),
    }
    if p_cls is not None:
        summary.update({
            "cls_prob_mean": float(np.nanmean(p_cls)),
            "cls_prob_p50": float(np.nanmedian(p_cls)),
            "cls_prob_p95": float(np.nanpercentile(p_cls, 95)),
        })
    if opc_prob is not None:
        summary.update({
            "opc_prob_mean": float(np.nanmean(opc_prob)),
            "opc_prob_p50": float(np.nanmedian(opc_prob)),
            "opc_prob_p95": float(np.nanpercentile(opc_prob, 95)),
        })
    if fused_score is not None:
        summary.update({
            "fused_score_mean": float(np.nanmean(fused_score)),
            "fused_score_p50": float(np.nanmedian(fused_score)),
            "fused_score_p95": float(np.nanpercentile(fused_score, 95)),
        })
    if seg4_pred is not None:
        vc = pd.Series(seg4_pred).value_counts(dropna=False).to_dict()
        summary["seg4_pred_counts"] = {str(k): int(v) for k, v in vc.items()}

    try:
        import json as _json
        with open(os.path.join(outdir, f"{name}_cls_debug_summary.json"), "w", encoding="utf-8") as f:
            _json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[DEBUG-VIS][WARN] 保存 debug_summary 失败: {e}")

    # ==================== 画图（分类-only） ====================
    fig, axes = plt.subplots(
        6, 1, figsize=(16, 13), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.05, 1.00, 1.05, 1.05, 0.28], "hspace": 0.220}
    )
    # 给右侧留空间放图例（避免遮挡曲线），同时拉开上下子图间距
    fig.subplots_adjust(left=0.07, right=0.82, top=0.95, bottom=0.06)

    # 统一配色/线型：显式指定，避免“每个子图都是默认蓝色”
    COL = {
        "ber": "black",
        "prob_cls": "tab:blue",
        "seg4_pred": "tab:cyan",
        "seg4_conf": "tab:orange",
        "opc": "tab:green",
        "fused": "tab:purple",
        "thr_y": "goldenrod",
        "thr_r": "firebrick",
    }
    LEG_KW = dict(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.9,
        fancybox=True
    )
    # 1) BER_true
    ax = axes[0]
    if ber_true is not None:
        ax.plot(depth, ber_true, label="BER_true", linewidth=1)
        ax.axhline(by, linestyle="--", linewidth=1, alpha=0.7, label=f"BER_Y({by:g})")
        ax.axhline(br, linestyle="--", linewidth=1, alpha=0.7, label=f"BER_R({br:g})")
    else:
        ax.text(0.02, 0.5, "缺少列: BER", transform=ax.transAxes)
    ax.set_ylabel("BER (%)")
    ax.set_title(f"{name} - Debug(1/6) BER 真值（对照）")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # 2) 分类概率
    ax = axes[1]
    if p_cls is not None:
        ax.plot(depth, p_cls, label=cls_col, linewidth=1)
        ax.axhline(cy, linestyle="--", linewidth=1, alpha=0.7, label=f"CLS_Y({cy:g})")
        ax.axhline(cr, linestyle="--", linewidth=1, alpha=0.7, label=f"CLS_R({cr:g})")
    else:
        ax.text(0.02, 0.5, f"缺少分类概率列: {cls_col}", transform=ax.transAxes)
    ax.set_ylabel("Cls Prob")
    ax.set_title(f"{name} - Debug(2/6) Seg-Now 概率 + 阈值")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # 3) 严重度 4 分类
    ax = axes[2]
    ax2 = None

    if seg4_pred is not None:
        ax.step(depth, seg4_pred, where="post", label="SEG_LEVEL4_PRED",
                linewidth=1.2, color=COL["seg4_pred"])
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["0(G)", "1(B)", "2(Y)", "3(R)"])
        ax.set_ylim(-0.5, 3.5)
    else:
        ax.text(0.02, 0.5, "缺少列: SEG_LEVEL4_PRED", transform=ax.transAxes)

    if seg4_conf is not None:
        ax2 = ax.twinx()
        ax2.plot(depth, seg4_conf, label="SEG_LEVEL4_CONF",
                 linewidth=1.0, alpha=0.85, color=COL["seg4_conf"])
        ax2.set_ylabel("Conf")
        ax2.set_ylim(0, 1.0)

    ax.set_ylabel("Level4")
    ax.set_title(f"{name} - Debug(3/6) 严重度 4 分类预测 + 置信度")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.invert_xaxis()

    # 合并左右轴图例，并放到右侧图外
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if ax2 is not None else ([], []))
    ax.legend(h1 + h2, l1 + l2, **LEG_KW)

    # 4) OPC 概率
    ax = axes[3]
    if opc_prob is not None:
        ax.plot(depth, opc_prob, label="OPC_PROB", linewidth=1)
        ax.axhline(oy, linestyle="--", linewidth=1, alpha=0.7, label=f"OPC_Y({oy:g})")
        ax.axhline(or_, linestyle="--", linewidth=1, alpha=0.7, label=f"OPC_R({or_:g})")
    else:
        ax.text(0.02, 0.5, "缺少 OPC_PROB", transform=ax.transAxes)
    ax.set_ylabel("OPC Prob")
    ax.set_title(f"{name} - Debug(4/6) OPC 概率 + 阈值")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # 5) 融合分数
    ax = axes[4]
    if fused_score is not None:
        ax.plot(depth, fused_score, label="FUSED_SCORE", linewidth=1)
        ax.axhline(fy, linestyle="--", linewidth=1, alpha=0.7, label=f"FUSE_Y({fy:g})")
        ax.axhline(fr, linestyle="--", linewidth=1, alpha=0.7, label=f"FUSE_R({fr:g})")
    else:
        ax.text(0.02, 0.5, "缺少 FUSED_SCORE", transform=ax.transAxes)
    ax.set_ylabel("Fused")
    ax.set_title(f"{name} - Debug(5/6) 融合分数 + 阈值")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # 6) 灯轨道（细线轨道，替代 imshow 色带）
    ax = axes[5]
    from matplotlib.patches import Patch

    lw = float(cfg.get("VIS_LIGHT_LW", 3.0))
    _plot_lights_as_lines(ax, depth, lights_main, y=0, lw=lw)
    _plot_lights_as_lines(ax, depth, lights_opc, y=1, lw=lw)
    _plot_lights_as_lines(ax, depth, lights_fused, y=2, lw=lw)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Main(Cls)", "OPC", "Fused"])
    ax.set_ylim(-0.5, 2.5)
    ax.set_ylabel("Lights")
    ax.set_xlabel("Depth")
    ax.set_title(f"{name} - Debug(6/6) 灯轨道 (R/Y/G)")
    ax.grid(False)
    ax.invert_xaxis()

    ax.legend(
        handles=[Patch(color="green", label="G"), Patch(color="gold", label="Y"), Patch(color="red", label="R")],
        loc="upper right",
        fontsize=8
    )

    fig_path = os.path.join(outdir, f"{name}_cls_debug.png")
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[DEBUG-VIS] 分类推理诊断图已保存: {fig_path}")



def main():
    # ===== 固定随机种子，保证同一数据多次训练结果尽量一致 =====
    set_global_seed(42)
    # ===== 设备选择：优先 CUDA =====
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] 检测到 CUDA，使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] 未检测到 CUDA，使用 CPU 进行训练")

    # ===== 输出目录设置 =====
    out_root = RUN_CFG["OUTDIR_ROOT"]
    os.makedirs(out_root, exist_ok=True)

    # 现在不再依赖 TEST_WELL 的 name，多井测试时 RUN_NAME 请手动在 RUN_CFG 里设置
    run_name = RUN_CFG.get("RUN_NAME") or "ber_timeseries_run"
    outdir = os.path.join(out_root, run_name)
    os.makedirs(outdir, exist_ok=True)
    print(f"[INFO] 本次运行输出目录: {outdir}")

    # ===== 1) 训练井预处理 =====
    # ===== 1) 训练井预处理 =====
    use_df_cache = RUN_CFG.get("USE_DF_CACHE", True)

    train_dfs = None
    if use_df_cache:
        train_dfs = load_cached_train_dfs(outdir)

    if train_dfs is None:
        print("[CACHE] 未找到训练井 df 缓存，开始从原始数据构建 train_dfs ...")
        train_dfs = build_train_well_dfs()
        if use_df_cache:
            cache_train_dfs(train_dfs, outdir)
    else:
        print(f"[CACHE] 使用缓存的训练井 df，共 {len(train_dfs)} 口井")

    # === 新增：分析各口井的未来扩径风险标签分布（B 方案标签体检） ===
    try:
        analyze_seg_future_labels(
            train_dfs,
            outdir=outdir,
            depth_col="DEPTH",
            time_col="WELLDATETIME",
            future_label_col="SEG_FUTURE_LABEL",
            seg_col="BER_IS_SEG",
        )
    except Exception as e:
        print(f"[ANALYSIS][WARN] 分析 SEG_FUTURE_LABEL 失败: {e}")
    if not train_dfs:
        print("[ERROR] 没有成功加载任何训练井，请先在 TRAIN_WELLS 中配置路径。")
        return

    # ===== 1.5) 扩径段模式分析 =====
    print("[INFO] 分析扩径段时间模式...")
    lookahead_steps = MODEL_CFG.get("LOOKAHEAD_STEPS", 60)
    for i, df in enumerate(train_dfs):
        well_name = list(TRAIN_WELLS.keys())[i]
        print(f"[ANALYSIS] 分析训练井 {well_name} 的扩径段模式")
        transitions = analyze_segmentation_patterns(df, lookahead_steps)

        # 可选：保存分析结果
        if transitions:
            analysis_df = pd.DataFrame(transitions)
            analysis_path = os.path.join(outdir, f"segmentation_analysis_{well_name}.csv")
            analysis_df.to_csv(analysis_path, index=False, encoding="utf-8-sig")
            print(f"[INFO] 扩径段分析结果已保存: {analysis_path}")
    # ===== 2) 特征列 + 标准化 =====
    print("[INFO] 选择特征列并拟合标准化...")
    feature_cols = select_feature_columns(train_dfs[0])
    feature_cols = sorted(feature_cols)
    # 保留一份“基础特征列”，标准化和 Seg-Now / 四档分类都只在这部分上做
    base_feature_cols = list(feature_cols)
    # === 只用“训练时间段”拟合标准化器，避免时间泄露 ===
    time_col = CLEAN_CFG.get("TIME_COL", "WELLDATETIME")
    seq_len = MODEL_CFG["SEQ_LEN"]
    val_ratio = float(MODEL_CFG.get("VAL_SPLIT", 0.2))

    scaler_fit_dfs = []
    for df in train_dfs:
        df_tmp = df.sort_values(time_col).reset_index(drop=True)
        n_rows = len(df_tmp)
        if n_rows <= seq_len * 2:
            # 太短的井：全部作为训练段参与拟合
            scaler_fit_dfs.append(df_tmp)
        else:
            split_row = int(n_rows * (1.0 - val_ratio))
            scaler_fit_dfs.append(df_tmp.iloc[:split_row].copy())

    scaler = fit_feature_scaler(scaler_fit_dfs, base_feature_cols)
    train_dfs = [apply_feature_scaler(df, base_feature_cols, scaler) for df in train_dfs]
    print(f"[INFO] 特征维度: {len(base_feature_cols)}")

    if RUN_CFG.get("SAVE_SCALERS", True):
        scaler_path = os.path.join(outdir, "feature_scaler.pkl")
        joblib.dump({"scaler": scaler, "feature_cols": base_feature_cols}, scaler_path)
        print(f"[INFO] 特征标准化器已保存到: {scaler_path}")



    # ===== 3.5) 可选：先训练一个“当前扩径段”二分类模型（Seg-Now） =====
    seg_model = None
    seg_level4_model = None
    if MODEL_CFG.get("SEG_CLS_ENABLE", False):
        # 为 Seg-Now 单独构造一份 DataLoader：
        # - 不做前看（LOOKAHEAD_STEPS = 0）
        # - 不只限于扩径段附近（LOOKAHEAD_ONLY_IN_SEG = False）
        # - 不需要 margin（LOOKAHEAD_MARGIN = 0）
        seg_model_cfg = dict(MODEL_CFG)
        seg_model_cfg["LOOKAHEAD_STEPS"] = 0
        seg_model_cfg["LOOKAHEAD_ONLY_IN_SEG"] = False
        seg_model_cfg["LOOKAHEAD_MARGIN"] = 0
        # Stage1 gate：训练“未来黄灯(>=BER_THRESH_YELLOW)”二分类（列名仍叫 SEG_FUTURE_LABEL）
        seg_model_cfg["SEG_LABEL_COL"] = "SEG_FUTURE_LABEL"

        seg_model_cfg["OVERSAMPLE_FACTOR"] = 1.2  # ★建议 1.0~1.5，别再 3.0 起步
        seg_model_cfg["SEG_CLS_POS_WEIGHT"] = "auto"
        # ★ 这里原来漏了：先构造 seg_train_loader / seg_val_loader
        seg_train_loader, seg_val_loader = build_dataloaders(
            train_dfs,
            base_feature_cols,
            seg_model_cfg,
        )

        seg_model = run_seg_cls_training(
            train_loader=seg_train_loader,
            val_loader=seg_val_loader,
            feature_cols=base_feature_cols,  # 和 DataLoader 保持一致
            device=device,
            outdir=outdir,
            model_cfg=seg_model_cfg,  # 注意这里传的是 seg_model_cfg
        )


        # ===== 3.6) 可选：训练“扩径严重度 4 分类”模型（只在扩径核心区样本） =====
        seg_level4_model = None
        if MODEL_CFG.get("SEG_LEVEL4_ENABLE", False):
            seg_level4_model = run_seg_level4_training(
                train_dfs=train_dfs,
                feature_cols=base_feature_cols,
                device=device,
                outdir=outdir,
                model_cfg=MODEL_CFG,
            )

        if RUN_CFG.get("SEG_CLS_DO_DIAG", True):
            diag_report, _ = classification_diagnostic_analysis(
                model=seg_model,
                train_loader=seg_train_loader,
                val_loader=seg_val_loader,
                device=device,
                outdir=outdir,
                recall_floor=seg_model_cfg.get("SEG_RECALL_FLOOR", 0.8),
            )

            best_floor = (diag_report.get("threshold_analysis", {}) or {}).get("best_with_recall_floor", None)
            if best_floor is not None:
                thr_y = float(best_floor["threshold"])
                thr_r = float(min(0.99, thr_y + 0.15))  # 红灯更保守一点（你也可改成 +0.10/+0.20）

                MODEL_CFG["MAIN_CLS_THRESH_Y"] = thr_y
                MODEL_CFG["MAIN_CLS_THRESH_R"] = thr_r
                MODEL_CFG["GATE_STAGE1_THR"] = thr_y  # gate 也用同一个阈值
                print(
                    f"[CFG][AUTO] MAIN_CLS_THRESH_Y={thr_y:.2f}, MAIN_CLS_THRESH_R={thr_r:.2f}, GATE_STAGE1_THR={thr_y:.2f}")

    # ===== 6) OPC 分支 =====
    print("[INFO] 训练 OPC 因果风险分支...")
    opc_feature_cols = [c for c in ["WOB", "RPM", "TOR", "SPP", "FLOWIN", "FLOWOUT", "HKLD",
                                    "dWOB_dt", "dRPM_dt", "dTOR_dt", "dSPP_dt"]
                        if c in train_dfs[0].columns]
    if opc_feature_cols:
        opc_clf, opc_feature_cols = train_opc_model(
            train_dfs,
            opc_feature_cols,
            ber_col="BER",
            ber_thresh=MODEL_CFG["BER_THRESH_YELLOW"],
            val_ratio=MODEL_CFG.get("VAL_SPLIT", 0.2),
            time_col=CLEAN_CFG.get("TIME_COL", "WELLDATETIME"),
        )

        if RUN_CFG.get("SAVE_OPC", True):
            opc_path = os.path.join(outdir, "opc_model.pkl")
            joblib.dump({"opc_clf": opc_clf,
                         "opc_feature_cols": opc_feature_cols},
                        opc_path)
            print(f"[INFO] OPC 模型已保存到: {opc_path}")
    else:
        opc_clf = None
        print("[WARN] 未找到 OPC 特征列，跳过 OPC 模型训练。")

    # ===== 7) PCA 解释 =====
    dynamic_cols_for_pca = [c for c in ["WOB", "RPM", "TOR", "SPP", "FLOWIN", "FLOWOUT", "HKLD"]
                            if c in train_dfs[0].columns]
    if dynamic_cols_for_pca:
        pca_scaler, pca_model = fit_pca_on_dynamic(
            train_dfs,
            dynamic_cols_for_pca,
            n_components=MODEL_CFG["PCA_N_COMPONENTS"],
        )
        if RUN_CFG.get("SAVE_PCA", True):
            pca_path = os.path.join(outdir, "pca_explainer.pkl")
            joblib.dump({
                "pca_scaler": pca_scaler,
                "pca_model": pca_model,
                "dynamic_cols": dynamic_cols_for_pca,
            }, pca_path)
            print(f"[INFO] PCA 解释器已保存到: {pca_path}")
    else:
        pca_scaler, pca_model = None, None
        print("[WARN] 未找到动态特征列，跳过 PCA 拟合。")

    # ===== 8) 测试井推理 =====
    if not TEST_WELLS:
        print("[INFO] 未配置 TEST_WELLS，跳过测试井推理。")
        return

    for test_name, test_cfg in TEST_WELLS.items():
        log_path = test_cfg.get("log")
        form_path = test_cfg.get("form")
        cal_path = test_cfg.get("cal")

        if isinstance(log_path, str):
            log_paths = [log_path]
        else:
            log_paths = list(log_path or [])

        valid_paths = []
        for p in log_paths:
            exists = os.path.exists(p)
            print(f"[TEST-LOG] {test_name}: {p}  存在={exists}")
            if exists:
                valid_paths.append(p)
            else:
                print(f"[WARN] 测试井 {test_name} 录井路径不存在，已跳过: {p}")

        if not valid_paths:
            print(f"[WARN] 测试井 {test_name} 没有可用录井文件，跳过。")
            continue

        use_df_cache = RUN_CFG.get("USE_DF_CACHE", True)

        # --- 先尝试从缓存恢复“基础特征 df” ---
        df_test_base = None
        if use_df_cache:
            df_test_base = load_cached_test_base_df(outdir, test_name)

        if df_test_base is None:
            print(f"[CACHE] 未找到测试井 {test_name} 的基础特征缓存，开始从原始文件构建...")
            print(f"[WELL] 处理测试井 {test_name}")

            # 拼接多个月录井
            df_logs = []
            for p in valid_paths:
                print(f"[INFO] 打开测试井Excel: {p}")
                df_i = pd.read_excel(p)
                print(f"[INFO] 读取完成: {p}  行数={len(df_i)}  列数={df_i.shape[1]}")
                df_logs.append(df_i)
            df_log_raw = pd.concat(df_logs, ignore_index=True)

            df_form = pd.read_excel(form_path)
            df_cal = pd.read_csv(cal_path, **CAL_TXT_READ_CFG)

            df_clean = clean_time_series_per_well(df_log_raw, well_name=test_name)
            dynamic_cols = [c for c in ["WOB", "RPM", "TOR", "SPP", "FLOWIN", "FLOWOUT", "HKLD"]
                            if c in df_clean.columns]
            if dynamic_cols:
                df_clean = add_derived_features(df_clean, dynamic_cols=dynamic_cols)

            # ★ 这里得到“基础特征 df”：已合并地层+CAL，并加了交互/趋势
            df_test_base = merge_formation_and_cal_to_time(df_clean, df_form, df_cal)
            df_test_base = add_interaction_features(df_test_base)
            df_test_base = add_trend_features(df_test_base)

            if use_df_cache:
                cache_test_base_df(df_test_base, outdir, test_name)
        else:
            print(f"[CACHE] 测试井 {test_name} 使用缓存的基础特征 df")

        # --- 从基础特征 df 出发，做标准化 + Seg-Now 分类 + 回归预测 + OPC + 灯光 ---
        df_test = df_test_base.copy()
        # 注意：标准化器只针对基础特征列 base_feature_cols，SEG_PROB_NOW 等派生特征不参与标准化
        df_test = apply_feature_scaler(df_test, base_feature_cols, scaler)

        # 1) Seg-Now 分类推理：生成“未来扩径风险概率”列
        main_cls_prob_col = MODEL_CFG.get("MAIN_CLS_PROB_COL", "SEG_PROB_NOW")
        if seg_model is not None:
            try:
                df_test = apply_seg_classifier_on_df(
                    df_test,
                    seg_model=seg_model,
                    feature_cols=base_feature_cols,  # Seg-Now 只看基础特征
                    seg_model_cfg=seg_model_cfg,
                    device=device,
                    out_col=main_cls_prob_col,
                )
            except Exception as e:
                print(f"[SEG][WARN] 测试井 {test_name} 应用 Seg-Now 分类器失败: {e}")
                df_test[main_cls_prob_col] = np.nan
        else:
            # 没训练或没启用 Seg-Now 时，分类概率留空
            df_test[main_cls_prob_col] = np.nan
        # ===== [STAGE2 ANCHOR] 1.5) Seg4 推理：生成 SEG_LEVEL4_PRED / CONF =====
        if MODEL_CFG.get("SEG_LEVEL4_ENABLE", False) and (seg_level4_model is not None):
            try:
                df_test = apply_seg_level4_classifier_on_df(
                    df_test,
                    seg4_model=seg_level4_model,
                    feature_cols=base_feature_cols,  # 与 seg4 训练保持一致（你 seg4 训练用的是 feature_cols）
                    seg4_seq_len=int(MODEL_CFG.get("SEG_LEVEL4_SEQ_LEN", 200)),
                    seg4_batch_size=int(MODEL_CFG.get("SEG_LEVEL4_BATCH_SIZE", 128)),
                    device=device,
                    out_pred_col="SEG_LEVEL4_PRED",
                    out_conf_col="SEG_LEVEL4_CONF",
                )
            except Exception as e:
                print(f"[SEG4][WARN] 测试井 {test_name} 应用 4 分类器失败: {e}")
                df_test["SEG_LEVEL4_PRED"] = np.nan
                df_test["SEG_LEVEL4_CONF"] = np.nan


        # 3) OPC 模块
        if opc_clf is not None and opc_feature_cols:
            df_test = apply_opc_model(df_test, opc_clf, opc_feature_cols)
        else:
            df_test["OPC_PROB"] = np.nan

        # 4) 红黄灯：主灯由分类结果派生（assign_traffic_lights 内部看 MAIN_LIGHT_MODE="cls"）
        df_test = assign_traffic_lights(
            df_test,
            opc_prob_col="OPC_PROB",
            cfg=MODEL_CFG,
        )

        # 红黄灯区域参数分析 + 调控建议（测试井）
        if dynamic_cols_for_pca:
            analyze_risk_segments_and_suggest_controls(
                df_test,
                dynamic_cols=dynamic_cols_for_pca,
                outdir=os.path.join(outdir, f"test_{test_name}"),
                name=test_name,
                cfg=MODEL_CFG,
                light_col="LIGHT_FUSED",
            )

        # PCA 解释前几个红/黄灯点
        if pca_scaler is not None and pca_model is not None and dynamic_cols_for_pca:
            risk_idx = df_test.index[df_test["LIGHT_FUSED"] != "G"].tolist()
            print(f"[INFO] 测试井 {test_name} 联合红黄灯非绿点数: {len(risk_idx)}")
            for idx in risk_idx[:10]:
                ex = explain_risk_point(
                    df_test,
                    idx,
                    dynamic_cols_for_pca,
                    pca_scaler,
                    pca_model,
                    top_k=5,
                )
                dep_col = _choose_depth_column(df_test, CLEAN_CFG["DEP_COL_CANDIDATES"])
                # 基于 PCA 的红/黄灯段雷达风险图
                plot_risk_segments_pca_radar(
                    df_test,
                    dynamic_cols=dynamic_cols_for_pca,
                    scaler=pca_scaler,
                    pca=pca_model,
                    outdir=os.path.join(outdir, f"test_{test_name}"),
                    name=test_name,
                    light_col="LIGHT_FUSED",
                )

                print(
                    f"  深度={float(df_test.iloc[idx][dep_col]):.2f} "
                    f"灯={df_test.iloc[idx]['LIGHT_FUSED']} 主导因子={ex}"
                )


        if "OPC_PROB" in df_test.columns:
            print(f"[DEBUG] {test_name} OPC_PROB 分布：")
            print(df_test["OPC_PROB"].describe())

        # 保存测试结果到 outdir/test_井名/
        test_subdir = os.path.join(outdir, f"test_{test_name}")
        os.makedirs(test_subdir, exist_ok=True)
        out_csv = os.path.join(test_subdir, f"test_{test_name}_prediction_with_lights.csv")
        df_test.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] 测试井 {test_name} 预测结果已保存: {out_csv}")
        plot_test_results(df_test, test_subdir, test_name)
    # ===== 9) 在每口训练井上导出预测结果 & 验证段可视化 =====
    well_names = list(TRAIN_WELLS.keys())
    val_split = MODEL_CFG["VAL_SPLIT"]

    for wi, (well_name, df_train_scaled) in enumerate(zip(well_names, train_dfs)):
        print(f"[TRAIN-WELL] 生成训练井 {well_name} 的预测结果与验证段可视化...")

        # 1) Seg-Now 分类推理（在训练井上也生成分类概率，方便分析）
        df_well = df_train_scaled.copy()
        main_cls_prob_col = MODEL_CFG.get("MAIN_CLS_PROB_COL", "SEG_PROB_NOW")
        if seg_model is not None:
            try:
                df_well = apply_seg_classifier_on_df(
                    df_well,
                    seg_model=seg_model,
                    feature_cols=base_feature_cols,  # Seg-Now 只看基础特征
                    seg_model_cfg=seg_model_cfg,
                    device=device,
                    out_col=main_cls_prob_col,
                )
            except Exception as e:
                print(f"[SEG][WARN] 训练井 {well_name} 应用 Seg-Now 分类器失败: {e}")
                df_well[main_cls_prob_col] = np.nan
        else:
            df_well[main_cls_prob_col] = np.nan
        # ===== [STAGE2 ANCHOR] 训练井也跑 Seg4 推理（可选但建议）=====
        if MODEL_CFG.get("SEG_LEVEL4_ENABLE", False) and (seg_level4_model is not None):
            try:
                df_well = apply_seg_level4_classifier_on_df(
                    df_well,
                    seg4_model=seg_level4_model,
                    feature_cols=base_feature_cols,
                    seg4_seq_len=int(MODEL_CFG.get("SEG_LEVEL4_SEQ_LEN", 200)),
                    seg4_batch_size=int(MODEL_CFG.get("SEG_LEVEL4_BATCH_SIZE", 128)),
                    device=device,
                    out_pred_col="SEG_LEVEL4_PRED",
                    out_conf_col="SEG_LEVEL4_CONF",
                )
            except Exception as e:
                print(f"[SEG4][WARN] 训练井 {well_name} 应用 4 分类器失败: {e}")
                df_well["SEG_LEVEL4_PRED"] = np.nan
                df_well["SEG_LEVEL4_CONF"] = np.nan
        # ===== [STAGE2 ANCHOR END] =====

        # 3) OPC + 联合红黄灯（和测试井同一套逻辑）
        if opc_clf is not None and opc_feature_cols:
            df_well = apply_opc_model(df_well, opc_clf, opc_feature_cols)
        else:
            df_well["OPC_PROB"] = np.nan

        df_well = assign_traffic_lights(
            df_well,
            opc_prob_col="OPC_PROB",
            cfg=MODEL_CFG,
        )

        # 3) 按时间排序，近似地把“最后 val_split 的时间段”当作验证段
        time_col = CLEAN_CFG["TIME_COL"]
        df_well = df_well.sort_values(time_col).reset_index(drop=True)

        n_total = len(df_well)
        n_val_rows = max(1, int(n_total * val_split))
        val_start = n_total - n_val_rows

        df_train_part = df_well.iloc[:val_start].copy()
        df_val_part = df_well.iloc[val_start:].copy()

        # 4) 保存整口井预测结果
        train_subdir = os.path.join(outdir, f"train_{well_name}")
        os.makedirs(train_subdir, exist_ok=True)

        csv_all = os.path.join(train_subdir, f"{well_name}_trainval_prediction_with_lights.csv")
        df_well.to_csv(csv_all, index=False, encoding="utf-8-sig")
        print(f"[INFO] 训练井 {well_name} 全井预测结果已保存: {csv_all}")

        # 5) 保存验证段预测结果（只包含后 val_split 时段）
        csv_val = os.path.join(train_subdir, f"{well_name}_val_prediction_with_lights.csv")
        df_val_part.to_csv(csv_val, index=False, encoding="utf-8-sig")
        print(f"[INFO] 训练井 {well_name} 验证段预测结果已保存: {csv_val}")
        # 全井范围的红黄灯区域参数分析
        if dynamic_cols_for_pca:
            analyze_risk_segments_and_suggest_controls(
                df_well,
                dynamic_cols=dynamic_cols_for_pca,
                outdir=train_subdir,
                name=f"{well_name}_trainval",
                cfg=MODEL_CFG,
                light_col="LIGHT_FUSED",
            )

        # 仅针对验证段的红黄灯区域做一次单独分析
        if dynamic_cols_for_pca:
            analyze_risk_segments_and_suggest_controls(
                df_val_part,
                dynamic_cols=dynamic_cols_for_pca,
                outdir=train_subdir,
                name=f"{well_name}_val",
                cfg=MODEL_CFG,
                light_col="LIGHT_FUSED",
            )
        # 验证段红/黄灯段 PCA 雷达图
        if dynamic_cols_for_pca and pca_scaler is not None and pca_model is not None:
            plot_risk_segments_pca_radar(
                df_val_part,
                dynamic_cols=dynamic_cols_for_pca,
                scaler=pca_scaler,
                pca=pca_model,
                outdir=train_subdir,
                name=f"{well_name}_val",
                light_col="LIGHT_FUSED",
            )

        # 6) 对验证段单独出一张深度剖面图
        #    （复用 plot_test_results，只是名字里标一下 _val）
        plot_test_results(df_val_part, train_subdir, f"{well_name}_val")


if __name__ == "__main__":
    main()
