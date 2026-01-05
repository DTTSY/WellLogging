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
# 2) 数据预处理（清洗/特征/标签/数据集构建）
# ============================

from .data_io import (
    RUN_CFG, CLEAN_CFG, MERGE_CFG, SEG_CFG, MODEL_CFG, AUG_CFG,
    TRAIN_WELLS, TEST_WELLS, CAL_TXT_READ_CFG,
    load_formation_xlsx, load_cal_file, _ensure_time_column, load_dynamic_log_excels,
    cache_train_dfs, load_cached_train_dfs,
)

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

def build_train_well_dfs() -> List[pd.DataFrame]:
    train_dfs: List[pd.DataFrame] = []

    for well, cfg in TRAIN_WELLS.items():
        print(f"[WELL] 处理训练井 {well}")

        # ========= 录井数据：IO 模块读取（兼容单文件/多文件） =========
        df_log_raw = load_dynamic_log_excels(cfg["log"], well_name=well, add_src_file=True, verbose=True)
        print(f"[INFO] 井 {well} 录井合并后行数={len(df_log_raw)}")


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

            from .viz import plot_ber_segments_for_well
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

