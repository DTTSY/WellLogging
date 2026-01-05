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
# 5) 推理（含融合/评估/主流程 main）
# ============================

from .data_io import RUN_CFG, CLEAN_CFG, MERGE_CFG, SEG_CFG, MODEL_CFG, TRAIN_WELLS, TEST_WELLS
from .data_io import load_dynamic_log_excels, load_formation_xlsx, load_cal_file, cache_test_base_df, load_cached_test_base_df
from .preprocess import (
    _choose_depth_column, clean_time_series_per_well, add_derived_features, add_trend_features,
    add_interaction_features, select_feature_columns, merge_formation_and_cal_to_time,
    fit_feature_scaler, apply_feature_scaler,
    build_future_ber_threshold_labels, build_seg_now_labels, add_seg_level4_label, add_time_pre_alert,
)
from .train_engine import (
    set_global_seed, run_seg_cls_training, run_seg_level4_training,
    train_opc_model, fit_pca_on_dynamic,
)
from .viz import plot_test_results, plot_classification_inference_debug, plot_risk_segments_pca_radar

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
    from .viz import _plot_cm
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

            # 拼接多个月录井（IO 模块）
            df_log_raw = load_dynamic_log_excels(valid_paths, well_name=test_name, add_src_file=True, verbose=True)
            df_form = load_formation_xlsx(form_path, well_name=test_name)
            df_cal = load_cal_file(cal_path, well_name=test_name)

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
