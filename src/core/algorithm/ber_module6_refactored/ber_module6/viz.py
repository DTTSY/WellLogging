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
# 6) 可视化（出图/结果保存）
# ============================

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

