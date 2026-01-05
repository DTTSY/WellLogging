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
# 4) 训练（含过程保存/阈值搜索/OPC训练/PCA拟合）
# ============================

from .data_io import RUN_CFG, SEG_CFG, MODEL_CFG, NUM_WORKERS, PIN_MEMORY
from .preprocess import (
    fit_feature_scaler, apply_feature_scaler,
    build_dataloaders, build_seg_level4_dataloaders,
)
from .models import EnhancedSegClassifier, ExpansionLevelClassifier, FocalLossWithLogits, ModelDiagnosticSystem

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

