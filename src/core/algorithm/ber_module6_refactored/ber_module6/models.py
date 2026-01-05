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
# 3) 模型网络（所有模型结构/损失/诊断）
# ============================

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

