from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class BinaryMetrics:
    acc: float
    precision: float
    recall: float
    f1: float

def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> BinaryMetrics:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return BinaryMetrics(acc=acc, precision=precision, recall=recall, f1=f1)

def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Lightweight ROC-AUC (no sklearn). For publication, consider sklearn for robustness."""
    y_true = y_true.astype(int)
    # Sort by descending probability
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")

    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0
    last_p = None
    for yt, p in zip(y_true, y_prob):
        if last_p is None or p != last_p:
            tpr.append(tp / pos)
            fpr.append(fp / neg)
            last_p = p
        if yt == 1:
            tp += 1
        else:
            fp += 1
    tpr.append(tp / pos)
    fpr.append(fp / neg)

    # Trapezoidal area
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return float(auc)
