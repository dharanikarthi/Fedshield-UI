"""Shared evaluation: threshold chosen on validation, metrics reported on test."""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def pick_threshold(p_val, y_val, mode="f1", target_recall=0.8):
    """Validation-set threshold. mode='f1' maximises F1; mode='recall' hits a target recall."""
    order = np.argsort(-p_val)
    ps, ys = p_val[order], y_val[order]
    tp = np.cumsum(ys)
    if mode == "recall":
        rec = tp / max(ys.sum(), 1)
        return float(ps[min(np.searchsorted(rec, target_recall), len(ps) - 1)])
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / max(ys.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(ps[int(np.argmax(f1))])


def scores(p, y, thr, k_frac=0.01):
    pred = (p >= thr).astype(int)
    k = max(1, int(round(k_frac * len(p))))
    top = np.argsort(-p)[:k]
    return {
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "flag_rate": float(pred.mean()),
        f"precision_at_{int(k_frac * 100)}pct": float(y[top].mean()),
        "threshold": float(thr),
        "n": int(len(y)), "positives": int(y.sum()),
    }


def fpr(p, y, thr):
    neg = y == 0
    return float((p[neg] >= thr).mean()) if neg.any() else float("nan")


def fairness(p, y, thr, groups):
    """False-positive (wrongful hold) and true-positive rates per group."""
    out = {}
    for g in np.unique(groups):
        m = groups == g
        pos = (y == 1) & m
        out[str(g)] = {"n": int(m.sum()), "fpr": fpr(p[m], y[m], thr),
                       "tpr": float((p[pos] >= thr).mean()) if pos.any() else float("nan"),
                       "flag_rate": float((p[m] >= thr).mean())}
    rates = [v["fpr"] for v in out.values() if v["n"] > 100]
    out["fpr_ratio_max_min"] = float(max(rates) / max(min(rates), 1e-9)) if rates else float("nan")
    return out
