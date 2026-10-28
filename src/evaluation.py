"""
evaluation.py
====================================================================
Metric computation, threshold selection, operating points, and
confidence intervals for the PCOS Screening Aid.

Provides:
  - metrics_at_threshold(y_true, prob, thr)
  - threshold_table(y_true, prob, thresholds)
  - select_threshold(tbl, min_recall)
  - operating_points(tbl, targets)
  - wilson_ci(k, n, z)
  - bootstrap_ci(y_true, prob, thr, n_boot, seed)
  - paired_bootstrap_diff(y, p1, thr1, p2, thr2)
  - print_test_with_ci(label, y_true, prob, thr)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import MIN_RECALL, THRESHOLDS, N_BOOT, RANDOM_STATE


# ------------------------------------------------------------------
# Core metric computation
# ------------------------------------------------------------------
def metrics_at_threshold(y_true, prob, thr):
    """Return a dict of metrics for a given decision threshold."""
    y_true = np.asarray(y_true)
    pred   = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Threshold":    thr,
        "Accuracy":     accuracy_score(y_true, pred),
        "Precision":    precision_score(y_true, pred, zero_division=0),
        "Recall":       recall_score(y_true, pred, zero_division=0),
        "F1":           f1_score(y_true, pred, zero_division=0),
        "Specificity":  tn / (tn + fp) if (tn + fp) else np.nan,
        "ROC-AUC":      roc_auc_score(y_true, prob),
        "PR-AUC":       average_precision_score(y_true, prob),
        "MCC":          matthews_corrcoef(y_true, pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def threshold_table(y_true, prob, thresholds):
    """Build a DataFrame of metrics across a grid of thresholds."""
    rows = []
    for t in thresholds:
        m = metrics_at_threshold(y_true, prob, t)
        rows.append({
            "Threshold":      t,
            "Accuracy":       m["Accuracy"],
            "Precision":      m["Precision"],
            "Recall":         m["Recall"],
            "F1":             m["F1"],
            "Specificity":    m["Specificity"],
            "False Negatives": m["FN"],
            "False Positives": m["FP"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Threshold selection
# ------------------------------------------------------------------
def select_threshold(tbl, min_recall=MIN_RECALL):
    """
    Clinical rule: among thresholds with recall >= min_recall pick
    the highest F1; if none qualifies pick the highest recall.
    Returns (row, rule_met).
    """
    eligible = tbl[tbl["Recall"] >= min_recall]
    if len(eligible) > 0:
        return eligible.loc[eligible["F1"].idxmax()], True
    return tbl.loc[tbl["Recall"].idxmax()], False


def operating_points(tbl, targets=(0.85, 0.90, 0.95)):
    """
    For each recall target: the threshold with the best specificity
    that still reaches the target (clinical trade-off view).
    """
    rows = []
    for tg in targets:
        el = tbl[tbl["Recall"] >= tg]
        if len(el) == 0:
            rows.append({"Recall target": tg, "Reachable": False})
        else:
            r = el.loc[el["Specificity"].idxmax()]
            rows.append({
                "Recall target":   tg,
                "Reachable":       True,
                "Threshold":       r["Threshold"],
                "Recall":          r["Recall"],
                "Specificity":     r["Specificity"],
                "Precision":       r["Precision"],
                "F1":              r["F1"],
                "False Negatives": int(r["False Negatives"]),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Confidence intervals
# ------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p      = k / n
    denom  = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(y_true, prob, thr, n_boot=None, seed=RANDOM_STATE):
    """Bootstrap 95% CI for recall, specificity, precision, F1, AUCs."""
    n_boot = N_BOOT if n_boot is None else n_boot
    y_true = np.asarray(y_true)
    prob   = np.asarray(prob)
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    store  = {k: [] for k in
              ("Recall", "Specificity", "Precision", "F1", "ROC-AUC", "PR-AUC")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pr = y_true[idx], prob[idx]
        if yt.min() == yt.max():
            continue
        m = metrics_at_threshold(yt, pr, thr)
        for k in store:
            store[k].append(m[k])
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
            for k, v in store.items()}


def paired_bootstrap_diff(y, p1, thr1, p2, thr2,
                          n_boot=None, seed=RANDOM_STATE):
    """
    Compare model 1 (lite) with model 2 (full) on the SAME test patients.
    Returns diff = model1 - model2 with 95% bootstrap CI.
    """
    n_boot = N_BOOT if n_boot is None else n_boot
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    rng    = np.random.default_rng(seed)
    keys   = ("ROC-AUC", "PR-AUC", "Recall", "Specificity")
    diffs  = {k: [] for k in keys}
    n      = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        m1 = metrics_at_threshold(y[idx], p1[idx], thr1)
        m2 = metrics_at_threshold(y[idx], p2[idx], thr2)
        for k in keys:
            diffs[k].append(m1[k] - m2[k])
    rows = []
    for k in keys:
        d = np.array(diffs[k])
        rows.append({
            "Metric":            k,
            "Lite - Full (mean)": d.mean(),
            "95% CI low":        np.nanpercentile(d, 2.5),
            "95% CI high":       np.nanpercentile(d, 97.5),
            "P(Lite <= Full)":   float((d <= 0).mean()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Pretty-print helper
# ------------------------------------------------------------------
def print_test_with_ci(label, y_true, prob, thr):
    """Print held-out test metrics with Wilson and bootstrap CIs."""
    m    = metrics_at_threshold(y_true, prob, thr)
    boot = bootstrap_ci(y_true, prob, thr)
    pos, neg = m["TP"] + m["FN"], m["TN"] + m["FP"]
    w_rec  = wilson_ci(m["TP"], pos)
    w_spec = wilson_ci(m["TN"], neg)
    print(f"\n{label}  (threshold={thr:.3f}, "
          f"{pos} positives / {neg} negatives in test set)")
    print(f"  Recall      {m['Recall']:.4f}   Wilson 95% CI "
          f"[{w_rec[0]:.3f}, {w_rec[1]:.3f}]   ({m['TP']}/{pos})")
    print(f"  Specificity {m['Specificity']:.4f}   Wilson 95% CI "
          f"[{w_spec[0]:.3f}, {w_spec[1]:.3f}]   ({m['TN']}/{neg})")
    for k in ("Precision", "F1", "ROC-AUC", "PR-AUC"):
        lo, hi = boot[k]
        print(f"  {k:<11} {m[k]:.4f}   bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  Confusion matrix  TN={m['TN']} FP={m['FP']} "
          f"FN={m['FN']} TP={m['TP']}")
    return m, boot
