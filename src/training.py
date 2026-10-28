"""
training.py
====================================================================
Model training, cross-validation, and comparison for the PCOS
Screening Aid.

Provides:
  - cv_summary(pipe, X, y, cv)
  - summary_row(label, summ, **extra)
  - print_summary_table(df, cols)
  - step_comparison(data, state)
  - step_matched_recall(data, state)
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from config import (
    OUT_DIR, RANDOM_STATE, THRESHOLDS, SCORING,
    MODEL_A_FEATURES, MODEL_B_FEATURES,
)
from preprocessing import build_preprocessor, make_clf, make_pipeline
from evaluation import (
    metrics_at_threshold, threshold_table,
    select_threshold,
)

# lazy import for the base cv (original 5-fold, not repeated)
import preprocessing as _pre
_base_cv = _pre.cv


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def cv_summary(pipe, X, y, cv):
    """Cross-validate and return {metric: (mean, std)} dict."""
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {m: (res[f"test_{m}"].mean(), res[f"test_{m}"].std())
            for m in SCORING}


def summary_row(label, summ, **extra):
    """Build a flat dict row from a cv_summary result."""
    row = {"Configuration": label}
    row.update(extra)
    for m, (mu, sd) in summ.items():
        row[f"{m}_mean"] = mu
        row[f"{m}_std"]  = sd
    return row


def print_summary_table(df, cols=("recall", "precision", "f1",
                                  "roc_auc", "pr_auc", "mcc")):
    """Print mean±std for selected metrics from a summary DataFrame."""
    show = pd.DataFrame({"Configuration": df["Configuration"]})
    for m in cols:
        show[m] = [f"{a:.3f}±{b:.3f}" for a, b in
                   zip(df[f"{m}_mean"], df[f"{m}_std"])]
    print(show.to_string(index=False))


# ------------------------------------------------------------------
# Step 1: Final model comparison (computed, not hard-coded)
# ------------------------------------------------------------------
def step_comparison(data, state):
    header("1. FINAL MODEL COMPARISON (computed, not hard-coded)")
    ytr, yte = data["y_train"], data["y_test"]
    rows = []
    for ab, Xtr, Xte in (("A", data["X_train_A"], data["X_test_A"]),
                          ("B", data["X_train_B"], data["X_test_B"])):
        for name in ("LR", "RF", "SVM"):
            pipe = make_pipeline(list(Xtr.columns), name, ab).fit(Xtr, ytr)
            prob = pipe.predict_proba(Xte)[:, 1]
            m    = metrics_at_threshold(yte, prob, 0.50)
            m.update(Model=f"{ab} - {name} @0.50")
            rows.append(m)
            state.setdefault("test_prob", {})[(ab, name)] = prob
    df = pd.DataFrame(rows)[["Model", "Accuracy", "Precision", "Recall", "F1",
                              "Specificity", "ROC-AUC", "PR-AUC", "MCC",
                              "TN", "FP", "FN", "TP"]]
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df.to_csv(os.path.join(OUT_DIR, "final_comparison_computed.csv"),
              index=False)
    state["comparison"] = df


# ------------------------------------------------------------------
# Step 2b: Fair comparison at matched recall
# ------------------------------------------------------------------
def step_matched_recall(data, state):
    header("2b. MODELS COMPARED AT MATCHED RECALL (out-of-fold, training set)")
    print("Comparing classifiers at the default 0.5 threshold is unfair "
          "(e.g. Model B RF has higher AUC than SVM but lower recall @0.5).\n"
          "Here every model is moved to the threshold that reaches the "
          "recall target, and we compare specificity / precision there.")
    ytr  = data["y_train"]
    rows = []
    for ab, Xtr in (("A", data["X_train_A"]), ("B", data["X_train_B"])):
        feats   = list(Xtr.columns)
        models  = {
            "LR":          make_pipeline(feats, "LR", ab),
            "SVM":         make_pipeline(feats, "SVM", ab),
            "RF":          make_pipeline(feats, "RF", ab),
            "Soft voting": Pipeline([
                ("preprocessing", build_preprocessor(feats, scale=True)),
                ("model", VotingClassifier(
                    estimators=[("lr",  make_clf("LR",  ab)),
                                ("svm", make_clf("SVM", ab)),
                                ("rf",  make_clf("RF",  ab))],
                    voting="soft")),
            ]),
        }
        for name, pipe in models.items():
            oof = cross_val_predict(clone(pipe), Xtr, ytr, cv=_base_cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]
            tbl = threshold_table(ytr, oof, THRESHOLDS)
            auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]) \
                  .roc_auc_score(ytr, oof)
            for tg in (0.85, 0.90, 0.95):
                el = tbl[tbl["Recall"] >= tg]
                if len(el) == 0:
                    rows.append({"Model set": ab, "Classifier": name,
                                 "Recall target": tg, "Reachable": False,
                                 "OOF ROC-AUC": auc})
                else:
                    r = el.loc[el["Specificity"].idxmax()]
                    rows.append({"Model set": ab, "Classifier": name,
                                 "Recall target": tg, "Reachable": True,
                                 "Threshold":       r["Threshold"],
                                 "Recall":          r["Recall"],
                                 "Specificity":     r["Specificity"],
                                 "Precision":       r["Precision"],
                                 "False Negatives": int(r["False Negatives"]),
                                 "OOF ROC-AUC":     auc})
    df = pd.DataFrame(rows)
    for ab in ("A", "B"):
        print(f"\n--- Model {ab} ---")
        print(df[df["Model set"] == ab].drop(columns="Model set").to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(os.path.join(OUT_DIR, "matched_recall_comparison.csv"),
              index=False)
    state["matched_recall"] = df
