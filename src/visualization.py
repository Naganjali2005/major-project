"""
visualization.py
====================================================================
ROC / PR curves and confusion matrices for the PCOS Screening Aid.

Provides:
  - step_plots(data, state)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from config import OUT_DIR, FIG_DIR
from explainability import _ensure_AB
from training import header


def step_plots(data, state):
    """Step 9: ROC / PR curves and confusion matrices (tuned SVM, held-out test)."""
    header("9. ROC / PR CURVES AND CONFUSION MATRICES")
    _ensure_AB(data, state)
    yte = np.asarray(data["y_test"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for ab in ("A", "B"):
        prob = state[ab]["prob_test"]
        fpr, tpr, _ = roc_curve(yte, prob)
        pr,  rc,  _ = precision_recall_curve(yte, prob)
        ax[0].plot(fpr, tpr, label=f"Model {ab} (AUC="
                                   f"{roc_auc_score(yte, prob):.3f})")
        ax[1].plot(rc,  pr,  label=f"Model {ab} (AP="
                                   f"{average_precision_score(yte, prob):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC - tuned SVM, held-out test")
    ax[1].axhline(yte.mean(), color="k", ls="--", lw=1,
                  label=f"prevalence {yte.mean():.2f}")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall - tuned SVM, held-out test")
    for a in ax:
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roc_pr_curves.png"), dpi=150)
    plt.close("all")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    for k, ab in enumerate(("A", "B")):
        thr  = state[ab]["thr"]
        pred = (state[ab]["prob_test"] >= thr).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            yte, pred, display_labels=["No PCOS", "PCOS"],
            ax=ax[k], colorbar=False)
        ax[k].set_title(f"Model {ab} - SVM @ threshold {thr:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=150)
    plt.close("all")
    print(f"Saved figures to {FIG_DIR}/")
