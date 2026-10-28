"""
cascade.py
====================================================================
Two-stage screening cascade (Model B -> ultrasound -> Model A) and
the Lite Model B (10 no-lab features) for the PCOS Screening Aid.

Provides:
  - step_cascade(data, state)
  - step_lite(data, state)
  - lite_features(extra)
  - pick_simplest_near_best(tuned, tol)
  - tune_lite_models(feats, Xtr, ytr, label)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV, cross_val_predict,
)

from config import (
    OUT_DIR, FIG_DIR, MODEL_DIR,
    THRESHOLDS, MIN_RECALL, RANDOM_STATE, QUICK,
    FEATURE_GROUPS_B, LITE_GROUPS, TARGET,
)
from preprocessing import make_pipeline, cv as _base_cv
from evaluation import (
    metrics_at_threshold, threshold_table,
    select_threshold, operating_points,
    print_test_with_ci, paired_bootstrap_diff,
)
from training import header, cv_summary, summary_row, print_summary_table
from preprocessing import get_repeated_cv
from explainability import _ensure_AB, _nested_threshold_eval

# keep a local alias
def _cascade_table(y, p_stage1, p_A, thrA):
    y   = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    rows = []
    for t in THRESHOLDS:
        referred  = p_stage1 >= t
        final_pos = referred & (p_A >= thrA)
        tp        = (final_pos & (y == 1)).sum()
        tn        = (~final_pos & (y == 0)).sum()
        rows.append({
            "Stage-1 threshold":        t,
            "Referred for ultrasound (%)": 100 * referred.mean(),
            "Ultrasounds saved (%)":    100 * (1 - referred.mean()),
            "Cascade recall":           tp / pos,
            "Cascade specificity":      tn / neg,
            "Missed PCOS (FN)":         int(pos - tp),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Step 8: B -> ultrasound -> A cascade
# ------------------------------------------------------------------
def step_cascade(data, state):
    header("8. SCREENING CASCADE: Model B screens, Model A confirms")
    _ensure_AB(data, state)
    y    = np.asarray(data["y_test"])
    pB   = state["B"]["prob_test"]
    pA   = state["A"]["prob_test"]
    thrA = state["A"]["thr"]
    pos  = (y == 1).sum()
    neg  = (y == 0).sum()

    rows = []
    for thrB in THRESHOLDS:
        referred  = pB >= thrB
        final_pos = referred & (pA >= thrA)
        tp        = (final_pos & (y == 1)).sum()
        tn        = (~final_pos & (y == 0)).sum()
        rows.append({
            "Model B threshold":           thrB,
            "Referred for ultrasound (%)": 100 * referred.mean(),
            "Ultrasounds saved (%)":       100 * (1 - referred.mean()),
            "Cascade recall":              tp / pos,
            "Cascade specificity":         tn / neg,
            "Cascade precision":           (tp / final_pos.sum()
                                           if final_pos.sum() else np.nan),
            "Missed PCOS (FN)":            int(pos - tp),
        })
    df = pd.DataFrame(rows)
    print(f"Model A confirmation threshold = {thrA:.3f}; test set has "
          f"{pos} positives / {neg} negatives.")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(os.path.join(OUT_DIR, "cascade_analysis.csv"), index=False)

    thrB_sel = state["B"]["thr"]
    sel = df.iloc[(df["Model B threshold"] - thrB_sel).abs().argmin()]
    print(f"\nAt the selected Model B threshold ({thrB_sel:.3f}): "
          f"{sel['Ultrasounds saved (%)']:.1f}% of patients avoid ultrasound, "
          f"cascade recall = {sel['Cascade recall']:.3f}, "
          f"specificity = {sel['Cascade specificity']:.3f}.")

    plt.figure(figsize=(7, 4.5))
    plt.plot(df["Ultrasounds saved (%)"], df["Cascade recall"],
             marker="o", label="Cascade recall")
    plt.plot(df["Ultrasounds saved (%)"], df["Cascade specificity"],
             marker="s", label="Cascade specificity")
    plt.xlabel("Ultrasounds saved (%)")
    plt.title("Cascade: ultrasounds saved vs. accuracy (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cascade_tradeoff.png"), dpi=150)
    plt.close("all")
    state["cascade"] = df


# ------------------------------------------------------------------
# Lite feature helpers
# ------------------------------------------------------------------
def lite_features(extra=()):
    """10 no-lab features: cycle (2) + symptoms (5) + lifestyle (2) + age."""
    feats = [f for g in LITE_GROUPS for f in FEATURE_GROUPS_B[g]]
    feats += [f for f in extra if f not in feats]
    from config import MODEL_B_FEATURES
    return [f for f in MODEL_B_FEATURES if f in feats]


def _lite_grids():
    if QUICK:
        return {
            "LR":  {"model__C": [0.1, 1, 10], "model__penalty": ["l1", "l2"],
                    "model__solver": ["liblinear"]},
            "SVM": {"model__C": [0.1, 1], "model__kernel": ["linear", "rbf"],
                    "model__gamma": ["scale"]},
            "RF":  {"model__n_estimators": [100], "model__max_depth": [5, 10],
                    "model__min_samples_split": [5],
                    "model__min_samples_leaf": [2]},
        }
    return {
        "LR":  {"model__C": [0.01, 0.1, 1, 10, 100],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["liblinear"]},
        "SVM": {"model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["rbf", "linear"]},
        "RF":  {"model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 5, 10, 15],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]},
    }


def tune_lite_models(feats, Xtr, ytr, label):
    """GridSearchCV (ROC-AUC, same 5-fold CV) for LR, SVM, RF."""
    out = {}
    for name, grid in _lite_grids().items():
        gs = GridSearchCV(make_pipeline(feats, name, "B"), grid,
                          cv=_base_cv, scoring="roc_auc",
                          n_jobs=-1).fit(Xtr[feats], ytr)
        out[name] = gs
        print(f"  {label:<14} {name:<4} best CV ROC-AUC {gs.best_score_:.4f}"
              f"   {gs.best_params_}")
    return out


def pick_simplest_near_best(tuned, tol=0.005):
    """Pick the simplest classifier (LR < SVM < RF) within tol of the best."""
    best = max(g.best_score_ for g in tuned.values())
    for name in ("LR", "SVM", "RF"):
        if name in tuned and tuned[name].best_score_ >= best - tol:
            return name
    return max(tuned, key=lambda n: tuned[n].best_score_)


# ------------------------------------------------------------------
# Step 8b: Lite Model B
# ------------------------------------------------------------------
def step_lite(data, state):
    header("8b. LITE MODEL B - 10 NO-LAB FEATURES (cycle, symptoms, "
           "lifestyle, age)")
    _ensure_AB(data, state)
    Xtr, Xte = data["X_train_B"], data["X_test_B"]
    ytr, yte = data["y_train"], np.asarray(data["y_test"])

    sets = {"Lite-10": lite_features(),
            "Lite-10+BMI": lite_features(extra=["BMI"])}
    print("Lite-10 features:")
    for f in sets["Lite-10"]:
        print("  -", f)
    print("These need NO ultrasound and NO blood test.\n")

    # (1) Tune LR / SVM / RF on each cheap set
    print("--- (1) Hyper-parameter tuning on the cheap feature sets ---")
    tuned, chosen = {}, {}
    for label, feats in sets.items():
        tuned[label]  = tune_lite_models(feats, Xtr, ytr, label)
        chosen[label] = pick_simplest_near_best(tuned[label])
        print(f"  -> {label}: using {chosen[label]}\n")

    # (2) Repeated 5x5 CV: lite vs full Model B
    print("--- (2) Repeated 5x5 CV on the training set ---")
    rep_cv     = get_repeated_cv()
    full_feats = list(Xtr.columns)
    rows = [summary_row(
        "Full Model B (29) SVM", cv_summary(
            make_pipeline(full_feats, "SVM", "B"), Xtr, ytr, rep_cv),
        n_features=len(full_feats))]
    for label, feats in sets.items():
        best = tuned[label][chosen[label]].best_estimator_
        rows.append(summary_row(
            f"{label} {chosen[label]} (tuned)",
            cv_summary(best, Xtr[feats], ytr, rep_cv),
            n_features=len(feats)))
    cmp_df = pd.DataFrame(rows)
    print_summary_table(cmp_df)
    cmp_df.to_csv(os.path.join(OUT_DIR, "lite_vs_full_cv.csv"), index=False)

    # (3) Primary lite model thresholds + operating points
    primary  = "Lite-10"
    feats    = sets[primary]
    clf_name = chosen[primary]
    best     = tuned[primary][clf_name].best_estimator_
    print(f"\n--- (3) Primary lite model: {primary} + {clf_name} ---")
    oof = cross_val_predict(clone(best), Xtr[feats], ytr, cv=_base_cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    tbl            = threshold_table(ytr, oof, THRESHOLDS)
    row_t, rule_met = select_threshold(tbl)
    thr            = float(row_t["Threshold"])
    op             = operating_points(tbl)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSelected threshold (recall>={MIN_RECALL:.0%}, then best F1): "
          f"{thr:.3f} | rule met: {rule_met}")
    print("\nOperating points - LITE model:")
    print(op.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nOperating points - FULL Model B (29 features), for comparison:")
    print(state["B"]["operating"].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    tbl.to_csv(os.path.join(OUT_DIR, "threshold_table_lite.csv"), index=False)
    op.to_csv(os.path.join(OUT_DIR, "operating_points_lite.csv"), index=False)

    fitted  = clone(best).fit(Xtr[feats], ytr)
    prob_te = fitted.predict_proba(Xte[feats])[:, 1]
    print_test_with_ci("LITE model - held-out test @0.50",      yte, prob_te, 0.50)
    print_test_with_ci("LITE model - held-out test @selected",  yte, prob_te, thr)

    # (4) Paired bootstrap: lite vs full on the SAME test patients
    print("\n--- (4) Paired bootstrap (Lite minus Full) ---")
    pf, thr_f = state["B"]["prob_test"], state["B"]["thr"]
    diff = paired_bootstrap_diff(yte, prob_te, thr, pf, thr_f)
    print(f"Lite threshold = {thr:.3f}, Full Model B threshold = {thr_f:.3f}")
    print(diff.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    auc_row = diff[diff["Metric"] == "ROC-AUC"].iloc[0]
    margin  = 0.05
    if auc_row["95% CI low"] > -margin:
        print(f"\nROC-AUC: the lite model is NON-INFERIOR to the full model "
              f"at a margin of {margin}.")
    else:
        print(f"\nROC-AUC: non-inferiority at margin {margin} is NOT shown.")
    diff.to_csv(os.path.join(OUT_DIR, "lite_vs_full_paired_bootstrap.csv"),
                index=False)

    # (5) Cascade: lite as stage 1
    print("\n--- (5) Cascade: LITE screen -> ultrasound + Model A ---")
    thrA = state["A"]["thr"]
    casc = _cascade_table(yte, prob_te, state["A"]["prob_test"], thrA)
    print(casc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    casc.to_csv(os.path.join(OUT_DIR, "cascade_lite.csv"), index=False)

    # (6) ROC comparison plot + save lite model
    plt.figure(figsize=(5.8, 5))
    for lab, pr in (("Model A (29+5 features)", state["A"]["prob_test"]),
                    ("Model B full (29)",        pf),
                    ("Lite (10, no labs)",        prob_te)):
        fpr, tpr, _ = roc_curve(yte, pr)
        plt.plot(fpr, tpr, label=f"{lab} AUC={roc_auc_score(yte, pr):.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Held-out test ROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roc_lite_vs_full.png"), dpi=150)
    plt.close("all")

    path = os.path.join(MODEL_DIR, "model_B_lite.joblib")
    joblib.dump(fitted, path)
    m = metrics_at_threshold(yte, prob_te, thr)
    with open(os.path.join(MODEL_DIR, "model_B_lite_metadata.json"), "w") as f:
        json.dump({
            "file": path, "classifier": clf_name,
            "best_params": {k: (v if isinstance(v, (int, float, str, type(None)))
                                else str(v))
                            for k, v in tuned[primary][clf_name]
                            .best_params_.items()},
            "threshold": thr, "threshold_rule_met": bool(rule_met),
            "features": feats, "target": TARGET,
            "test_metrics_at_threshold": {
                k: (float(v) if not isinstance(v, int) else v)
                for k, v in m.items()},
            "intended_use": "screening aid, NOT a diagnosis",
        }, f, indent=2)
    print(f"\nSaved {path} and models/model_B_lite_metadata.json")
    state["lite"] = dict(feats=feats, clf=clf_name, pipe=fitted, thr=thr,
                         prob_test=prob_te, table=tbl, operating=op,
                         diff=diff, cascade=casc, tuned=tuned)
