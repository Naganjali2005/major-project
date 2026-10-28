"""
explainability.py
====================================================================
SHAP analysis, SVM coefficients, feature ablation, and engineered
features for the PCOS Screening Aid (Model B).

Provides:
  - step_thresholds(data, state)
  - step_shap_b(data, state)
  - step_ablation(data, state)
  - step_engineered(data, state)
  - step_families(data, state)
  - step_nested(data, state)
  - add_engineered_features(X, regular_code)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_predict, cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from config import (
    OUT_DIR, FIG_DIR,
    THRESHOLDS, MIN_RECALL, RANDOM_STATE,
    FEATURE_GROUPS_B, COST_TIERS_B, SYMPTOM_COLS,
    NESTED_OUTER_SPLITS, NESTED_OUTER_REPEATS,
    QUICK,
)
from preprocessing import (
    build_preprocessor, make_clf, make_pipeline, get_repeated_cv, cv as _base_cv,
)
from evaluation import (
    metrics_at_threshold, threshold_table, select_threshold,
    operating_points, wilson_ci, print_test_with_ci,
)
from training import header, cv_summary, summary_row, print_summary_table

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _ensure_AB(data, state):
    """Make sure fitted A/B SVMs + thresholds exist in state."""
    if "A" not in state or "B" not in state:
        step_thresholds(data, state)


def _class1_shap(sv):
    """Extract class-1 SHAP values regardless of shap version output shape."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


# ------------------------------------------------------------------
# Step 2: Thresholds, calibration, CIs
# ------------------------------------------------------------------
def step_thresholds(data, state):
    header("2. THRESHOLD ANALYSIS (grid 0.10-0.60) + CALIBRATION + CIs")
    ytr, yte = data["y_train"], data["y_test"]
    fig_tr, ax_tr = plt.subplots(1, 2, figsize=(13, 4.5))
    fig_cal, ax_cal = plt.subplots(figsize=(5.5, 5))
    ax_cal.plot([0, 1], [0, 1], "k--", label="perfect")

    for k, (ab, Xtr, Xte) in enumerate((
            ("A", data["X_train_A"], data["X_test_A"]),
            ("B", data["X_train_B"], data["X_test_B"]))):

        pipe = make_pipeline(list(Xtr.columns), "SVM", ab)
        oof  = cross_val_predict(clone(pipe), Xtr, ytr, cv=_base_cv,
                                 method="predict_proba", n_jobs=-1)[:, 1]
        tbl  = threshold_table(ytr, oof, THRESHOLDS)
        row, rule_met = select_threshold(tbl)
        thr  = float(row["Threshold"])

        print(f"\n--- Model {ab}: tuned SVM, OOF threshold table ---")
        print(tbl.to_string(index=False,
                            float_format=lambda x: f"{x:.4f}"))
        print(f"\nSelected threshold (rule: recall>={MIN_RECALL:.0%}, then "
              f"best F1): {thr:.3f}   | rule met: {rule_met}")
        if thr in (THRESHOLDS.min(), THRESHOLDS.max()):
            print("  WARNING: selected threshold is at the edge of the grid.")
        if not rule_met:
            print(f"  The {MIN_RECALL:.0%} recall target is NOT reachable "
                  f"for Model {ab} even with the wider grid.")
        op = operating_points(tbl)
        print(f"\nOperating points for Model {ab}:")
        print(op.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        tbl.to_csv(os.path.join(OUT_DIR, f"threshold_table_model_{ab}.csv"),
                   index=False)
        op.to_csv(os.path.join(OUT_DIR, f"operating_points_model_{ab}.csv"),
                  index=False)

        brier = brier_score_loss(ytr, oof)
        frac_pos, mean_pred = calibration_curve(ytr, oof, n_bins=8,
                                                strategy="quantile")
        ax_cal.plot(mean_pred, frac_pos, marker="o",
                    label=f"Model {ab} SVM (Brier={brier:.3f})")
        print(f"\nModel {ab} OOF Brier score: {brier:.4f}")

        ax = ax_tr[k]
        for col in ("Recall", "Specificity", "Precision", "F1"):
            ax.plot(tbl["Threshold"], tbl[col], marker="o", ms=3, label=col)
        ax.axvline(thr, color="k", ls="--", lw=1, label=f"selected {thr:.3f}")
        ax.axhline(MIN_RECALL, color="grey", ls=":", lw=1)
        ax.set_title(f"Model {ab} - tuned SVM (OOF, training set)")
        ax.set_xlabel("Decision threshold")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)

        fitted     = clone(pipe).fit(Xtr, ytr)
        prob_te    = fitted.predict_proba(Xte)[:, 1]
        print_test_with_ci(f"Model {ab} - held-out test @0.50",       yte, prob_te, 0.50)
        print_test_with_ci(f"Model {ab} - held-out test @selected",   yte, prob_te, thr)

        state[ab] = dict(pipe=fitted, thr=thr, rule_met=rule_met,
                         oof=oof, prob_test=prob_te, table=tbl, operating=op)

    fig_tr.tight_layout()
    fig_tr.savefig(os.path.join(FIG_DIR, "threshold_tradeoff.png"), dpi=150)
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Observed fraction of PCOS")
    ax_cal.set_title("Calibration (out-of-fold, training set)")
    ax_cal.legend(fontsize=8)
    fig_cal.tight_layout()
    fig_cal.savefig(os.path.join(FIG_DIR, "calibration.png"), dpi=150)
    plt.close("all")


# ------------------------------------------------------------------
# Step 3: SHAP + SVM coefficients for Model B
# ------------------------------------------------------------------
def step_shap_b(data, state):
    header("3. MODEL B (NO ULTRASOUND): SHAP + LINEAR-SVM COEFFICIENTS")
    import shap

    Xtr, ytr = data["X_train_B"], data["y_train"]

    # (a) SHAP on the tuned Random Forest
    rf          = make_pipeline(list(Xtr.columns), "RF", "B").fit(Xtr, ytr)
    pre         = rf.named_steps["preprocessing"]
    rf_model    = rf.named_steps["model"]
    Xp          = pre.transform(Xtr)
    names       = pre.get_feature_names_out()
    explainer   = shap.TreeExplainer(rf_model)
    sv          = _class1_shap(explainer.shap_values(Xp))
    imp = pd.DataFrame({"Feature": names,
                        "Mean |SHAP|": np.abs(sv).mean(axis=0)}
                       ).sort_values("Mean |SHAP|", ascending=False)
    print("\n--- Model B Random Forest: SHAP importance (top 15) ---")
    print(imp.head(15).to_string(index=False))
    imp.to_csv(os.path.join(OUT_DIR, "shap_importance_model_B.csv"),
               index=False)

    top = imp.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["Feature"], top["Mean |SHAP|"])
    plt.title("Model B (no ultrasound) - mean |SHAP| (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_bar_model_B.png"), dpi=150)
    plt.close("all")
    try:
        shap.summary_plot(sv, Xp, feature_names=names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "shap_summary_model_B.png"), dpi=150)
    except Exception as e:
        print("  (SHAP summary plot skipped:", e, ")")
    plt.close("all")

    # (b) Coefficients of the final deployed linear SVM
    svm   = make_pipeline(list(Xtr.columns), "SVM", "B").fit(Xtr, ytr)
    model = svm.named_steps["model"]
    if getattr(model, "kernel", None) == "linear":
        coef = pd.DataFrame({
            "Feature":     svm.named_steps["preprocessing"].get_feature_names_out(),
            "Coefficient": model.coef_.ravel(),
        })
        coef["|Coefficient|"] = coef["Coefficient"].abs()
        coef = coef.sort_values("|Coefficient|", ascending=False)
        print("\n--- Model B linear SVM: standardised coefficients (top 15) ---")
        print(coef.head(15).to_string(index=False))
        coef.to_csv(os.path.join(OUT_DIR, "svm_coefficients_model_B.csv"),
                    index=False)
    state["shap_B"] = imp


# ------------------------------------------------------------------
# Step 4: Feature-group ablation + cost tiers
# ------------------------------------------------------------------
def step_ablation(data, state):
    header("4. FEATURE-GROUP ABLATION + COST TIERS (MODEL B, linear SVM)")
    from config import MODEL_B_FEATURES
    all_feats = [f for g in FEATURE_GROUPS_B.values() for f in g]
    assert sorted(all_feats) == sorted(MODEL_B_FEATURES), \
        "FEATURE_GROUPS_B must cover exactly the Model B features"

    Xtr, ytr = data["X_train_B"], data["y_train"]
    rep_cv   = get_repeated_cv()

    # (a) Cumulative cost tiers
    rows, used = [], []
    for label, groups in COST_TIERS_B:
        for g in groups:
            used += FEATURE_GROUPS_B[g]
        feats = list(dict.fromkeys(used))
        summ  = cv_summary(make_pipeline(feats, "SVM", "B"),
                           Xtr[feats], ytr, rep_cv)
        rows.append(summary_row(label, summ, n_features=len(feats)))
    tiers = pd.DataFrame(rows)
    print("\n--- Cumulative cost tiers (cheapest first) ---")
    print_summary_table(tiers)
    tiers.to_csv(os.path.join(OUT_DIR, "ablation_cost_tiers.csv"), index=False)

    # (b) Leave-one-group-out from the full Model B
    full = list(MODEL_B_FEATURES)
    rows = [summary_row("Full Model B", cv_summary(
        make_pipeline(full, "SVM", "B"), Xtr[full], ytr, rep_cv),
        n_features=len(full))]
    for g, gf in FEATURE_GROUPS_B.items():
        feats = [f for f in full if f not in gf]
        summ  = cv_summary(make_pipeline(feats, "SVM", "B"),
                           Xtr[feats], ytr, rep_cv)
        rows.append(summary_row(f"minus {g}", summ, n_features=len(feats)))
    logo = pd.DataFrame(rows)
    print("\n--- Leave-one-group-out ---")
    print_summary_table(logo)
    logo.to_csv(os.path.join(OUT_DIR, "ablation_leave_one_group_out.csv"),
                index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].barh(tiers["Configuration"][::-1], tiers["roc_auc_mean"][::-1],
               xerr=tiers["roc_auc_std"][::-1])
    ax[0].set_xlim(0.5, 1.0)
    ax[0].set_title("ROC-AUC by cumulative cost tier")
    ax[1].barh(logo["Configuration"][::-1], logo["recall_mean"][::-1],
               xerr=logo["recall_std"][::-1], color="tab:orange")
    ax[1].set_xlim(0.4, 1.0)
    ax[1].set_title("Recall: leave-one-group-out")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ablation_model_B.png"), dpi=150)
    plt.close("all")
    state["ablation_tiers"], state["ablation_logo"] = tiers, logo


# ------------------------------------------------------------------
# Step 5: Engineered features
# ------------------------------------------------------------------
def add_engineered_features(X, regular_code=None):
    """Add clinically motivated derived features to a Model B feature set."""
    X    = X.copy()
    X["Symptom_count"] = X[SYMPTOM_COLS].apply(
        pd.to_numeric, errors="coerce").sum(axis=1)
    cyc  = pd.to_numeric(X["Cycle(R/I)"], errors="coerce")
    code = regular_code if regular_code is not None else cyc.min()
    irregular = (cyc != code).astype(float).where(cyc.notna())
    length    = pd.to_numeric(X["Cycle length(days)"], errors="coerce")
    X["Irregular_x_CycleLen"] = irregular * length
    X["AMH_x_LH"] = (pd.to_numeric(X["AMH(ng/mL)"], errors="coerce")
                     * pd.to_numeric(X["LH(mIU/mL)"], errors="coerce"))
    bmi = pd.to_numeric(X["BMI"], errors="coerce")
    X["BMI_over_23"] = (bmi >= 23).astype(float).where(bmi.notna())
    return X


ENGINEERED_NAMES = ["Symptom_count", "Irregular_x_CycleLen",
                    "AMH_x_LH", "BMI_over_23"]


def step_engineered(data, state):
    header("5. ENGINEERED FEATURES (MODEL B)")
    counts = data["X_B"]["Cycle(R/I)"].value_counts().sort_index()
    code   = int(counts.index.min())
    print("Cycle(R/I) value counts:", counts.to_dict())
    print(f"Treating {code} as 'regular'. VERIFY against your data dictionary.")

    Xtr, ytr    = data["X_train_B"], data["y_train"]
    Xtr_eng     = add_engineered_features(Xtr, regular_code=code)
    feats_plain = list(Xtr.columns)
    feats_eng   = feats_plain + ENGINEERED_NAMES
    rep_cv      = get_repeated_cv()

    rows = []
    for name in ("LR", "SVM", "RF"):
        rows.append(summary_row(f"B plain - {name}", cv_summary(
            make_pipeline(feats_plain, name, "B"), Xtr, ytr, rep_cv)))
        rows.append(summary_row(f"B + engineered - {name}", cv_summary(
            make_pipeline(feats_eng, name, "B"),
            Xtr_eng[feats_eng], ytr, rep_cv)))
    df = pd.DataFrame(rows)
    print_summary_table(df)
    df.to_csv(os.path.join(OUT_DIR, "engineered_features_comparison.csv"),
              index=False)
    state["engineered_regular_code"] = code
    state["engineered_cmp"]          = df


# ------------------------------------------------------------------
# Step 6: Other model families (XGBoost, voting, stacking)
# ------------------------------------------------------------------
def step_families(data, state):
    header("6. OTHER MODEL FAMILIES FOR MODEL B (repeated 5x5 CV)")
    Xtr, ytr = data["X_train_B"], data["y_train"]
    feats    = list(Xtr.columns)
    rep_cv   = get_repeated_cv()
    n_pos    = int(ytr.sum())
    n_neg    = int((ytr == 0).sum())

    def shared_prep():
        return build_preprocessor(feats, scale=True)

    candidates = {
        "LR (tuned)":            make_pipeline(feats, "LR",  "B"),
        "SVM linear (tuned)":    make_pipeline(feats, "SVM", "B"),
        "RF (tuned)":            make_pipeline(feats, "RF",  "B"),
        "Soft voting LR+SVM+RF": Pipeline([
            ("preprocessing", shared_prep()),
            ("model", VotingClassifier(
                estimators=[("lr",  make_clf("LR",  "B")),
                            ("svm", make_clf("SVM", "B")),
                            ("rf",  make_clf("RF",  "B"))],
                voting="soft")),
        ]),
        "Stacking LR+SVM+RF -> LR": Pipeline([
            ("preprocessing", shared_prep()),
            ("model", StackingClassifier(
                estimators=[("lr",  make_clf("LR",  "B")),
                            ("svm", make_clf("SVM", "B")),
                            ("rf",  make_clf("RF",  "B"))],
                final_estimator=LogisticRegression(
                    max_iter=1000, class_weight="balanced",
                    random_state=RANDOM_STATE),
                cv=5)),
        ]),
    }

    if HAS_XGB:
        xgb_pipe = Pipeline([
            ("preprocessing", shared_prep()),
            ("model", XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss",
                scale_pos_weight=n_neg / max(n_pos, 1),
                subsample=0.8, colsample_bytree=0.8, n_jobs=1)),
        ])
        candidates["XGBoost (tuned, nested)"] = GridSearchCV(
            xgb_pipe,
            {"model__max_depth": [2, 3, 4],
             "model__n_estimators": [100, 200],
             "model__learning_rate": [0.03, 0.1]},
            cv=3, scoring="roc_auc", n_jobs=1)
    else:
        print("xgboost not installed -> skipping (pip install xgboost).")

    rows = []
    for label, est in candidates.items():
        print(f"  evaluating: {label}")
        rows.append(summary_row(label, cv_summary(est, Xtr, ytr, rep_cv)))
    df = pd.DataFrame(rows)
    print()
    print_summary_table(df)
    df.to_csv(os.path.join(OUT_DIR, "model_families_model_B.csv"), index=False)
    state["families"] = df


# ------------------------------------------------------------------
# Step 7: Nested CV of the whole procedure
# ------------------------------------------------------------------
def _nested_threshold_eval(X, y, ab, param_grid, outer_cv, inner_cv):
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    pipe = make_pipeline(list(X.columns), "SVM", ab)
    fold_rows, pooled_true, pooled_pred, pooled_prob = [], [], [], []
    for i, (tr, te) in enumerate(outer_cv.split(X, y), 1):
        Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
        gs   = GridSearchCV(clone(pipe), param_grid, cv=inner_cv,
                            scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
        best = gs.best_estimator_
        oof  = cross_val_predict(clone(best), Xtr, ytr, cv=inner_cv,
                                 method="predict_proba", n_jobs=-1)[:, 1]
        row_t, _ = select_threshold(threshold_table(ytr, oof, THRESHOLDS))
        thr  = float(row_t["Threshold"])
        prob = best.predict_proba(Xte)[:, 1]
        m    = metrics_at_threshold(yte, prob, thr)
        fold_rows.append({"fold": i, "threshold": thr,
                          "recall": m["Recall"], "specificity": m["Specificity"],
                          "precision": m["Precision"], "f1": m["F1"],
                          "roc_auc": m["ROC-AUC"],
                          "best_params": str(gs.best_params_)})
        pooled_true += list(yte)
        pooled_pred += list((prob >= thr).astype(int))
        pooled_prob += list(prob)
    return (pd.DataFrame(fold_rows), np.array(pooled_true),
            np.array(pooled_pred), np.array(pooled_prob))


def step_nested(data, state):
    header("7. NESTED CV OF THE WHOLE PROCEDURE (tune + pick threshold)")
    grid  = {"model__C": [0.1, 1, 10, 100],
              "model__gamma": ["scale", "auto"],
              "model__kernel": ["rbf", "linear"]}
    outer = RepeatedStratifiedKFold(n_splits=NESTED_OUTER_SPLITS,
                                    n_repeats=NESTED_OUTER_REPEATS,
                                    random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=5, shuffle=True,
                            random_state=RANDOM_STATE)
    for ab, X in (("A", data["X_A"]), ("B", data["X_B"])):
        print(f"\n--- Model {ab}: nested CV on all {len(X)} patients ---")
        folds, yt, yp, _ = _nested_threshold_eval(
            X, data["y"], ab, grid, outer, inner)
        print(folds.drop(columns="best_params").to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nMean ± SD over outer folds:")
        for c in ("threshold", "recall", "specificity", "precision",
                  "f1", "roc_auc"):
            print(f"  {c:<12}{folds[c].mean():.4f} ± {folds[c].std():.4f}")
        tp  = int(((yt == 1) & (yp == 1)).sum())
        pos = int((yt == 1).sum())
        tn  = int(((yt == 0) & (yp == 0)).sum())
        neg = int((yt == 0).sum())
        lo, hi   = wilson_ci(tp, pos)
        slo, shi = wilson_ci(tn, neg)
        print(f"  pooled recall      {tp}/{pos} = {tp/pos:.4f}  "
              f"Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  pooled specificity {tn}/{neg} = {tn/neg:.4f}  "
              f"Wilson 95% CI [{slo:.3f}, {shi:.3f}]")
        folds.to_csv(os.path.join(OUT_DIR, f"nested_cv_model_{ab}.csv"),
                     index=False)
        state[f"nested_{ab}"] = folds
