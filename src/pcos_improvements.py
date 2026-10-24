"""
pcos_improvements.py
====================================================================
Add-on for the PCOS major project.  Your original script (src/preprocess.py)
is NOT modified: this file imports its features, pipelines, CV object and
data loader, and adds everything we discussed on top of it.

Run from the project root (same place you run preprocess.py from, so that
data/pcos_updated.xlsx is found):

    python src/pcos_improvements.py            # everything
    python src/pcos_improvements.py --quick    # fast smoke run (fewer repeats)
    python src/pcos_improvements.py --steps thresholds cascade freeze

Steps (all outputs go to outputs/, models to models/):
  comparison  1. Final model comparison computed from fitted models
                 (replaces the hard-coded numbers in final_comparison)
  matched_recall 2b. LR/SVM/RF/voting compared at the SAME recall (fair comparison)
  thresholds  2. Wider threshold grid (0.10-0.60), recall/specificity trade-off,
                 operating points at 85/90/95% recall, calibration + Brier,
                 bootstrap / Wilson confidence intervals on test metrics
  shap_b      3. SHAP + linear-SVM coefficients for Model B (no ultrasound)
  ablation    4. Feature-group ablation + cost-tier analysis for Model B
  engineered  5. Engineered features (symptom count, interactions, ...)
  families    6. XGBoost, soft-voting and stacking ensembles for Model B
  nested      7. Nested CV of the whole procedure (tuning + threshold choice)
  cascade     8. Two-stage screening cascade: Model B -> ultrasound -> Model A
  lite        8b. LOW-COST Model B: only the 10 no-lab features (cycle, symptoms,
                 lifestyle, age) re-tuned, thresholds + CIs, paired bootstrap vs
                 the full 29-feature Model B, cascade with the lite model
  plots       9. ROC / PR curves and confusion matrices
  freeze     10. Save final pipelines + thresholds (joblib / json)
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
    cross_validate,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
    make_scorer,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Import everything from the ORIGINAL script (left untouched)
# ------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess as base  # noqa: E402  (your existing src/preprocess.py)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # xgboost not installed -> that part is skipped
    HAS_XGB = False

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
RANDOM_STATE = 42
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
MODEL_DIR = "models"

MIN_RECALL = 0.90                                   # same target as original
THRESHOLDS = np.round(np.arange(0.10, 0.601, 0.025), 3)   # was 0.30-0.60 only

CV_REPEATS = 5          # repeated stratified 5-fold (5x5) for stable estimates
NESTED_OUTER_SPLITS = 5
NESTED_OUTER_REPEATS = 2
N_BOOT = 2000
QUICK = False           # set by --quick (smaller grids for the lite step)

# Best hyper-parameters reported in your progress report (GridSearchCV results)
REPORTED_PARAMS = {
    ("A", "LR"): dict(C=1, penalty="l1", solver="liblinear"),
    ("B", "LR"): dict(C=1, penalty="l1", solver="liblinear"),
    ("A", "SVM"): dict(C=1, kernel="linear", gamma="scale"),
    ("B", "SVM"): dict(C=0.1, kernel="linear", gamma="scale"),
    ("A", "RF"): dict(n_estimators=300, max_depth=None,
                      min_samples_split=2, min_samples_leaf=1),
    ("B", "RF"): dict(n_estimators=200, max_depth=10,
                      min_samples_split=10, min_samples_leaf=4),
}

SYMPTOM_COLS = [
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
]

# Feature groups for Model B (29 features) - used for the ablation / cost tiers
FEATURE_GROUPS_B = {
    "Cycle": ["Cycle(R/I)", "Cycle length(days)"],
    "Symptoms": SYMPTOM_COLS,
    "Lifestyle": ["Fast food (Y/N)", "Reg.Exercise(Y/N)"],
    "Demographics": ["Age (yrs)"],
    "Anthropometrics": ["Weight (Kg)", "Height(Cm)", "BMI", "Hip(inch)",
                        "Waist(inch)", "Waist:Hip Ratio"],
    "Vitals": ["Pulse rate(bpm)", "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)"],
    "BasicLabs": ["Hb(g/dl)", "RBS(mg/dl)"],
    "Hormones_noAMH": ["FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "TSH (mIU/L)",
                       "PRL(ng/mL)", "PRG(ng/mL)", "Vit D3 (ng/mL)"],
    "AMH": ["AMH(ng/mL)"],
}

# Cumulative cost tiers (cheapest -> most expensive)
COST_TIERS_B = [
    ("T1 symptoms+cycle+lifestyle+age",
     ["Cycle", "Symptoms", "Lifestyle", "Demographics"]),
    ("T2 + anthropometrics+vitals", ["Anthropometrics", "Vitals"]),
    ("T3 + basic labs (Hb, RBS)", ["BasicLabs"]),
    ("T4 + hormone panel (no AMH)", ["Hormones_noAMH"]),
    ("T5 + AMH  (= full Model B)", ["AMH"]),
]

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "balanced_acc": "balanced_accuracy",
    "mcc": make_scorer(matthews_corrcoef),
}


# ==================================================================
# GENERIC HELPERS
# ==================================================================
def ensure_dirs():
    for d in (OUT_DIR, FIG_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def build_preprocessor(features, scale=True):
    """Same preprocessing logic as preprocess.py, but for ANY feature subset.
    numeric -> median impute (+ standardise if scale)
    binary  -> most-frequent impute
    cycle   -> most-frequent impute + one-hot
    Unknown / engineered columns are treated as numeric."""
    binary = [f for f in features if f in base.BINARY_FEATURES]
    cycle = [f for f in features if f == base.CYCLE_FEATURE]
    numeric = [f for f in features if f not in binary and f not in cycle]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(num_steps), numeric))
    if binary:
        transformers.append(
            ("binary", SimpleImputer(strategy="most_frequent"), binary))
    if cycle:
        transformers.append((
            "cycle",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=False)),
            ]),
            cycle,
        ))
    return ColumnTransformer(transformers=transformers)


def make_clf(name, ab):
    p = dict(REPORTED_PARAMS[(ab, name)])
    if name == "LR":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                  class_weight="balanced", **p)
    if name == "SVM":
        return SVC(probability=True, random_state=RANDOM_STATE,
                   class_weight="balanced", **p)
    if name == "RF":
        return RandomForestClassifier(random_state=RANDOM_STATE,
                                      class_weight="balanced", **p)
    raise ValueError(name)


def make_pipeline(features, name, ab):
    """Pipeline with reported tuned params. RF is unscaled (as in original)."""
    return Pipeline([
        ("preprocessing", build_preprocessor(features, scale=(name != "RF"))),
        ("model", make_clf(name, ab)),
    ])


def get_repeated_cv():
    return RepeatedStratifiedKFold(
        n_splits=5, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)


def cv_summary(pipe, X, y, cv):
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {m: (res[f"test_{m}"].mean(), res[f"test_{m}"].std())
            for m in SCORING}


def summary_row(label, summ, **extra):
    row = {"Configuration": label}
    row.update(extra)
    for m, (mu, sd) in summ.items():
        row[f"{m}_mean"] = mu
        row[f"{m}_std"] = sd
    return row


def print_summary_table(df, cols=("recall", "precision", "f1",
                                  "roc_auc", "pr_auc", "mcc")):
    show = pd.DataFrame({"Configuration": df["Configuration"]})
    for m in cols:
        show[m] = [f"{a:.3f}±{b:.3f}" for a, b in
                   zip(df[f"{m}_mean"], df[f"{m}_std"])]
    print(show.to_string(index=False))


# ------------------------------------------------------------------
# Metrics, thresholds, confidence intervals
# ------------------------------------------------------------------
def metrics_at_threshold(y_true, prob, thr):
    y_true = np.asarray(y_true)
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Threshold": thr,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ROC-AUC": roc_auc_score(y_true, prob),
        "PR-AUC": average_precision_score(y_true, prob),
        "MCC": matthews_corrcoef(y_true, pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def threshold_table(y_true, prob, thresholds):
    rows = []
    for t in thresholds:
        m = metrics_at_threshold(y_true, prob, t)
        rows.append({
            "Threshold": t, "Accuracy": m["Accuracy"],
            "Precision": m["Precision"], "Recall": m["Recall"],
            "F1": m["F1"], "Specificity": m["Specificity"],
            "False Negatives": m["FN"], "False Positives": m["FP"],
        })
    return pd.DataFrame(rows)


def select_threshold(tbl, min_recall=MIN_RECALL):
    """Same rule as your original code: among thresholds with recall >=
    min_recall pick the highest F1; if none qualifies pick the highest recall.
    Returns (row, rule_met)."""
    eligible = tbl[tbl["Recall"] >= min_recall]
    if len(eligible) > 0:
        return eligible.loc[eligible["F1"].idxmax()], True
    return tbl.loc[tbl["Recall"].idxmax()], False


def operating_points(tbl, targets=(0.85, 0.90, 0.95)):
    """For each recall target: the threshold with the best specificity that
    still reaches the target (a cleaner clinical way to read the trade-off)."""
    rows = []
    for tg in targets:
        el = tbl[tbl["Recall"] >= tg]
        if len(el) == 0:
            rows.append({"Recall target": tg, "Reachable": False})
        else:
            r = el.loc[el["Specificity"].idxmax()]
            rows.append({
                "Recall target": tg, "Reachable": True,
                "Threshold": r["Threshold"], "Recall": r["Recall"],
                "Specificity": r["Specificity"], "Precision": r["Precision"],
                "F1": r["F1"], "False Negatives": int(r["False Negatives"]),
            })
    return pd.DataFrame(rows)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(y_true, prob, thr, n_boot=None, seed=RANDOM_STATE):
    n_boot = N_BOOT if n_boot is None else n_boot
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    store = {k: [] for k in
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


def print_test_with_ci(label, y_true, prob, thr):
    m = metrics_at_threshold(y_true, prob, thr)
    boot = bootstrap_ci(y_true, prob, thr)
    pos, neg = m["TP"] + m["FN"], m["TN"] + m["FP"]
    w_rec = wilson_ci(m["TP"], pos)
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


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load_split():
    """Exactly the same split as preprocess.py (80/20, stratified, rs=42)."""
    df = base.load_data()
    X_A = df[base.MODEL_A_FEATURES]
    X_B = df[base.MODEL_B_FEATURES]
    y = df[base.TARGET]

    X_train_A, X_test_A, y_train, y_test = train_test_split(
        X_A, y, test_size=0.2, random_state=42, stratify=y)
    X_train_B = X_B.loc[X_train_A.index]
    X_test_B = X_B.loc[X_test_A.index]

    print(f"Dataset: {df.shape}  | train {X_train_A.shape[0]} / "
          f"test {X_test_A.shape[0]}  | positives in test: "
          f"{int(y_test.sum())}")
    return dict(df=df, X_A=X_A, X_B=X_B, y=y,
                X_train_A=X_train_A, X_test_A=X_test_A,
                X_train_B=X_train_B, X_test_B=X_test_B,
                y_train=y_train, y_test=y_test)


# ==================================================================
# STEP 1  Final comparison computed from fitted models (no hard-coding)
# ==================================================================
def step_comparison(data, state):
    header("1. FINAL MODEL COMPARISON (computed, not hard-coded)")
    ytr, yte = data["y_train"], data["y_test"]
    rows = []
    for ab, Xtr, Xte in (("A", data["X_train_A"], data["X_test_A"]),
                         ("B", data["X_train_B"], data["X_test_B"])):
        for name in ("LR", "RF", "SVM"):
            pipe = make_pipeline(list(Xtr.columns), name, ab).fit(Xtr, ytr)
            prob = pipe.predict_proba(Xte)[:, 1]
            m = metrics_at_threshold(yte, prob, 0.50)
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


# ==================================================================
# STEP 2  Thresholds (wider grid), calibration, confidence intervals
# ==================================================================
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
        oof = cross_val_predict(clone(pipe), Xtr, ytr, cv=base.cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        tbl = threshold_table(ytr, oof, THRESHOLDS)
        row, rule_met = select_threshold(tbl)
        thr = float(row["Threshold"])

        print(f"\n--- Model {ab}: tuned SVM, OOF threshold table ---")
        print(tbl.to_string(index=False,
                            float_format=lambda x: f"{x:.4f}"))
        print(f"\nSelected threshold (rule: recall>={MIN_RECALL:.0%}, then "
              f"best F1): {thr:.3f}   | rule met: {rule_met}")
        if thr in (THRESHOLDS.min(), THRESHOLDS.max()):
            print("  WARNING: selected threshold is at the edge of the grid; "
                  "consider widening it further.")
        if not rule_met:
            print(f"  The {MIN_RECALL:.0%} recall target is NOT reachable "
                  f"for Model {ab} even with the wider grid.")
        op = operating_points(tbl)
        print(f"\nOperating points for Model {ab} "
              f"(best specificity at each recall target):")
        print(op.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        tbl.to_csv(os.path.join(OUT_DIR, f"threshold_table_model_{ab}.csv"),
                   index=False)
        op.to_csv(os.path.join(OUT_DIR, f"operating_points_model_{ab}.csv"),
                  index=False)

        # calibration on OOF probabilities
        brier = brier_score_loss(ytr, oof)
        frac_pos, mean_pred = calibration_curve(ytr, oof, n_bins=8,
                                                strategy="quantile")
        ax_cal.plot(mean_pred, frac_pos, marker="o",
                    label=f"Model {ab} SVM (Brier={brier:.3f})")
        print(f"\nModel {ab} OOF Brier score: {brier:.4f} "
              f"(lower is better; class_weight='balanced' shifts "
              f"probabilities, so check the calibration plot)")

        # trade-off plot
        ax = ax_tr[k]
        for col in ("Recall", "Specificity", "Precision", "F1"):
            ax.plot(tbl["Threshold"], tbl[col], marker="o", ms=3, label=col)
        ax.axvline(thr, color="k", ls="--", lw=1, label=f"selected {thr:.3f}")
        ax.axhline(MIN_RECALL, color="grey", ls=":", lw=1)
        ax.set_title(f"Model {ab} - tuned SVM (OOF, training set)")
        ax.set_xlabel("Decision threshold")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)

        # fit on full training set, evaluate on held-out test with CIs
        fitted = clone(pipe).fit(Xtr, ytr)
        prob_te = fitted.predict_proba(Xte)[:, 1]
        print_test_with_ci(f"Model {ab} - held-out test @0.50", yte,
                           prob_te, 0.50)
        print_test_with_ci(f"Model {ab} - held-out test @selected", yte,
                           prob_te, thr)

        state[ab] = dict(pipe=fitted, thr=thr, rule_met=rule_met, oof=oof,
                         prob_test=prob_te, table=tbl, operating=op)

    fig_tr.tight_layout()
    fig_tr.savefig(os.path.join(FIG_DIR, "threshold_tradeoff.png"), dpi=150)
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Observed fraction of PCOS")
    ax_cal.set_title("Calibration (out-of-fold, training set)")
    ax_cal.legend(fontsize=8)
    fig_cal.tight_layout()
    fig_cal.savefig(os.path.join(FIG_DIR, "calibration.png"), dpi=150)
    plt.close("all")


# ==================================================================
# STEP 2b  Fair model comparison at a MATCHED recall (OOF probabilities)
# ==================================================================
def step_matched_recall(data, state):
    header("2b. MODELS COMPARED AT MATCHED RECALL (out-of-fold, training set)")
    print("Comparing classifiers at the default 0.5 threshold is unfair "
          "(e.g. Model B RF has higher AUC than SVM but lower recall @0.5).\n"
          "Here every model is moved to the threshold that reaches the "
          "recall target, and we compare specificity / precision there.")
    ytr = data["y_train"]
    rows = []
    for ab, Xtr in (("A", data["X_train_A"]), ("B", data["X_train_B"])):
        feats = list(Xtr.columns)
        models = {
            "LR": make_pipeline(feats, "LR", ab),
            "SVM": make_pipeline(feats, "SVM", ab),
            "RF": make_pipeline(feats, "RF", ab),
            "Soft voting": Pipeline([
                ("preprocessing", build_preprocessor(feats, scale=True)),
                ("model", VotingClassifier(
                    estimators=[("lr", make_clf("LR", ab)),
                                ("svm", make_clf("SVM", ab)),
                                ("rf", make_clf("RF", ab))],
                    voting="soft"))]),
        }
        for name, pipe in models.items():
            oof = cross_val_predict(clone(pipe), Xtr, ytr, cv=base.cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]
            tbl = threshold_table(ytr, oof, THRESHOLDS)
            auc = roc_auc_score(ytr, oof)
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
                                 "Threshold": r["Threshold"],
                                 "Recall": r["Recall"],
                                 "Specificity": r["Specificity"],
                                 "Precision": r["Precision"],
                                 "False Negatives": int(r["False Negatives"]),
                                 "OOF ROC-AUC": auc})
    df = pd.DataFrame(rows)
    for ab in ("A", "B"):
        print(f"\n--- Model {ab} ---")
        print(df[df["Model set"] == ab].drop(columns="Model set").to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(os.path.join(OUT_DIR, "matched_recall_comparison.csv"),
              index=False)
    state["matched_recall"] = df



def _ensure_AB(data, state):
    """Make sure fitted A/B SVMs + thresholds exist (cheap if not done)."""
    if "A" not in state or "B" not in state:
        step_thresholds(data, state)


# ==================================================================
# STEP 3  SHAP + coefficients for Model B
# ==================================================================
def _class1_shap(sv):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def step_shap_b(data, state):
    header("3. MODEL B (NO ULTRASOUND): SHAP + LINEAR-SVM COEFFICIENTS")
    import shap

    Xtr, ytr = data["X_train_B"], data["y_train"]

    # (a) SHAP on the tuned Random Forest for Model B
    rf = make_pipeline(list(Xtr.columns), "RF", "B").fit(Xtr, ytr)
    pre, rf_model = rf.named_steps["preprocessing"], rf.named_steps["model"]
    Xp = pre.transform(Xtr)
    names = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(rf_model)
    sv = _class1_shap(explainer.shap_values(Xp))
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
    except Exception as e:  # plotting is best-effort
        print("  (SHAP summary plot skipped:", e, ")")
    plt.close("all")

    # (b) coefficients of the FINAL model (linear SVM) - matches what we ship
    svm = make_pipeline(list(Xtr.columns), "SVM", "B").fit(Xtr, ytr)
    model = svm.named_steps["model"]
    if getattr(model, "kernel", None) == "linear":
        coef = pd.DataFrame({
            "Feature": svm.named_steps["preprocessing"]
                          .get_feature_names_out(),
            "Coefficient": model.coef_.ravel(),
        })
        coef["|Coefficient|"] = coef["Coefficient"].abs()
        coef = coef.sort_values("|Coefficient|", ascending=False)
        print("\n--- Model B linear SVM: standardised coefficients (top 15) ---")
        print(coef.head(15).to_string(index=False))
        coef.to_csv(os.path.join(OUT_DIR, "svm_coefficients_model_B.csv"),
                    index=False)
    state["shap_B"] = imp


# ==================================================================
# STEP 4  Feature-group ablation + cost tiers (Model B)
# ==================================================================
def step_ablation(data, state):
    header("4. FEATURE-GROUP ABLATION + COST TIERS (MODEL B, linear SVM)")
    all_feats = [f for g in FEATURE_GROUPS_B.values() for f in g]
    assert sorted(all_feats) == sorted(base.MODEL_B_FEATURES), \
        "FEATURE_GROUPS_B must cover exactly the Model B features"

    Xtr, ytr = data["X_train_B"], data["y_train"]
    cv = get_repeated_cv()

    # (a) cumulative cost tiers
    rows, used = [], []
    for label, groups in COST_TIERS_B:
        for g in groups:
            used += FEATURE_GROUPS_B[g]
        feats = list(dict.fromkeys(used))
        summ = cv_summary(make_pipeline(feats, "SVM", "B"), Xtr[feats], ytr, cv)
        rows.append(summary_row(label, summ, n_features=len(feats)))
    tiers = pd.DataFrame(rows)
    print("\n--- Cumulative cost tiers (cheapest first) ---")
    print_summary_table(tiers)
    tiers.to_csv(os.path.join(OUT_DIR, "ablation_cost_tiers.csv"), index=False)

    # (b) leave-one-group-out from the full Model B
    full = list(base.MODEL_B_FEATURES)
    rows = [summary_row("Full Model B", cv_summary(
        make_pipeline(full, "SVM", "B"), Xtr[full], ytr, cv),
        n_features=len(full))]
    for g, gf in FEATURE_GROUPS_B.items():
        feats = [f for f in full if f not in gf]
        summ = cv_summary(make_pipeline(feats, "SVM", "B"),
                          Xtr[feats], ytr, cv)
        rows.append(summary_row(f"minus {g}", summ, n_features=len(feats)))
    logo = pd.DataFrame(rows)
    print("\n--- Leave-one-group-out (drop in AUC/recall = group importance) ---")
    print_summary_table(logo)
    logo.to_csv(os.path.join(OUT_DIR, "ablation_leave_one_group_out.csv"),
                index=False)

    # plot
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
    print("\nNote: hyper-parameters were NOT re-tuned per subset "
          "(reported Model B SVM params reused), so differences are "
          "indicative.")
    state["ablation_tiers"], state["ablation_logo"] = tiers, logo


# ==================================================================
# STEP 5  Engineered features
# ==================================================================
def add_engineered_features(X, regular_code=None):
    """Adds clinically motivated features (all derived row-wise):
       Symptom_count         - number of the 5 symptom flags that are 1
       Irregular_x_CycleLen  - irregular-cycle flag * cycle length
       AMH_x_LH              - AMH * LH
       BMI_over_23           - 1 if BMI >= 23 (Asian overweight cut-off)
    Cycle(R/I) in this dataset has codes 2, 4 and 5 (seen in your output).
    regular_code = the code meaning 'regular' (2 in the Kaggle PCOS data);
    every other code (4, 5) is treated as irregular. If None, the smallest
    code present is used as 'regular'."""
    X = X.copy()
    X["Symptom_count"] = X[SYMPTOM_COLS].apply(
        pd.to_numeric, errors="coerce").sum(axis=1)

    cyc = pd.to_numeric(X["Cycle(R/I)"], errors="coerce")
    code = regular_code if regular_code is not None else cyc.min()
    irregular = (cyc != code).astype(float).where(cyc.notna())
    length = pd.to_numeric(X["Cycle length(days)"], errors="coerce")
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
    code = int(counts.index.min())
    print("Cycle(R/I) value counts:", counts.to_dict())
    print(f"Treating {code} as 'regular' and every other code as "
          f"'irregular'. VERIFY this against your data dictionary.")

    Xtr, ytr = data["X_train_B"], data["y_train"]
    Xtr_eng = add_engineered_features(Xtr, regular_code=code)
    feats_plain = list(Xtr.columns)
    feats_eng = feats_plain + ENGINEERED_NAMES
    cv = get_repeated_cv()

    rows = []
    for name in ("LR", "SVM", "RF"):
        rows.append(summary_row(f"B plain - {name}", cv_summary(
            make_pipeline(feats_plain, name, "B"), Xtr, ytr, cv)))
        rows.append(summary_row(f"B + engineered - {name}", cv_summary(
            make_pipeline(feats_eng, name, "B"), Xtr_eng[feats_eng], ytr, cv)))
    df = pd.DataFrame(rows)
    print_summary_table(df)
    df.to_csv(os.path.join(OUT_DIR, "engineered_features_comparison.csv"),
              index=False)
    state["engineered_regular_code"] = code
    state["engineered_cmp"] = df


# ==================================================================
# STEP 6  Other model families (XGBoost, voting, stacking)
# ==================================================================
def step_families(data, state):
    header("6. OTHER MODEL FAMILIES FOR MODEL B (repeated 5x5 CV)")
    Xtr, ytr = data["X_train_B"], data["y_train"]
    feats = list(Xtr.columns)
    cv = get_repeated_cv()
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())

    def shared_prep():
        return build_preprocessor(feats, scale=True)

    candidates = {
        "LR (tuned)": make_pipeline(feats, "LR", "B"),
        "SVM linear (tuned)": make_pipeline(feats, "SVM", "B"),
        "RF (tuned)": make_pipeline(feats, "RF", "B"),
        "Soft voting LR+SVM+RF": Pipeline([
            ("preprocessing", shared_prep()),
            ("model", VotingClassifier(
                estimators=[("lr", make_clf("LR", "B")),
                            ("svm", make_clf("SVM", "B")),
                            ("rf", make_clf("RF", "B"))],
                voting="soft")),
        ]),
        "Stacking LR+SVM+RF -> LR": Pipeline([
            ("preprocessing", shared_prep()),
            ("model", StackingClassifier(
                estimators=[("lr", make_clf("LR", "B")),
                            ("svm", make_clf("SVM", "B")),
                            ("rf", make_clf("RF", "B"))],
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
        xgb_grid = {
            "model__max_depth": [2, 3, 4],
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.1],
        }
        # nested: the grid search is re-run inside every outer fold, so this
        # score is not optimistic (unlike the others, which reuse params
        # tuned earlier on the same training data)
        candidates["XGBoost (tuned, nested)"] = GridSearchCV(
            xgb_pipe, xgb_grid, cv=3, scoring="roc_auc", n_jobs=1)
    else:
        print("xgboost not installed -> skipping XGBoost "
              "(pip install xgboost).")

    rows = []
    for label, est in candidates.items():
        print(f"  evaluating: {label}")
        rows.append(summary_row(label, cv_summary(est, Xtr, ytr, cv)))
    df = pd.DataFrame(rows)
    print()
    print_summary_table(df)
    df.to_csv(os.path.join(OUT_DIR, "model_families_model_B.csv"), index=False)
    print("\nNote: all except XGBoost reuse hyper-parameters tuned earlier on "
          "this same training set, so they are slightly optimistic relative "
          "to the nested XGBoost number.")
    state["families"] = df


# ==================================================================
# STEP 7  Nested CV of the full procedure (tuning + threshold choice)
# ==================================================================
def nested_threshold_eval(X, y, ab, param_grid, outer_cv, inner_cv):
    """For every outer fold: tune on the outer-training part (inner CV), choose
    the threshold by the recall rule on inner OOF predictions, then apply both
    to the untouched outer-test part."""
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    pipe = make_pipeline(list(X.columns), "SVM", ab)

    fold_rows, pooled_true, pooled_pred, pooled_prob = [], [], [], []
    for i, (tr, te) in enumerate(outer_cv.split(X, y), 1):
        Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
        gs = GridSearchCV(clone(pipe), param_grid, cv=inner_cv,
                          scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
        best = gs.best_estimator_
        oof = cross_val_predict(clone(best), Xtr, ytr, cv=inner_cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        row, _ = select_threshold(threshold_table(ytr, oof, THRESHOLDS))
        thr = float(row["Threshold"])
        prob = best.predict_proba(Xte)[:, 1]
        m = metrics_at_threshold(yte, prob, thr)
        fold_rows.append({"fold": i, "threshold": thr,
                          "recall": m["Recall"],
                          "specificity": m["Specificity"],
                          "precision": m["Precision"], "f1": m["F1"],
                          "roc_auc": m["ROC-AUC"],
                          "best_params": str(gs.best_params_)})
        pooled_true += list(yte)
        pooled_pred += list((prob >= thr).astype(int))
        pooled_prob += list(prob)
    return pd.DataFrame(fold_rows), np.array(pooled_true), \
        np.array(pooled_pred), np.array(pooled_prob)


def step_nested(data, state):
    header("7. NESTED CV OF THE WHOLE PROCEDURE (tune + pick threshold)")
    grid = {"model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "linear"]}
    outer = RepeatedStratifiedKFold(n_splits=NESTED_OUTER_SPLITS,
                                    n_repeats=NESTED_OUTER_REPEATS,
                                    random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=5, shuffle=True,
                            random_state=RANDOM_STATE)
    for ab, X in (("A", data["X_A"]), ("B", data["X_B"])):
        print(f"\n--- Model {ab}: nested CV on all {len(X)} patients ---")
        folds, yt, yp, _ = nested_threshold_eval(
            X, data["y"], ab, grid, outer, inner)
        print(folds.drop(columns="best_params").to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nMean ± SD over outer folds:")
        for c in ("threshold", "recall", "specificity", "precision",
                  "f1", "roc_auc"):
            print(f"  {c:<12}{folds[c].mean():.4f} ± {folds[c].std():.4f}")
        tp = int(((yt == 1) & (yp == 1)).sum())
        pos = int((yt == 1).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        neg = int((yt == 0).sum())
        lo, hi = wilson_ci(tp, pos)
        slo, shi = wilson_ci(tn, neg)
        print(f"  pooled recall      {tp}/{pos} = {tp/pos:.4f}  "
              f"Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  pooled specificity {tn}/{neg} = {tn/neg:.4f}  "
              f"Wilson 95% CI [{slo:.3f}, {shi:.3f}]")
        folds.to_csv(os.path.join(OUT_DIR, f"nested_cv_model_{ab}.csv"),
                     index=False)
        state[f"nested_{ab}"] = folds


# ==================================================================
# STEP 8  Two-stage screening cascade: B -> ultrasound -> A
# ==================================================================
def step_cascade(data, state):
    header("8. SCREENING CASCADE: Model B screens, Model A confirms")
    _ensure_AB(data, state)
    y = np.asarray(data["y_test"])
    pB, pA = state["B"]["prob_test"], state["A"]["prob_test"]
    thrA = state["A"]["thr"]
    pos, neg = (y == 1).sum(), (y == 0).sum()

    rows = []
    for thrB in THRESHOLDS:
        referred = pB >= thrB                   # sent for ultrasound
        final_pos = referred & (pA >= thrA)     # confirmed by Model A
        tp = (final_pos & (y == 1)).sum()
        tn = (~final_pos & (y == 0)).sum()
        rows.append({
            "Model B threshold": thrB,
            "Referred for ultrasound (%)": 100 * referred.mean(),
            "Ultrasounds saved (%)": 100 * (1 - referred.mean()),
            "Cascade recall": tp / pos,
            "Cascade specificity": tn / neg,
            "Cascade precision": tp / final_pos.sum()
            if final_pos.sum() else np.nan,
            "Missed PCOS (FN)": int(pos - tp),
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
    print("Baseline (everyone gets ultrasound + Model A): "
          f"recall = {(((pA >= thrA) & (y == 1)).sum() / pos):.3f}, "
          "ultrasounds saved = 0%.")

    plt.figure(figsize=(7, 4.5))
    plt.plot(df["Ultrasounds saved (%)"], df["Cascade recall"], marker="o",
             label="Cascade recall")
    plt.plot(df["Ultrasounds saved (%)"], df["Cascade specificity"],
             marker="s", label="Cascade specificity")
    plt.xlabel("Ultrasounds saved (%)")
    plt.title("Cascade: ultrasounds saved vs. accuracy (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cascade_tradeoff.png"), dpi=150)
    plt.close("all")
    state["cascade"] = df


# ==================================================================
# STEP 8b  LITE Model B: cheap features only (no ultrasound, no labs)
# ==================================================================
LITE_GROUPS = ["Cycle", "Symptoms", "Lifestyle", "Demographics"]


def lite_features(extra=()):
    """The cheap feature set: cycle (2) + symptoms (5) + lifestyle (2) + age
    (= 10 features), optionally plus extras such as 'BMI'. Order follows
    MODEL_B_FEATURES so column order is stable."""
    feats = [f for g in LITE_GROUPS for f in FEATURE_GROUPS_B[g]]
    feats += [f for f in extra if f not in feats]
    return [f for f in base.MODEL_B_FEATURES if f in feats]


def _lite_grids():
    if QUICK:
        return {
            "LR": {"model__C": [0.1, 1, 10], "model__penalty": ["l1", "l2"],
                   "model__solver": ["liblinear"]},
            "SVM": {"model__C": [0.1, 1], "model__kernel": ["linear", "rbf"],
                    "model__gamma": ["scale"]},
            "RF": {"model__n_estimators": [100], "model__max_depth": [5, 10],
                   "model__min_samples_split": [5],
                   "model__min_samples_leaf": [2]},
        }
    return {
        "LR": {"model__C": [0.01, 0.1, 1, 10, 100],
               "model__penalty": ["l1", "l2"],
               "model__solver": ["liblinear"]},
        "SVM": {"model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["rbf", "linear"]},
        "RF": {"model__n_estimators": [100, 200, 300],
               "model__max_depth": [None, 5, 10, 15],
               "model__min_samples_split": [2, 5, 10],
               "model__min_samples_leaf": [1, 2, 4]},
    }


def tune_lite_models(feats, Xtr, ytr, label):
    """GridSearchCV (ROC-AUC, same 5-fold CV as your original) for LR, SVM
    and RF on a feature subset."""
    out = {}
    for name, grid in _lite_grids().items():
        gs = GridSearchCV(make_pipeline(feats, name, "B"), grid, cv=base.cv,
                          scoring="roc_auc", n_jobs=-1).fit(Xtr[feats], ytr)
        out[name] = gs
        print(f"  {label:<14} {name:<4} best CV ROC-AUC {gs.best_score_:.4f}"
              f"   {gs.best_params_}")
    return out


def pick_simplest_near_best(tuned, tol=0.005):
    """Classifiers are within noise of each other, so pick the SIMPLEST one
    (LR < SVM < RF) whose CV AUC is within `tol` of the best."""
    best = max(g.best_score_ for g in tuned.values())
    for name in ("LR", "SVM", "RF"):
        if name in tuned and tuned[name].best_score_ >= best - tol:
            return name
    return max(tuned, key=lambda n: tuned[n].best_score_)


def paired_bootstrap_diff(y, p1, thr1, p2, thr2, n_boot=None,
                          seed=RANDOM_STATE):
    """Compare model 1 (lite) with model 2 (full) on the SAME test patients.
    Returns diff = model1 - model2 for several metrics, each at the model's
    own threshold, with a 95% bootstrap CI."""
    n_boot = N_BOOT if n_boot is None else n_boot
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    rng = np.random.default_rng(seed)
    keys = ("ROC-AUC", "PR-AUC", "Recall", "Specificity")
    diffs = {k: [] for k in keys}
    n = len(y)
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
        rows.append({"Metric": k,
                     "Lite - Full (mean)": d.mean(),
                     "95% CI low": np.nanpercentile(d, 2.5),
                     "95% CI high": np.nanpercentile(d, 97.5),
                     "P(Lite <= Full)": float((d <= 0).mean())})
    return pd.DataFrame(rows)


def _cascade_table(y, p_stage1, p_A, thrA):
    y = np.asarray(y)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    rows = []
    for t in THRESHOLDS:
        referred = p_stage1 >= t
        final_pos = referred & (p_A >= thrA)
        tp = (final_pos & (y == 1)).sum()
        tn = (~final_pos & (y == 0)).sum()
        rows.append({"Stage-1 threshold": t,
                     "Referred for ultrasound (%)": 100 * referred.mean(),
                     "Ultrasounds saved (%)": 100 * (1 - referred.mean()),
                     "Cascade recall": tp / pos,
                     "Cascade specificity": tn / neg,
                     "Missed PCOS (FN)": int(pos - tp)})
    return pd.DataFrame(rows)


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
    print("These need NO ultrasound and NO blood test: only a short history "
          "questionnaire (+ age; BMI needs weight and height).\n")

    # ---- (1) re-tune LR / SVM / RF on each cheap set --------------------
    print("--- (1) Hyper-parameter tuning on the cheap feature sets ---")
    tuned, chosen = {}, {}
    for label, feats in sets.items():
        tuned[label] = tune_lite_models(feats, Xtr, ytr, label)
        chosen[label] = pick_simplest_near_best(tuned[label])
        print(f"  -> {label}: using {chosen[label]} (simplest classifier "
              f"within 0.005 CV-AUC of the best)\n")

    # ---- (2) repeated 5x5 CV: lite vs full Model B ----------------------
    print("--- (2) Repeated 5x5 CV on the training set ---")
    cv = get_repeated_cv()
    full_feats = list(Xtr.columns)
    rows = [summary_row(
        "Full Model B (29) SVM", cv_summary(
            make_pipeline(full_feats, "SVM", "B"), Xtr, ytr, cv),
        n_features=len(full_feats))]
    for label, feats in sets.items():
        best = tuned[label][chosen[label]].best_estimator_
        rows.append(summary_row(
            f"{label} {chosen[label]} (tuned)",
            cv_summary(best, Xtr[feats], ytr, cv), n_features=len(feats)))
    cmp_df = pd.DataFrame(rows)
    print_summary_table(cmp_df)
    cmp_df.to_csv(os.path.join(OUT_DIR, "lite_vs_full_cv.csv"), index=False)

    # ---- (3) primary lite model: thresholds + operating points ----------
    primary = "Lite-10"
    feats = sets[primary]
    clf_name = chosen[primary]
    best = tuned[primary][clf_name].best_estimator_
    print(f"\n--- (3) Primary lite model: {primary} + {clf_name} ---")
    oof = cross_val_predict(clone(best), Xtr[feats], ytr, cv=base.cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    tbl = threshold_table(ytr, oof, THRESHOLDS)
    row, rule_met = select_threshold(tbl)
    thr = float(row["Threshold"])
    op = operating_points(tbl)
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

    fitted = clone(best).fit(Xtr[feats], ytr)
    prob_te = fitted.predict_proba(Xte[feats])[:, 1]
    print_test_with_ci("LITE model - held-out test @0.50", yte, prob_te, 0.50)
    print_test_with_ci("LITE model - held-out test @selected", yte, prob_te,
                       thr)

    # ---- (4) paired bootstrap: lite vs full on the SAME test patients ---
    print("\n--- (4) Paired bootstrap on the same test patients "
          "(Lite minus Full) ---")
    pf, thr_f = state["B"]["prob_test"], state["B"]["thr"]
    diff = paired_bootstrap_diff(yte, prob_te, thr, pf, thr_f)
    print(f"Lite threshold = {thr:.3f}, Full Model B threshold = {thr_f:.3f} "
          f"(each uses its own recall-rule threshold)")
    print(diff.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    auc_row = diff[diff["Metric"] == "ROC-AUC"].iloc[0]
    margin = 0.05
    if auc_row["95% CI low"] > -margin:
        print(f"\nROC-AUC: the lite model is NON-INFERIOR to the full model "
              f"at a margin of {margin} (lower CI bound "
              f"{auc_row['95% CI low']:+.3f} > -{margin}).")
    else:
        print(f"\nROC-AUC: non-inferiority at margin {margin} is NOT shown "
              f"(lower CI bound {auc_row['95% CI low']:+.3f}).")
    print(f"(With only {int(yte.sum())} positive test cases these intervals "
          "are wide; a CI that contains 0 means 'no detectable difference', "
          "not 'equal'.)")
    diff.to_csv(os.path.join(OUT_DIR, "lite_vs_full_paired_bootstrap.csv"),
                index=False)

    # ---- (5) cascade with the lite model as stage 1 ---------------------
    print("\n--- (5) Cascade: LITE screen -> ultrasound + Model A ---")
    thrA = state["A"]["thr"]
    casc = _cascade_table(yte, prob_te, state["A"]["prob_test"], thrA)
    print(casc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    casc.to_csv(os.path.join(OUT_DIR, "cascade_lite.csv"), index=False)

    # ---- (6) ROC comparison plot + freeze lite model --------------------
    plt.figure(figsize=(5.8, 5))
    for lab, pr in (("Model A (29+5 features)", state["A"]["prob_test"]),
                    ("Model B full (29)", pf),
                    ("Lite (10, no labs)", prob_te)):
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
            "features": feats, "target": base.TARGET,
            "test_metrics_at_threshold": {
                k: (float(v) if not isinstance(v, int) else v)
                for k, v in m.items()},
            "intended_use": "screening aid, NOT a diagnosis",
        }, f, indent=2)
    print(f"\nSaved {path} and models/model_B_lite_metadata.json")
    state["lite"] = dict(feats=feats, clf=clf_name, pipe=fitted, thr=thr,
                         prob_test=prob_te, table=tbl, operating=op,
                         diff=diff, cascade=casc, tuned=tuned)


# ==================================================================
# STEP 9  ROC / PR curves + confusion matrices
# ==================================================================
def step_plots(data, state):
    header("9. ROC / PR CURVES AND CONFUSION MATRICES")
    _ensure_AB(data, state)
    yte = np.asarray(data["y_test"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for ab in ("A", "B"):
        prob = state[ab]["prob_test"]
        fpr, tpr, _ = roc_curve(yte, prob)
        pr, rc, _ = precision_recall_curve(yte, prob)
        ax[0].plot(fpr, tpr, label=f"Model {ab} (AUC="
                                   f"{roc_auc_score(yte, prob):.3f})")
        ax[1].plot(rc, pr, label=f"Model {ab} (AP="
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
        thr = state[ab]["thr"]
        pred = (state[ab]["prob_test"] >= thr).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            yte, pred, display_labels=["No PCOS", "PCOS"],
            ax=ax[k], colorbar=False)
        ax[k].set_title(f"Model {ab} - SVM @ threshold {thr:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=150)
    plt.close("all")
    print(f"Saved figures to {FIG_DIR}/")


# ==================================================================
# STEP 10  Freeze final pipelines + thresholds
# ==================================================================
def step_freeze(data, state):
    header("10. FREEZE FINAL PIPELINES + THRESHOLDS")
    _ensure_AB(data, state)
    meta = {}
    for ab, feats in (("A", base.MODEL_A_FEATURES),
                      ("B", base.MODEL_B_FEATURES)):
        path = os.path.join(MODEL_DIR, f"model_{ab}_svm.joblib")
        joblib.dump(state[ab]["pipe"], path)
        thr = state[ab]["thr"]
        yte = data["y_test"]
        m = metrics_at_threshold(yte, state[ab]["prob_test"], thr)
        meta[f"model_{ab}"] = {
            "file": path,
            "classifier": "SVC (tuned, class_weight=balanced, "
                          "probability=True)",
            "params": REPORTED_PARAMS[(ab, "SVM")],
            "threshold": thr,
            "threshold_rule": f"recall>={MIN_RECALL} then max F1 on "
                              "out-of-fold training predictions",
            "threshold_rule_met": bool(state[ab]["rule_met"]),
            "features": list(feats),
            "target": base.TARGET,
            "test_metrics_at_threshold": {
                k: (float(v) if not isinstance(v, int) else v)
                for k, v in m.items()},
            "trained_on": "80% stratified training split (random_state=42)",
            "intended_use": "screening aid, NOT a diagnosis",
        }
        print(f"Saved {path}   (threshold {thr:.3f})")
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {MODEL_DIR}/model_metadata.json")
    print("\nLoad later with:  pipe = joblib.load('models/model_B_svm.joblib');"
          "  prob = pipe.predict_proba(X_new[features])[:, 1];"
          "  pred = prob >= threshold")


# ==================================================================
# MAIN
# ==================================================================
ALL_STEPS = ["comparison", "thresholds", "matched_recall", "shap_b", "ablation", "engineered",
             "families", "nested", "cascade", "lite", "plots", "freeze"]
STEP_FUNCS = {
    "comparison": step_comparison, "thresholds": step_thresholds,
    "matched_recall": step_matched_recall,
    "shap_b": step_shap_b, "ablation": step_ablation,
    "engineered": step_engineered, "families": step_families,
    "nested": step_nested, "cascade": step_cascade,
    "lite": step_lite,
    "plots": step_plots, "freeze": step_freeze,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="*", default=ALL_STEPS,
                        choices=ALL_STEPS)
    parser.add_argument("--quick", action="store_true",
                        help="fewer CV repeats / bootstrap draws (smoke test)")
    args = parser.parse_args()

    if args.quick:
        QUICK = True
        CV_REPEATS = 1
        NESTED_OUTER_REPEATS = 1
        NESTED_OUTER_SPLITS = 3
        N_BOOT = 200

    ensure_dirs()
    data = load_split()
    state = {}
    for s in args.steps:
        STEP_FUNCS[s](data, state)

    header("DONE - tables in outputs/, figures in outputs/figures/, "
          "models in models/")