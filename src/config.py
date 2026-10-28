"""
config.py
====================================================================
Central configuration for the PCOS Screening Aid project.

All feature lists, constants, hyperparameters, and settings used
across the project are defined here.  Nothing in this file trains
or evaluates models — it only holds static definitions.
"""

import numpy as np
from sklearn.metrics import make_scorer, matthews_corrcoef

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_PATH = "data/pcos_updated.xlsx"
OUT_DIR = "outputs"
FIG_DIR = "outputs/figures"
MODEL_DIR = "models"

# ------------------------------------------------------------------
# Target variable
# ------------------------------------------------------------------
TARGET = "PCOS (Y/N)"

# ------------------------------------------------------------------
# Feature lists
# ------------------------------------------------------------------

# Model A: Full clinical model INCLUDING ultrasound features
MODEL_A_FEATURES = [
    "Age (yrs)",
    "Weight (Kg)",
    "Height(Cm)",
    "BMI",
    "Pulse rate(bpm)",
    "Hb(g/dl)",
    "Cycle(R/I)",
    "Cycle length(days)",
    "FSH(mIU/mL)",
    "LH(mIU/mL)",
    "FSH/LH",
    "Hip(inch)",
    "Waist(inch)",
    "Waist:Hip Ratio",
    "TSH (mIU/L)",
    "AMH(ng/mL)",
    "PRL(ng/mL)",
    "Vit D3 (ng/mL)",
    "PRG(ng/mL)",
    "RBS(mg/dl)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
    "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)",
    "Follicle No. (L)",
    "Follicle No. (R)",
    "Avg. F size (L) (mm)",
    "Avg. F size (R) (mm)",
    "Endometrium (mm)"
]

# Model B: Same clinical information, but WITHOUT ultrasound
MODEL_B_FEATURES = [
    feature
    for feature in MODEL_A_FEATURES
    if feature not in [
        "Follicle No. (L)",
        "Follicle No. (R)",
        "Avg. F size (L) (mm)",
        "Avg. F size (R) (mm)",
        "Endometrium (mm)"
    ]
]

BINARY_FEATURES = [
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)"
]

CYCLE_FEATURE = "Cycle(R/I)"

NUMERIC_FEATURES = [
    feature
    for feature in MODEL_A_FEATURES
    if feature not in BINARY_FEATURES
    and feature != CYCLE_FEATURE
]

NUMERIC_FEATURES_B = [
    feature
    for feature in MODEL_B_FEATURES
    if feature not in BINARY_FEATURES
    and feature != CYCLE_FEATURE
]

SYMPTOM_COLS = [
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
]

# ------------------------------------------------------------------
# Feature groups for Model B (used for ablation / cost tiers)
# ------------------------------------------------------------------
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

# Lite model: cheap features only (no ultrasound, no labs)
LITE_GROUPS = ["Cycle", "Symptoms", "Lifestyle", "Demographics"]

# ------------------------------------------------------------------
# Hyperparameters / settings
# ------------------------------------------------------------------
RANDOM_STATE = 42
MIN_RECALL = 0.90                                   # clinical target
THRESHOLDS = np.round(np.arange(0.10, 0.601, 0.025), 3)

CV_REPEATS = 5          # repeated stratified 5-fold (5x5)
NESTED_OUTER_SPLITS = 5
NESTED_OUTER_REPEATS = 2
N_BOOT = 2000
QUICK = False           # set by --quick

# Best hyper-parameters reported in the progress report (GridSearchCV results)
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

# Engineered feature names
ENGINEERED_NAMES = ["Symptom_count", "Irregular_x_CycleLen",
                    "AMH_x_LH", "BMI_over_23"]

# ------------------------------------------------------------------
# Scoring dictionary for cross_validate
# ------------------------------------------------------------------
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
