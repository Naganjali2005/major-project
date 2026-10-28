"""
preprocessing.py
====================================================================
Preprocessor builders and sklearn Pipeline factories for the PCOS
Screening Aid.

Provides:
  - build_preprocessor(features, scale)  : generic preprocessor for any
                                           feature subset
  - make_clf(name, ab)                   : classifier with reported params
  - make_pipeline(features, name, ab)    : full preprocessing + model pipeline
  - get_repeated_cv()                    : RepeatedStratifiedKFold helper

The module-level objects (preprocessor, preprocessor_B, rf_preprocessor,
rf_preprocessor_B, cv) are kept for backward compatibility with any code
that imports them directly from here.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from config import (
    BINARY_FEATURES, CYCLE_FEATURE,
    NUMERIC_FEATURES, NUMERIC_FEATURES_B,
    RANDOM_STATE, CV_REPEATS,
    REPORTED_PARAMS,
)


# ------------------------------------------------------------------
# Generic preprocessor builder (works for any feature subset)
# ------------------------------------------------------------------
def build_preprocessor(features, scale=True):
    """
    Build a ColumnTransformer for any feature subset:
      numeric  -> median impute  (+ StandardScaler if scale=True)
      binary   -> most-frequent impute
      cycle    -> most-frequent impute + OneHotEncoder
    Unknown / engineered columns are treated as numeric.
    """
    binary  = [f for f in features if f in BINARY_FEATURES]
    cycle   = [f for f in features if f == CYCLE_FEATURE]
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


# ------------------------------------------------------------------
# Classifier factory (with reported tuned hyper-parameters)
# ------------------------------------------------------------------
def make_clf(name, ab):
    """Return a fresh classifier with the reported best params for (ab, name)."""
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
    raise ValueError(f"Unknown classifier name: {name}")


# ------------------------------------------------------------------
# Full pipeline factory
# ------------------------------------------------------------------
def make_pipeline(features, name, ab):
    """Pipeline with reported tuned params. RF is unscaled (tree-based)."""
    return Pipeline([
        ("preprocessing", build_preprocessor(features, scale=(name != "RF"))),
        ("model", make_clf(name, ab)),
    ])


# ------------------------------------------------------------------
# CV helpers
# ------------------------------------------------------------------
def get_repeated_cv():
    """Repeated Stratified K-Fold (5 splits × CV_REPEATS repeats)."""
    return RepeatedStratifiedKFold(
        n_splits=5, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)


# ------------------------------------------------------------------
# Module-level CV object (used by several scripts via `import preprocessing`)
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ------------------------------------------------------------------
# Module-level preprocessors (kept for backward-compatibility)
# ------------------------------------------------------------------
# SVM / LR preprocessors (with scaling)
preprocessor   = build_preprocessor(NUMERIC_FEATURES   + BINARY_FEATURES + [CYCLE_FEATURE], scale=True)
preprocessor_B = build_preprocessor(NUMERIC_FEATURES_B + BINARY_FEATURES + [CYCLE_FEATURE], scale=True)

# RF preprocessors (no scaling)
rf_preprocessor   = build_preprocessor(NUMERIC_FEATURES   + BINARY_FEATURES + [CYCLE_FEATURE], scale=False)
rf_preprocessor_B = build_preprocessor(NUMERIC_FEATURES_B + BINARY_FEATURES + [CYCLE_FEATURE], scale=False)
