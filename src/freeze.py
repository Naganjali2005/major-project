"""
freeze.py
====================================================================
Save the final trained pipelines + thresholds to disk.

Saves:
  models/model_A_svm.joblib
  models/model_B_svm.joblib
  models/model_metadata.json

Run as part of the full pipeline via run_all.py --steps freeze, or:
    python src/freeze.py
"""

import os
import json
import joblib

from config import (
    MODEL_DIR, OUT_DIR, FIG_DIR,
    MODEL_A_FEATURES, MODEL_B_FEATURES,
    MIN_RECALL, REPORTED_PARAMS,
    TARGET,
)
from evaluation import metrics_at_threshold
from explainability import _ensure_AB
from training import header


def step_freeze(data, state):
    """Step 10: Freeze final pipelines + thresholds to disk."""
    header("10. FREEZE FINAL PIPELINES + THRESHOLDS")
    _ensure_AB(data, state)

    meta = {}
    for ab, feats in (("A", MODEL_A_FEATURES),
                      ("B", MODEL_B_FEATURES)):
        path = os.path.join(MODEL_DIR, f"model_{ab}_svm.joblib")
        joblib.dump(state[ab]["pipe"], path)
        thr = state[ab]["thr"]
        yte = data["y_test"]
        m   = metrics_at_threshold(yte, state[ab]["prob_test"], thr)
        meta[f"model_{ab}"] = {
            "file":       path,
            "classifier": "SVC (tuned, class_weight=balanced, probability=True)",
            "params":     REPORTED_PARAMS[(ab, "SVM")],
            "threshold":  thr,
            "threshold_rule": f"recall>={MIN_RECALL} then max F1 on "
                              "out-of-fold training predictions",
            "threshold_rule_met": bool(state[ab]["rule_met"]),
            "features":   list(feats),
            "target":     TARGET,
            "test_metrics_at_threshold": {
                k: (float(v) if not isinstance(v, int) else v)
                for k, v in m.items()},
            "trained_on": "80% stratified training split (random_state=42)",
            "intended_use": "screening aid, NOT a diagnosis",
        }
        print(f"Saved {path}   (threshold {thr:.3f})")

    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")
    print("\nLoad later with:"
          "  pipe = joblib.load('models/model_B_svm.joblib');"
          "  prob = pipe.predict_proba(X_new[features])[:, 1];"
          "  pred = prob >= threshold")


if __name__ == "__main__":
    import os as _os
    for d in (OUT_DIR, FIG_DIR, MODEL_DIR):
        _os.makedirs(d, exist_ok=True)
    from data_loader import load_split
    data  = load_split()
    state = {}
    step_freeze(data, state)
