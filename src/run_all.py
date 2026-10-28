"""
run_all.py
====================================================================
CLI entry point for the PCOS Screening Aid training pipeline.

Run from the project root (where data/, models/, outputs/ live):

    python src/run_all.py                          # all steps
    python src/run_all.py --quick                  # fast smoke-test
    python src/run_all.py --steps thresholds shap_b freeze

Available steps (in recommended run order):
  comparison     1.  Final model comparison (LR / RF / SVM @ 0.50)
  matched_recall 2b. Fair comparison at matched recall
  thresholds     2.  Wider threshold grid + calibration + CIs
  shap_b         3.  SHAP + linear-SVM coefficients (Model B)
  ablation       4.  Feature-group ablation + cost-tier analysis
  engineered     5.  Engineered feature experiments
  families       6.  XGBoost, soft-voting, stacking ensembles
  nested         7.  Nested CV (tune + threshold) for Model A & B
  cascade        8.  Two-stage B -> ultrasound -> A screening cascade
  lite           8b. Lite Model B (10 no-lab features)
  plots          9.  ROC / PR curves + confusion matrices
  freeze         10. Save final pipelines + thresholds to disk
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg                        # central settings (mutable for --quick)
from data_loader import load_split
from training import step_comparison, step_matched_recall, header
from preprocessing import get_repeated_cv  # noqa: F401 — re-exported for any step that needs it
from explainability import (
    step_thresholds, step_shap_b, step_ablation,
    step_engineered, step_families, step_nested,
)
from cascade import step_cascade, step_lite
from visualization import step_plots
from freeze import step_freeze


# ------------------------------------------------------------------
# Step registry
# ------------------------------------------------------------------
ALL_STEPS = [
    "comparison", "matched_recall", "thresholds",
    "shap_b", "ablation", "engineered", "families",
    "nested", "cascade", "lite", "plots", "freeze",
]

STEP_FUNCS = {
    "comparison":     step_comparison,
    "matched_recall": step_matched_recall,
    "thresholds":     step_thresholds,
    "shap_b":         step_shap_b,
    "ablation":       step_ablation,
    "engineered":     step_engineered,
    "families":       step_families,
    "nested":         step_nested,
    "cascade":        step_cascade,
    "lite":           step_lite,
    "plots":          step_plots,
    "freeze":         step_freeze,
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run the PCOS Screening Aid training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps", nargs="*", default=ALL_STEPS, choices=ALL_STEPS,
        help="Which steps to run (default: all).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fewer CV repeats / bootstrap draws — for a fast smoke test.",
    )
    args = parser.parse_args()

    # Apply --quick overrides
    if args.quick:
        cfg.QUICK                = True
        cfg.CV_REPEATS           = 1
        cfg.NESTED_OUTER_REPEATS = 1
        cfg.NESTED_OUTER_SPLITS  = 3
        cfg.N_BOOT               = 200
        print("Running in QUICK mode (reduced CV repeats / bootstrap draws).\n")

    # Ensure output directories exist
    for d in (cfg.OUT_DIR, cfg.FIG_DIR, cfg.MODEL_DIR):
        os.makedirs(d, exist_ok=True)

    # Load data once
    data  = load_split()
    state = {}

    # Run selected steps
    for step in args.steps:
        STEP_FUNCS[step](data, state)

    header("DONE - tables in outputs/, figures in outputs/figures/, "
           "models in models/")


if __name__ == "__main__":
    main()
