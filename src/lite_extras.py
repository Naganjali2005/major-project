"""
lite_extras.py
====================================================================
Extra analyses for the Lite Model B:
  - Like-for-like comparison vs the full Model B Random Forest
  - Nested CV of the whole lite procedure
  - Paper table at 85% and 90% recall operating points

Run from the project root:
    python src/lite_extras.py                    # all three steps
    python src/lite_extras.py --quick
    python src/lite_extras.py --steps lite_rf lite_nested lite_paper
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_predict,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OUT_DIR, RANDOM_STATE,
    NESTED_OUTER_SPLITS, NESTED_OUTER_REPEATS, THRESHOLDS,
)
from preprocessing import make_pipeline, cv as _base_cv
from evaluation import (
    metrics_at_threshold, threshold_table, select_threshold,
    wilson_ci, bootstrap_ci, paired_bootstrap_diff,
    print_test_with_ci,
)
from training import header, cv_summary, summary_row, print_summary_table
from cascade import (
    lite_features, tune_lite_models, pick_simplest_near_best,
    step_lite, _lite_grids,
)
from explainability import _ensure_AB, step_thresholds

OP_TARGETS = (0.85, 0.90)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _ensure_lite(data, state):
    if "lite" not in state:
        step_lite(data, state)


def op_threshold(tbl, target):
    """Best-specificity threshold that reaches `target` recall."""
    el = tbl[tbl["Recall"] >= target]
    if len(el) == 0:
        r = tbl.loc[tbl["Recall"].idxmax()]
        return float(r["Threshold"]), False
    r = el.loc[el["Specificity"].idxmax()]
    return float(r["Threshold"]), True


def oof_profile(est, Xtr, ytr, Xte):
    """OOF threshold table + three operating thresholds + fitted model."""
    oof = cross_val_predict(clone(est), Xtr, ytr, cv=_base_cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    tbl = threshold_table(ytr, oof, THRESHOLDS)
    row, rule_met = select_threshold(tbl)
    prof = {"oof": oof, "table": tbl, "thr": float(row["Threshold"]),
            "rule_met": bool(rule_met)}
    for tg in OP_TARGETS:
        t, ok = op_threshold(tbl, tg)
        prof[f"thr{int(round(tg * 100))}"] = t
        prof[f"reach{int(round(tg * 100))}"] = ok
    fitted = clone(est).fit(Xtr, ytr)
    prof["fitted"]     = fitted
    prof["prob_test"]  = fitted.predict_proba(Xte)[:, 1]
    return prof


def point_estimates(prob, y, thr):
    y = np.asarray(y)
    m = metrics_at_threshold(y, prob, thr)
    pos, neg   = m["TP"] + m["FN"], m["TN"] + m["FP"]
    r_lo, r_hi = wilson_ci(m["TP"], pos)
    s_lo, s_hi = wilson_ci(m["TN"], neg)
    boot = bootstrap_ci(y, prob, thr)
    return {
        "Threshold":     thr,
        "Recall": m["Recall"], "Recall CI": (r_lo, r_hi),
        "Specificity": m["Specificity"], "Specificity CI": (s_lo, s_hi),
        "Precision": m["Precision"], "Precision CI": boot["Precision"],
        "F1": m["F1"], "FN": m["FN"], "FP": m["FP"],
        "TP": m["TP"], "TN": m["TN"],
        "Referred (%)": 100 * float((np.asarray(prob) >= thr).mean()),
    }


def _fmt_ci(v, ci):
    return f"{v:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"


# ------------------------------------------------------------------
# Step 8c: Lite vs full Model B Random Forest (like-for-like)
# ------------------------------------------------------------------
def step_lite_vs_rf(data, state):
    header("8c. LITE MODEL vs FULL MODEL B RANDOM FOREST (like-for-like)")
    _ensure_lite(data, state)
    Xtr, Xte = data["X_train_B"], data["X_test_B"]
    ytr, yte = data["y_train"], np.asarray(data["y_test"])
    L         = state["lite"]
    lite_feats = L["feats"]
    full_feats = list(Xtr.columns)
    clf        = L["clf"]

    models = {
        "Full B (29) RF":             (make_pipeline(full_feats, "RF", "B"), full_feats),
        "Full B (29) SVM":            (make_pipeline(full_feats, "SVM", "B"), full_feats),
        "Lite-10 RF (tuned)":         (L["tuned"]["Lite-10"]["RF"].best_estimator_, lite_feats),
        f"Lite-10 {clf} (chosen)":    (L["pipe"], lite_feats),
    }
    key_full_rf = "Full B (29) RF"
    key_lite_rf = "Lite-10 RF (tuned)"
    key_lite    = f"Lite-10 {clf} (chosen)"

    # (1) Repeated 5x5 CV
    from preprocessing import get_repeated_cv
    rep_cv = get_repeated_cv()
    print("--- (1) Repeated 5x5 CV, training set ---")
    rows = []
    for label, (est, feats) in models.items():
        summ = cv_summary(clone(est), Xtr[feats], ytr, rep_cv)
        rows.append(summary_row(label, summ, n_features=len(feats)))
    cvdf = pd.DataFrame(rows)
    print_summary_table(cvdf)
    cvdf.to_csv(os.path.join(OUT_DIR, "lite_vs_full_rf_cv.csv"), index=False)

    # (2) Thresholds + held-out test
    print("\n--- (2) Thresholds from out-of-fold predictions, then held-out test ---")
    profs = {}
    for label, (est, feats) in models.items():
        profs[label] = oof_profile(est, Xtr[feats], ytr, Xte[feats])
        p = profs[label]
        print(f"  {label:<26} rule thr {p['thr']:.3f} | "
              f"85%-recall thr {p['thr85']:.3f} | "
              f"90%-recall thr {p['thr90']:.3f}")
    for label in (key_full_rf, key_lite_rf, key_lite):
        p = profs[label]
        print_test_with_ci(f"{label} @0.50",           yte, p["prob_test"], 0.50)
        print_test_with_ci(f"{label} @rule threshold", yte, p["prob_test"], p["thr"])

    # (3) Matched-recall table on held-out test
    print("\n--- (3) Matched-recall operating points on the held-out test ---")
    rows = []
    for label, p in profs.items():
        for tg in OP_TARGETS:
            tag = int(round(tg * 100))
            thr = p[f"thr{tag}"]
            pt  = point_estimates(p["prob_test"], yte, thr)
            rows.append({
                "Model": label, "Recall target": tg, "Threshold": thr,
                "OOF target reachable":  p[f"reach{tag}"],
                "Test recall":           pt["Recall"],
                "Recall CI low":         pt["Recall CI"][0],
                "Recall CI high":        pt["Recall CI"][1],
                "Test specificity":      pt["Specificity"],
                "Spec CI low":           pt["Specificity CI"][0],
                "Spec CI high":          pt["Specificity CI"][1],
                "Test precision":        pt["Precision"],
                "False negatives":       pt["FN"],
                "Referred (%)":          pt["Referred (%)"],
            })
    mr = pd.DataFrame(rows)
    print(mr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    mr.to_csv(os.path.join(OUT_DIR, "lite_vs_full_rf_matched_recall.csv"),
              index=False)

    # (4) Paired bootstrap
    print("\n--- (4) Paired bootstrap (Lite minus Full RF) ---")
    comps  = []
    margin = 0.05
    for lite_label in (key_lite, key_lite_rf):
        for tag, name in (("thr", "own rule threshold"),
                          ("thr85", "85%-recall point"),
                          ("thr90", "90%-recall point")):
            pl, pf = profs[lite_label], profs[key_full_rf]
            d = paired_bootstrap_diff(
                yte, pl["prob_test"], pl[tag], pf["prob_test"], pf[tag])
            d.insert(0, "Operating point", name)
            d.insert(0, "Comparison", f"{lite_label}  minus  {key_full_rf}")
            comps.append(d)
            print(f"\n{lite_label} minus {key_full_rf}  @ {name}")
            print(d.drop(columns=["Comparison", "Operating point"])
                  .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    allc = pd.concat(comps, ignore_index=True)
    allc.to_csv(os.path.join(OUT_DIR, "lite_vs_full_rf_paired_bootstrap.csv"),
                index=False)
    state["lite_rf"] = dict(cv=cvdf, matched=mr, paired=allc, profiles=profs)


# ------------------------------------------------------------------
# Step 8d: Nested CV of the whole lite procedure
# ------------------------------------------------------------------
def nested_lite_eval(X, y, feats, outer_cv, inner_cv):
    X = X[feats].reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    grids      = _lite_grids()
    keys       = ["rule", "op85", "op90"]
    pooled     = {k: {"y": [], "pred": []} for k in keys}
    pooled_prob, pooled_y = [], []
    fold_rows  = []

    for i, (tr, te) in enumerate(outer_cv.split(X, y), 1):
        Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
        tuned = {}
        for name, grid in grids.items():
            tuned[name] = GridSearchCV(
                make_pipeline(feats, name, "B"), grid, cv=inner_cv,
                scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
        chosen = pick_simplest_near_best(tuned)
        best   = tuned[chosen].best_estimator_

        oof = cross_val_predict(clone(best), Xtr, ytr, cv=inner_cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        tbl      = threshold_table(ytr, oof, THRESHOLDS)
        row_t, _ = select_threshold(tbl)
        thr      = {"rule": float(row_t["Threshold"])}
        reach    = {"rule": True}
        for tg in OP_TARGETS:
            k        = f"op{int(round(tg * 100))}"
            thr[k], reach[k] = op_threshold(tbl, tg)

        prob = best.predict_proba(Xte)[:, 1]
        rec  = {"fold": i, "classifier": chosen,
                "inner_cv_auc": tuned[chosen].best_score_}
        for k in keys:
            m = metrics_at_threshold(yte, prob, thr[k])
            rec[f"thr_{k}"]   = thr[k]
            rec[f"reach_{k}"] = reach[k]
            rec[f"recall_{k}"] = m["Recall"]
            rec[f"spec_{k}"]  = m["Specificity"]
            rec[f"prec_{k}"]  = m["Precision"]
            pooled[k]["y"]    += list(np.asarray(yte))
            pooled[k]["pred"] += list((prob >= thr[k]).astype(int))
        rec["outer_roc_auc"] = metrics_at_threshold(yte, prob, 0.5)["ROC-AUC"]
        rec["best_params"]   = str(tuned[chosen].best_params_)
        fold_rows.append(rec)
        pooled_y    += list(np.asarray(yte))
        pooled_prob += list(prob)
        print(f"  outer fold {i}: {chosen:<3} inner AUC "
              f"{tuned[chosen].best_score_:.3f} | outer AUC "
              f"{rec['outer_roc_auc']:.3f} | recall@85/90 "
              f"{rec['recall_op85']:.2f}/{rec['recall_op90']:.2f}")

    return (pd.DataFrame(fold_rows),
            {k: (np.array(v["y"]), np.array(v["pred"]))
             for k, v in pooled.items()},
            np.array(pooled_y), np.array(pooled_prob))


def step_lite_nested(data, state):
    header("8d. NESTED CV OF THE WHOLE LITE PROCEDURE")
    feats = lite_features()
    X, y  = data["X_B"], data["y"]
    print(f"Lite features ({len(feats)}): {feats}")
    print(f"Nested CV on all {len(X)} patients: "
          f"{NESTED_OUTER_SPLITS}-fold outer x {NESTED_OUTER_REPEATS} repeats.\n")

    outer = RepeatedStratifiedKFold(n_splits=NESTED_OUTER_SPLITS,
                                    n_repeats=NESTED_OUTER_REPEATS,
                                    random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=5, shuffle=True,
                            random_state=RANDOM_STATE)
    folds, pooled, yall, pall = nested_lite_eval(X, y, feats, outer, inner)

    show = folds.drop(columns=["best_params"])
    print("\n" + show.to_string(index=False,
                                float_format=lambda x: f"{x:.3f}"))
    print("\nChosen classifier per outer fold:",
          folds["classifier"].value_counts().to_dict())

    print("\nMean ± SD over outer folds:")
    for k, label in (("rule", "rule (recall>=90%, best F1)"),
                     ("op85", "85%-recall operating point"),
                     ("op90", "90%-recall operating point")):
        print(f"  {label}")
        for c, nm in ((f"thr_{k}", "threshold"), (f"recall_{k}", "recall"),
                      (f"spec_{k}", "specificity"), (f"prec_{k}", "precision")):
            print(f"    {nm:<12}{folds[c].mean():.4f} ± {folds[c].std():.4f}")
    print(f"  outer ROC-AUC {folds['outer_roc_auc'].mean():.4f} ± "
          f"{folds['outer_roc_auc'].std():.4f}")

    rows = []
    print("\nPooled over all outer-test predictions:")
    for k, label in (("rule", "rule"), ("op85", "85% recall"),
                     ("op90", "90% recall")):
        yt, yp = pooled[k]
        tp  = int(((yt == 1) & (yp == 1)).sum())
        pos = int((yt == 1).sum())
        tn  = int(((yt == 0) & (yp == 0)).sum())
        neg = int((yt == 0).sum())
        fp  = neg - tn
        r_lo, r_hi = wilson_ci(tp, pos)
        s_lo, s_hi = wilson_ci(tn, neg)
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        print(f"  {label:<11} recall {tp}/{pos} = {tp/pos:.4f} "
              f"[{r_lo:.3f}, {r_hi:.3f}]   specificity {tn}/{neg} = "
              f"{tn/neg:.4f} [{s_lo:.3f}, {s_hi:.3f}]   precision {prec:.3f}")
        rows.append({"Operating point": label,
                     "Mean threshold": folds[f"thr_{k}"].mean(),
                     "Pooled recall": tp / pos, "Recall CI low": r_lo,
                     "Recall CI high": r_hi,
                     "Pooled specificity": tn / neg,
                     "Spec CI low": s_lo, "Spec CI high": s_hi,
                     "Pooled precision": prec,
                     "Mean fold recall": folds[f"recall_{k}"].mean(),
                     "SD fold recall": folds[f"recall_{k}"].std(),
                     "Mean fold specificity": folds[f"spec_{k}"].mean(),
                     "SD fold specificity": folds[f"spec_{k}"].std()})
    pooled_df  = pd.DataFrame(rows)
    pooled_auc = metrics_at_threshold(yall, pall, 0.5)["ROC-AUC"]
    print(f"  pooled outer-test ROC-AUC {pooled_auc:.4f}")

    folds.to_csv(os.path.join(OUT_DIR, "nested_cv_lite_folds.csv"),  index=False)
    pooled_df.to_csv(os.path.join(OUT_DIR, "nested_cv_lite_pooled.csv"), index=False)
    state["nested_lite"] = dict(folds=folds, pooled=pooled_df,
                                pooled_auc=pooled_auc)


# ------------------------------------------------------------------
# Step 8e: Paper table
# ------------------------------------------------------------------
def step_lite_paper(data, state):
    header("8e. PAPER TABLE - LITE MODEL AT 85% AND 90% RECALL")
    _ensure_lite(data, state)
    if "nested_lite" not in state:
        step_lite_nested(data, state)
    yte  = np.asarray(data["y_test"])
    thrA = state["A"]["thr"]
    pA   = state["A"]["prob_test"]
    pos  = int((yte == 1).sum())
    neg  = int((yte == 0).sum())
    nest = state["nested_lite"]["pooled"].set_index("Operating point")

    sources = ((f"Lite-10 ({state['lite']['clf']}, no labs)",
                state["lite"], True),
               ("Full Model B SVM (29) - reference", state["B"], False))
    rows = []
    for label, S, is_lite in sources:
        for tg in OP_TARGETS:
            tag        = f"{int(round(tg * 100))}% recall"
            thr, ok    = op_threshold(S["table"], tg)
            pt         = point_estimates(S["prob_test"], yte, thr)
            referred   = S["prob_test"] >= thr
            final_pos  = referred & (pA >= thrA)
            c_tp       = int((final_pos & (yte == 1)).sum())
            c_tn       = int((~final_pos & (yte == 0)).sum())
            row = {
                "Model": label, "Operating point": tag, "Threshold": thr,
                "OOF target reachable": ok,
                "Test recall":          pt["Recall"],
                "Test recall CI":       _fmt_ci(pt["Recall"], pt["Recall CI"]),
                "Test specificity":     pt["Specificity"],
                "Test specificity CI":  _fmt_ci(pt["Specificity"], pt["Specificity CI"]),
                "Test precision":       pt["Precision"],
                "Test precision CI":    _fmt_ci(pt["Precision"], pt["Precision CI"]),
                "FN": pt["FN"], "FP": pt["FP"],
                "Ultrasounds saved (%)": 100 - pt["Referred (%)"],
                "Cascade recall":        c_tp / pos,
                "Cascade specificity":   c_tn / neg,
            }
            if is_lite:
                n = nest.loc[f"{int(round(tg * 100))}% recall"]
                row["Nested recall"]          = n["Pooled recall"]
                row["Nested recall CI"]       = _fmt_ci(
                    n["Pooled recall"],
                    (n["Recall CI low"], n["Recall CI high"]))
                row["Nested specificity"]     = n["Pooled specificity"]
                row["Nested specificity CI"]  = _fmt_ci(
                    n["Pooled specificity"],
                    (n["Spec CI low"], n["Spec CI high"]))
            rows.append(row)
    df = pd.DataFrame(rows)

    cols = ["Model", "Operating point", "Threshold", "Test recall CI",
            "Test specificity CI", "Test precision CI", "FN", "FP",
            "Ultrasounds saved (%)", "Cascade recall", "Cascade specificity"]
    print(f"Held-out test set: {pos} positives / {neg} negatives. "
          f"Model A confirmation threshold = {thrA:.3f}.")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    ncols = ["Model", "Operating point", "Nested recall CI", "Nested specificity CI"]
    print("\nNested-CV estimate for the lite model:")
    print(df[df["Model"].str.startswith("Lite")][ncols].to_string(index=False))

    df.to_csv(os.path.join(OUT_DIR, "lite_paper_operating_points.csv"),
              index=False)

    # Markdown table
    lite  = df[df["Model"].str.startswith("Lite")]
    lines = ["| Operating point | Threshold | Test recall (95% CI) | "
             "Test specificity (95% CI) | Test precision (95% CI) | "
             "Ultrasounds saved | Nested recall (95% CI) | "
             "Nested specificity (95% CI) |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in lite.iterrows():
        lines.append(
            f"| {r['Operating point']} | {r['Threshold']:.3f} | "
            f"{r['Test recall CI']} | {r['Test specificity CI']} | "
            f"{r['Test precision CI']} | "
            f"{r['Ultrasounds saved (%)']:.1f}% | "
            f"{r['Nested recall CI']} | {r['Nested specificity CI']} |")
    md = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "lite_paper_operating_points.md"), "w") as f:
        f.write(md + "\n")
    print("\nMarkdown table saved to outputs/lite_paper_operating_points.md")
    print(md)
    state["lite_paper"] = df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
EXTRA_STEPS = ["lite_rf", "lite_nested", "lite_paper"]
EXTRA_FUNCS = {
    "lite_rf":    step_lite_vs_rf,
    "lite_nested": step_lite_nested,
    "lite_paper": step_lite_paper,
}

if __name__ == "__main__":
    import config as _cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="*", default=EXTRA_STEPS,
                        choices=EXTRA_STEPS)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        _cfg.QUICK                = True
        _cfg.CV_REPEATS           = 1
        _cfg.NESTED_OUTER_REPEATS = 1
        _cfg.NESTED_OUTER_SPLITS  = 3
        _cfg.N_BOOT               = 200

    from data_loader import load_split
    import preprocessing as _pre
    _pre.ensure_dirs() if hasattr(_pre, "ensure_dirs") else None

    import os as _os
    for d in (OUT_DIR, _cfg.FIG_DIR, _cfg.MODEL_DIR):
        _os.makedirs(d, exist_ok=True)

    data  = load_split()
    state = {}
    for s in args.steps:
        EXTRA_FUNCS[s](data, state)

    header("DONE - tables in outputs/ (lite_vs_full_rf_*.csv, "
           "nested_cv_lite_*.csv, lite_paper_operating_points.*)")
