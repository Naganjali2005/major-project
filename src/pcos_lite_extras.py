"""
pcos_lite_extras.py
====================================================================
Second add-on for the PCOS major project.  NEITHER src/preprocess.py NOR
src/pcos_improvements.py is modified: this file imports both and adds the
three things needed for the LITE Model B (10 no-lab features) section of
the paper.

Run from the project root, next to the other two scripts:

    python src/pcos_lite_extras.py                    # all three steps
    python src/pcos_lite_extras.py --quick            # fast smoke run
    python src/pcos_lite_extras.py --steps lite_nested lite_paper

Steps (outputs -> outputs/, figures unchanged, models unchanged):

  lite_rf      8c. LIKE-FOR-LIKE comparison of the lite model against the
                   FULL Model B RANDOM FOREST (previously only the full SVM
                   was the reference): repeated 5x5 CV, OOF thresholds,
                   held-out test with CIs, matched-recall operating points,
                   paired bootstrap (lite chosen / lite RF  vs  full RF).
  lite_nested  8d. NESTED CV of the whole lite procedure on ALL patients
                   (541 in your data): inside every outer fold the classifier
                   (LR/SVM/RF) is tuned, the simplest near-best one is
                   chosen, and the thresholds (rule, 85%-recall and
                   90%-recall operating points) are chosen on inner
                   out-of-fold predictions; only then is the untouched
                   outer-test part scored.  Free of selection effects.
  lite_paper   8e. Paper table: TWO operating points (85% and 90% recall)
                   for the lite model, each with held-out-test recall /
                   specificity (Wilson 95% CI), precision, cascade effect and
                   the nested-CV estimate next to it; full Model B SVM is
                   shown as a reference.  Writes a CSV and a Markdown table.

Dependencies between steps are handled automatically: if the 'lite' step
(or Model A/B thresholds) has not been run in this process, it is run first.
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_predict,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Import the two existing scripts (left untouched)
# ------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcos_improvements as pi  # noqa: E402
base = pi.base

OP_TARGETS = (0.85, 0.90)          # the two operating points for the paper


# ==================================================================
# HELPERS
# ==================================================================
def _ensure_lite(data, state):
    """Run the existing 'lite' step (and A/B) if it has not been run yet."""
    if "lite" not in state:
        pi.step_lite(data, state)


def op_threshold(tbl, target):
    """Threshold with the best specificity among those reaching `target`
    recall (same rule as pi.operating_points).  If the target is not
    reachable, falls back to the highest-recall row.
    Returns (threshold, reachable)."""
    el = tbl[tbl["Recall"] >= target]
    if len(el) == 0:
        r = tbl.loc[tbl["Recall"].idxmax()]
        return float(r["Threshold"]), False
    r = el.loc[el["Specificity"].idxmax()]
    return float(r["Threshold"]), True


def oof_profile(est, Xtr, ytr, Xte):
    """Out-of-fold threshold table (training set, original 5-fold CV object),
    rule threshold, 85%/90% operating thresholds, fitted model and held-out
    test probabilities for ONE (unfitted) estimator."""
    oof = cross_val_predict(clone(est), Xtr, ytr, cv=base.cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    tbl = pi.threshold_table(ytr, oof, pi.THRESHOLDS)
    row, rule_met = pi.select_threshold(tbl)
    prof = {"oof": oof, "table": tbl, "thr": float(row["Threshold"]),
            "rule_met": bool(rule_met)}
    for tg in OP_TARGETS:
        t, ok = op_threshold(tbl, tg)
        prof[f"thr{int(round(tg * 100))}"] = t
        prof[f"reach{int(round(tg * 100))}"] = ok
    fitted = clone(est).fit(Xtr, ytr)
    prof["fitted"] = fitted
    prof["prob_test"] = fitted.predict_proba(Xte)[:, 1]
    return prof


def point_estimates(prob, y, thr):
    """Held-out-test numbers at one threshold with Wilson CIs for recall and
    specificity and a bootstrap CI for precision."""
    y = np.asarray(y)
    m = pi.metrics_at_threshold(y, prob, thr)
    pos, neg = m["TP"] + m["FN"], m["TN"] + m["FP"]
    r_lo, r_hi = pi.wilson_ci(m["TP"], pos)
    s_lo, s_hi = pi.wilson_ci(m["TN"], neg)
    boot = pi.bootstrap_ci(y, prob, thr)
    return {
        "Threshold": thr,
        "Recall": m["Recall"], "Recall CI": (r_lo, r_hi),
        "Specificity": m["Specificity"], "Specificity CI": (s_lo, s_hi),
        "Precision": m["Precision"], "Precision CI": boot["Precision"],
        "F1": m["F1"], "FN": m["FN"], "FP": m["FP"],
        "TP": m["TP"], "TN": m["TN"],
        "Referred (%)": 100 * float((np.asarray(prob) >= thr).mean()),
    }


def _fmt_ci(v, ci):
    return f"{v:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"


# ==================================================================
# STEP 8c  Lite vs FULL Model B RANDOM FOREST (like-for-like)
# ==================================================================
def step_lite_vs_rf(data, state):
    pi.header("8c. LITE MODEL vs FULL MODEL B RANDOM FOREST (like-for-like)")
    _ensure_lite(data, state)
    Xtr, Xte = data["X_train_B"], data["X_test_B"]
    ytr, yte = data["y_train"], np.asarray(data["y_test"])
    L = state["lite"]
    lite_feats = L["feats"]
    full_feats = list(Xtr.columns)
    clf = L["clf"]

    print("Why: the lite model was so far compared with the full Model B "
          "SVM only.\nA reviewer can ask for the strongest full-feature "
          "competitor, which for AUC is the tuned Random Forest, so the lite "
          "model is also compared against it here with the same split, the "
          "same CV and the same threshold rules.\n")

    models = {
        "Full B (29) RF": (pi.make_pipeline(full_feats, "RF", "B"),
                           full_feats),
        "Full B (29) SVM": (pi.make_pipeline(full_feats, "SVM", "B"),
                            full_feats),
        "Lite-10 RF (tuned)": (
            L["tuned"]["Lite-10"]["RF"].best_estimator_, lite_feats),
        f"Lite-10 {clf} (chosen)": (L["pipe"], lite_feats),
    }
    key_full_rf = "Full B (29) RF"
    key_lite_rf = "Lite-10 RF (tuned)"
    key_lite = f"Lite-10 {clf} (chosen)"

    # ---- (1) repeated 5x5 CV on the training set ------------------------
    print("--- (1) Repeated 5x5 CV, training set ---")
    cv = pi.get_repeated_cv()
    rows = []
    for label, (est, feats) in models.items():
        summ = pi.cv_summary(clone(est), Xtr[feats], ytr, cv)
        rows.append(pi.summary_row(label, summ, n_features=len(feats)))
    cvdf = pd.DataFrame(rows)
    pi.print_summary_table(cvdf)
    cvdf.to_csv(os.path.join(pi.OUT_DIR, "lite_vs_full_rf_cv.csv"),
                index=False)

    # ---- (2) thresholds + held-out test for every model -----------------
    print("\n--- (2) Thresholds from out-of-fold predictions, "
          "then held-out test ---")
    profs = {}
    for label, (est, feats) in models.items():
        profs[label] = oof_profile(est, Xtr[feats], ytr, Xte[feats])
        p = profs[label]
        print(f"  {label:<26} rule thr {p['thr']:.3f} | "
              f"85%-recall thr {p['thr85']:.3f} (reachable {p['reach85']}) | "
              f"90%-recall thr {p['thr90']:.3f} (reachable {p['reach90']})")

    for label in (key_full_rf, key_lite_rf, key_lite):
        p = profs[label]
        pi.print_test_with_ci(f"{label} - held-out test @0.50", yte,
                              p["prob_test"], 0.50)
        pi.print_test_with_ci(f"{label} - held-out test @rule threshold",
                              yte, p["prob_test"], p["thr"])

    # ---- (3) matched-recall table on the held-out test ------------------
    print("\n--- (3) Matched-recall operating points on the held-out test "
          "(threshold chosen on OOF training predictions) ---")
    rows = []
    for label, p in profs.items():
        for tg in OP_TARGETS:
            tag = int(round(tg * 100))
            thr = p[f"thr{tag}"]
            pt = point_estimates(p["prob_test"], yte, thr)
            rows.append({
                "Model": label, "Recall target": tg, "Threshold": thr,
                "OOF target reachable": p[f"reach{tag}"],
                "Test recall": pt["Recall"],
                "Recall CI low": pt["Recall CI"][0],
                "Recall CI high": pt["Recall CI"][1],
                "Test specificity": pt["Specificity"],
                "Spec CI low": pt["Specificity CI"][0],
                "Spec CI high": pt["Specificity CI"][1],
                "Test precision": pt["Precision"],
                "False negatives": pt["FN"], "Referred (%)": pt["Referred (%)"],
            })
    mr = pd.DataFrame(rows)
    print(mr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    mr.to_csv(os.path.join(pi.OUT_DIR, "lite_vs_full_rf_matched_recall.csv"),
              index=False)

    # ---- (4) paired bootstrap on the same test patients -----------------
    print("\n--- (4) Paired bootstrap on the same test patients "
          "(Lite minus Full RF) ---")
    comps = []
    margin = 0.05
    for lite_label in (key_lite, key_lite_rf):
        for tag, name in (("thr", "own rule threshold"),
                          ("thr85", "85%-recall point"),
                          ("thr90", "90%-recall point")):
            pl, pf = profs[lite_label], profs[key_full_rf]
            d = pi.paired_bootstrap_diff(
                yte, pl["prob_test"], pl[tag], pf["prob_test"], pf[tag])
            d.insert(0, "Operating point", name)
            d.insert(0, "Comparison", f"{lite_label}  minus  {key_full_rf}")
            comps.append(d)
            print(f"\n{lite_label} minus {key_full_rf}  @ {name} "
                  f"(lite thr {pl[tag]:.3f}, full RF thr {pf[tag]:.3f})")
            print(d.drop(columns=["Comparison", "Operating point"])
                  .to_string(index=False,
                             float_format=lambda x: f"{x:+.4f}"))
            if tag == "thr":
                a = d[d["Metric"] == "ROC-AUC"].iloc[0]
                if a["95% CI low"] > -margin:
                    print(f"  ROC-AUC: non-inferior to the full RF at "
                          f"margin {margin} (lower CI bound "
                          f"{a['95% CI low']:+.3f}).")
                else:
                    print(f"  ROC-AUC: non-inferiority at margin {margin} "
                          f"NOT shown (lower CI bound "
                          f"{a['95% CI low']:+.3f}).")
    allc = pd.concat(comps, ignore_index=True)
    allc.to_csv(os.path.join(pi.OUT_DIR,
                             "lite_vs_full_rf_paired_bootstrap.csv"),
                index=False)
    print(f"\n(With only {int(yte.sum())} positive test cases these "
          "intervals are wide; a CI containing 0 means 'no detectable "
          "difference', not 'equal'.)")
    state["lite_rf"] = dict(cv=cvdf, matched=mr, paired=allc, profiles=profs)


# ==================================================================
# STEP 8d  Nested CV of the whole LITE procedure (all patients)
# ==================================================================
def nested_lite_eval(X, y, feats, outer_cv, inner_cv):
    """For every outer fold, using ONLY the outer-training part:
         1. tune LR, SVM and RF (same grids as step 'lite') with inner CV,
         2. choose the simplest classifier within 0.005 inner CV-AUC of the
            best (same rule as step 'lite'),
         3. get inner out-of-fold probabilities for it and choose three
            thresholds: the original rule (recall>=90% then best F1), and the
            85%- and 90%-recall operating points (best specificity),
       then score the untouched outer-test part at those thresholds."""
    X = X[feats].reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    grids = pi._lite_grids()
    keys = ["rule", "op85", "op90"]
    pooled = {k: {"y": [], "pred": []} for k in keys}
    pooled_prob, pooled_y = [], []
    fold_rows = []

    for i, (tr, te) in enumerate(outer_cv.split(X, y), 1):
        Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]

        tuned = {}
        for name, grid in grids.items():
            tuned[name] = GridSearchCV(
                pi.make_pipeline(feats, name, "B"), grid, cv=inner_cv,
                scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
        chosen = pi.pick_simplest_near_best(tuned)
        best = tuned[chosen].best_estimator_

        oof = cross_val_predict(clone(best), Xtr, ytr, cv=inner_cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        tbl = pi.threshold_table(ytr, oof, pi.THRESHOLDS)
        row, _ = pi.select_threshold(tbl)
        thr = {"rule": float(row["Threshold"])}
        reach = {"rule": True}
        for tg in OP_TARGETS:
            k = f"op{int(round(tg * 100))}"
            thr[k], reach[k] = op_threshold(tbl, tg)

        prob = best.predict_proba(Xte)[:, 1]
        rec = {"fold": i, "classifier": chosen,
               "inner_cv_auc": tuned[chosen].best_score_}
        for k in keys:
            m = pi.metrics_at_threshold(yte, prob, thr[k])
            rec[f"thr_{k}"] = thr[k]
            rec[f"reach_{k}"] = reach[k]
            rec[f"recall_{k}"] = m["Recall"]
            rec[f"spec_{k}"] = m["Specificity"]
            rec[f"prec_{k}"] = m["Precision"]
            pooled[k]["y"] += list(np.asarray(yte))
            pooled[k]["pred"] += list((prob >= thr[k]).astype(int))
        rec["outer_roc_auc"] = pi.metrics_at_threshold(
            yte, prob, 0.5)["ROC-AUC"]
        rec["best_params"] = str(tuned[chosen].best_params_)
        fold_rows.append(rec)
        pooled_y += list(np.asarray(yte))
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
    pi.header("8d. NESTED CV OF THE WHOLE LITE PROCEDURE "
              "(tune + choose classifier + choose thresholds)")
    feats = pi.lite_features()
    X, y = data["X_B"], data["y"]
    print(f"Lite features ({len(feats)}): {feats}")
    print(f"Nested CV on all {len(X)} patients: "
          f"{pi.NESTED_OUTER_SPLITS}-fold outer x {pi.NESTED_OUTER_REPEATS} "
          "repeats, 5-fold inner CV.")
    print("Everything that is selected on data (hyper-parameters, the "
          "classifier, and the thresholds) is selected INSIDE each outer "
          "fold, so the numbers below contain no selection effect.\n")

    outer = RepeatedStratifiedKFold(n_splits=pi.NESTED_OUTER_SPLITS,
                                    n_repeats=pi.NESTED_OUTER_REPEATS,
                                    random_state=pi.RANDOM_STATE)
    inner = StratifiedKFold(n_splits=5, shuffle=True,
                            random_state=pi.RANDOM_STATE)
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
                      (f"spec_{k}", "specificity"),
                      (f"prec_{k}", "precision")):
            print(f"    {nm:<12}{folds[c].mean():.4f} ± "
                  f"{folds[c].std():.4f}")
        n_short = int((folds[f'recall_{k}'] < {'rule': 0.90, 'op85': 0.85,
                                               'op90': 0.90}[k]).sum())
        print(f"    folds whose outer recall fell below the target: "
              f"{n_short}/{len(folds)}")
    print(f"  outer ROC-AUC {folds['outer_roc_auc'].mean():.4f} ± "
          f"{folds['outer_roc_auc'].std():.4f}")

    rows = []
    print("\nPooled over all outer-test predictions:")
    for k, label in (("rule", "rule"), ("op85", "85% recall"),
                     ("op90", "90% recall")):
        yt, yp = pooled[k]
        tp = int(((yt == 1) & (yp == 1)).sum())
        pos = int((yt == 1).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        neg = int((yt == 0).sum())
        fp = neg - tn
        r_lo, r_hi = pi.wilson_ci(tp, pos)
        s_lo, s_hi = pi.wilson_ci(tn, neg)
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        print(f"  {label:<11} recall {tp}/{pos} = {tp / pos:.4f} "
              f"[{r_lo:.3f}, {r_hi:.3f}]   specificity {tn}/{neg} = "
              f"{tn / neg:.4f} [{s_lo:.3f}, {s_hi:.3f}]   "
              f"precision {prec:.3f}")
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
    pooled_df = pd.DataFrame(rows)
    pooled_auc = pi.metrics_at_threshold(yall, pall, 0.5)["ROC-AUC"]
    print(f"  pooled outer-test ROC-AUC {pooled_auc:.4f}")
    if pi.NESTED_OUTER_REPEATS > 1:
        print(f"  NOTE: each patient is scored {pi.NESTED_OUTER_REPEATS}x "
              "(repeats), so the pooled Wilson intervals are slightly "
              "narrower than for one pass; the fold-level SDs above are the "
              "more conservative uncertainty statement.")

    if "nested_B" in state:
        nb = state["nested_B"]
        print("\nReference - full Model B SVM nested CV (step 'nested', "
              "its own rule threshold):")
        print(f"  recall {nb['recall'].mean():.4f} ± "
              f"{nb['recall'].std():.4f}   specificity "
              f"{nb['specificity'].mean():.4f} ± "
              f"{nb['specificity'].std():.4f}   ROC-AUC "
              f"{nb['roc_auc'].mean():.4f}")

    folds.to_csv(os.path.join(pi.OUT_DIR, "nested_cv_lite_folds.csv"),
                 index=False)
    pooled_df.to_csv(os.path.join(pi.OUT_DIR, "nested_cv_lite_pooled.csv"),
                     index=False)
    state["nested_lite"] = dict(folds=folds, pooled=pooled_df,
                                pooled_auc=pooled_auc)


# ==================================================================
# STEP 8e  Paper table: TWO operating points for the lite model
# ==================================================================
def step_lite_paper(data, state):
    pi.header("8e. PAPER TABLE - LITE MODEL AT 85% AND 90% RECALL")
    _ensure_lite(data, state)
    if "nested_lite" not in state:
        step_lite_nested(data, state)
    yte = np.asarray(data["y_test"])
    thrA = state["A"]["thr"]
    pA = state["A"]["prob_test"]
    pos, neg = int((yte == 1).sum()), int((yte == 0).sum())
    nest = state["nested_lite"]["pooled"].set_index("Operating point")

    sources = (("Lite-10 (" + state["lite"]["clf"] + ", no labs)",
                state["lite"], True),
               ("Full Model B SVM (29) - reference", state["B"], False))
    rows = []
    for label, S, is_lite in sources:
        for tg in OP_TARGETS:
            tag = f"{int(round(tg * 100))}% recall"
            thr, ok = op_threshold(S["table"], tg)
            pt = point_estimates(S["prob_test"], yte, thr)
            referred = S["prob_test"] >= thr
            final_pos = referred & (pA >= thrA)
            c_tp = int((final_pos & (yte == 1)).sum())
            c_tn = int((~final_pos & (yte == 0)).sum())
            row = {
                "Model": label, "Operating point": tag, "Threshold": thr,
                "OOF target reachable": ok,
                "Test recall": pt["Recall"],
                "Test recall CI": _fmt_ci(pt["Recall"], pt["Recall CI"]),
                "Test specificity": pt["Specificity"],
                "Test specificity CI": _fmt_ci(pt["Specificity"],
                                               pt["Specificity CI"]),
                "Test precision": pt["Precision"],
                "Test precision CI": _fmt_ci(pt["Precision"],
                                             pt["Precision CI"]),
                "FN": pt["FN"], "FP": pt["FP"],
                "Ultrasounds saved (%)": 100 - pt["Referred (%)"],
                "Cascade recall": c_tp / pos,
                "Cascade specificity": c_tn / neg,
            }
            if is_lite:
                n = nest.loc[f"{int(round(tg * 100))}% recall"]
                row["Nested recall"] = n["Pooled recall"]
                row["Nested recall CI"] = _fmt_ci(
                    n["Pooled recall"],
                    (n["Recall CI low"], n["Recall CI high"]))
                row["Nested specificity"] = n["Pooled specificity"]
                row["Nested specificity CI"] = _fmt_ci(
                    n["Pooled specificity"],
                    (n["Spec CI low"], n["Spec CI high"]))
            rows.append(row)
    df = pd.DataFrame(rows)

    cols = ["Model", "Operating point", "Threshold", "Test recall CI",
            "Test specificity CI", "Test precision CI", "FN", "FP",
            "Ultrasounds saved (%)", "Cascade recall", "Cascade specificity"]
    print(f"Held-out test set: {pos} positives / {neg} negatives. "
          f"Model A confirmation threshold = {thrA:.3f}.")
    print("Thresholds were chosen on OUT-OF-FOLD training predictions only.")
    print(df[cols].to_string(index=False,
                             float_format=lambda x: f"{x:.3f}"))
    ncols = ["Model", "Operating point", "Nested recall CI",
             "Nested specificity CI"]
    print("\nNested-CV estimate for the lite model (selection-effect free):")
    print(df[df["Model"].str.startswith("Lite")][ncols].to_string(
        index=False))
    print("\nHow to read it: the target recall (85% / 90%) is reached on "
          "the OOF training predictions by construction; the held-out test "
          "and the nested CV show what recall is actually delivered on "
          "unseen patients. Report all three and say which is which.")

    df.to_csv(os.path.join(pi.OUT_DIR, "lite_paper_operating_points.csv"),
              index=False)

    # Markdown table ready to paste / convert for the paper
    lite = df[df["Model"].str.startswith("Lite")]
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
    with open(os.path.join(pi.OUT_DIR, "lite_paper_operating_points.md"),
              "w") as f:
        f.write(md + "\n")
    print("\nMarkdown table (lite model) saved to "
          "outputs/lite_paper_operating_points.md:\n")
    print(md)
    state["lite_paper"] = df


# ==================================================================
# MAIN
# ==================================================================
EXTRA_STEPS = ["lite_rf", "lite_nested", "lite_paper"]
EXTRA_FUNCS = {"lite_rf": step_lite_vs_rf,
               "lite_nested": step_lite_nested,
               "lite_paper": step_lite_paper}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="*", default=EXTRA_STEPS,
                        choices=EXTRA_STEPS)
    parser.add_argument("--quick", action="store_true",
                        help="fewer CV repeats / bootstrap draws / smaller "
                             "grids (smoke test)")
    args = parser.parse_args()

    if args.quick:
        pi.QUICK = True
        pi.CV_REPEATS = 1
        pi.NESTED_OUTER_REPEATS = 1
        pi.NESTED_OUTER_SPLITS = 3
        pi.N_BOOT = 200

    pi.ensure_dirs()
    data = pi.load_split()
    state = {}
    for s in args.steps:
        EXTRA_FUNCS[s](data, state)

    pi.header("DONE - tables in outputs/ (lite_vs_full_rf_*.csv, "
              "nested_cv_lite_*.csv, lite_paper_operating_points.*)")
