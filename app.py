

"""
PCOS screening aid - Streamlit demo (3 models: Lite / Model B / Model A)
Clinic-style interface.
 
Put this file in the PROJECT ROOT (next to data/, models/, src/) and run:
    streamlit run app.py
Optional: put .streamlit/config.toml (teal clinic theme) in the project root too.
 
It loads the frozen pipelines + thresholds that your scripts saved:
    models/model_A_svm.joblib, models/model_B_svm.joblib, models/model_metadata.json
    models/model_B_lite.joblib, models/model_B_lite_metadata.json
Feature lists and thresholds are read from those JSON files.
"""
import os
import sys
import json
from pathlib import Path
 
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
 
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
os.chdir(ROOT)                       # base.load_data() uses relative data/ path
sys.path.insert(0, str(ROOT / "src"))
 
st.set_page_config(page_title="PCOS Screening Aid", page_icon="🩺",
                   layout="wide")
 
DERIVED = {"BMI", "Waist:Hip Ratio", "FSH/LH"}     # computed, not typed in
NA = "Not provided"
 
# Clinic palette
TEAL, TEAL_DK, TEAL_LT = "#0F766E", "#134E4A", "#E6F4F2"
RED, RED_LT = "#DC2626", "#FEE2E2"
GREEN, GREEN_LT = "#16A34A", "#DCFCE7"
AMBER, AMBER_LT = "#D97706", "#FEF3C7"
SLATE = "#475569"
 
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
.stApp { background: #F4F8F8; }
.block-container { padding-top: 1.2rem; max-width: 1180px; }
[data-testid="stAppDeployButton"], .stDeployButton, footer { display: none !important; }
 
/* hero */
.hero {
  background: linear-gradient(120deg, #0F766E 0%, #0E9488 60%, #14B8A6 100%);
  border-radius: 18px; padding: 26px 32px; color: #fff;
  display: flex; justify-content: space-between; align-items: center;
  gap: 20px; flex-wrap: wrap; margin-bottom: 18px;
  box-shadow: 0 6px 18px rgba(15,118,110,.25);
}
.hero-title { font-size: 1.9rem; font-weight: 700; letter-spacing: -.5px; }
.hero-sub { opacity: .92; margin-top: 4px; font-size: .98rem; }
.hero-badge {
  background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.4);
  padding: 8px 16px; border-radius: 999px; font-size: .82rem; font-weight: 500;
}
 
/* stepper */
.stepper { display: flex; align-items: center; margin: 4px 0 20px 0; }
.step { display: flex; align-items: center; gap: 10px; }
.step .dot {
  width: 34px; height: 34px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; font-weight: 700;
  font-size: .95rem; background: #CBD5E1; color: #fff;
}
.step.on .dot { background: #0F766E; box-shadow: 0 0 0 4px #CCEBE7; }
.step .lbl { font-weight: 600; color: #94A3B8; font-size: .92rem; }
.step.on .lbl { color: #134E4A; }
.line { flex: 1; height: 3px; background: #CBD5E1; margin: 0 14px; border-radius: 2px; }
.line.on { background: #0F766E; }
 
/* section titles */
.sec-title { display: flex; align-items: center; gap: 10px;
  font-size: 1.12rem; font-weight: 700; color: #134E4A; margin-bottom: 2px; }
.sec-title .num { background: #0F766E; color: #fff; width: 26px; height: 26px;
  border-radius: 50%; display: inline-flex; align-items: center;
  justify-content: center; font-size: .82rem; }
.sec-cap { color: #64748B; font-size: .86rem; margin-bottom: 8px; }
 
/* bordered containers -> white cards */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: #fff; border-radius: 16px !important;
  border-color: #D9E5E4 !important;
}
 
/* model result cards */
.card { background: #fff; border: 1px solid #D9E5E4; border-radius: 16px;
  padding: 20px 22px 18px 22px; min-height: 360px;
  box-shadow: 0 2px 8px rgba(15,118,110,.07); }
.card.pos { border-top: 6px solid #DC2626; }
.card.neg { border-top: 6px solid #16A34A; }
.card.off { border-top: 6px solid #CBD5E1; background: #F8FAFC; }
.tier { font-size: .7rem; font-weight: 700; letter-spacing: 1.2px; color: #0F766E; }
.mname { font-size: 1.35rem; font-weight: 700; color: #134E4A; margin-top: 2px; }
.mdesc { color: #64748B; font-size: .85rem; margin-bottom: 14px; min-height: 36px; }
.score { font-size: 3rem; font-weight: 700; line-height: 1; color: #0F172A; }
.score span { font-size: 1.4rem; color: #64748B; margin-left: 2px; }
.slabel { color: #64748B; font-size: .78rem; margin: 2px 0 10px 0; }
.bar { position: relative; height: 12px; border-radius: 6px; background: #E2E8F0; }
.bar .fill { position: absolute; left: 0; top: 0; height: 12px; border-radius: 6px; }
.bar .thr { position: absolute; top: -5px; width: 3px; height: 22px;
  background: #0F172A; border-radius: 2px; }
.barlab { position: relative; height: 20px; margin-top: 8px; font-size: .74rem;
  color: #475569; }
.barlab .t { position: absolute; transform: translateX(-50%); white-space: nowrap;
  font-weight: 600; }
.pill { display: inline-block; margin-top: 14px; padding: 7px 14px;
  border-radius: 999px; font-weight: 700; font-size: .8rem; letter-spacing: .4px; }
.pill.pos { background: #FEE2E2; color: #B91C1C; }
.pill.neg { background: #DCFCE7; color: #15803D; }
.pill.off { background: #E2E8F0; color: #64748B; }
.cnote { margin-top: 8px; color: #334155; font-size: .86rem; }
.cwarn { margin-top: 8px; color: #B45309; font-size: .78rem; }
 
/* recommendation banner */
.rec { border-radius: 14px; padding: 16px 20px; display: flex; gap: 14px;
  align-items: flex-start; border: 1px solid; }
.rec .ic { font-size: 1.5rem; line-height: 1.2; }
.rec .tt { font-weight: 700; font-size: 1.02rem; margin-bottom: 2px; }
.rec.danger { background: #FEE2E2; border-color: #FCA5A5; color: #7F1D1D; }
.rec.ok { background: #DCFCE7; border-color: #86EFAC; color: #14532D; }
.rec.warn { background: #FEF3C7; border-color: #FCD34D; color: #78350F; }
.rec.info { background: #E6F4F2; border-color: #99D5CE; color: #134E4A; }
 
.chip { display: inline-block; background: #E6F4F2; color: #134E4A;
  border: 1px solid #99D5CE; padding: 5px 12px; border-radius: 999px;
  font-size: .82rem; font-weight: 600; margin-top: 10px; }
 
/* sidebar */
[data-testid="stSidebar"] { background: #EAF3F2; border-right: 1px solid #D0E3E1; }
.sb-title { font-weight: 700; color: #134E4A; font-size: 1.02rem; margin: 6px 0; }
.ptable { width: 100%; border-collapse: collapse; font-size: .82rem; background: #fff;
  border-radius: 10px; overflow: hidden; }
.ptable th { background: #0F766E; color: #fff; padding: 6px 8px; text-align: left;
  font-weight: 600; }
.ptable td { padding: 6px 8px; border-bottom: 1px solid #E2E8F0; color: #0F172A; }
.legend { font-size: .84rem; color: #334155; line-height: 1.55; }
.legend b { color: #134E4A; }
.disc { color: #64748B; font-size: .78rem; margin-top: 8px; }
 
/* primary button */
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: #0F766E !important; border: none !important; color: #fff !important;
  font-weight: 600 !important; padding: .6rem 1.6rem !important;
  border-radius: 10px !important; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  background: #115E59 !important; }
.footer-note { text-align: center; color: #94A3B8; font-size: .78rem; margin: 26px 0 6px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
 
 
# ------------------------------------------------------------------
# Load models / data
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    meta = json.load(open(MODELS / "model_metadata.json"))
    lite = json.load(open(MODELS / "model_B_lite_metadata.json"))
    specs = (
        ("lite", "Lite model", "STEP 1 · QUESTIONNAIRE",
         "10 history questions - no exam, no tests", lite),
        ("B", "Model B", "STEP 2 · EXAM + BLOOD TESTS",
         "Clinical exam and blood tests, no ultrasound", meta["model_B"]),
        ("A", "Model A", "STEP 3 · ULTRASOUND",
         "Full model including ultrasound findings", meta["model_A"]),
    )
    out = {}
    for key, name, tier, desc, m in specs:
        out[key] = dict(
            name=name, tier=tier, desc=desc, thr=float(m["threshold"]),
            feats=list(m["features"]), metrics=m.get(
                "test_metrics_at_threshold", {}),
            pipe=joblib.load(MODELS / Path(m["file"]).name))
    return out
 
 
@st.cache_data
def load_test_patients():
    import preprocess as base
    from sklearn.model_selection import train_test_split
    df = base.load_data()
    _, test = train_test_split(df, test_size=0.2, random_state=42,
                               stratify=df[base.TARGET])
    return test, base.TARGET
 
 
@st.cache_data
def load_train():
    import preprocess as base
    from sklearn.model_selection import train_test_split
    df = base.load_data()
    train, _ = train_test_split(df, test_size=0.2, random_state=42,
                                stratify=df[base.TARGET])
    return train
 
 
try:
    M = load_models()
except Exception as e:
    st.error(f"Could not load models from {MODELS}: {e}\n\n"
             "Run `python src/pcos_improvements.py --steps freeze lite` first.")
    st.stop()
 
lite_f, b_f, a_f = M["lite"]["feats"], M["B"]["feats"], M["A"]["feats"]
sec1 = lite_f
sec2 = [f for f in b_f if f not in lite_f and f not in DERIVED]
sec3 = [f for f in a_f if f not in b_f]
ALL_INPUTS = sec1 + sec2 + sec3
 
 
@st.cache_resource
def get_background(key):
    """Training data in the model's preprocessed space (SHAP baseline)."""
    m = M[key]
    pre = m["pipe"].named_steps["preprocessing"]
    return np.asarray(pre.transform(load_train()[m["feats"]]))
 
 
# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
def kind(f):
    if "(Y/N)" in f:
        return "yn"
    if f == "Cycle(R/I)":
        return "cycle"
    return "num"
 
 
def widget(f, col):
    k = f"in_{f}"
    if kind(f) == "yn":
        return col.selectbox(f.replace("(Y/N)", "").strip(),
                             [NA, "No", "Yes"], key=k)
    if kind(f) == "cycle":
        return col.selectbox("Menstrual cycle", [NA, "Regular", "Irregular"],
                             key=k)
    return col.number_input(f, value=None, min_value=0.0, format="%g",
                            key=k, placeholder="enter value")
 
 
def to_value(f, raw):
    if kind(f) == "yn":
        return {"No": 0.0, "Yes": 1.0}.get(raw, np.nan)
    if kind(f) == "cycle":
        return {"Regular": 2.0, "Irregular": 4.0}.get(raw, np.nan)
    return np.nan if raw is None else float(raw)
 
 
def fields(feats):
    cols = st.columns(3)
    vals = {}
    for i, f in enumerate(feats):
        vals[f] = to_value(f, widget(f, cols[i % 3]))
    return vals
 
 
def sec_title(n, title, caption=""):
    st.markdown(f'<div class="sec-title"><span class="num">{n}</span>{title}'
                f'</div><div class="sec-cap">{caption}</div>',
                unsafe_allow_html=True)
 
 
def add_derived(v):
    w, h = v.get("Weight (Kg)", np.nan), v.get("Height(Cm)", np.nan)
    v["BMI"] = w / ((h / 100) ** 2) if h and h > 0 and not np.isnan(w) \
        and not np.isnan(h) else np.nan
    wa, hi = v.get("Waist(inch)", np.nan), v.get("Hip(inch)", np.nan)
    v["Waist:Hip Ratio"] = wa / hi if hi and hi > 0 and not np.isnan(wa) \
        and not np.isnan(hi) else np.nan
    fsh, lh = v.get("FSH(mIU/mL)", np.nan), v.get("LH(mIU/mL)", np.nan)
    v["FSH/LH"] = fsh / lh if lh and lh > 0 and not np.isnan(fsh) \
        and not np.isnan(lh) else np.nan
    return v
 
 
def load_random_patient():
    test, target = load_test_patients()
    row = test.sample(1).iloc[0]
    for f in ALL_INPUTS:
        k, x = f"in_{f}", row.get(f, np.nan)
        if kind(f) == "yn":
            st.session_state[k] = NA if pd.isna(x) else (
                "Yes" if float(x) == 1 else "No")
        elif kind(f) == "cycle":
            st.session_state[k] = NA if pd.isna(x) else (
                "Regular" if float(x) == 2 else "Irregular")
        else:
            st.session_state[k] = None if pd.isna(x) else float(x)
    st.session_state["truth"] = int(row[target])
    st.session_state["use_labs"] = True
    st.session_state["use_us"] = True
 
 
def clear_all():
    for f in ALL_INPUTS:
        st.session_state.pop(f"in_{f}", None)
    st.session_state.pop("truth", None)
 
 
# ------------------------------------------------------------------
# Prediction + explanation
# ------------------------------------------------------------------
def predict(key, values):
    m = M[key]
    X = pd.DataFrame([{f: values.get(f, np.nan) for f in m["feats"]}],
                     columns=m["feats"]).astype(float)
    p = float(m["pipe"].predict_proba(X)[:, 1][0])
    n_missing = int(X.isna().sum(axis=1).iloc[0])
    return p, p >= m["thr"], n_missing
 
 
def explain(key, values):
    """Per-patient SHAP contributions.
    Linear SVM (A, B): exact SHAP = coef * (x - training mean of x), in
    SVM-score units. Random Forest (Lite): shap.TreeExplainer, in
    probability units."""
    m = M[key]
    pre = m["pipe"].named_steps["preprocessing"]
    model = m["pipe"].named_steps["model"]
    X = pd.DataFrame([{f: values.get(f, np.nan) for f in m["feats"]}],
                     columns=m["feats"]).astype(float)
    xp = np.asarray(pre.transform(X))[0]
    names = [n.split("__", 1)[1] if "__" in n else n
             for n in pre.get_feature_names_out()]
    names = ["Menstrual cycle" if n.startswith("Cycle(R/I)") else
             n.replace("(Y/N)", "").strip() for n in names]
    if hasattr(model, "coef_") and getattr(model, "kernel", "linear") == "linear":
        base_x = get_background(key).mean(axis=0)
        contrib = model.coef_.ravel() * (xp - base_x)
        unit = "Contribution to SVM score"
    else:
        import shap
        sv = shap.TreeExplainer(model).shap_values(xp.reshape(1, -1))
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        contrib = sv[0, :, 1] if sv.ndim == 3 else sv[0]
        unit = "Contribution to risk (probability points)"
    df = pd.DataFrame({"feature": names, "contribution": contrib})
    return df.groupby("feature", as_index=False).sum(), unit
 
 
def shap_chart(df, unit, top_n=8):
    top = df.reindex(df["contribution"].abs().sort_values(
        ascending=False).index).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    fig.patch.set_facecolor("white")
    ax.barh(top["feature"], top["contribution"], height=0.62,
            color=[RED if v > 0 else TEAL for v in top["contribution"]])
    ax.axvline(0, color="#334155", lw=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="y", length=0, labelsize=9, colors="#0F172A")
    ax.tick_params(axis="x", labelsize=8, colors="#64748B")
    ax.xaxis.grid(True, color="#E2E8F0", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(unit, fontsize=8.5, color="#475569")
    fig.tight_layout()
    return fig
 
 
# ------------------------------------------------------------------
# HTML builders
# ------------------------------------------------------------------
def stepper_html(labs, us):
    def step(n, label, on):
        return (f'<div class="step{" on" if on else ""}"><div class="dot">{n}'
                f'</div><div class="lbl">{label}</div></div>')
    return ('<div class="stepper">' + step(1, "Questionnaire", True)
            + f'<div class="line{" on" if labs else ""}"></div>'
            + step(2, "Exam &amp; blood tests", labs)
            + f'<div class="line{" on" if (labs and us) else ""}"></div>'
            + step(3, "Ultrasound", labs and us) + '</div>')
 
 
def model_card(m, p=None, pos=None, n_miss=0):
    if p is None:
        return (f'<div class="card off"><div class="tier">{m["tier"]}</div>'
                f'<div class="mname">{m["name"]}</div>'
                f'<div class="mdesc">{m["desc"]}</div>'
                '<div class="score" style="color:#94A3B8">-</div>'
                '<div class="slabel">Risk score</div>'
                '<div class="bar"></div><div class="barlab"></div>'
                '<span class="pill off">NOT RUN</span>'
                '<div class="cnote">Inputs for this step were not provided.</div>'
                '</div>')
    state = "pos" if pos else "neg"
    color = RED if pos else GREEN
    pct = max(0.0, min(100.0, p * 100))
    thr = m["thr"] * 100
    label = "● POSITIVE SCREEN" if pos else "● NEGATIVE SCREEN"
    note = ("Above threshold - refer for the next test." if pos
            else "Below threshold - low likelihood of PCOS.")
    warn = (f'<div class="cwarn">⚠ {n_miss} of {len(m["feats"])} inputs empty '
            '(filled with training median) - interpret with caution.</div>'
            if n_miss else "")
    return (f'<div class="card {state}"><div class="tier">{m["tier"]}</div>'
            f'<div class="mname">{m["name"]}</div>'
            f'<div class="mdesc">{m["desc"]}</div>'
            f'<div class="score">{pct:.0f}<span>%</span></div>'
            '<div class="slabel">Risk score (model output, not a probability)</div>'
            f'<div class="bar"><div class="fill" style="width:{pct:.1f}%;'
            f'background:{color}"></div><div class="thr" '
            f'style="left:calc({thr:.1f}% - 1px)"></div></div>'
            f'<div class="barlab"><span class="t" style="left:{thr:.1f}%">'
            f'▲ threshold {thr:.0f}%</span></div>'
            f'<span class="pill {state}">{label}</span>'
            f'<div class="cnote">{note}</div>{warn}</div>')
 
 
def rec_banner(kind_, icon, title, text):
    return (f'<div class="rec {kind_}"><div class="ic">{icon}</div><div>'
            f'<div class="tt">{title}</div><div>{text}</div></div></div>')
 
 
# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-title">Demo tools</div>',
                unsafe_allow_html=True)
    st.button("🎲 Load a random held-out patient",
              on_click=load_random_patient, use_container_width=True)
    st.button("Clear form", on_click=clear_all, use_container_width=True)
    st.markdown('<div class="disc">Held-out patients were never used to '
                'train the models.</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="sb-title" style="margin-top:22px">'
                'How to read the result</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="legend"><b>Lite</b> - 10 questions, no tests<br>'
        '<b>Model B</b> - adds exam &amp; blood tests<br>'
        '<b>Model A</b> - adds ultrasound<br><br>'
        'Each threshold is set to catch about <b>90% of PCOS cases</b>, so '
        'false alarms are expected: a positive screen means '
        '<i>go to the next test</i>, not <i>has PCOS</i>.</div>',
        unsafe_allow_html=True)
 
    st.markdown('<div class="sb-title" style="margin-top:22px">'
                'Held-out test performance</div>', unsafe_allow_html=True)
    rows = ""
    for k in ("lite", "B", "A"):
        t = M[k]["metrics"]
        if t:
            rows += (f"<tr><td>{M[k]['name'].replace(' model', '')}</td>"
                     f"<td>{t.get('Recall', float('nan')):.2f}</td>"
                     f"<td>{t.get('Specificity', float('nan')):.2f}</td>"
                     f"<td>{t.get('ROC-AUC', float('nan')):.2f}</td></tr>")
    if rows:
        st.markdown('<table class="ptable"><tr><th>Model</th><th>Recall</th>'
                    '<th>Spec.</th><th>AUC</th></tr>' + rows + '</table>',
                    unsafe_allow_html=True)
    st.markdown('<div class="disc">109 held-out patients (36 with PCOS) - '
                'wide uncertainty. Recall = share of PCOS cases caught; '
                'Spec. = share of non-PCOS correctly cleared.</div>',
                unsafe_allow_html=True)
 
 
# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------
st.markdown(
    '<div class="hero"><div><div class="hero-title">🩺 PCOS Screening Aid</div>'
    '<div class="hero-sub">Stepwise screening: questionnaire → exam &amp; '
    'blood tests → ultrasound confirmation</div></div>'
    '<div class="hero-badge">Clinical decision support · Not a diagnosis'
    '</div></div>', unsafe_allow_html=True)
 
st.markdown(stepper_html(bool(st.session_state.get("use_labs", False)),
                         bool(st.session_state.get("use_us", False))
                         and bool(st.session_state.get("use_labs", False))),
            unsafe_allow_html=True)
 
values = {}
with st.container(border=True):
    sec_title(1, "Questionnaire",
              "Quick history questions - needed for every model.")
    values.update(fields(sec1))
 
with st.container(border=True):
    sec_title(2, "Examination &amp; blood tests",
              "BMI, waist:hip ratio and FSH/LH are calculated automatically. "
              "Leave unknown values empty (filled with training medians).")
    use_labs = st.toggle("Exam & blood tests available", key="use_labs")
    if use_labs:
        values.update(fields(sec2))
 
with st.container(border=True):
    sec_title(3, "Ultrasound findings",
              "Needs step 2. Enables the confirmation model (Model A).")
    use_us = st.toggle("Ultrasound available", key="use_us",
                       disabled=not use_labs)
    if use_us and use_labs:
        values.update(fields(sec3))
 
values = add_derived(values)
 
st.write("")
go = st.button("▶  Run screening", type="primary")
 
if go:
    missing_q = [f for f in lite_f if np.isnan(values.get(f, np.nan))]
    if missing_q:
        st.warning("Please answer all questionnaire fields first: "
                   + ", ".join(missing_q))
        st.stop()
 
    run = ["lite"] + (["B"] if use_labs else []) + \
          (["A"] if use_labs and use_us else [])
    results = {}
 
    st.write("")
    st.markdown('<div class="sec-title" style="font-size:1.3rem">'
                'Screening results</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, key in zip(cols, ("lite", "B", "A")):
        with col:
            if key not in run:
                st.markdown(model_card(M[key]), unsafe_allow_html=True)
                continue
            p, pos, n_miss = predict(key, values)
            results[key] = (p, pos)
            st.markdown(model_card(M[key], p, pos, n_miss),
                        unsafe_allow_html=True)
 
    # ---- combined recommendation ----------------------------------------
    st.write("")
    st.markdown('<div class="sec-title" style="font-size:1.3rem">'
                'Suggested next step</div>', unsafe_allow_html=True)
    if "A" in results:
        if results["A"][1]:
            st.markdown(rec_banner(
                "danger", "🔴", "High likelihood of PCOS",
                "The ultrasound-based model is positive. Recommend clinician "
                "confirmation (Rotterdam criteria) and management planning."),
                unsafe_allow_html=True)
        else:
            st.markdown(rec_banner(
                "ok", "🟢", "PCOS unlikely",
                "The ultrasound-based model is negative. Follow up if "
                "symptoms persist."), unsafe_allow_html=True)
    else:
        stage1 = results.get("B", results["lite"])
        if stage1[1]:
            st.markdown(rec_banner(
                "warn", "🟠", "Screen positive - refer for the next test",
                "Recommend ultrasound (and blood tests if not yet done), "
                "then run Model A to confirm."), unsafe_allow_html=True)
        else:
            st.markdown(rec_banner(
                "info", "🔵", "Screen negative - routine follow-up",
                "This screen catches roughly 9 in 10 cases, so keep clinical "
                "judgement for symptomatic patients."),
                unsafe_allow_html=True)
 
    # ---- per-patient SHAP explanation ------------------------------------
    st.write("")
    st.markdown('<div class="sec-title" style="font-size:1.3rem">'
                'Why did the models say this? (SHAP)</div>',
                unsafe_allow_html=True)
    st.caption("Red bars push the score toward PCOS, teal bars push it away, "
               "compared with an average training patient. Lite is a Random "
               "Forest (probability points); Models A and B are linear SVMs "
               "(SVM-score units), so bar sizes are not comparable across tabs.")
    tabs = st.tabs([M[k]["name"] for k in run])
    for tab, key in zip(tabs, run):
        with tab:
            try:
                exp, unit = explain(key, values)
                fig = shap_chart(exp, unit)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.info(f"Explanation not available for this model: {e}")
 
    if "truth" in st.session_state:
        t = st.session_state["truth"]
        st.markdown('<span class="chip">Recorded diagnosis for this demo '
                    'patient: ' + ("PCOS" if t == 1 else "No PCOS")
                    + '</span>', unsafe_allow_html=True)
 
# ---- global SHAP figures saved by src/pcos_improvements.py (step shap_b) ----
fig_dir = ROOT / "outputs" / "figures"
bar, summ = fig_dir / "shap_bar_model_B.png", fig_dir / "shap_summary_model_B.png"
if bar.exists() or summ.exists():
    st.write("")
    with st.expander("Overall feature importance (SHAP, Model B - no ultrasound)"):
        st.caption("Computed on the tuned Random Forest for Model B. The "
                   "deployed linear SVM ranks the same top features: skin "
                   "darkening, hair growth, weight gain and fast food.")
        c1, c2 = st.columns(2)
        if bar.exists():
            c1.image(str(bar))
        if summ.exists():
            c2.image(str(summ))
 
st.markdown('<div class="footer-note">Prototype for clinical evaluation · '
            'screening aid only, not a medical device · final decisions rest '
            'with the clinician</div>', unsafe_allow_html=True)
 














