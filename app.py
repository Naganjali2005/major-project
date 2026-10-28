"""
PMOS Screening Aid — Streamlit app
Features: sequential models, risk cards, dark mode, disclaimer,
          symptom tracker, PDF export, suggestions.
"""
import os
import sys
import json
import io
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parent
MODELS = ROOT / "models"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from chatbot import render_chatbot_page, build_knowledge_base
from symptom_tracker import render_symptom_tracker

st.set_page_config(page_title="PMOS Screening Aid", page_icon="🩺", layout="wide")

DERIVED = {"BMI", "Waist:Hip Ratio", "FSH/LH"}
NA = "Not provided"

# ── Palette ──────────────────────────────────────────────────────────────────
TEAL, TEAL_DK, TEAL_LT = "#0F766E", "#134E4A", "#E6F4F2"
RED,  RED_LT            = "#DC2626", "#FEE2E2"
GREEN,GREEN_LT          = "#16A34A", "#DCFCE7"
AMBER,AMBER_LT          = "#D97706", "#FEF3C7"
SLATE                   = "#475569"

# ── Dark-mode flag ────────────────────────────────────────────────────────────
dark = st.session_state.get("dark_mode", False)

# ── CSS ───────────────────────────────────────────────────────────────────────
_bg       = "#1A1F2E" if dark else "#F4F8F8"
_card_bg  = "#242938" if dark else "#ffffff"
_card_brd = "#3A4055" if dark else "#D9E5E4"
_txt      = "#E2E8F0" if dark else "#0F172A"
_sub      = "#94A3B8" if dark else "#475569"
_sb_bg    = "#1E2536" if dark else "#EAF3F2"
_sb_brd   = "#2E3650" if dark else "#D0E3E1"
_inp_bg   = "#2A3045" if dark else "#ffffff"
_inp_brd  = "#4A5268" if dark else "#CBD5E1"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Source+Serif+4:wght@400;600&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"], .stMarkdown, .stButton button, p, div, label, span, h1, h2, h3, h4 {{
  font-family: 'Source Sans 3', 'Segoe UI', system-ui, sans-serif;
  color: {_txt};
}}
.stApp {{ background: {_bg}; }}
.block-container {{ padding-top: 4.5rem !important; max-width: 1140px; }}
[data-testid="stAppDeployButton"], .stDeployButton, footer {{ display: none !important; }}

/* ── Page header bar ── */
.page-header {{
  background: {'#1B2A3B' if dark else '#1A3C5E'};
  border-bottom: 3px solid #2B7A5F;
  padding: 18px 28px 16px 28px;
  border-radius: 6px;
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,.15);
}}
.page-header-left {{ display:flex; align-items:center; gap:14px; }}
.page-header-logo {{
  width: 42px; height: 42px; border-radius: 10px;
  background: #2B7A5F;
  display:flex; align-items:center; justify-content:center;
  font-size: 1.3rem; color:#fff; font-weight:700;
  letter-spacing:-1px; border: 2px solid rgba(255,255,255,.2);
}}
.page-header-title {{ color:#fff; font-size:1.28rem; font-weight:700; letter-spacing:-.3px; line-height:1.2; }}
.page-header-sub   {{ color:rgba(255,255,255,.65); font-size:.78rem; margin-top:2px; letter-spacing:.2px; }}
.page-header-badge {{
  background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.25);
  color: rgba(255,255,255,.85); padding: 5px 14px; border-radius: 4px;
  font-size: .74rem; font-weight: 600; letter-spacing:.5px; text-transform:uppercase;
}}

/* ── Cards / Containers ── */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {_card_bg} !important;
  border: 1px solid {_card_brd} !important;
  border-radius: 6px !important;
  box-shadow: 0 1px 4px rgba(0,0,0,.06) !important;
}}

.clinical-card {{
  background: {_card_bg};
  border: 1px solid {_card_brd};
  border-radius: 6px;
  padding: 20px 24px;
  margin-bottom: 14px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}}
.clinical-card.result-pos {{
  border-left: 4px solid #C0392B;
}}
.clinical-card.result-neg {{
  border-left: 4px solid #1E8449;
}}
.clinical-card.result-off {{
  border-left: 4px solid #AEB6BF;
  background: {'#1A2030' if dark else '#F7F9FC'};
}}

/* ── Section headers ── */
.section-header {{
  display: flex; align-items: center; gap: 10px;
  border-bottom: 1.5px solid {_card_brd};
  padding-bottom: 10px; margin-bottom: 16px;
}}
.section-header .step-badge {{
  background: #1A3C5E; color:#fff;
  font-size:.72rem; font-weight:700;
  padding: 3px 10px; border-radius:3px; letter-spacing:.6px;
  text-transform: uppercase;
}}
.section-header .step-title {{
  font-size:1rem; font-weight:700;
  color: {'#A8D5C8' if dark else '#1A3C5E'};
  letter-spacing:-.2px;
}}
.section-caption {{
  font-size:.83rem; color:{_sub};
  margin-bottom:12px; line-height:1.5;
}}

/* ── Progress stepper ── */
.stepper {{
  display: flex; align-items: center;
  background: {_card_bg};
  border: 1px solid {_card_brd};
  border-radius: 6px; padding: 14px 20px;
  margin-bottom: 20px;
  box-shadow: 0 1px 4px rgba(0,0,0,.05);
}}
.step {{ display:flex; align-items:center; gap:10px; }}
.step .dot {{
  width:30px; height:30px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:.88rem;
  background: {'#2E3650' if dark else '#DDE6EF'}; color: {'#6B7A99' if dark else '#5D7A9A'};
  border: 2px solid {'#3A4060' if dark else '#B8CCE0'};
}}
.step.on .dot {{
  background: #1A3C5E; color:#fff;
  border-color: #2B7A5F;
  box-shadow: 0 0 0 3px rgba(43,122,95,.2);
}}
.step .lbl {{ font-weight:600; color:{_sub}; font-size:.88rem; }}
.step.on .lbl {{ color: {'#A8D5C8' if dark else '#1A3C5E'}; }}
.line {{ flex:1; height:2px; background:{'#2E3650' if dark else '#DDE6EF'}; margin:0 16px; border-radius:1px; }}
.line.on {{ background: #2B7A5F; }}

/* ── Model result cards ── */
.card {{
  background: {_card_bg}; border: 1px solid {_card_brd};
  border-radius: 6px; padding: 20px 22px; min-height: 340px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}}
.card.pos {{ border-top: 4px solid #C0392B; }}
.card.neg {{ border-top: 4px solid #1E8449; }}
.card.off {{
  border-top: 4px solid {'#3A4060' if dark else '#AEB6BF'};
  background: {'#181F2E' if dark else '#F7F9FC'};
}}
.tier {{
  font-size:.68rem; font-weight:700; letter-spacing:1.4px;
  color: {'#7EB8AC' if dark else '#2B7A5F'};
  text-transform:uppercase; margin-bottom:4px;
}}
.mname {{
  font-size:1.15rem; font-weight:700;
  color: {'#A8D5C8' if dark else '#1A3C5E'}; margin-bottom:4px;
}}
.mdesc {{ color:{_sub}; font-size:.82rem; margin-bottom:14px; min-height:34px; line-height:1.45; }}
.score {{
  font-size: 2.8rem; font-weight:700; line-height:1; letter-spacing:-1px;
  color:{_txt}; font-variant-numeric: tabular-nums;
}}
.score span {{ font-size:1.2rem; color:{_sub}; margin-left:2px; font-weight:400; }}
.slabel {{ color:{_sub}; font-size:.75rem; margin:3px 0 12px 0; letter-spacing:.3px; text-transform:uppercase; }}
.bar {{
  position:relative; height:8px; border-radius:3px;
  background: {'#2E3650' if dark else '#E0E8EF'};
}}
.bar .fill {{ position:absolute; left:0; top:0; height:8px; border-radius:3px; }}
.bar .thr {{
  position:absolute; top:-6px; width:2px; height:20px;
  background: {'#C0C8D8' if dark else '#2C3E50'}; border-radius:1px;
}}
.barlab {{
  position:relative; height:18px; margin-top:6px;
  font-size:.7rem; color:{_sub};
}}
.barlab .t {{ position:absolute; transform:translateX(-50%); white-space:nowrap; font-weight:600; }}

/* ── Status pills ── */
.pill {{
  display:inline-block; margin-top:12px; padding:5px 14px;
  border-radius:3px; font-weight:700; font-size:.75rem;
  letter-spacing:.8px; text-transform:uppercase;
}}
.pill.pos {{ background:#FADBD8; color:#922B21; border:1px solid #E6B0AA; }}
.pill.neg {{ background:#D5F5E3; color:#1D6A3A; border:1px solid #A9DFBF; }}
.pill.off {{ background:{'#252D40' if dark else '#EAECEE'}; color:{_sub}; border:1px solid {_card_brd}; }}

/* ── Recommendation banners ── */
.rec {{
  border-radius:5px; padding:14px 18px;
  display:flex; gap:14px; align-items:flex-start; border:1px solid;
  margin-top:10px;
}}
.rec .ic {{ font-size:1.3rem; line-height:1.3; flex-shrink:0; }}
.rec .tt {{ font-weight:700; font-size:.95rem; margin-bottom:3px; }}
.rec .body {{ font-size:.88rem; line-height:1.5; }}
.rec.danger {{ background:{'#2D1515' if dark else '#FDEDEC'}; border-color:#E6B0AA; color:{'#EDBBBB' if dark else '#6E2219'}; }}
.rec.ok     {{ background:{'#0D2010' if dark else '#EAF7EE'}; border-color:#A9DFBF; color:{'#8BDBA8' if dark else '#1D6A3A'}; }}
.rec.warn   {{ background:{'#2A1E00' if dark else '#FEF9E7'}; border-color:#F9E79F; color:{'#D4AC0D' if dark else '#6E4B00'}; }}
.rec.info   {{ background:{'#0D1E2A' if dark else '#EAF2FB'}; border-color:#AED6F1; color:{'#85C1E9' if dark else '#1A5276'}; }}

/* ── Risk summary card ── */
.risk-card {{
  background:{'#1A2232' if dark else '#F4F8FC'};
  border:1px solid {'#2A3550' if dark else '#BDD5E8'};
  border-radius:5px; padding:16px 20px; margin:12px 0;
}}
.risk-card h4 {{
  color:{'#A8D5C8' if dark else '#1A3C5E'};
  margin:0 0 10px 0; font-size:.92rem; font-weight:700;
  text-transform:uppercase; letter-spacing:.5px;
}}
.risk-factor {{ padding:5px 0; font-size:.86rem; color:{_txt}; border-bottom:1px solid {_card_brd}; }}
.risk-factor:last-child {{ border-bottom:none; }}

/* ── Suggestion cards ── */
.sug-card {{
  background:{'#111E2D' if dark else '#F0F7FF'};
  border:1px solid {'#1E3048' if dark else '#C2D9F0'};
  border-left:3px solid #2B7A5F;
  border-radius:5px; padding:13px 18px; margin:8px 0;
  font-size:.88rem; line-height:1.6; color:{_txt};
}}
.sug-header {{
  font-size:.68rem; font-weight:700; letter-spacing:1px;
  color:{'#7EB8AC' if dark else '#2B7A5F'}; text-transform:uppercase;
  margin-bottom:8px;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background: {_sb_bg} !important;
  border-right: 1px solid {_sb_brd} !important;
}}
.sb-title {{
  font-weight:700; letter-spacing:.3px;
  color:{'#A8D5C8' if dark else '#1A3C5E'};
  font-size:.88rem; text-transform:uppercase; margin:10px 0 6px 0;
}}
.ptable {{
  width:100%; border-collapse:collapse; font-size:.8rem;
  background:{_card_bg}; border-radius:4px; overflow:hidden;
  border: 1px solid {_card_brd};
}}
.ptable th {{
  background:{'#1A3C5E' if not dark else '#1A2A3A'}; color:#fff;
  padding:6px 8px; text-align:left; font-weight:600; font-size:.75rem;
  letter-spacing:.3px; text-transform:uppercase;
}}
.ptable td {{ padding:5px 8px; border-bottom:1px solid {_card_brd}; color:{_txt}; }}
.ptable tr:last-child td {{ border-bottom:none; }}
.legend {{ font-size:.82rem; color:{_txt}; line-height:1.6; }}
.legend b {{ color:{'#A8D5C8' if dark else '#1A3C5E'}; }}
.disc {{ color:{_sub}; font-size:.75rem; margin-top:6px; line-height:1.5; }}

/* ── Tabs ── */
[data-baseweb="tab-list"] {{
  background: {'#161E2E' if dark else '#EBF0F5'} !important;
  border-radius: 5px 5px 0 0 !important;
  border-bottom: 2px solid {'#2A3550' if dark else '#BDD5E8'} !important;
  padding: 0 4px !important;
}}
[data-baseweb="tab"] {{
  padding: 10px 22px !important; font-weight: 600 !important;
  font-size: .88rem !important; letter-spacing:.2px !important;
  color: {_sub} !important; border-radius: 0 !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
  color: {'#A8D5C8' if dark else '#1A3C5E'} !important;
  border-bottom: 2px solid {'#2B7A5F' if not dark else '#2B7A5F'} !important;
  background: {_card_bg} !important;
}}

/* ── Primary buttons ── */
button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: #1A3C5E !important; border: none !important;
  color: #fff !important; font-weight: 600 !important;
  padding: .55rem 1.6rem !important; border-radius: 4px !important;
  letter-spacing: .3px !important; font-size:.9rem !important;
}}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
  background: #1D4E80 !important;
}}

/* ── Chip ── */
.chip {{
  display:inline-block; background:{'#1A2A3A' if dark else '#E8F1F9'};
  color:{'#A8D5C8' if dark else '#1A3C5E'};
  border:1px solid {'#2A3E55' if dark else '#BDD5E8'};
  padding:4px 12px; border-radius:3px; font-size:.78rem;
  font-weight:600; margin-top:10px; letter-spacing:.3px;
}}

/* == COMPREHENSIVE VISIBILITY FIX (dark + light) == */

/* All text inputs */
input, textarea {{
  background: {_inp_bg} !important;
  color: {_txt} !important;
  border-color: {_inp_brd} !important;
}}
[data-baseweb="input"] input {{
  background: {_inp_bg} !important; color: {_txt} !important;
}}

/* Selectbox / dropdown */
[data-baseweb="select"] > div,
[data-baseweb="select"] input {{
  background: {_inp_bg} !important;
  color: {_txt} !important;
  border-color: {_inp_brd} !important;
}}
[role="listbox"] {{ background: {_inp_bg} !important; }}
[role="option"]  {{ background: {_inp_bg} !important; color: {_txt} !important; }}
[role="option"]:hover {{ background: {'#2E3A50' if dark else '#E8F0F8'} !important; }}

/* ALL widget labels — essential for dark mode readability */
label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stToggle label {{
  color: {_txt} !important;
  font-size: .92rem !important;
}}

/* Toggle text */
[data-testid="stToggle"] p,
[data-testid="stToggle"] span {{ color: {_txt} !important; font-size: .92rem !important; }}

/* Number input */
[data-testid="stNumberInput"] input {{
  background: {_inp_bg} !important; color: {_txt} !important;
  border-color: {_inp_brd} !important; font-size: .92rem !important;
}}

/* Alerts / info / warning */
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {{
  color: {_txt} !important; font-size: .9rem !important;
}}

/* Expander */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {{
  color: {_txt} !important; font-size: .92rem !important;
}}

/* ── DISCLAIMER DIALOG — always high contrast ── */
[data-testid="stModal"] {{
  background: {'#1C2233' if dark else '#ffffff'} !important;
}}
[data-testid="stModal"] div,
[data-testid="stModal"] p,
[data-testid="stModal"] span,
[data-testid="stModal"] li,
[data-testid="stModal"] h1,
[data-testid="stModal"] h2,
[data-testid="stModal"] h3,
[data-testid="stModal"] label,
[data-testid="stModal"] strong,
[data-testid="stModal"] em {{
  color: {'#E8EDF5' if dark else '#1A1F2E'} !important;
  font-size: 1rem !important;
  line-height: 1.7 !important;
}}
[data-testid="stModal"] h3 {{
  font-size: 1.25rem !important; font-weight: 700 !important;
  color: {'#7ECFC4' if dark else '#1A3C5E'} !important;
  margin-bottom: 10px !important;
}}
/* Dialog acknowledge button — always visible */
[data-testid="stModal"] button,
[data-testid="stModal"] [data-testid="stBaseButton-primary"] {{
  background: #1A5276 !important;
  color: #ffffff !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  padding: .75rem 2rem !important;
  border-radius: 5px !important;
  border: 2px solid #2874A6 !important;
  width: 100% !important;
  letter-spacing: .3px !important;
}}
[data-testid="stModal"] button:hover {{
  background: #1F618D !important;
}}

/* Secondary / default buttons */
button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {{
  color: {_txt} !important;
  background: {_inp_bg} !important;
  border: 1px solid {_inp_brd} !important;
  font-size: .9rem !important;
}}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] p {{
  color: {_sub} !important; font-size: .82rem !important;
}}

/* Sidebar widget labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
  color: {_txt} !important; font-size: .88rem !important;
}}

/* Markdown body text */
.stMarkdown p, .stMarkdown li {{
  color: {_txt}; font-size: .92rem; line-height: 1.65;
}}

/* Divider */
.med-divider {{
  height:1px; background:{'#2A3550' if dark else '#DDE6EF'};
  margin: 22px 0; border:none;
}}

/* Footer */
.footer-note {{
  text-align:center; color:{_sub}; font-size:.74rem;
  margin:28px 0 8px 0; letter-spacing:.2px;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ── Step 7: First-visit disclaimer ───────────────────────────────────────────
if not st.session_state.get("disclaimer_accepted", False):
    @st.dialog("Important Notice", width="large")
    def show_disclaimer():
        st.markdown("""
        ### 🩺 PMOS Screening Aid — Disclaimer
        
        This application is a **screening and health-education tool**, not a diagnostic system.
        
        - Its predictions are **not a confirmed diagnosis** of PMOS/PCOS.
        - A positive screen means *proceed to the next clinical step*, not that you have PMOS.
        - **Always consult a qualified healthcare professional** for diagnosis and treatment decisions.
        - Results depend on the accuracy of inputs provided.
        - This tool is for **research and educational purposes only**.
        """)
        st.markdown("---")
        if st.button("✅ I understand — Continue to the app", type="primary", use_container_width=True):
            st.session_state["disclaimer_accepted"] = True
            st.rerun()
    show_disclaimer()
    st.stop()


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    meta = json.load(open(MODELS / "model_metadata.json"))
    lite = json.load(open(MODELS / "model_B_lite_metadata.json"))
    specs = (
        ("lite", "Lite model", "STEP 1 · QUESTIONNAIRE",
         "10 history questions - no exam, no tests", lite),
        ("B",    "Model B",    "STEP 2 · EXAM + BLOOD TESTS",
         "Clinical exam and blood tests, no ultrasound", meta["model_B"]),
        ("A",    "Model A",    "STEP 3 · ULTRASOUND",
         "Full model including ultrasound findings",    meta["model_A"]),
    )
    out = {}
    for key, name, tier, desc, m in specs:
        out[key] = dict(
            name=name, tier=tier, desc=desc, thr=float(m["threshold"]),
            feats=list(m["features"]),
            metrics=m.get("test_metrics_at_threshold", {}),
            pipe=joblib.load(MODELS / Path(m["file"]).name))
    return out


@st.cache_data
def load_test_patients():
    import preprocess as base
    from sklearn.model_selection import train_test_split
    df = base.load_data()
    _, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[base.TARGET])
    return test, base.TARGET


@st.cache_data
def load_train():
    import preprocess as base
    from sklearn.model_selection import train_test_split
    df = base.load_data()
    train, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df[base.TARGET])
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

build_knowledge_base()


@st.cache_resource
def get_background(key):
    m   = M[key]
    pre = m["pipe"].named_steps["preprocessing"]
    return np.asarray(pre.transform(load_train()[m["feats"]]))


# ── Input helpers ─────────────────────────────────────────────────────────────
def kind(f):
    if "(Y/N)" in f: return "yn"
    if f == "Cycle(R/I)": return "cycle"
    return "num"


def widget(f, col):
    k = f"in_{f}"
    if kind(f) == "yn":
        return col.selectbox(f.replace("(Y/N)", "").strip(), [NA, "No", "Yes"], key=k)
    if kind(f) == "cycle":
        return col.selectbox("Menstrual cycle", [NA, "Regular", "Irregular"], key=k)
    return col.number_input(f, value=None, min_value=0.0, format="%g", key=k, placeholder="enter value")


def to_value(f, raw):
    if kind(f) == "yn":   return {"No": 0.0, "Yes": 1.0}.get(raw, np.nan)
    if kind(f) == "cycle": return {"Regular": 2.0, "Irregular": 4.0}.get(raw, np.nan)
    return np.nan if raw is None else float(raw)


def fields(feats):
    cols = st.columns(3)
    vals = {}
    for i, f in enumerate(feats):
        vals[f] = to_value(f, widget(f, cols[i % 3]))
    return vals


def sec_title(n, title, caption=""):
    st.markdown(
        f'<div class="section-header">'
        f'<span class="step-badge">Step {n}</span>'
        f'<span class="step-title">{title}</span>'
        f'</div>'
        + (f'<div class="section-caption">{caption}</div>' if caption else ''),
        unsafe_allow_html=True)


def add_derived(v):
    w, h = v.get("Weight (Kg)", np.nan), v.get("Height(Cm)", np.nan)
    v["BMI"] = w / ((h/100)**2) if h and h > 0 and not np.isnan(w) and not np.isnan(h) else np.nan
    wa, hi  = v.get("Waist(inch)", np.nan), v.get("Hip(inch)", np.nan)
    v["Waist:Hip Ratio"] = wa/hi if hi and hi > 0 and not np.isnan(wa) and not np.isnan(hi) else np.nan
    fsh, lh = v.get("FSH(mIU/mL)", np.nan), v.get("LH(mIU/mL)", np.nan)
    v["FSH/LH"] = fsh/lh if lh and lh > 0 and not np.isnan(fsh) and not np.isnan(lh) else np.nan
    return v


def load_random_patient():
    test, target = load_test_patients()
    row = test.sample(1).iloc[0]
    for f in ALL_INPUTS:
        k, x = f"in_{f}", row.get(f, np.nan)
        if kind(f) == "yn":
            st.session_state[k] = NA if pd.isna(x) else ("Yes" if float(x) == 1 else "No")
        elif kind(f) == "cycle":
            st.session_state[k] = NA if pd.isna(x) else ("Regular" if float(x) == 2 else "Irregular")
        else:
            st.session_state[k] = None if pd.isna(x) else float(x)
    st.session_state["truth"] = int(row[target])
    st.session_state["use_labs"] = True
    st.session_state["use_us"]   = True
    # Reset sequential state on new patient load
    for k in ("results_lite","results_B","results_A","step_reached","shap_lite","shap_B","shap_A"):
        st.session_state.pop(k, None)


def clear_all():
    for f in ALL_INPUTS:
        st.session_state.pop(f"in_{f}", None)
    for k in ("truth","results_lite","results_B","results_A","step_reached",
               "shap_lite","shap_B","shap_A","values_snapshot"):
        st.session_state.pop(k, None)


# ── Prediction + explanation ──────────────────────────────────────────────────
def predict(key, values):
    m   = M[key]
    X   = pd.DataFrame([{f: values.get(f, np.nan) for f in m["feats"]}],
                       columns=m["feats"]).astype(float)
    p   = float(m["pipe"].predict_proba(X)[:, 1][0])
    n_miss = int(X.isna().sum(axis=1).iloc[0])
    return p, p >= m["thr"], n_miss


def explain(key, values):
    m   = M[key]
    pre = m["pipe"].named_steps["preprocessing"]
    mod = m["pipe"].named_steps["model"]
    X   = pd.DataFrame([{f: values.get(f, np.nan) for f in m["feats"]}],
                       columns=m["feats"]).astype(float)
    xp  = np.asarray(pre.transform(X))[0]
    names = [n.split("__", 1)[1] if "__" in n else n for n in pre.get_feature_names_out()]
    names = ["Menstrual cycle" if n.startswith("Cycle(R/I)") else
             n.replace("(Y/N)", "").strip() for n in names]
    if hasattr(mod, "coef_") and getattr(mod, "kernel", "linear") == "linear":
        base_x = get_background(key).mean(axis=0)
        contrib = mod.coef_.ravel() * (xp - base_x)
        unit = "Contribution to SVM score"
    else:
        import shap
        sv = shap.TreeExplainer(mod).shap_values(xp.reshape(1, -1))
        if isinstance(sv, list): sv = sv[1]
        sv = np.asarray(sv)
        contrib = sv[0, :, 1] if sv.ndim == 3 else sv[0]
        unit = "Contribution to risk (probability points)"
    df = pd.DataFrame({"feature": names, "contribution": contrib})
    return df.groupby("feature", as_index=False).sum(), unit


def shap_chart(df, unit, top_n=8):
    top = df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    fig.patch.set_facecolor("#242938" if dark else "white")
    ax.set_facecolor("#242938" if dark else "white")
    ax.barh(top["feature"], top["contribution"], height=0.62,
            color=[RED if v > 0 else TEAL for v in top["contribution"]])
    ax.axvline(0, color="#334155", lw=1)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    tc = "#E2E8F0" if dark else "#0F172A"
    ax.tick_params(axis="y", length=0, labelsize=9, colors=tc)
    ax.tick_params(axis="x", labelsize=8, colors="#64748B")
    ax.xaxis.grid(True, color="#E2E8F0" if not dark else "#3A4055", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(unit, fontsize=8.5, color="#475569")
    fig.tight_layout()
    return fig


# ── HTML builders ─────────────────────────────────────────────────────────────
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
    pct   = max(0.0, min(100.0, p * 100))
    risk_badge = ("🟢 Low Risk" if pct < 30 else "🟡 Moderate Risk" if pct < 65 else "🔴 High Risk")
    thr   = m["thr"] * 100
    label = "● POSITIVE SCREEN" if pos else "● NEGATIVE SCREEN"
    note  = ("Above threshold - refer for the next test." if pos
             else "Below threshold - low likelihood of PMOS.")
    warn  = (f'<div class="cwarn">⚠ {n_miss} of {len(m["feats"])} inputs empty '
             '(filled with training median) - interpret with caution.</div>' if n_miss else "")
    return (f'<div class="card {state}"><div class="tier">{m["tier"]}</div>'
            f'<div class="mname">{m["name"]}</div>'
            f'<div class="mdesc">{m["desc"]}</div>'
            f'<div class="score">{pct:.0f}<span>%</span></div>'
            f'<div class="slabel">{risk_badge} &nbsp;|&nbsp; Risk score (model output)</div>'
            f'<div class="bar"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div>'
            f'<div class="thr" style="left:calc({thr:.1f}% - 1px)"></div></div>'
            f'<div class="barlab"><span class="t" style="left:{thr:.1f}%">▲ threshold {thr:.0f}%</span></div>'
            f'<span class="pill {state}">{label}</span>'
            f'<div class="cnote">{note}</div>{warn}</div>')


def rec_banner(kind_, icon, title, text):
    return (f'<div class="rec {kind_}"><div class="ic">{icon}</div><div>'
            f'<div class="tt">{title}</div><div>{text}</div></div></div>')


# ── Step 3: Risk Summary Card ─────────────────────────────────────────────────
def risk_summary_card(key, p, pos, values):
    """Show a card with risk %, top 3 SHAP factors, and next steps."""
    pct = p * 100
    try:
        exp_df, _ = explain(key, values)
        top3 = exp_df.reindex(
            exp_df["contribution"].abs().sort_values(ascending=False).index
        ).head(3)
        factors_html = "".join(
            f'<div class="risk-factor">{"🔺" if row["contribution"] > 0 else "🔻"} '
            f'<b>{row["feature"]}</b> — {"increases" if row["contribution"] > 0 else "decreases"} risk</div>'
            for _, row in top3.iterrows()
        )
    except Exception:
        factors_html = "<div class='risk-factor'>SHAP not available for this model.</div>"

    next_steps = {
        "lite": ("Questionnaire screen complete. "
                 + ("Proceed to exam & blood tests (Model B) to refine." if pos
                    else "Low questionnaire risk. Continue if clinical suspicion.")),
        "B":    ("Clinical screen complete. "
                 + ("Consider ultrasound assessment (Model A) to confirm." if pos
                    else "Blood-test screen negative. Clinical judgement applies.")),
        "A":    ("Full screening complete. "
                 + ("Consult a gynaecologist or endocrinologist for Rotterdam criteria assessment." if pos
                    else "All three screens negative. Follow up if symptoms persist.")),
    }

    st.markdown(f"""
    <div class="risk-card">
      <h4>📊 Risk Summary — {M[key]['name']}</h4>
      <div style="font-size:.95rem;margin-bottom:8px;">
        Risk score: <b>{pct:.1f}%</b> &nbsp;|&nbsp;
        Threshold: <b>{M[key]['thr']*100:.0f}%</b> &nbsp;|&nbsp;
        Result: <b>{"🔴 POSITIVE" if pos else "🟢 NEGATIVE"}</b>
      </div>
      <div style="margin-bottom:8px;font-size:.85rem;color:{SLATE};">
        <i>Top contributing factors (SHAP):</i>
      </div>
      {factors_html}
      <div style="margin-top:10px;font-size:.84rem;color:{SLATE};">
        💡 {next_steps[key]}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Step 5: Suggestions based on report ──────────────────────────────────────
def render_suggestions(results: dict):
    st.markdown('<div style="font-size:1.05rem;font-weight:700;letter-spacing:.3px;color:#1A3C5E;text-transform:uppercase;border-bottom:2px solid #2B7A5F;padding-bottom:8px;margin-bottom:10px;">Personalised Suggestions</div>',
                unsafe_allow_html=True)
    st.caption("Based on your actual screening results. These are educational suggestions — "
               "not medical advice. Consult a qualified healthcare professional.")

    last_key = "A" if "A" in results else ("B" if "B" in results else "lite")
    p, pos = results[last_key]
    pct = p * 100

    suggestions = []

    # Based on result
    if pos:
        suggestions += [
            ("🏥 Doctor consultation", "Your screen is positive. Book an appointment with a gynaecologist "
             "or endocrinologist. Bring this report and mention your symptoms."),
            ("🔬 Questions to ask your doctor",
             "Ask about: Rotterdam criteria assessment, hormone panel (LH, FSH, AMH, testosterone, insulin), "
             "ultrasound if not yet done, and management options."),
        ]
    else:
        suggestions += [
            ("✅ Routine follow-up",
             "Your screen is currently negative. If symptoms persist or worsen, "
             "revisit this screening tool and consult a doctor."),
        ]

    # Always include lifestyle
    suggestions += [
        ("🥗 Lifestyle (general evidence-based)",
         "Regardless of result: regular physical activity (150 min/week), balanced diet (low glycaemic index foods), "
         "adequate sleep (7–9 hours), and stress management may support hormonal health."),
        ("📋 Track your symptoms",
         "Use the Symptom Tracker tab to log symptoms over time. "
         "A trend log is useful evidence when consulting a doctor."),
        ("⚠️ Important disclaimer",
         "These suggestions are general health education. "
         "This tool does not diagnose PMOS. Do not change any medication or treatment based on this screen alone."),
    ]

    for title, text in suggestions:
        st.markdown(f'<div class="sug-card"><b>{title}</b><br><span style="font-size:.88rem">{text}</span></div>',
                    unsafe_allow_html=True)


# ── PDF field interpretation helper ──────────────────────────────────────────
def _interpret_field(field: str, value):
    """Return (display_value, interpretation_str, is_flagged) for a raw field value."""
    import math as _math
    if isinstance(value, float) and _math.isnan(value):
        return "\u2014", "\u2014", False

    # Boolean (Y/N) fields
    yn_map = {
        "Weight gain(Y/N)":     ("Weight gain reported",                      "No weight gain"),
        "hair growth(Y/N)":     ("Excess hair growth (hirsutism) \u2014 Yes", "No excess hair growth"),
        "Skin darkening (Y/N)": ("Skin darkening (acanthosis nigricans)",      "No skin darkening"),
        "Hair loss(Y/N)":       ("Hair loss / thinning \u2014 Yes",            "No hair loss"),
        "Pimples(Y/N)":         ("Acne / pimples \u2014 Yes",                 "No acne"),
        "Fast food (Y/N)":      ("High fast-food intake",                      "Low fast-food intake"),
        "Reg.Exercise(Y/N)":    ("Regular exercise \u2014 Yes",                "No regular exercise"),
        "Pregnant(Y/N)":        ("Currently pregnant \u2014 Yes",              "Not pregnant"),
    }
    if field in yn_map:
        yes_txt, no_txt = yn_map[field]
        is_yes = (value == 1.0)
        display = "Yes" if is_yes else "No"
        flag = is_yes and field in (
            "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
            "Hair loss(Y/N)", "Pimples(Y/N)"
        )
        return display, (yes_txt if is_yes else no_txt), flag

    # Menstrual cycle regularity
    if field == "Cycle(R/I)":
        if value == 2.0:
            return "Regular", "Regular cycle (21\u201335 days) \u2014 Normal", False
        if value == 4.0:
            return "Irregular", "Irregular / absent cycle \u2014 Common PMOS indicator", True
        return str(value), "\u2014", False

    # Age
    if field == "Age (yrs)":
        a = int(value)
        note = "Reproductive age" if 15 <= a <= 45 else "Outside typical screening age"
        return str(a) + " yrs", note, False

    # Cycle length
    if field == "Cycle length(days)":
        v = int(value)
        if 21 <= v <= 35:
            return f"{v} days", f"{v} days \u2014 Normal (21\u201335 days)", False
        if v < 21:
            return f"{v} days", f"{v} days \u2014 Short cycle (<21 days); evaluate", True
        return f"{v} days", f"{v} days \u2014 Long cycle (>35 days); possible anovulation", True

    # BMI
    if field == "BMI":
        v = round(float(value), 1)
        if v < 18.5:
            return f"{v}", f"{v} \u2014 Underweight (Normal: 18.5\u201324.9)", True
        if v < 25:
            return f"{v}", f"{v} \u2014 Normal weight (18.5\u201324.9)", False
        if v < 30:
            return f"{v}", f"{v} \u2014 Overweight (25\u201329.9); mild risk", True
        return f"{v}", f"{v} \u2014 Obese (\u226530); insulin resistance risk", True

    # Weight / Height
    if field == "Weight (Kg)":
        return f"{float(value):.1f} kg", "Body weight", False
    if field == "Height(Cm)":
        return f"{float(value):.0f} cm", "Height", False

    # Waist:Hip Ratio
    if field == "Waist:Hip Ratio":
        v = round(float(value), 3)
        if v > 0.85:
            return f"{v}", f"{v} \u2014 >0.85; central obesity risk (Normal \u22640.85 for women)", True
        return f"{v}", f"{v} \u2014 Normal (\u22640.85 for women)", False

    # Waist / Hip raw
    if "Waist(inch)" in field:
        v = round(float(value), 1)
        note_w = (f"{v} in \u2014 >35 in; central obesity") if v > 35 else f"{v} in"
        return f"{v} in", note_w, v > 35
    if "Hip(inch)" in field:
        return f"{round(float(value),1)} in", "Hip circumference", False

    # Pulse rate
    if "Pulse rate" in field:
        v = int(value)
        if 60 <= v <= 100:
            return f"{v} bpm", f"{v} bpm \u2014 Normal (60\u2013100 bpm)", False
        if v < 60:
            return f"{v} bpm", f"{v} bpm \u2014 Bradycardia (<60 bpm)", True
        return f"{v} bpm", f"{v} bpm \u2014 Tachycardia (>100 bpm)", True

    # Hemoglobin
    if "Hb(" in field:
        v = round(float(value), 1)
        if v < 11.0:
            return f"{v} g/dL", f"{v} g/dL \u2014 Anaemia (<11; Normal: 12\u201316 g/dL for women)", True
        if v < 12.0:
            return f"{v} g/dL", f"{v} g/dL \u2014 Borderline low (Normal: 12\u201316 g/dL)", True
        if v <= 16.0:
            return f"{v} g/dL", f"{v} g/dL \u2014 Normal (12\u201316 g/dL for women)", False
        return f"{v} g/dL", f"{v} g/dL \u2014 Above normal (>16 g/dL)", True

    # FSH
    if field == "FSH(mIU/mL)":
        v = round(float(value), 2)
        if v < 3.0:
            return f"{v} mIU/mL", f"{v} \u2014 Low FSH (<3; Normal follicular: 3\u201310 mIU/mL)", True
        if v <= 10.0:
            return f"{v} mIU/mL", f"{v} \u2014 Normal follicular phase (3\u201310 mIU/mL)", False
        if v <= 20.0:
            return f"{v} mIU/mL", f"{v} \u2014 Mildly elevated FSH (10\u201320 mIU/mL)", True
        return f"{v} mIU/mL", f"{v} \u2014 High FSH (>20; possible diminished ovarian reserve)", True

    # LH
    if field == "LH(mIU/mL)":
        v = round(float(value), 2)
        if v < 2.0:
            return f"{v} mIU/mL", f"{v} \u2014 Low LH (Normal: 2\u201315 mIU/mL)", True
        if v <= 15.0:
            return f"{v} mIU/mL", f"{v} \u2014 Normal follicular phase (2\u201315 mIU/mL)", False
        if v <= 25.0:
            return f"{v} mIU/mL", f"{v} \u2014 Elevated LH (15\u201325); associated with PMOS", True
        return f"{v} mIU/mL", f"{v} \u2014 High LH (>25); strongly associated with PMOS", True

    # FSH/LH ratio
    if field == "FSH/LH":
        v = round(float(value), 2)
        if v < 1.0:
            return f"{v}", f"{v} \u2014 LH dominant (<1.0); typical PMOS pattern", True
        if v <= 2.0:
            return f"{v}", f"{v} \u2014 Normal FSH:LH ratio (1\u20132)", False
        return f"{v}", f"{v} \u2014 FSH dominant (>2); possible diminished reserve", True

    # AMH
    if "AMH" in field:
        v = round(float(value), 2)
        if v < 1.0:
            return f"{v} ng/mL", f"{v} \u2014 Low AMH (<1.0; reduced ovarian reserve)", True
        if v <= 3.5:
            return f"{v} ng/mL", f"{v} \u2014 Normal (1.0\u20133.5 ng/mL)", False
        if v <= 6.9:
            return f"{v} ng/mL", f"{v} \u2014 High-normal AMH; possible PMOS (3.5\u20136.9)", True
        return f"{v} ng/mL", f"{v} \u2014 Elevated AMH (>7.0); PMOS indicator", True

    # PRG (Progesterone)
    if "PRG(" in field:
        v = round(float(value), 2)
        if v < 1.0:
            return f"{v} ng/mL", f"{v} \u2014 Low progesterone; possible anovulation", True
        if v <= 10.0:
            return f"{v} ng/mL", f"{v} \u2014 Follicular / low-luteal range (0.1\u201310 ng/mL)", False
        return f"{v} ng/mL", f"{v} \u2014 Mid-luteal range (suggests ovulation occurred)", False

    # RBS (Random Blood Sugar)
    if "RBS(" in field:
        v = round(float(value), 1)
        if v < 140:
            return f"{v} mg/dL", f"{v} \u2014 Normal (<140 mg/dL)", False
        if v < 200:
            return f"{v} mg/dL", f"{v} \u2014 Pre-diabetic range (140\u2013199 mg/dL)", True
        return f"{v} mg/dL", f"{v} \u2014 Diabetic range (\u2265200 mg/dL)", True

    # Blood pressure
    if "BP _Systolic" in field:
        v = int(value)
        if v < 120:
            return f"{v} mmHg", f"{v} \u2014 Normal systolic (<120 mmHg)", False
        if v < 130:
            return f"{v} mmHg", f"{v} \u2014 Elevated (120\u2013129 mmHg)", True
        if v < 140:
            return f"{v} mmHg", f"{v} \u2014 Stage 1 hypertension (130\u2013139 mmHg)", True
        return f"{v} mmHg", f"{v} \u2014 Stage 2 hypertension (\u2265140 mmHg)", True
    if "BP _Diastolic" in field:
        v = int(value)
        if v < 80:
            return f"{v} mmHg", f"{v} \u2014 Normal diastolic (<80 mmHg)", False
        if v < 90:
            return f"{v} mmHg", f"{v} \u2014 Stage 1 hypertension (80\u201389 mmHg)", True
        return f"{v} mmHg", f"{v} \u2014 Stage 2 hypertension (\u226590 mmHg)", True

    # Follicle counts (ultrasound)
    if "Follicle No." in field:
        v = int(value)
        side = "Right" if "(R)" in field else "Left"
        if v >= 12:
            return str(v), f"{v} follicles ({side}) \u2014 \u226512 per ovary: polycystic threshold (PMOS indicator)", True
        return str(v), f"{v} follicles ({side}) \u2014 Normal (<12 per ovary)", False

    # Avg follicle size
    if "Avg. F size" in field:
        v = round(float(value), 1)
        side = "Right" if "(R)" in field else "Left"
        if v <= 9.0:
            return f"{v} mm", f"{v} mm ({side}) \u2014 Small antral follicles (\u22649 mm): PMOS pattern", True
        return f"{v} mm", f"{v} mm ({side}) \u2014 Normal follicle size", False

    # Endometrium
    if "Endometrium" in field:
        v = round(float(value), 1)
        if v < 7.0:
            return f"{v} mm", f"{v} mm \u2014 Thin endometrium (<7 mm); evaluate", True
        if v <= 14.0:
            return f"{v} mm", f"{v} mm \u2014 Normal range (7\u201314 mm)", False
        return f"{v} mm", f"{v} mm \u2014 Thick endometrium (>14 mm); evaluate", True

    # TSH
    if "TSH" in field:
        v = round(float(value), 2)
        if v < 0.4:
            return f"{v} mIU/L", f"{v} \u2014 Low TSH; hyperthyroidism risk (Normal: 0.4\u20134.0)", True
        if v <= 4.0:
            return f"{v} mIU/L", f"{v} \u2014 Normal (0.4\u20134.0 mIU/L)", False
        return f"{v} mIU/L", f"{v} \u2014 Elevated TSH; hypothyroidism risk (>4.0)", True

    # PRL (Prolactin)
    if "PRL" in field:
        v = round(float(value), 1)
        if v <= 25.0:
            return f"{v} ng/mL", f"{v} \u2014 Normal (\u226425 ng/mL for non-pregnant women)", False
        return f"{v} ng/mL", f"{v} \u2014 Elevated prolactin (>25 ng/mL; may suppress ovulation)", True

    # Vit D3
    if "Vit D3" in field:
        v = round(float(value), 1)
        if v < 20:
            return f"{v} ng/mL", f"{v} \u2014 Deficient (<20; Normal: 20\u201350 ng/mL)", True
        if v <= 50:
            return f"{v} ng/mL", f"{v} \u2014 Normal range (20\u201350 ng/mL)", False
        return f"{v} ng/mL", f"{v} \u2014 Above optimal (>50 ng/mL)", True

    # Beta-HCG
    if "beta-HCG" in field.lower() or "betahcg" in field.lower():
        v = round(float(value), 1)
        note_hcg = (f"{v} mIU/mL - Positive (>5)") if v > 5 else (f"{v} mIU/mL - Negative (<=5 mIU/mL)")
        return f"{v} mIU/mL", note_hcg, v > 5

    # Default
    dv = f"{float(value):.2f}" if isinstance(value, float) else str(value)
    return dv, "\u2014", False


# ── Step 4: PDF Export ────────────────────────────────────────────────────────
def generate_pdf_report(values: dict, results: dict, shap_figs: dict) -> bytes:
    """Generate a PDF report using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, Image as RLImage, HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return b""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    teal   = colors.HexColor("#0F766E")
    red    = colors.HexColor("#DC2626")
    amber  = colors.HexColor("#B45309")
    green  = colors.HexColor("#16A34A")
    slate  = colors.HexColor("#475569")
    head1  = ParagraphStyle("h1", fontSize=18, textColor=teal, spaceAfter=6, fontName="Helvetica-Bold")
    head2  = ParagraphStyle("h2", fontSize=13, textColor=teal, spaceAfter=4, fontName="Helvetica-Bold")
    body   = ParagraphStyle("body", fontSize=10, spaceAfter=4, leading=14, textColor=colors.HexColor("#1E293B"))
    small  = ParagraphStyle("small", fontSize=8.5, textColor=slate, leading=12)
    disc   = ParagraphStyle("disc", fontSize=8, textColor=slate, leading=11, borderPadding=6,
                            backColor=colors.HexColor("#FEF3C7"))

    story = []

    # Header
    story.append(Paragraph("PMOS Screening Aid \u2014 Patient Report", head1))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}", small))
    story.append(HRFlowable(width="100%", thickness=1, color=teal, spaceAfter=12))

    # Disclaimer
    story.append(Paragraph(
        "DISCLAIMER: This report is a screening aid, not a diagnosis. "
        "Results are not a confirmed diagnosis of PMOS/PCOS. "
        "Consult a qualified healthcare professional for clinical decisions.",
        disc))
    story.append(Spacer(1, 12))

    # ── Patient inputs summary — 3 columns with interpretation ──────────────────
    story.append(Paragraph("Patient Input Summary", head2))
    story.append(Paragraph(
        "Values are as entered. Reference ranges are population-based guidelines "
        "(WHO, ASRM, ADA) and vary by individual. Fields marked \u26d1 are outside "
        "typical reference ranges. Consult your clinician for personalised interpretation.",
        ParagraphStyle("note", fontSize=7.5, textColor=slate, leading=11)))
    story.append(Spacer(1, 5))

    cell_n = ParagraphStyle("cn", fontSize=8,   leading=11, textColor=colors.HexColor("#1E293B"))
    cell_f = ParagraphStyle("cf", fontSize=8,   leading=11, textColor=colors.HexColor("#B91C1C"))
    cell_h = ParagraphStyle("ch", fontSize=8.5, leading=11, textColor=colors.white,
                            fontName="Helvetica-Bold")

    hdr_row = [Paragraph("Field",  cell_h),
               Paragraph("Value",  cell_h),
               Paragraph("Interpretation / Reference Range", cell_h)]
    input_rows = [hdr_row]
    flag_row_idxs = []
    ri = 1
    for f, v in values.items():
        if isinstance(v, float) and np.isnan(v):
            continue
        disp_v, interp, flagged = _interpret_field(f, v)
        label = f.replace("(Y/N)", "").replace("(R/I)", "").strip()
        if flagged:
            flag_row_idxs.append(ri)
            pl = Paragraph("\u26d1 " + label, cell_f)
            pv = Paragraph(disp_v, cell_f)
            pi = Paragraph(interp,  cell_f)
        else:
            pl = Paragraph(label,  cell_n)
            pv = Paragraph(disp_v, cell_n)
            pi = Paragraph(interp, cell_n)
        input_rows.append([pl, pv, pi])
        ri += 1

    if len(input_rows) > 1:
        t = Table(input_rows, colWidths=[4.5*cm, 2.5*cm, 9.5*cm])
        ts = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), teal),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F9F8")]),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E5E4")),
            ("PADDING",       (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ])
        for ri_flag in flag_row_idxs:
            ts.add("BACKGROUND", (0, ri_flag), (-1, ri_flag), colors.HexColor("#FFF7ED"))
        t.setStyle(ts)
        story.append(t)
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "\u26d1 = outside typical reference range.  "
        "These values require clinical evaluation, not automated interpretation.",
        ParagraphStyle("fnote", fontSize=7, textColor=slate, leading=10)))
    story.append(Spacer(1, 14))

    # Screening results
    story.append(Paragraph("Screening Results", head2))
    MODEL_ORDER = [("lite", "Lite Model"), ("B", "Model B"), ("A", "Model A")]
    for key, mname in MODEL_ORDER:
        if key not in results:
            continue
        p, pos = results[key]
        pct = p * 100
        clr = red if pos else green
        story.append(Paragraph(f"{mname}: {pct:.1f}% risk — {'POSITIVE SCREEN' if pos else 'NEGATIVE SCREEN'}",
                                ParagraphStyle("res", fontSize=11, textColor=clr,
                                               fontName="Helvetica-Bold", spaceAfter=4)))
        story.append(Paragraph(
            f"Threshold: {M[key]['thr']*100:.0f}% | {'Above' if pos else 'Below'} threshold",
            small))

        # SHAP chart
        if key in shap_figs:
            img_buf = io.BytesIO()
            shap_figs[key].savefig(img_buf, format="png", dpi=120, bbox_inches="tight",
                                    facecolor="white")
            img_buf.seek(0)
            story.append(Spacer(1, 4))
            story.append(RLImage(img_buf, width=14*cm, height=6*cm))

        story.append(Spacer(1, 10))

    # Suggestions
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#D9E5E4"), spaceAfter=8))
    story.append(Paragraph("Suggested Next Steps", head2))
    last_key = "A" if "A" in results else ("B" if "B" in results else "lite")
    _, pos = results[last_key]
    if pos:
        story.append(Paragraph("• Your screen is positive. Book an appointment with a gynaecologist or endocrinologist.", body))
        story.append(Paragraph("• Ask about: Rotterdam criteria, hormone panel, ultrasound, management options.", body))
    else:
        story.append(Paragraph("• Your screen is negative. If symptoms persist, revisit this tool and consult a doctor.", body))
    story.append(Paragraph("• General lifestyle: regular exercise, balanced diet, adequate sleep, stress management.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "This report is generated by a research/educational screening tool. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare professional.",
        disc))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    _hdr_bg = '#1A2A3A' if dark else '#1A3C5E'
    st.markdown(
        f'<div style="background:{_hdr_bg};margin:-1rem -1rem 1rem -1rem;'
        'padding:18px 18px 14px 18px;border-bottom:3px solid #2B7A5F;">'
        '<div style="color:#fff;font-weight:700;font-size:.9rem;'
        'letter-spacing:.5px;text-transform:uppercase;margin-bottom:3px;">'
        'PMOS Screening Aid</div>'
        '<div style="color:rgba(255,255,255,.55);font-size:.72rem;">'
        'Clinical Decision Support</div>'
        '</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sb-title">Patient Tools</div>', unsafe_allow_html=True)
    st.button('Load Demo Patient', on_click=load_random_patient, use_container_width=True)
    st.button('Reset Form', on_click=clear_all, use_container_width=True)
    st.markdown(
        '<div class="disc">Demo patients are from the held-out test set '
        '— never used in model training.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="med-divider"></div>', unsafe_allow_html=True)
    dm_new = st.toggle('Dark mode', value=dark, key='dm_toggle')
    if dm_new != dark:
        st.session_state['dark_mode'] = dm_new
        st.rerun()

    st.markdown('<div class="sb-title" style="margin-top:18px">Screening Protocol</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="legend">'
        '<b>Step 1 — Lite</b><br>'
        '10 symptom &amp; history questions. No investigations required.<br><br>'
        '<b>Step 2 — Model B</b><br>'
        'Clinical exam + blood tests (FSH, LH, AMH, Hb, RBS).<br><br>'
        '<b>Step 3 — Model A</b><br>'
        'Adds transvaginal ultrasound findings.<br><br>'
        'Each threshold detects <b>~90% of PMOS cases</b>. '
        'A positive screen means <i>proceed to the next clinical step</i>, '
        'not a confirmed diagnosis.'
        '</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sb-title" style="margin-top:18px">Model Performance (Test Set)</div>',
                unsafe_allow_html=True)
    rows = ''
    for k in ('lite', 'B', 'A'):
        t = M[k]['metrics']
        if t:
            rows += (f"<tr><td>{M[k]['name'].replace(' model','')}</td>"
                     f"<td>{t.get('Recall', float('nan')):.2f}</td>"
                     f"<td>{t.get('Specificity', float('nan')):.2f}</td>"
                     f"<td>{t.get('ROC-AUC', float('nan')):.2f}</td></tr>")
    if rows:
        st.markdown('<table class="ptable"><tr><th>Model</th><th>Recall</th>'
                    '<th>Spec.</th><th>AUC</th></tr>' + rows + '</table>',
                    unsafe_allow_html=True)
    st.markdown(
        '<div class="disc">n=109 held-out patients (36 PMOS positive). '
        'Recall = sensitivity; Spec. = specificity.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="med-divider"></div>', unsafe_allow_html=True)
    if st.button('Open PMOSBot Assistant', use_container_width=True, key='open_chat_btn',
                 help='AI assistant for PMOS / PCOS clinical questions'):
        st.session_state['view'] = 'chatbot'
        st.rerun()
    st.markdown(
        '<div class="disc">AI-powered clinical Q&amp;A on PMOS symptoms, '
        'hormones, and screening results.</div>',
        unsafe_allow_html=True)


# ── View routing ──────────────────────────────────────────────────────────────
if st.session_state.get("view", "dashboard") == "chatbot":
    render_chatbot_page()
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_screen, tab_tracker, tab_insights = st.tabs(["PMOS Screening", "Symptom Tracker", "Project Insights"])

# ══════════════════════════════════════════════════════════════════════════════
# SCREENING TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_screen:
    st.markdown(
        '<div class="page-header">'
        '<div class="page-header-left">'
        '<div class="page-header-logo">Rx</div>'
        '<div><div class="page-header-title">PMOS Screening Aid</div>'
        '<div class="page-header-sub">Polycystic Metabolic Ovary Syndrome · Stepwise ML Screening: Questionnaire → Exam &amp; Blood Tests → Ultrasound</div>'
        '</div></div>'
        '<div class="page-header-badge">Clinical Decision Support</div>'
        '</div>', unsafe_allow_html=True)

    _step_r = st.session_state.get("step_reached", 0)
    st.markdown(stepper_html(
        bool(st.session_state.get("use_labs", False)) or _step_r >= 2,
        (bool(st.session_state.get("use_us", False)) and bool(st.session_state.get("use_labs", False))) or _step_r >= 3
    ), unsafe_allow_html=True)

    values = {}
    with st.container(border=True):
        sec_title(1, "Questionnaire", "Quick history questions — needed for every model.")
        values.update(fields(sec1))

    with st.container(border=True):
        sec_title(2, "Examination &amp; blood tests",
                  "BMI, waist:hip ratio and FSH/LH are calculated automatically. "
                  "Leave unknown values empty (filled with training medians).")
        use_labs = st.toggle("Exam & blood tests available", key="use_labs")
        if use_labs:
            values.update(fields(sec2))

    with st.container(border=True):
        sec_title(3, "Ultrasound findings", "Needs step 2. Enables the confirmation model (Model A).")
        use_us = st.toggle("Ultrasound available", key="use_us", disabled=not use_labs)
        if use_us and use_labs:
            values.update(fields(sec3))

    values = add_derived(values)
    st.write("")

    # ── Step 2: Sequential model execution ───────────────────────────────────
    # Retrieve persisted results
    results    = {}
    shap_cache = {}
    for k in ("lite", "B", "A"):
        if f"results_{k}" in st.session_state:
            results[k]    = st.session_state[f"results_{k}"]
        if f"shap_{k}" in st.session_state:
            shap_cache[k] = st.session_state[f"shap_{k}"]

    step_reached = st.session_state.get("step_reached", 0)

    # ── Run Lite ──────────────────────────────────────────────────────────────
    if step_reached == 0:
        go_lite = st.button("▶  Run Lite Model (Step 1)", type="primary")
        if go_lite:
            missing_q = [f for f in lite_f if np.isnan(values.get(f, np.nan))]
            if missing_q:
                st.warning("Please answer all questionnaire fields first: " + ", ".join(missing_q))
            else:
                p, pos, n_miss = predict("lite", values)
                results["lite"] = (p, pos)
                st.session_state["results_lite"] = (p, pos)
                st.session_state["values_snapshot"] = dict(values)
                st.session_state["step_reached"]    = 1
                st.rerun()

    # ── Show Lite result + progress buttons ───────────────────────────────────
    if step_reached >= 1:
        snap = st.session_state.get("values_snapshot", values)

        st.markdown('<div style="font-size:1.05rem;font-weight:700;letter-spacing:.3px;color:#1A3C5E;text-transform:uppercase;border-bottom:2px solid #2B7A5F;padding-bottom:8px;margin-bottom:14px;">Screening Results</div>',
                    unsafe_allow_html=True)

        # ── Lite card ─────────────────────────────────────────────────────────
        p_l, pos_l = results["lite"]
        _, _, n_miss_l = predict("lite", snap)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(model_card(M["lite"], p_l, pos_l, n_miss_l), unsafe_allow_html=True)
        with c2:
            if "B" in results:
                p_b, pos_b = results["B"]
                _, _, n_miss_b = predict("B", snap)
                st.markdown(model_card(M["B"], p_b, pos_b, n_miss_b), unsafe_allow_html=True)
            else:
                st.markdown(model_card(M["B"]), unsafe_allow_html=True)
        with c3:
            if "A" in results:
                p_a, pos_a = results["A"]
                _, _, n_miss_a = predict("A", snap)
                st.markdown(model_card(M["A"], p_a, pos_a, n_miss_a), unsafe_allow_html=True)
            else:
                st.markdown(model_card(M["A"]), unsafe_allow_html=True)

        # ── Risk summary for each completed model ─────────────────────────────
        st.write("")
        for k in [kk for kk in ("lite","B","A") if kk in results]:
            p_k, pos_k = results[k]
            risk_summary_card(k, p_k, pos_k, snap)

        # ── Proceed to Model B ────────────────────────────────────────────────
        if step_reached == 1 and "B" not in results:
            st.write("")
            if not use_labs:
                st.info("💡 Enable **Exam & blood tests** above to unlock Model B.")
            else:
                p_l, pos_l = results["lite"]
                if pos_l:
                    st.markdown(rec_banner("warn", "🟠",
                        "Lite screen positive — proceed to Model B",
                        "The questionnaire screen suggests elevated risk. "
                        "Click below to run the clinical exam + blood test model."),
                        unsafe_allow_html=True)
                else:
                    st.info("Lite screen negative. You may still proceed to Model B if clinically indicated.")
                col_b1, col_b2 = st.columns([1,4])
                with col_b1:
                    if st.button("▶  Run Model B", type="primary"):
                        p, pos, n_miss = predict("B", snap)
                        results["B"] = (p, pos)
                        st.session_state["results_B"]    = (p, pos)
                        st.session_state["step_reached"] = 2
                        st.rerun()

        # ── Proceed to Model A ────────────────────────────────────────────────
        if step_reached == 2 and "A" not in results:
            st.write("")
            if not use_us or not use_labs:
                st.info("💡 Enable **Ultrasound available** above to unlock Model A.")
            else:
                p_b, pos_b = results["B"]
                if pos_b:
                    st.markdown(rec_banner("warn", "🟠",
                        "Model B positive — proceed to Model A",
                        "Blood-test model is positive. Run the ultrasound-based confirmation model."),
                        unsafe_allow_html=True)
                else:
                    st.info("Model B negative. You may still run Model A if ultrasound is available.")
                col_a1, col_a2 = st.columns([1,4])
                with col_a1:
                    if st.button("▶  Run Model A", type="primary"):
                        p, pos, n_miss = predict("A", snap)
                        results["A"] = (p, pos)
                        st.session_state["results_A"]    = (p, pos)
                        st.session_state["step_reached"] = 3
                        st.rerun()

        # ── Final recommendation ──────────────────────────────────────────────
        if step_reached >= 1:
            st.write("")
            st.markdown('<div style="font-size:1.05rem;font-weight:700;letter-spacing:.3px;color:#1A3C5E;text-transform:uppercase;border-bottom:2px solid #2B7A5F;padding-bottom:8px;margin-bottom:14px;">Suggested Next Step</div>',
                        unsafe_allow_html=True)
            if "A" in results:
                if results["A"][1]:
                    st.markdown(rec_banner("danger","🔴","High likelihood of PMOS",
                        "The ultrasound-based model is positive. Recommend clinician "
                        "confirmation (Rotterdam criteria) and management planning."), unsafe_allow_html=True)
                else:
                    st.markdown(rec_banner("ok","🟢","PMOS unlikely",
                        "The ultrasound-based model is negative. Follow up if symptoms persist."),
                        unsafe_allow_html=True)
            elif "B" in results:
                if results["B"][1]:
                    st.markdown(rec_banner("warn","🟠","Screen positive - refer for ultrasound",
                        "Recommend ultrasound, then run Model A to confirm."), unsafe_allow_html=True)
                else:
                    st.markdown(rec_banner("info","🔵","Screen negative - routine follow-up",
                        "Blood-test screen negative. Keep clinical judgement for symptomatic patients."),
                        unsafe_allow_html=True)
            else:
                if results["lite"][1]:
                    st.markdown(rec_banner("warn","🟠","Questionnaire positive - proceed to clinical tests",
                        "Recommend exam & blood tests, then run Model B."), unsafe_allow_html=True)
                else:
                    st.markdown(rec_banner("info","🔵","Questionnaire negative",
                        "Low questionnaire risk. Run Model B if clinical suspicion exists."),
                        unsafe_allow_html=True)

        # ── SHAP explanations ─────────────────────────────────────────────────
        run_keys = [k for k in ("lite","B","A") if k in results]
        if run_keys:
            st.write("")
            st.markdown('<div style="font-size:1.05rem;font-weight:700;letter-spacing:.3px;color:#1A3C5E;text-transform:uppercase;border-bottom:2px solid #2B7A5F;padding-bottom:8px;margin-bottom:14px;">Model Explanation (SHAP)</div>',
                        unsafe_allow_html=True)
            st.caption("Red bars push the score toward PMOS, teal bars push it away. "
                       "Lite is a Random Forest (probability points); Models A & B are linear SVMs (score units).")
            shap_tabs = st.tabs([M[k]["name"] for k in run_keys])
            for tab, key in zip(shap_tabs, run_keys):
                with tab:
                    try:
                        exp, unit = explain(key, snap)
                        fig = shap_chart(exp, unit)
                        st.pyplot(fig)
                        shap_cache[key] = fig
                        st.session_state[f"shap_{key}"] = fig
                        # Note: do NOT close fig here — kept open for PDF export
                    except Exception as e:
                        st.info(f"Explanation not available: {e}")

        # ── Step 5: Suggestions ───────────────────────────────────────────────
        if results:
            st.write("")
            render_suggestions(results)

        # ── Step 4: PDF Export ────────────────────────────────────────────────
        if results:
            st.write("")
            st.markdown('<div style="font-size:1rem;font-weight:700;letter-spacing:.3px;color:#1A3C5E;text-transform:uppercase;margin-bottom:8px;">Download Report</div>',
                        unsafe_allow_html=True)
            with st.spinner("Preparing PDF…"):
                _pdf_bytes = generate_pdf_report(snap, results, shap_cache)
            if _pdf_bytes:
                fname = f"pmos_screening_{datetime.date.today()}.pdf"
                st.download_button(
                    "📥 Download My Results (PDF)",
                    data=_pdf_bytes, file_name=fname,
                    mime="application/pdf", key="pdf_dl",
                    help="Download a PDF report with your screening results"
                )
            else:
                st.warning("PDF library not available. Run: `pip install reportlab` in the venv.")

        if "truth" in st.session_state:
            t = st.session_state["truth"]
            st.markdown('<span class="chip">Recorded diagnosis for this demo patient: '
                        + ("PMOS" if t == 1 else "No PMOS") + '</span>', unsafe_allow_html=True)

    # Global SHAP figures
    fig_dir = ROOT / "outputs" / "figures"
    bar, summ = fig_dir / "shap_bar_model_B.png", fig_dir / "shap_summary_model_B.png"
    if bar.exists() or summ.exists():
        st.write("")
        with st.expander("Overall feature importance (SHAP, Model B — no ultrasound)"):
            st.caption("Computed on the tuned Random Forest for Model B.")
            c1, c2 = st.columns(2)
            if bar.exists():  c1.image(str(bar))
            if summ.exists(): c2.image(str(summ))

    st.markdown('<div class="footer-note">Prototype for clinical evaluation · '
                'screening aid only, not a medical device — final decisions rest '
                'with the clinician</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SYMPTOM TRACKER TAB — Step 6
# ══════════════════════════════════════════════════════════════════════════════
with tab_tracker:
    render_symptom_tracker()


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHTS TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown('<div style="font-size:1.2rem;font-weight:700;letter-spacing:.3px;color:#1A3C5E;text-transform:uppercase;border-bottom:2px solid #2B7A5F;padding-bottom:8px;margin-bottom:14px;">Project Insights</div>',
                unsafe_allow_html=True)
    st.caption("Summary of model training results, SHAP feature importance, and evaluation figures.")

    fig_dir = ROOT / "outputs" / "figures"
    out_dir = ROOT / "outputs"

    st.subheader("Feature Importance (SHAP)")
    _bar  = fig_dir / "shap_bar_model_B.png"
    _summ = fig_dir / "shap_summary_model_B.png"
    _c1, _c2 = st.columns(2)
    if _bar.exists():  _c1.image(str(_bar),  caption="SHAP Bar — Model B", use_container_width=True)
    if _summ.exists(): _c2.image(str(_summ), caption="SHAP Beeswarm — Model B", use_container_width=True)

    st.subheader("Evaluation Plots")
    _fig_cols = st.columns(3)
    for _idx, (_fname, _cap) in enumerate([
        ("roc_pr_curves.png","ROC & PR Curves"),
        ("roc_lite_vs_full.png","Lite vs Full ROC"),
        ("confusion_matrices.png","Confusion Matrices"),
        ("calibration.png","Calibration Curves"),
        ("cascade_tradeoff.png","Cascade Trade-off"),
        ("threshold_tradeoff.png","Threshold Trade-off"),
    ]):
        _fpath = fig_dir / _fname
        if _fpath.exists():
            _fig_cols[_idx % 3].image(str(_fpath), caption=_cap, use_container_width=True)

    st.subheader("Key Result Tables")
    _tables = {
        "Model Comparison":         out_dir / "final_comparison_computed.csv",
        "Nested CV (Model B)":      out_dir / "nested_cv_model_B.csv",
        "Cascade Analysis":         out_dir / "cascade_analysis.csv",
        "SHAP Importance":          out_dir / "shap_importance_model_B.csv",
        "Lite vs Full (Bootstrap)": out_dir / "lite_vs_full_paired_bootstrap.csv",
    }
    _chosen = st.selectbox("Choose table", list(_tables.keys()))
    _tpath  = _tables[_chosen]
    if _tpath.exists():
        import pandas as _pd
        st.dataframe(_pd.read_csv(_tpath), use_container_width=True)
    else:
        st.info(f"Table not found: {_tpath.name}. Run: python src/run_all.py")
