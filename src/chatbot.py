"""
chatbot.py
====================================================================
PCOS-specific RAG chatbot for the PCOS Screening Aid Streamlit app.

Uses:
  - Groq API  (llama-3.3-70b-versatile — free tier, ultra-fast LPU)
  - Simple in-memory vector store with cosine similarity (no extra deps)
  - Knowledge base built from:
      (a) Your project's own output CSV/MD files   (RAG over YOUR results)
      (b) Curated PCOS clinical knowledge           (medical Q&A)

Usage inside app.py:
    from chatbot import render_chatbot_tab
    with tab:
        render_chatbot_tab()
"""

import os
import json
import re
import warnings
from pathlib import Path
from typing import List, Tuple

# Suppress deprecation noise from older google packages
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent   # project root
OUT_DIR  = ROOT / "outputs"
MODELS   = ROOT / "models"


# ==================================================================
# 1.  KNOWLEDGE BASE
# ==================================================================

# --- (a) Curated PMOS/PCOS clinical knowledge ---
PCOS_KNOWLEDGE = [
    {
        "id": "pmos_def",
        "title": "What is PMOS? (Polycystic Metabolic Ovary Syndrome)",
        "content": """PMOS (Polycystic Metabolic Ovary Syndrome) is the current and preferred medical term 
for the condition previously known as PCOS (Polycystic Ovary Syndrome). 
The name was updated because:
- "Polycystic" was misleading — not all patients have visible cysts on their ovaries
- The old name focused too narrowly on the ovaries, ignoring the metabolic/systemic nature
- PMOS emphasizes the hormonal, metabolic, and reproductive aspects of the disorder

PMOS is one of the most common hormonal disorders affecting women of reproductive age (6-12% globally).
Key features of PMOS:
- Irregular or absent menstrual cycles (oligo/anovulation)
- Elevated androgens (hyperandrogenism) — causing excess hair growth, acne, hair loss
- Polycystic ovaries on ultrasound (≥12 follicles per ovary or ovarian volume >10 mL)
- Insulin resistance and metabolic dysfunction
Diagnosis requires at least 2 of the 3 ovarian/hormonal criteria (Rotterdam criteria, 2003).

Note: This application uses PMOS as the preferred term. When you see PCOS in older documents or 
medical literature, it refers to the same condition now called PMOS."""
    },
    {
        "id": "pcos_def",
        "title": "What is PCOS? (old name for PMOS)",
        "content": """PCOS (Polycystic Ovary Syndrome) is the OLD name for PMOS (Polycystic Metabolic Ovary Syndrome).
The name PCOS has been replaced by PMOS to better reflect the metabolic and systemic nature of the condition.
PCOS/PMOS is one of the most common hormonal disorders affecting women of reproductive age, affecting 6-12% of women globally. It is characterized by:
- Irregular or absent menstrual cycles (oligo/anovulation)
- Elevated androgens (hyperandrogenism) — causing excess hair growth, acne, hair loss
- Polycystic ovaries on ultrasound (≥12 follicles per ovary or ovarian volume >10 mL)
Diagnosis requires at least 2 of these 3 criteria (Rotterdam criteria, 2003)."""
    },
    {
        "id": "rotterdam",
        "title": "Rotterdam Criteria for PCOS Diagnosis",
        "content": """The Rotterdam criteria (2003) define PCOS as the presence of at least 2 of:
1. Oligo- or anovulation (irregular periods, cycle >35 days or <8 per year)
2. Clinical or biochemical signs of hyperandrogenism 
   (excess hair growth, acne, elevated testosterone/DHEAS)
3. Polycystic ovaries on ultrasound (≥12 follicles 2-9mm per ovary, or ovarian volume >10 mL)
Other causes must be excluded: thyroid disease, hyperprolactinemia, congenital adrenal hyperplasia."""
    },
    {
        "id": "symptoms",
        "title": "PCOS Symptoms",
        "content": """Common PCOS symptoms include:
- Irregular periods (oligomenorrhea) or absent periods (amenorrhea)
- Excess facial/body hair (hirsutism) — most common androgen sign
- Acne or oily skin
- Hair thinning/loss on the scalp (androgenic alopecia)
- Weight gain, especially around the abdomen
- Skin darkening (acanthosis nigricans) — often around neck, groin, underarms
- Difficulty getting pregnant (infertility)
- Mood changes, anxiety, depression"""
    },
    {
        "id": "hormones",
        "title": "PCOS Hormone Markers",
        "content": """Key hormonal findings in PCOS:
- LH:FSH ratio elevated (often >2:1 or >3:1) — LH is high, FSH is relatively low
- AMH (Anti-Müllerian Hormone): elevated in PCOS (often >3.4 ng/mL), reflects antral follicle count
- Testosterone: elevated free or total testosterone
- DHEAS: elevated in some patients
- Insulin: elevated (insulin resistance affects ~70% of PCOS patients)
- TSH: checked to rule out thyroid disease
- Prolactin: checked to rule out hyperprolactinemia
Note: AMH is a reliable marker and is increasingly used as an alternative to ultrasound."""
    },
    {
        "id": "treatment",
        "title": "PCOS Treatment Options",
        "content": """PCOS treatment depends on the patient's goals:

For cycle regulation & hyperandrogenism:
- Combined oral contraceptive pills (OCP) — first-line
- Progestin-only therapy

For insulin resistance / metabolic:
- Lifestyle changes (diet, exercise) — most effective long-term
- Metformin — improves insulin sensitivity, restores ovulation
- Weight loss (even 5-10%) can restore cycles in overweight patients

For fertility:
- Letrozole — first-line ovulation induction (better than clomiphene)
- Clomiphene citrate
- Gonadotropins (FSH injections) — second line
- IVF — if other treatments fail

For hirsutism/acne:
- Anti-androgens: spironolactone, finasteride, flutamide (not in pregnancy)
- Topical treatments, laser hair removal"""
    },
    {
        "id": "risk_factors",
        "title": "PCOS Risk Factors and Long-term Risks",
        "content": """Risk factors for PCOS:
- Family history (mother/sister with PCOS)
- Obesity / overweight (worsens symptoms but not required for diagnosis)
- Insulin resistance / type 2 diabetes history in family

Long-term health risks with PCOS:
- Type 2 diabetes (up to 10x higher risk)
- Cardiovascular disease (dyslipidemia, hypertension)
- Endometrial cancer (due to chronic anovulation — unopposed estrogen)
- Sleep apnea
- Anxiety and depression
- Non-alcoholic fatty liver disease (NAFLD)
Regular monitoring is essential."""
    },
    {
        "id": "bmi_waist",
        "title": "BMI and Waist:Hip Ratio in PCOS",
        "content": """BMI and body measurements are important in PCOS:
- BMI ≥23 kg/m² is the Asian overweight cutoff (used in this model's BMI_over_23 feature)
- Normal Western BMI cutoff: 25 kg/m²
- Waist:Hip Ratio >0.85 in women indicates central obesity
- Central obesity is strongly associated with insulin resistance in PCOS
- Weight loss of 5-10% of body weight in overweight women can restore ovulation
- Even normal-weight women can have PCOS (lean PCOS — ~20-30% of cases)"""
    },
    {
        "id": "screening_cascade",
        "title": "Stepwise PCOS Screening — How This App Works",
        "content": """This app implements a 3-tier clinical screening cascade:

Step 1 — Lite Model (questionnaire only, no tests):
  Uses 10 features: age, menstrual cycle, 5 symptom flags, 2 lifestyle factors.
  No blood tests or physical exam required. Catches ~97% of PCOS cases.
  Purpose: First triage — who needs further investigation?

Step 2 — Model B (exam + blood tests, no ultrasound):
  Uses 29 features including BMI, hormones (FSH, LH, AMH, TSH, etc.), vitals.
  No ultrasound required. Catches ~94% of PCOS cases.
  Purpose: Should we order an ultrasound?

Step 3 — Model A (full model with ultrasound):
  Uses 34 features including follicle counts, average follicle size, endometrium.
  Highest specificity (89%). Catches ~92% of PCOS cases.
  Purpose: Confirmation before clinical diagnosis.

All thresholds are set to catch ≥90% of PCOS cases (high recall).
A positive screen means "refer for the next step", not "has PCOS".
Final diagnosis always requires a clinician."""
    },
    {
        "id": "follicle",
        "title": "Ultrasound Findings in PCOS",
        "content": """Ultrasound criteria for polycystic ovaries (Rotterdam 2003):
- ≥12 follicles measuring 2-9mm in diameter in either ovary, AND/OR
- Ovarian volume >10 mL (calculated as 0.5 × length × width × depth)
- The "string of pearls" appearance — small follicles arranged around the periphery

Updated ESHRE/ASRM 2018 threshold: ≥20 follicles (with modern high-frequency transducers)

In this app:
- Follicle No. (L): number of follicles in left ovary
- Follicle No. (R): number of follicles in right ovary
- Avg. F size (L/R): average follicle size in mm
- Endometrium (mm): endometrial thickness (thickened in anovulatory cycles)"""
    },
    {
        "id": "disclaimer",
        "title": "Important Medical Disclaimer",
        "content": """This PCOS Screening Aid is a research/educational tool only.
- It is NOT a medical device or diagnostic tool
- It is NOT a substitute for clinical evaluation by a qualified healthcare provider
- A positive screen result does NOT mean you have PCOS
- A negative screen result does NOT guarantee you don't have PCOS
- All clinical decisions must be made by a licensed physician

If you have concerns about PCOS, please consult a gynecologist or endocrinologist.
The models were trained on a dataset of 541 patients — results may not generalize 
to all populations."""
    },
    {
        "id": "cycle_coding",
        "title": "Menstrual Cycle Coding in This Model",
        "content": """In this project's dataset, the Cycle(R/I) feature is numerically encoded:
- Code 2 = Regular cycle (normal, 21-35 days)
- Code 4 = Irregular cycle (>35 days or <21 days, unpredictable)
- Code 5 = Rare variant (treated as irregular)

In the Streamlit app:
- "Regular" → encoded as 2.0
- "Irregular" → encoded as 4.0

An irregular cycle is one of the strongest predictors of PCOS in all three models."""
    },
]


# --- (b) Load project outputs as additional RAG documents ---
def _load_project_docs() -> List[dict]:
    """Load your project's own CSV/MD outputs as RAG documents."""
    docs = []

    # Threshold tables
    for name, path in [
        ("Lite Model Threshold Table", OUT_DIR / "threshold_table_lite.csv"),
        ("Model A Threshold Table",    OUT_DIR / "threshold_table_model_A.csv"),
        ("Model B Threshold Table",    OUT_DIR / "threshold_table_model_B.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            docs.append({
                "id": f"threshold_{path.stem}",
                "title": name,
                "content": f"{name}:\n{df.to_string(index=False, float_format=lambda x: f'{x:.4f}')}"
            })

    # Operating points
    for name, path in [
        ("Lite Model Operating Points (85%/90%/95% recall)",     OUT_DIR / "operating_points_lite.csv"),
        ("Model A Operating Points (85%/90%/95% recall)",        OUT_DIR / "operating_points_model_A.csv"),
        ("Model B Operating Points (85%/90%/95% recall)",        OUT_DIR / "operating_points_model_B.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            docs.append({
                "id": f"op_{path.stem}",
                "title": name,
                "content": f"{name}:\n{df.to_string(index=False, float_format=lambda x: f'{x:.4f}')}"
            })

    # Paper operating points markdown
    md_path = OUT_DIR / "lite_paper_operating_points.md"
    if md_path.exists():
        docs.append({
            "id": "lite_paper_table",
            "title": "Lite Model Paper Table (85% and 90% recall operating points)",
            "content": md_path.read_text()
        })

    # Final model comparison
    cmp_path = OUT_DIR / "final_comparison_computed.csv"
    if cmp_path.exists():
        df = pd.read_csv(cmp_path)
        docs.append({
            "id": "final_comparison",
            "title": "Final Model Comparison (LR / RF / SVM at default threshold 0.50)",
            "content": f"Final model comparison:\n{df.to_string(index=False, float_format=lambda x: f'{x:.4f}')}"
        })

    # SHAP importance
    shap_path = OUT_DIR / "shap_importance_model_B.csv"
    if shap_path.exists():
        df = pd.read_csv(shap_path)
        docs.append({
            "id": "shap_importance",
            "title": "SHAP Feature Importance for Model B (Random Forest)",
            "content": f"Top features by mean |SHAP| value for Model B:\n{df.head(20).to_string(index=False)}"
        })

    # SVM coefficients
    coef_path = OUT_DIR / "svm_coefficients_model_B.csv"
    if coef_path.exists():
        df = pd.read_csv(coef_path)
        docs.append({
            "id": "svm_coef",
            "title": "Model B Linear SVM Standardised Coefficients",
            "content": f"Model B (deployed linear SVM) — top feature coefficients:\n{df.head(20).to_string(index=False)}"
        })

    # Ablation results
    abl_path = OUT_DIR / "ablation_cost_tiers.csv"
    if abl_path.exists():
        df = pd.read_csv(abl_path)
        cols = ["Configuration", "n_features", "recall_mean", "recall_std",
                "roc_auc_mean", "roc_auc_std"]
        cols = [c for c in cols if c in df.columns]
        docs.append({
            "id": "ablation_tiers",
            "title": "Feature Cost Tiers Ablation Study (Model B)",
            "content": f"Cost-tier ablation (cheapest to most expensive feature sets):\n{df[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}')}"
        })

    # Model metadata
    meta_path = MODELS / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for ab in ("model_A", "model_B"):
            m = meta.get(ab, {})
            docs.append({
                "id": f"meta_{ab}",
                "title": f"{ab.replace('_', ' ').title()} — Trained Model Metadata",
                "content": (
                    f"{ab.replace('_', ' ').title()} metadata:\n"
                    f"Classifier: {m.get('classifier','')}\n"
                    f"Params: {m.get('params','')}\n"
                    f"Threshold: {m.get('threshold','')}\n"
                    f"Threshold rule: {m.get('threshold_rule','')}\n"
                    f"Features ({len(m.get('features',[]))}): {m.get('features','')}\n"
                    f"Test metrics:\n{json.dumps(m.get('test_metrics_at_threshold',{}), indent=2)}"
                )
            })

    # Lite model metadata
    lite_path = MODELS / "model_B_lite_metadata.json"
    if lite_path.exists():
        with open(lite_path) as f:
            lite = json.load(f)
        docs.append({
            "id": "meta_lite",
            "title": "Lite Model — Trained Model Metadata",
            "content": (
                f"Lite Model metadata:\n"
                f"Classifier: {lite.get('classifier','')}\n"
                f"Best params: {lite.get('best_params','')}\n"
                f"Threshold: {lite.get('threshold','')}\n"
                f"Features ({len(lite.get('features',[]))}): {lite.get('features','')}\n"
                f"Test metrics:\n{json.dumps(lite.get('test_metrics_at_threshold',{}), indent=2)}"
            )
        })

    return docs


# ==================================================================
# 2.  SIMPLE VECTOR STORE (TF-IDF cosine, no external deps)
# ==================================================================
class SimpleVectorStore:
    """
    Lightweight TF-IDF vector store for semantic search.
    No faiss, no sentence-transformers — just numpy + stdlib.
    """

    def __init__(self):
        self.docs: List[dict] = []
        self.vocab: dict      = {}
        self.idf: np.ndarray  = np.array([])
        self.tfidf: np.ndarray = np.array([])

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _tf(self, tokens: List[str]) -> dict:
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = max(len(tokens), 1)
        return {k: v / n for k, v in tf.items()}

    def build(self, docs: List[dict]):
        self.docs = docs
        # Build vocab
        all_tokens = set()
        for d in docs:
            all_tokens.update(self._tokenize(d["title"] + " " + d["content"]))
        self.vocab = {t: i for i, t in enumerate(sorted(all_tokens))}
        V = len(self.vocab)
        N = len(docs)

        # TF matrix
        tf_mat = np.zeros((N, V), dtype=np.float32)
        for i, d in enumerate(docs):
            tokens = self._tokenize(d["title"] + " " + d["content"])
            tf = self._tf(tokens)
            for word, val in tf.items():
                if word in self.vocab:
                    tf_mat[i, self.vocab[word]] = val

        # IDF
        df = (tf_mat > 0).sum(axis=0) + 1
        self.idf = np.log((N + 1) / df).astype(np.float32)

        # TF-IDF (L2-normalised)
        self.tfidf = tf_mat * self.idf
        norms = np.linalg.norm(self.tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.tfidf = self.tfidf / norms

    def search(self, query: str, top_k: int = 4) -> List[Tuple[dict, float]]:
        if not self.docs:
            return []
        tokens = self._tokenize(query)
        tf = self._tf(tokens)
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        for word, val in tf.items():
            if word in self.vocab:
                q_vec[self.vocab[word]] = val * self.idf[self.vocab[word]]
        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return []
        q_vec /= norm
        scores = self.tfidf @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], float(scores[i])) for i in top_idx if scores[i] > 0.01]


# ==================================================================
# 3.  KNOWLEDGE BASE INITIALIZATION (cached per session)
# ==================================================================
@st.cache_resource
def build_knowledge_base() -> SimpleVectorStore:
    """Build and cache the RAG vector store."""
    store = SimpleVectorStore()
    docs  = list(PCOS_KNOWLEDGE) + _load_project_docs()
    store.build(docs)
    return store


# ==================================================================
# 4.  GROQ API HELPER  (ultra-fast, free tier, Llama 3 models)
# ==================================================================
_GROQ_MODELS = [
    "qwen/qwen3.8-27b",      # best quality, fast
    "openai/gpt-oss-20b",    # fallback
    "allam-2-7b",            # last resort
]

def _get_groq_client(api_key: str):
    """Return a configured Groq client, or None if unavailable."""
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        # groq package not installed — we'll use requests directly
        return {"_type": "requests", "api_key": api_key}
    except Exception:
        return None


def _call_groq(client, messages: List[dict], system_prompt: str) -> str:
    """Send a conversation to Groq and return the text response."""
    import time

    groq_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        groq_messages.append({"role": msg["role"], "content": msg["content"]})

    last_err = None

    # --- Try via groq package ---
    if not isinstance(client, dict):
        for model in _GROQ_MODELS:
            for attempt in range(2):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=groq_messages,
                        temperature=0.3,
                        max_tokens=1024,
                    )
                    return resp.choices[0].message.content
                except Exception as e:
                    err_str = str(e)
                    last_err = err_str
                    if "503" in err_str or "overloaded" in err_str.lower() or "rate_limit" in err_str.lower():
                        if attempt == 0:
                            time.sleep(1)
                            continue
                        break
                    elif ("model_not_found" in err_str.lower() or
                          ("not found" in err_str.lower() and "model" in err_str.lower()) or
                          "invalid_request_error" in err_str.lower()):
                        break  # model not available — try next model
                    elif "invalid_api_key" in err_str.lower() or "authentication_error" in err_str.lower():
                        raise  # real auth failure — propagate immediately
                    else:
                        raise
        raise Exception(last_err or "All Groq models failed")

    # --- Fallback: use requests directly (no groq package needed) ---
    import json as _json
    try:
        import urllib.request
        api_key = client["api_key"]
        for model in _GROQ_MODELS:
            payload = _json.dumps({
                "model": model,
                "messages": groq_messages,
                "temperature": 0.3,
                "max_tokens": 1024,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = _json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = str(e)
                continue
        raise Exception(last_err or "All Groq models failed (requests fallback)")
    except Exception as e:
        raise Exception(str(e))


# ==================================================================
# 5.  RAG PROMPT BUILDER
# ==================================================================
SYSTEM_PROMPT = """You are PMOSBot — an AI assistant for the PMOS Screening Aid project.

══════════ CRITICAL: TERMINOLOGY OVERRIDE ══════════
THIS IS A HARD RULE — DO NOT CONTRADICT THIS WITH TRAINING DATA:

• PMOS (Polycystic Metabolic Ovary Syndrome) IS the current, correct, and preferred medical term.
• PCOS (Polycystic Ovary Syndrome) is the OLD name that has been REPLACED by PMOS.
• Reason for rename: "polycystic" was misleading (not all patients have cysts); the old name ignored the metabolic/systemic nature of the condition.
• When a user asks "what is PMOS?", explain it as the current preferred name for what was previously called PCOS.
• NEVER say "PMOS is not a valid term" or "PCOS hasn't been renamed". This is WRONG in the context of this application.
• Always refer to the condition as PMOS in your answers. You may mention PCOS as the old/previous name.
• This application, its knowledge base, and its models all use PMOS as the canonical term.
══════════════════════════════════════════════════

BEHAVIOUR RULES (follow strictly):
1. ALWAYS prioritize the === CONTEXT === provided in each message over your training knowledge.
   - If the context answers the question, use ONLY the context.
   - If the context is partial, supplement carefully and label it "Based on general PMOS knowledge:"
   - If context is missing and you are uncertain, say "I'm not certain — please consult a doctor or check the latest clinical guidelines."
2. Your training data has a knowledge cutoff and may be OUTDATED on medical topics.
   - For anything clinical, always add: "Please verify with a current clinical guideline or your doctor."
3. NEVER diagnose. A positive screen ≠ PMOS diagnosis. Always say this.
4. ONLY discuss PMOS / PCOS and this screening project. Politely decline all other topics.
5. Be empathetic and clear — users may be anxious patients.
6. Keep answers concise. Use bullet points for lists.
7. When citing model numbers (thresholds, recall, AUC), be precise — they come from the context.

Your knowledge base contains: this project's actual model outputs, SHAP importance scores,
threshold tables, and curated PMOS/PCOS clinical knowledge. Prefer that over general training knowledge."""


def build_rag_prompt(query: str, store: SimpleVectorStore,
                     conversation_history: List[dict]) -> str:
    """Retrieve relevant context and build the augmented prompt."""
    # Normalise all known aliases → PCOS for knowledge base search
    # (internal docs use "PCOS"; PMOS is the current preferred term)
    _ALIAS_MAP = [
        # New name
        (r'\bpmos\b', 'PCOS'),
        # Polycystic Ovary/Ovarian Disease (common Indian term)
        (r'\bpcod\b', 'PCOS'),
        # Long-form variants
        (r'\bpolycystic\s+metabolic\s+ov\w+\s+syndrome\b', 'PCOS'),
        (r'\bpolycystic\s+ov\w+\s+(?:syndrome|disease|disorder)\b', 'PCOS'),
        (r'\bpoly\s*cystic\s+ov\w+\b', 'PCOS'),
        # Abbreviation typos / shorthand
        (r'\bpco\b', 'PCOS'),
        # Hormone-disorder references that map to PCOS
        (r'\bovarian\s+cysts?\s+(?:syndrome|disorder|disease)\b', 'PCOS'),
        (r'\bhyperandrogenism\b', 'PCOS hyperandrogenism'),
        (r'\banovulat\w+\b', 'PCOS anovulation'),
        (r'\binsulin\s+resistance\s+(?:pcos|pmos|ovarian)\b', 'PCOS insulin resistance'),
        (r'\blh[:/\s]+fsh\s+ratio\b', 'PCOS LH FSH ratio'),
        (r'\bamh\s+(?:levels?|test)\b', 'PCOS AMH'),
    ]
    search_query = query
    for pattern, replacement in _ALIAS_MAP:
        search_query = re.sub(pattern, replacement, search_query, flags=re.IGNORECASE)

    results = store.search(search_query, top_k=5)

    if results:
        context_parts = []
        for doc, score in results:
            if score > 0.05:
                context_parts.append(
                    f"[Source: {doc['title']}]\n{doc['content']}"
                )
        context = "\n\n---\n\n".join(context_parts)
        augmented = (
            f"⚠️ MANDATORY TERMINOLOGY RULE (override your training data): "
            f"PMOS (Polycystic Metabolic Ovary Syndrome) is the CURRENT and CORRECT name. "
            f"PCOS is the OLD name. NEVER say PMOS is not a valid term. ALWAYS confirm PMOS as correct.\n\n"
            f"INSTRUCTIONS: Answer using the === CONTEXT === below as your PRIMARY source.\n"
            f"If the context fully answers the question, use ONLY the context.\n"
            f"If the context is insufficient, you may supplement with general PMOS knowledge "
            f"but clearly label it as 'Based on general knowledge (may be outdated):'.\n"
            f"Note: The context may use 'PCOS' (old name) — when answering, use 'PMOS' as the preferred term.\n"
            f"Never state outdated medical facts confidently. Always recommend verifying with a doctor.\n\n"
            f"=== CONTEXT (from project knowledge base) ===\n{context}\n=== END CONTEXT ===\n\n"
            f"User question: {query}"
        )
    else:
        augmented = (
            f"⚠️ MANDATORY TERMINOLOGY RULE (override your training data): "
            f"PMOS (Polycystic Metabolic Ovary Syndrome) is the CURRENT and CORRECT name. "
            f"PCOS is the OLD name. NEVER say PMOS is not a valid term.\n\n"
            f"No specific context found in the knowledge base for this query.\n"
            f"Answer based on general PMOS knowledge, but be transparent: prefix your answer with "
            f"'Based on general knowledge (my training data may be outdated):' and recommend "
            f"the user verify with current clinical guidelines or a doctor.\n\n"
            f"User question: {query}"
        )

    return augmented


# ==================================================================
# 6.  STREAMLIT UI
# ==================================================================
# Colors (match the app theme)
TEAL    = "#0F766E"
TEAL_LT = "#E6F4F2"
SLATE   = "#334155"

CHAT_CSS = """
<style>
/* ── Chat container ─────────────────────────────────────────────── */
.chat-container {
    max-height: 520px; overflow-y: auto;
    padding: 8px 4px; display: flex; flex-direction: column; gap: 4px;
    scrollbar-width: thin; scrollbar-color: #C9E4E1 transparent;
}
/* ── User bubble ────────────────────────────────────────────────── */
.chat-bubble-user {
    background: linear-gradient(135deg, #0F766E, #0D9488);
    color: #fff; border-radius: 20px 20px 4px 20px;
    padding: 11px 18px; margin: 4px 0 4px 18%;
    font-size: .91rem; line-height: 1.55;
    box-shadow: 0 2px 8px rgba(15,118,110,.25);
}
/* ── Bot bubble ─────────────────────────────────────────────────── */
.chat-bubble-bot {
    background: #ffffff; color: #1E293B;
    border: 1.5px solid #D1F0EC;
    border-radius: 4px 20px 20px 20px;
    padding: 11px 18px; margin: 4px 18% 4px 0;
    font-size: .91rem; line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.chat-bubble-bot b { color: #0F766E; }
.chat-bubble-bot ul, .chat-bubble-bot ol { margin: 6px 0 6px 18px; }
/* ── Labels ─────────────────────────────────────────────────────── */
.chat-label-user {
    text-align: right; font-size: .72rem;
    color: #94A3B8; margin-bottom: 2px; padding-right: 4px;
}
.chat-label-bot {
    font-size: .72rem; color: #94A3B8;
    margin-bottom: 2px; padding-left: 2px;
    display: flex; align-items: center; gap: 4px;
}
/* ── API info banner ────────────────────────────────────────────── */
.api-info {
    background: #FEF3C7; border: 1px solid #FCD34D;
    border-radius: 10px; padding: 12px 16px;
    font-size: .84rem; color: #78350F; margin-bottom: 12px;
}
</style>
"""


def _render_message(role: str, content: str, sources: List[str] = None):
    """Render a single chat bubble."""
    if role == "user":
        st.markdown(
            f'<div class="chat-label-user">You</div>'
            f'<div class="chat-bubble-user">{content}</div>',
            unsafe_allow_html=True
        )
    else:
        source_html = ""
        # Convert markdown-style **bold** for display
        content_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
        content_html = content_html.replace("\n", "<br>")
        st.markdown(
            f'<div class="chat-label-bot">🤖 PCOSBot</div>'
            f'<div class="chat-bubble-bot">{content_html}{source_html}</div>',
            unsafe_allow_html=True
        )


def render_chatbot_tab():
    """Main entry point — renders the chatbot tab."""
    try:
        _render_chatbot_inner()
    except Exception as exc:
        st.error(f"PCOSBot encountered an error: {exc}")
        st.exception(exc)


def _render_chatbot_inner():
    """Inner implementation — all rendering logic lives here."""
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    # --- API key setup ---
    st.markdown("### 🤖 PMOSBot — PMOS / PCOS Screening AI Assistant")
    st.caption(
        "Ask anything about PMOS symptoms, hormones, diagnosis, treatment, "
        "or how this screening model works. Powered by Groq + RAG."
    )

    # Check for API key in secrets or environment first
    api_key = (
        st.secrets.get("GROQ_API_KEY", "")
        if hasattr(st, "secrets") else ""
    ) or os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        st.markdown(
            '<div class="api-info">'
            '🔑 <b>Enter your free Groq API key to activate the chatbot.</b><br>'
            'Free key at <a href="https://console.groq.com/keys" '
            'target="_blank">console.groq.com/keys</a> — no payment required, generous free tier.'
            '</div>',
            unsafe_allow_html=True
        )
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            key="groq_api_key_input",
            help="Your key is used only in this session and never stored."
        )
        if api_key:
            st.session_state["groq_api_key"] = api_key
    else:
        st.session_state["groq_api_key"] = api_key

    # Use stored key from session
    api_key = st.session_state.get("groq_api_key", "")

    if not api_key:
        # Show sample questions even without key
        st.markdown("---")
        st.markdown("**💡 Sample questions you can ask:**")
        samples = [
            "What are the symptoms of PCOS?",
            "What threshold does the Lite model use and why?",
            "Which features are most important for predicting PCOS?",
            "What does a positive screen result mean?",
            "How is AMH related to PCOS?",
            "What is the sensitivity of Model A at 90% recall?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(samples):
            cols[i % 2].markdown(f"- _{q}_")
        return

    # --- Initialize session state ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_sources" not in st.session_state:
        st.session_state.chat_sources = {}

    # --- Load RAG store (already pre-warmed by app.py, instant from cache) ---
    store = build_knowledge_base()

    # --- Groq client ---
    client = _get_groq_client(api_key)
    if client is None:
        st.error(
            "Could not load Google Generative AI library. "
            "Please run: `pip install groq`"
        )
        return

    # --- Chat history display ---
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, msg in enumerate(st.session_state.chat_history):
            sources = st.session_state.chat_sources.get(i, [])
            _render_message(msg["role"], msg["content"], sources)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    # --- Input form ---
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        user_input = col1.text_input(
            "Ask PCOSBot",
            placeholder="Ask me anything about PCOS...",
            label_visibility="collapsed",
            key="chat_input",
        )
        submitted = col2.form_submit_button("Send ➤", use_container_width=True)

    # Handle suggestion buttons – pop safely
    pending = st.session_state.pop("_pending_question", None)
    if pending:
        user_input = pending
        submitted  = True

    if submitted and user_input.strip():
        query = user_input.strip()

        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        msg_idx = len(st.session_state.chat_history) - 1

        # RAG retrieval
        results = store.search(query, top_k=4)
        source_names = [doc["title"] for doc, score in results if score > 0.05]

        # Build augmented prompt
        augmented_query = build_rag_prompt(
            query, store, st.session_state.chat_history[:-1]
        )

        # Add to history for context
        history_with_augmented = (
            st.session_state.chat_history[:-1]
            + [{"role": "user", "content": augmented_query}]
        )

        # Call Groq
        with st.spinner("PCOSBot is thinking..."):
            try:
                response_text = _call_groq(client, history_with_augmented, SYSTEM_PROMPT)
            except Exception as e:
                err = str(e)
                if "invalid_api_key" in err.lower() or "authentication_error" in err.lower():
                    response_text = (
                        "⚠️ Invalid API key. Please check your Groq API key and try again. "
                        "Free key at https://console.groq.com/keys"
                    )
                elif "quota" in err.lower() or "429" in err or "resource_exhausted" in err.lower():
                    response_text = (
                        "⚠️ Rate limit reached. Please wait a moment and try again. "
                        "The free tier allows 15 requests per minute."
                    )
                elif "model_not_found" in err.lower() or "invalid_request_error" in err.lower():
                    response_text = "⚠️ No Groq model responded. Please try again later."
                else:
                    response_text = f"⚠️ Error calling Groq: {err}"

        # Store response
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_text}
        )
        # Store sources for the bot message index
        bot_idx = len(st.session_state.chat_history) - 1
        st.session_state.chat_sources[bot_idx] = source_names

        st.rerun()

    # --- Clear chat ---
    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation", use_container_width=False):
            st.session_state.chat_history = []
            st.session_state.chat_sources = {}
            st.rerun()

    # --- Disclaimer ---
    st.markdown(
        '<div style="color:#94A3B8;font-size:.75rem;margin-top:16px;text-align:center">'
        '⚕️ PCOSBot provides general information only. '
        'Not a substitute for medical advice. Always consult a qualified clinician.'
        '</div>',
        unsafe_allow_html=True
    )

# ============================================================
# SIDEBAR CHATBOT  (compact, always-visible version)
# ============================================================

SB_CHAT_CSS = """<style>
.sb-chat { max-height: 340px; overflow-y: auto; padding: 2px 0; margin-bottom: 6px; }
.sb-chat-label-u { font-size:.68rem; color:#94A3B8; text-align:right; margin-bottom:1px; }
.sb-chat-label-b { font-size:.68rem; color:#94A3B8; margin-bottom:1px; }
.sb-chat-user {
    background:#0F766E; color:#fff;
    border-radius:10px 10px 3px 10px;
    padding:7px 10px; margin:2px 0;
    font-size:.8rem; line-height:1.45; word-break:break-word;
}
.sb-chat-bot {
    background:#F0F9F8; color:#0F172A;
    border:1px solid #C9E4E1;
    border-radius:10px 10px 10px 3px;
    padding:7px 10px; margin:2px 0;
    font-size:.8rem; line-height:1.45; word-break:break-word;
}
.sb-no-key { background:#FEF3C7; border:1px solid #FCD34D;
    border-radius:8px; padding:8px 10px; font-size:.78rem; color:#78350F; }
</style>"""


def render_chatbot_sidebar():
    """
    Compact chatbot rendered inside the Streamlit sidebar.
    All Groq / RAG logic is shared with the full version.
    """
    import re as _re
    st.markdown(SB_CHAT_CSS, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 1. API key resolution
    # ------------------------------------------------------------------
    api_key = (
        st.secrets.get("GROQ_API_KEY", "")
        if hasattr(st, "secrets") else ""
    ) or os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        api_key = st.session_state.get("sb_groq_key_stored", "")

    if not api_key:
        st.markdown(
            '<div class="sb-no-key">🔑 Add <b>GROQ_API_KEY</b> to '
            '<code>.streamlit/secrets.toml</code> to activate PCOSBot.</div>',
            unsafe_allow_html=True,
        )
        typed = st.text_input(
            "Or paste key here",
            type="password",
            placeholder="gsk_...",
            key="sb_groq_key_input",
            label_visibility="visible",
        )
        if typed:
            st.session_state["sb_groq_key_stored"] = typed
            st.rerun()
        return

    st.session_state["sb_groq_key_stored"] = api_key

    # ------------------------------------------------------------------
    # 2. Session state init
    # ------------------------------------------------------------------
    if "sb_chat_history" not in st.session_state:
        st.session_state.sb_chat_history = []

    # ------------------------------------------------------------------
    # 3. RAG store + Groq client (both cached)
    # ------------------------------------------------------------------
    store  = build_knowledge_base()
    client = _get_groq_client(api_key)
    if client is None:
        st.error("Install `groq`: pip install groq")
        return

    # ------------------------------------------------------------------
    # 4. Chat history display
    # ------------------------------------------------------------------
    if st.session_state.sb_chat_history:
        parts = ['<div class="sb-chat">']
        for msg in st.session_state.sb_chat_history:
            txt = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", msg["content"])
            txt = txt.replace("\n", "<br>")
            if msg["role"] == "user":
                parts.append('<div class="sb-chat-label-u">You</div>')
                parts.append(f'<div class="sb-chat-user">{txt}</div>')
            else:
                parts.append('<div class="sb-chat-label-b">🤖 PCOSBot</div>')
                parts.append(f'<div class="sb-chat-bot">{txt}</div>')
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)
    else:
        pass  # No chat history yet — nothing to display

    # ------------------------------------------------------------------
    # 5. Input form
    # ------------------------------------------------------------------
    with st.form("sb_chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Message",
            placeholder="Ask about PCOS...",
            label_visibility="collapsed",
            key="sb_chat_input",
        )
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

    # Consume any pending suggestion
    pending = st.session_state.pop("sb_pending_q", None)
    if pending:
        user_input = pending
        submitted  = True

    # ------------------------------------------------------------------
    # 6. Handle submission
    # ------------------------------------------------------------------
    if submitted and user_input and user_input.strip():
        query = user_input.strip()
        st.session_state.sb_chat_history.append({"role": "user", "content": query})

        augmented = build_rag_prompt(
            query, store, st.session_state.sb_chat_history[:-1]
        )
        history_for_groq = (
            st.session_state.sb_chat_history[:-1]
            + [{"role": "user", "content": augmented}]
        )

        with st.spinner("Thinking..."):
            try:
                reply = _call_groq(client, history_for_groq, SYSTEM_PROMPT)
            except Exception as exc:
                err = str(exc)
                if "invalid_api_key" in err.lower() or "authentication_error" in err.lower():
                    reply = "⚠️ Invalid API key — check your Groq key."
                elif "quota" in err.lower() or "429" in err or "exhausted" in err.lower():
                    reply = "⚠️ Rate limit reached — wait a moment and retry."
                elif "model_not_found" in err.lower() or "invalid_request_error" in err.lower():
                    reply = "⚠️ No Groq model responded. Please try again."
                else:
                    reply = f"⚠️ Error: {err}"

        st.session_state.sb_chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # ------------------------------------------------------------------
    # 7. Clear button + disclaimer
    # ------------------------------------------------------------------
    if st.session_state.sb_chat_history:
        if st.button("🗑️ Clear chat", use_container_width=True, key="sb_clear"):
            st.session_state.sb_chat_history = []
            st.rerun()

    st.markdown(
        '<div style="color:#94A3B8;font-size:.68rem;text-align:center;margin-top:6px">'
        "⚕️ Not medical advice · consult a clinician</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# FULL-PAGE CHATBOT  (separate view, proper back navigation)
# ============================================================

PAGE_CHAT_CSS = """<style>
/* Back button */
.back-btn-row { margin-bottom: 12px; }

/* Page chat header */
.chat-page-header {
    background: linear-gradient(120deg, #0F766E 0%, #0D9488 60%, #14B8A6 100%);
    border-radius: 18px; padding: 24px 30px; color: #fff;
    margin-bottom: 22px;
    box-shadow: 0 4px 18px rgba(15,118,110,.28);
    display: flex; align-items: center; gap: 16px;
}
.chat-page-header h2 {
    margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -.3px;
}
.chat-page-header p {
    margin: 5px 0 0 0; opacity: .9; font-size: .93rem; line-height: 1.4;
}

/* Chat scroll area */
.cp-chat {
    max-height: 520px; overflow-y: auto;
    padding: 6px 0 14px 0;
    scrollbar-width: thin; scrollbar-color: #C9E4E1 transparent;
    display: flex; flex-direction: column; gap: 2px;
}
.cp-label-u {
    text-align: right; font-size: .71rem;
    color: #94A3B8; margin-bottom: 2px; padding-right: 4px;
}
.cp-label-b {
    font-size: .71rem; color: #94A3B8; margin-bottom: 2px; padding-left: 2px;
}
/* User bubble */
.cp-user {
    background: linear-gradient(135deg, #0F766E, #0D9488);
    color: #fff; border-radius: 20px 20px 4px 20px;
    padding: 12px 18px; margin: 3px 0 3px 15%;
    font-size: .9rem; line-height: 1.55; word-break: break-word;
    box-shadow: 0 2px 8px rgba(15,118,110,.22);
}
/* Bot bubble */
.cp-bot {
    background: #ffffff; color: #1E293B;
    border: 1.5px solid #D1F0EC;
    border-radius: 4px 20px 20px 20px;
    padding: 12px 18px; margin: 3px 15% 3px 0;
    font-size: .9rem; line-height: 1.6; word-break: break-word;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.cp-bot b { color: #0F766E; }
.cp-bot ul, .cp-bot ol { margin: 6px 0 6px 18px; }

/* Welcome card */
.cp-welcome {
    background: linear-gradient(135deg, #F0FDF9, #E6F4F2);
    border: 1.5px solid #C9E4E1;
    border-radius: 16px; padding: 24px 28px;
    color: #334155; font-size: .93rem; line-height: 1.7;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(15,118,110,.08);
}
</style>"""


def render_chatbot_page():
    """
    Full-page chatbot view.
    Navigated to via session_state['view'] = 'chatbot'.
    Has a ← Back button to return to the dashboard.
    """
    import re as _re

    st.markdown(PAGE_CHAT_CSS, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Back button
    # ------------------------------------------------------------------
    if st.button("← Back to Dashboard", key="cp_back_btn"):
        st.session_state["view"] = "dashboard"
        st.rerun()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="chat-page-header">'
        '<div>'
        '<h2>🤖 PMOSBot</h2>'
        '<p>Ask anything about PMOS / PCOS — symptoms, hormones, diagnosis, treatment, '
        'or how the screening models in this app work.</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # API key
    # ------------------------------------------------------------------
    api_key = (
        st.secrets.get("GROQ_API_KEY", "")
        if hasattr(st, "secrets") else ""
    ) or os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        api_key = st.session_state.get("groq_api_key_stored", "")

    if not api_key:
        st.warning(
            "🔑 **Groq API key not found.** "
            "Add `GROQ_API_KEY` to `.streamlit/secrets.toml` — "
            "or paste it below (free key from [aistudio.google.com](https://console.groq.com/keys))."
        )
        typed = st.text_input(
            "Groq API key",
            type="password",
            placeholder="gsk_...",
            key="groq_cp_key_input",
            label_visibility="visible",
        )
        if typed:
            st.session_state["groq_api_key_stored"] = typed
            st.rerun()
        return

    st.session_state["groq_api_key_stored"] = api_key

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------
    if "cp_chat_history" not in st.session_state:
        st.session_state.cp_chat_history = []
    if "cp_chat_sources" not in st.session_state:
        st.session_state.cp_chat_sources = {}

    # ------------------------------------------------------------------
    # RAG + Groq client
    # ------------------------------------------------------------------
    store  = build_knowledge_base()
    client = _get_groq_client(api_key)
    if client is None:
        st.error("❌ `groq` not installed. Run: `pip install groq`")
        return

    # ------------------------------------------------------------------
    # Chat history display  OR  welcome message
    # ------------------------------------------------------------------
    if st.session_state.cp_chat_history:
        parts = ['<div class="cp-chat">']
        for i, msg in enumerate(st.session_state.cp_chat_history):
            txt = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", msg["content"])
            txt = txt.replace("\n", "<br>")
            if msg["role"] == "user":
                parts.append('<div class="cp-label-u">You</div>')
                parts.append(f'<div class="cp-user">{txt}</div>')
            else:
                parts.append('<div class="cp-label-b">🤖 PCOSBot</div>')
                parts.append(f'<div class="cp-bot">{txt}</div>')
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)
        st.markdown("---")
    else:
        st.markdown(
            '<div class="cp-welcome">'
            "👋 <b>Hi! I'm PMOSBot.</b><br><br>"
            "I can help you understand:<br>"
            "• <b>PMOS / PCOS</b> symptoms, hormones, and diagnosis<br>"
            "• Treatment options and lifestyle changes<br>"
            "• How this screening app's models work (Lite, Model B, Model A)<br>"
            "• What your screening result means<br>"
            "• Related terms: PCOD, hyperandrogenism, anovulation, AMH, LH/FSH<br><br>"
            "Type your question below and press <b>Send ➤</b>."
            "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # Input form  (clean — no pre-suggested questions)
    # ------------------------------------------------------------------
    with st.form("cp_chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question",
            placeholder="Ask me anything about PCOS...",
            label_visibility="collapsed",
            key="cp_chat_input",
        )
        c1, c2 = st.columns([5, 1])
        submitted = c2.form_submit_button("Send ➤", use_container_width=True)

    # ------------------------------------------------------------------
    # Handle message
    # ------------------------------------------------------------------
    if submitted and user_input and user_input.strip():
        query = user_input.strip()
        st.session_state.cp_chat_history.append({"role": "user", "content": query})

        results     = store.search(query, top_k=4)
        src_names   = [d["title"] for d, s in results if s > 0.05]
        augmented   = build_rag_prompt(query, store, st.session_state.cp_chat_history[:-1])
        history_aug = (
            st.session_state.cp_chat_history[:-1]
            + [{"role": "user", "content": augmented}]
        )

        with st.spinner("PCOSBot is thinking..."):
            try:
                reply = _call_groq(client, history_aug, SYSTEM_PROMPT)
            except Exception as exc:
                err = str(exc)
                if "invalid_api_key" in err.lower() or "authentication_error" in err.lower():
                    reply = "⚠️ Invalid API key — please check your Groq API key."
                elif "quota" in err.lower() or "429" in err or "exhausted" in err.lower():
                    reply = "⚠️ Rate limit reached — please wait a moment and try again."
                elif "model_not_found" in err.lower() or "invalid_request_error" in err.lower():
                    reply = "⚠️ No available Groq model responded. Please try again later."
                else:
                    reply = f"⚠️ Error from Groq API: {err}"

        bot_idx = len(st.session_state.cp_chat_history)
        st.session_state.cp_chat_history.append({"role": "assistant", "content": reply})
        st.session_state.cp_chat_sources[bot_idx] = src_names
        st.rerun()

    # ------------------------------------------------------------------
    # Clear + disclaimer
    # ------------------------------------------------------------------
    if st.session_state.cp_chat_history:
        if st.button("🗑️ Clear conversation", key="cp_clear"):
            st.session_state.cp_chat_history = []
            st.session_state.cp_chat_sources = {}
            st.rerun()

    st.markdown(
        '<div style="color:#94A3B8;font-size:.75rem;text-align:center;margin-top:24px">'
        "⚕️ PCOSBot provides general information only — "
        "not a substitute for medical advice. Always consult a qualified clinician."
        "</div>",
        unsafe_allow_html=True,
    )
