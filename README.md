# PCOS Screening Aid

A machine learning project for stepwise PCOS (Polycystic Ovary Syndrome) screening,
built as a clinical decision support tool.

---

## Project Overview

The system implements a **three-tier screening cascade**:

| Tier | Model | Features | Requires |
|------|-------|----------|---------|
| Step 1 | **Lite** | 10 history questions | Nothing — just a questionnaire |
| Step 2 | **Model B** | 29 features | Physical exam + blood tests |
| Step 3 | **Model A** | 34 features | Step 2 + ultrasound findings |

Each model's threshold is set to catch **≥ 90% of PCOS cases** (high recall), so a positive screen means *"refer for the next step"*, not *"has PCOS"*.

---

## Project Structure

```
pmos-prediction/
├── app.py                        # Streamlit web app (clinic-style UI)
├── requirements.txt              # Python dependencies
├── .streamlit/
│   └── config.toml               # Teal clinic theme
├── data/
│   └── pcos_updated.xlsx         # Cleaned dataset (541 patients)
├── models/                       # Trained & frozen model files
│   ├── model_A_svm.joblib
│   ├── model_B_svm.joblib
│   ├── model_B_lite.joblib
│   ├── model_metadata.json
│   └── model_B_lite_metadata.json
├── outputs/                      # All analysis outputs
│   ├── figures/                  # 9 saved plots
│   └── *.csv                     # 27 result tables
└── src/                          # All training & analysis code
    ├── config.py                 # Feature lists, constants, hyperparameters
    ├── data_loader.py            # load_data(), load_split(), audit_data()
    ├── preprocessing.py          # Preprocessor builders, pipeline factory
    ├── evaluation.py             # Metrics, thresholds, CIs, bootstrap
    ├── training.py               # CV comparison, matched-recall analysis
    ├── explainability.py         # SHAP, ablation, engineered features, nested CV
    ├── cascade.py                # Cascade analysis + Lite Model B
    ├── lite_extras.py            # Lite vs RF, nested lite CV, paper table
    ├── visualization.py          # ROC/PR curves, confusion matrices
    ├── freeze.py                 # Save final models to disk
    └── run_all.py                # CLI entry point
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the Streamlit App

```bash
# From the project root (pmos-prediction/)
streamlit run app.py
```

The app loads the pre-trained models from `models/`. No retraining required.

---

## Reproducing the Training

Run from the project root:

```bash
# Full pipeline (all steps — takes ~10–30 min)
python src/run_all.py

# Quick smoke-test (reduced CV repeats)
python src/run_all.py --quick

# Specific steps only (e.g., re-freeze models after changes)
python src/run_all.py --steps thresholds lite freeze

# Lite model extras (nested CV, paper table)
python src/lite_extras.py
```

### Available Steps

| Step | Description |
|------|-------------|
| `comparison` | Final model comparison (LR / RF / SVM) |
| `matched_recall` | Fair comparison at matched recall |
| `thresholds` | Threshold grid + calibration + CIs |
| `shap_b` | SHAP + linear-SVM coefficients (Model B) |
| `ablation` | Feature-group ablation + cost tiers |
| `engineered` | Engineered feature experiments |
| `families` | XGBoost, voting, stacking ensembles |
| `nested` | Nested CV (tune + threshold) |
| `cascade` | Two-stage B → ultrasound → A cascade |
| `lite` | Lite Model B (10 no-lab features) |
| `plots` | ROC / PR curves + confusion matrices |
| `freeze` | Save final pipelines + thresholds to disk |

---

## Model Performance (Held-out test set, 109 patients)

| Model | Recall | Specificity | ROC-AUC | Threshold |
|-------|--------|-------------|---------|-----------|
| Lite (10 features, RF) | 0.972 | 0.644 | 0.921 | 0.25 |
| Model B (29 features, SVM) | 0.944 | 0.575 | 0.877 | 0.15 |
| Model A (34 features, SVM) | 0.917 | 0.890 | 0.960 | 0.325 |

> Each threshold is set to catch ≥ 90% of PCOS cases (recall ≥ 0.90).
> Wide confidence intervals apply — test set has 36 positive cases.

---

## Dataset

- **Source:** Kaggle PCOS Dataset (without infertility data)
- **Size:** 541 patients | 38 columns
- **Target:** `PCOS (Y/N)` — binary classification
- **Split:** 80% train / 20% test (stratified, `random_state=42`)
- **Prevalence:** ~33% PCOS positive

---

## Important Disclaimer

> **This is a screening aid, not a medical device.**
> All results are for research and educational purposes only.
> Final clinical decisions must rest with a qualified clinician.
