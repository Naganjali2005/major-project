import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

DATA_PATH = "data/pcos_updated.xlsx"

# Target: what we want the model to predict
TARGET = "PCOS (Y/N)"


# Model A:
# Full clinical model INCLUDING ultrasound features
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


# Model B:
# Same clinical information, but WITHOUT ultrasound
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

# Preprocessing for numerical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing for binary features
binary_imputer = SimpleImputer(
    strategy="most_frequent"
)

# Preprocessing for cycle category
cycle_imputer = SimpleImputer(
    strategy="most_frequent"
)
 ######
cycle_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# Combine the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "binary",
            binary_imputer,
            BINARY_FEATURES
        ),
        (
            "cycle",
            Pipeline([
                ("imputer", cycle_imputer),
                ("encoder", cycle_encoder)
            ]),
            [CYCLE_FEATURE]
        )
    ]
)

preprocessor_B = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES_B
        ),
        (
            "binary",
            binary_imputer,
            BINARY_FEATURES
        ),
        (
            "cycle",
            Pipeline([
                ("imputer", cycle_imputer),
                ("encoder", cycle_encoder)
            ]),
            [CYCLE_FEATURE]
        )
    ]
)


# Preprocessing for Random Forest
# No scaling is required for tree-based models

rf_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            NUMERIC_FEATURES
        ),
        (
            "binary",
            binary_imputer,
            BINARY_FEATURES
        ),
        (
            "cycle",
            Pipeline([
                ("imputer", cycle_imputer),
                ("encoder", cycle_encoder)
            ]),
            [CYCLE_FEATURE]
        )
    ]
)

rf_preprocessor_B = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            NUMERIC_FEATURES_B
        ),
        (
            "binary",
            binary_imputer,
            BINARY_FEATURES
        ),
        (
            "cycle",
            Pipeline([
                ("imputer", cycle_imputer),
                ("encoder", cycle_encoder)
            ]),
            [CYCLE_FEATURE]
        )
    ]
)

model_A_rf_pipeline = Pipeline([
    ("preprocessing", rf_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])


rf_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10, 15],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}


model_B_rf_pipeline = Pipeline([
    ("preprocessing", rf_preprocessor_B),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# SVM PIPELINES

model_A_svm_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight="balanced"
    ))
])

model_B_svm_pipeline = Pipeline([
    ("preprocessing", preprocessor_B),
    ("model", SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight="balanced"
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


model_A_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])


model_B_pipeline = Pipeline([
    ("preprocessing", preprocessor_B),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])


def load_data():
    """Load the original PCOS dataset."""

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="Sheet1"
    )

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()
    df["AMH(ng/mL)"] = pd.to_numeric(
        df["AMH(ng/mL)"],
        errors="coerce"
    )

    return df


if __name__ == "__main__":

    # Load data
    df = load_data()

    # Separate features and target
    X_A = df[MODEL_A_FEATURES]
    X_B = df[MODEL_B_FEATURES]

    print("\n--- Missing values in Model A ---")
    print(X_A.isnull().sum())

    print("\n--- Missing values in Model B ---")
    print(X_B.isnull().sum())

    y = df[TARGET]

    # Split Model A
    X_train_A, X_test_A, y_train, y_test = train_test_split(
        X_A,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Use exactly the same patients for Model B
    X_train_B = X_B.loc[X_train_A.index]
    X_test_B = X_B.loc[X_test_A.index]

    cv_results_A = cross_validate(
        model_A_pipeline,
        X_train_A,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    )

    cv_results_B = cross_validate(
        model_B_pipeline,
        X_train_B,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    )

    print("\n--- Model A 5-Fold Cross Validation ---")
    print("Accuracy :", cv_results_A["test_accuracy"].mean())
    print("Precision:", cv_results_A["test_precision"].mean())
    print("Recall   :", cv_results_A["test_recall"].mean())
    print("F1 Score :", cv_results_A["test_f1"].mean())
    print("ROC-AUC  :", cv_results_A["test_roc_auc"].mean())

    print("\n--- Model B 5-Fold Cross Validation ---")
    print("Accuracy :", cv_results_B["test_accuracy"].mean())
    print("Precision:", cv_results_B["test_precision"].mean())
    print("Recall   :", cv_results_B["test_recall"].mean())
    print("F1 Score :", cv_results_B["test_f1"].mean())
    print("ROC-AUC  :", cv_results_B["test_roc_auc"].mean())

    # -----------------------------
    # RANDOM FOREST - 5-FOLD CV
    # -----------------------------

    cv_results_A_rf = cross_validate(
        model_A_rf_pipeline,
        X_train_A,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    )

    cv_results_B_rf = cross_validate(
        model_B_rf_pipeline,
        X_train_B,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    )

    print("\n--- Model A Random Forest 5-Fold CV ---")
    print("Accuracy :", cv_results_A_rf["test_accuracy"].mean())
    print("Precision:", cv_results_A_rf["test_precision"].mean())
    print("Recall   :", cv_results_A_rf["test_recall"].mean())
    print("F1 Score :", cv_results_A_rf["test_f1"].mean())
    print("ROC-AUC  :", cv_results_A_rf["test_roc_auc"].mean())

    print("\n--- Model B Random Forest 5-Fold CV ---")
    print("Accuracy :", cv_results_B_rf["test_accuracy"].mean())
    print("Precision:", cv_results_B_rf["test_precision"].mean())
    print("Recall   :", cv_results_B_rf["test_recall"].mean())
    print("F1 Score :", cv_results_B_rf["test_f1"].mean())
    print("ROC-AUC  :", cv_results_B_rf["test_roc_auc"].mean())

    # -----------------------------
    # SVM - 5-FOLD CV
    # -----------------------------

    cv_results_A_svm = cross_validate(
        model_A_svm_pipeline,
        X_train_A,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    )

    cv_results_B_svm = cross_validate(
        model_B_svm_pipeline,
        X_train_B,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    )

    print("\n--- Model A SVM 5-Fold CV ---")
    print("Accuracy :", cv_results_A_svm["test_accuracy"].mean())
    print("Precision:", cv_results_A_svm["test_precision"].mean())
    print("Recall   :", cv_results_A_svm["test_recall"].mean())
    print("F1 Score :", cv_results_A_svm["test_f1"].mean())
    print("ROC-AUC  :", cv_results_A_svm["test_roc_auc"].mean())

    print("\n--- Model B SVM 5-Fold CV ---")
    print("Accuracy :", cv_results_B_svm["test_accuracy"].mean())
    print("Precision:", cv_results_B_svm["test_precision"].mean())
    print("Recall   :", cv_results_B_svm["test_recall"].mean())
    print("F1 Score :", cv_results_B_svm["test_f1"].mean())
    print("ROC-AUC  :", cv_results_B_svm["test_roc_auc"].mean())


    # ============================================================
    # LOGISTIC REGRESSION HYPERPARAMETER TUNING - MODEL A
    # ============================================================

    lr_param_grid_A = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l1", "l2"],
        "model__solver": ["liblinear"]
    }

    lr_grid_search_A = GridSearchCV(
        estimator=model_A_pipeline,
        param_grid=lr_param_grid_A,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    lr_grid_search_A.fit(X_train_A, y_train)

    print("\n==============================")
    print("TUNED LOGISTIC REGRESSION - MODEL A")
    print("==============================")

    print("Best LR Parameters:")
    print(lr_grid_search_A.best_params_)

    print("Best CV ROC-AUC:")
    print(lr_grid_search_A.best_score_)



    # ============================================================
    # EVALUATE TUNED LOGISTIC REGRESSION - MODEL A
    # ============================================================

    best_lr_model_A = lr_grid_search_A.best_estimator_

    y_pred_lr_A = best_lr_model_A.predict(X_test_A)
    y_prob_lr_A = best_lr_model_A.predict_proba(X_test_A)[:, 1]

    print("\n--- Tuned Logistic Regression Model A Test Results ---")

    print("Accuracy:",
        accuracy_score(y_test, y_pred_lr_A))

    print("Precision:",
        precision_score(y_test, y_pred_lr_A))

    print("Recall:",
        recall_score(y_test, y_pred_lr_A))

    print("F1:",
        f1_score(y_test, y_pred_lr_A))

    print("ROC-AUC:",
        roc_auc_score(y_test, y_prob_lr_A))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr_A))


    # ============================================================
    # LOGISTIC REGRESSION HYPERPARAMETER TUNING - MODEL B
    # ============================================================

    lr_param_grid_B = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l1", "l2"],
        "model__solver": ["liblinear"]
    }

    lr_grid_search_B = GridSearchCV(
        estimator=model_B_pipeline,
        param_grid=lr_param_grid_B,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    lr_grid_search_B.fit(X_train_B, y_train)

    print("\n==============================")
    print("TUNED LOGISTIC REGRESSION - MODEL B")
    print("==============================")

    print("Best LR Parameters:")
    print(lr_grid_search_B.best_params_)

    print("Best CV ROC-AUC:")
    print(lr_grid_search_B.best_score_)



    # ============================================================
    # EVALUATE TUNED LOGISTIC REGRESSION - MODEL B
    # ============================================================

    best_lr_model_B = lr_grid_search_B.best_estimator_

    y_pred_lr_B = best_lr_model_B.predict(X_test_B)
    y_prob_lr_B = best_lr_model_B.predict_proba(X_test_B)[:, 1]

    print("\n--- Tuned Logistic Regression Model B Test Results ---")

    print("Accuracy:",
        accuracy_score(y_test, y_pred_lr_B))

    print("Precision:",
        precision_score(y_test, y_pred_lr_B))

    print("Recall:",
        recall_score(y_test, y_pred_lr_B))

    print("F1:",
        f1_score(y_test, y_pred_lr_B))

    print("ROC-AUC:",
        roc_auc_score(y_test, y_prob_lr_B))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr_B))


    # ============================================================
    # SVM HYPERPARAMETER TUNING - MODEL A
    # ============================================================

    svm_param_grid_A = {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto"],
        "model__kernel": ["rbf", "linear"]
    }

    svm_grid_search_A = GridSearchCV(
        estimator=model_A_svm_pipeline,
        param_grid=svm_param_grid_A,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    svm_grid_search_A.fit(X_train_A, y_train)

    print("\n==============================")
    print("TUNED SVM - MODEL A")
    print("==============================")

    print("Best SVM Parameters:")
    print(svm_grid_search_A.best_params_)

    print("Best CV ROC-AUC:")
    print(svm_grid_search_A.best_score_)


    # ============================================================
    # EVALUATE TUNED SVM - MODEL A
    # ============================================================

    best_svm_model_A = svm_grid_search_A.best_estimator_

    y_pred_svm_A = best_svm_model_A.predict(X_test_A)
    y_prob_svm_A = best_svm_model_A.predict_proba(X_test_A)[:, 1]

    print("\n--- Tuned SVM Model A Test Results ---")

    print("Accuracy:", accuracy_score(y_test, y_pred_svm_A))
    print("Precision:", precision_score(y_test, y_pred_svm_A))
    print("Recall:", recall_score(y_test, y_pred_svm_A))
    print("F1:", f1_score(y_test, y_pred_svm_A))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_svm_A))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm_A))




    # ============================================================
    # THRESHOLD ANALYSIS - MODEL A TUNED SVM
    # ============================================================

    from sklearn.model_selection import cross_val_predict

    oof_prob_svm_A = cross_val_predict(
        best_svm_model_A,
        X_train_A,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]


    thresholds = np.arange(0.30, 0.61, 0.05)

    threshold_results = []

    for threshold in thresholds:

        y_pred_threshold = (
            oof_prob_svm_A >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_train,
            y_pred_threshold
        ).ravel()

        accuracy = accuracy_score(
            y_train,
            y_pred_threshold
        )

        precision = precision_score(
            y_train,
            y_pred_threshold,
            zero_division=0
        )

        recall = recall_score(
            y_train,
            y_pred_threshold,
            zero_division=0
        )

        f1 = f1_score(
            y_train,
            y_pred_threshold,
            zero_division=0
        )

        specificity = tn / (tn + fp)

        threshold_results.append({
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Specificity": specificity,
            "False Negatives": fn
        })


    threshold_results_df = pd.DataFrame(threshold_results)

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS - MODEL A TUNED SVM")
    print("=" * 70)

    print(
        threshold_results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # ============================================================
    # SELECT THRESHOLD
    # ============================================================

    minimum_recall = 0.90

    eligible_thresholds = threshold_results_df[
        threshold_results_df["Recall"] >= minimum_recall
    ]

    if len(eligible_thresholds) > 0:

        selected_row = eligible_thresholds.loc[
            eligible_thresholds["F1"].idxmax()
        ]

        selected_threshold = selected_row["Threshold"]

    else:

        selected_row = threshold_results_df.loc[
            threshold_results_df["Recall"].idxmax()
        ]

        selected_threshold = selected_row["Threshold"]


    print("\nSelected threshold:")
    print(f"{selected_threshold:.2f}")

    print("\nSelected threshold performance on OOF training predictions:")
    print(selected_row.to_string())


    # ============================================================
    # APPLY SELECTED THRESHOLD TO TEST SET
    # ============================================================

    test_prob_svm_A = best_svm_model_A.predict_proba(
        X_test_A
    )[:, 1]

    y_test_threshold = (
        test_prob_svm_A >= selected_threshold
    ).astype(int)


    # ============================================================
    # FINAL TEST PERFORMANCE
    # ============================================================

    test_accuracy = accuracy_score(
        y_test,
        y_test_threshold
    )

    test_precision = precision_score(
        y_test,
        y_test_threshold,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        y_test_threshold,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        y_test_threshold,
        zero_division=0
    )

    test_cm = confusion_matrix(
        y_test,
        y_test_threshold
    )

    tn, fp, fn, tp = test_cm.ravel()

    test_specificity = tn / (tn + fp)

    test_roc_auc = roc_auc_score(
        y_test,
        test_prob_svm_A
    )


    print("\n" + "=" * 70)
    print("FINAL TEST PERFORMANCE - SELECTED THRESHOLD")
    print("=" * 70)

    print(f"Threshold   : {selected_threshold:.2f}")
    print(f"Accuracy    : {test_accuracy:.4f}")
    print(f"Precision   : {test_precision:.4f}")
    print(f"Recall      : {test_recall:.4f}")
    print(f"F1 Score    : {test_f1:.4f}")
    print(f"Specificity : {test_specificity:.4f}")
    print(f"ROC-AUC     : {test_roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(test_cm)

    print(f"\nTrue Negatives  : {tn}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print(f"True Positives  : {tp}")


    # ============================================================
    # FINAL MODEL COMPARISON - MODEL A
    # ============================================================

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON - MODEL A")
    print("=" * 80)

    final_comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "SVM (0.50 threshold)",
            "SVM (0.30 threshold)"
        ],
        "Accuracy": [
            0.8899,
            0.9358,
            0.9083,
            test_accuracy
        ],
        "Precision": [
            0.8000,
            0.9677,
            0.8250,
            test_precision
        ],
        "Recall": [
            0.8889,
            0.8333,
            0.9167,
            test_recall
        ],
        "F1": [
            0.8421,
            0.8955,
            0.8684,
            test_f1
        ],
        "Specificity": [
            0.8904,
            0.9863,
            0.9041,
            test_specificity
        ],
        "ROC-AUC": [
            0.9551,
            0.9416,
            0.9600,
            test_roc_auc
        ]
    })

    print(
        final_comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ============================================================
    # SVM HYPERPARAMETER TUNING - MODEL B
    # ============================================================

    svm_param_grid_B = {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto"],
        "model__kernel": ["rbf", "linear"]
    }

    svm_grid_search_B = GridSearchCV(
        estimator=model_B_svm_pipeline,
        param_grid=svm_param_grid_B,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    svm_grid_search_B.fit(X_train_B, y_train)

    print("\n==============================")
    print("TUNED SVM - MODEL B")
    print("==============================")

    print("Best SVM Parameters:")
    print(svm_grid_search_B.best_params_)

    print("Best CV ROC-AUC:")
    print(svm_grid_search_B.best_score_)


    # ============================================================
    # EVALUATE TUNED SVM - MODEL B
    # ============================================================

    best_svm_model_B = svm_grid_search_B.best_estimator_

    y_pred_svm_B = best_svm_model_B.predict(X_test_B)
    y_prob_svm_B = best_svm_model_B.predict_proba(X_test_B)[:, 1]

    print("\n--- Tuned SVM Model B Test Results ---")

    print("Accuracy:", accuracy_score(y_test, y_pred_svm_B))
    print("Precision:", precision_score(y_test, y_pred_svm_B))
    print("Recall:", recall_score(y_test, y_pred_svm_B))
    print("F1:", f1_score(y_test, y_pred_svm_B))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_svm_B))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm_B))



    # ======================================================================
    # THRESHOLD ANALYSIS - MODEL B TUNED SVM
    # ======================================================================

    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS - MODEL B TUNED SVM")
    print("=" * 70)

    # Generate out-of-fold probabilities using training data only
    y_prob_oof_B = cross_val_predict(
        best_svm_model_B,
        X_train_B,
        y_train,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    threshold_results_B = []

    for threshold in thresholds:

        y_pred_threshold = (y_prob_oof_B >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_train,
            y_pred_threshold
        ).ravel()

        accuracy = accuracy_score(y_train, y_pred_threshold)
        precision = precision_score(y_train, y_pred_threshold, zero_division=0)
        recall = recall_score(y_train, y_pred_threshold, zero_division=0)
        f1 = f1_score(y_train, y_pred_threshold, zero_division=0)
        specificity = tn / (tn + fp)

        threshold_results_B.append({
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Specificity": specificity,
            "False Negatives": fn
        })

    threshold_df_B = pd.DataFrame(threshold_results_B)

    print(threshold_df_B.to_string(index=False))

    # ---------------------------------------------------------
    # Select threshold
    # Minimum required recall = 90%
    # Among thresholds satisfying this, select highest F1
    # ---------------------------------------------------------

    minimum_recall = 0.90

    eligible_thresholds_B = threshold_df_B[
        threshold_df_B["Recall"] >= minimum_recall
    ]

    if not eligible_thresholds_B.empty:

        selected_row_B = eligible_thresholds_B.loc[
            eligible_thresholds_B["F1"].idxmax()
        ]

    else:

        selected_row_B = threshold_df_B.loc[
            threshold_df_B["Recall"].idxmax()
        ]

    selected_threshold_B = selected_row_B["Threshold"]

    print("\nSelected threshold:")
    print(selected_threshold_B)

    print("\nSelected threshold performance on OOF training predictions:")

    for column in [
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "False Negatives"
    ]:
        print(f"{column:<18}: {selected_row_B[column]:.6f}")


    # ======================================================================
    # FINAL TEST PERFORMANCE AT SELECTED THRESHOLD
    # ======================================================================

    y_prob_svm_B = best_svm_model_B.predict_proba(X_test_B)[:, 1]

    y_pred_svm_B_threshold = (
        y_prob_svm_B >= selected_threshold_B
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred_svm_B_threshold
    ).ravel()

    accuracy_B_threshold = accuracy_score(
        y_test,
        y_pred_svm_B_threshold
    )

    precision_B_threshold = precision_score(
        y_test,
        y_pred_svm_B_threshold,
        zero_division=0
    )

    recall_B_threshold = recall_score(
        y_test,
        y_pred_svm_B_threshold,
        zero_division=0
    )

    f1_B_threshold = f1_score(
        y_test,
        y_pred_svm_B_threshold,
        zero_division=0
    )

    specificity_B_threshold = tn / (tn + fp)

    roc_auc_B = roc_auc_score(
        y_test,
        y_prob_svm_B
    )

    print("\n" + "=" * 70)
    print("FINAL TEST PERFORMANCE - MODEL B SELECTED THRESHOLD")
    print("=" * 70)

    print(f"Threshold   : {selected_threshold_B:.2f}")
    print(f"Accuracy    : {accuracy_B_threshold:.4f}")
    print(f"Precision   : {precision_B_threshold:.4f}")
    print(f"Recall      : {recall_B_threshold:.4f}")
    print(f"F1 Score    : {f1_B_threshold:.4f}")
    print(f"Specificity : {specificity_B_threshold:.4f}")
    print(f"ROC-AUC     : {roc_auc_B:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm_B_threshold))

    print(f"\nTrue Negatives  : {tn}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print(f"True Positives  : {tp}")

    # =====================================================
    # RANDOM FOREST - GRID SEARCH
    # =====================================================

    rf_grid_search = GridSearchCV(
        estimator=model_A_rf_pipeline,
        param_grid=rf_param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    rf_grid_search.fit(X_train_A, y_train)

    print("\n--- Model A Random Forest Grid Search ---")
    print("Best parameters:")
    print(rf_grid_search.best_params_)

    print("\nBest CV ROC-AUC:")
    print(rf_grid_search.best_score_)

    # =====================================================
    # FINAL TUNED MODEL - TEST SET EVALUATION
    # =====================================================

    # Best model found by GridSearchCV
    best_rf_model = rf_grid_search.best_estimator_

    # Train the best model on the complete training set
    best_rf_model.fit(X_train_A, y_train)

    # Predictions on the untouched test set
    y_pred_A_rf_tuned = best_rf_model.predict(X_test_A)

    # Prediction probabilities for ROC-AUC
    y_prob_A_rf_tuned = best_rf_model.predict_proba(X_test_A)[:, 1]

    # Evaluation
    accuracy_A_rf_tuned = accuracy_score(
        y_test,
        y_pred_A_rf_tuned
    )

    precision_A_rf_tuned = precision_score(
        y_test,
        y_pred_A_rf_tuned
    )

    recall_A_rf_tuned = recall_score(
        y_test,
        y_pred_A_rf_tuned
    )

    f1_A_rf_tuned = f1_score(
        y_test,
        y_pred_A_rf_tuned
    )

    roc_auc_A_rf_tuned = roc_auc_score(
        y_test,
        y_prob_A_rf_tuned
    )

    cm_A_rf_tuned = confusion_matrix(
        y_test,
        y_pred_A_rf_tuned
    )

    print("\n--- Model A Tuned Random Forest - Test Set ---")
    print("Accuracy :", accuracy_A_rf_tuned)
    print("Precision:", precision_A_rf_tuned)
    print("Recall   :", recall_A_rf_tuned)
    print("F1 Score :", f1_A_rf_tuned)
    print("ROC-AUC  :", roc_auc_A_rf_tuned)

    print("\nConfusion Matrix:")
    print(cm_A_rf_tuned)

    # =====================================================
    # MODEL B - RANDOM FOREST GRID SEARCH
    # =====================================================

    rf_grid_search_B = GridSearchCV(
    estimator=model_B_rf_pipeline,
    param_grid=rf_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
    )

    rf_grid_search_B.fit(X_train_B, y_train)

    print("\n--- Model B Random Forest Grid Search ---")
    print("Best parameters:")
    print(rf_grid_search_B.best_params_)

    print("\nBest CV ROC-AUC:")
    print(rf_grid_search_B.best_score_)


    # =====================================================
    # MODEL B - TUNED RANDOM FOREST TEST SET EVALUATION
    # =====================================================

    best_rf_model_B = rf_grid_search_B.best_estimator_

    # Train best model on complete Model B training set
    best_rf_model_B.fit(X_train_B, y_train)

    # Predictions
    y_pred_B_rf_tuned = best_rf_model_B.predict(X_test_B)

    # Prediction probabilities
    y_prob_B_rf_tuned = best_rf_model_B.predict_proba(X_test_B)[:, 1]


    # Evaluation
    accuracy_B_rf_tuned = accuracy_score(
    y_test,
    y_pred_B_rf_tuned
    )

    precision_B_rf_tuned = precision_score(
    y_test,
    y_pred_B_rf_tuned
    )

    recall_B_rf_tuned = recall_score(
    y_test,
    y_pred_B_rf_tuned
    )

    f1_B_rf_tuned = f1_score(
    y_test,
    y_pred_B_rf_tuned
    )

    roc_auc_B_rf_tuned = roc_auc_score(
    y_test,
    y_prob_B_rf_tuned
    )

    cm_B_rf_tuned = confusion_matrix(
    y_test,
    y_pred_B_rf_tuned
    )


    print("\n--- Model B Tuned Random Forest - Test Set ---")
    print("Accuracy :", accuracy_B_rf_tuned)
    print("Precision:", precision_B_rf_tuned)
    print("Recall   :", recall_B_rf_tuned)
    print("F1 Score :", f1_B_rf_tuned)
    print("ROC-AUC  :", roc_auc_B_rf_tuned)

    print("\nConfusion Matrix:")
    print(cm_B_rf_tuned)

    # =====================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # =====================================================

    # Get the trained Random Forest from the pipeline
    rf_model = best_rf_model.named_steps["model"]

    # Get feature importance values
    importances = rf_model.feature_importances_

    # Get the processed feature names
    feature_names = best_rf_model.named_steps["preprocessing"].get_feature_names_out()

    # Combine feature names and importance
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    # Sort from most important to least important
    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n--- Random Forest Feature Importance ---")
    print(feature_importance)

    # =====================================================
    # PREPARE DATA FOR SHAP
    # =====================================================

    # Get fitted preprocessing step from the final pipeline
    fitted_preprocessor = best_rf_model.named_steps["preprocessing"]

    # Transform the training data
    X_train_A_processed = fitted_preprocessor.transform(X_train_A)

    # Get the actual Random Forest model
    rf_model = best_rf_model.named_steps["model"]

    # Get processed feature names
    shap_feature_names = fitted_preprocessor.get_feature_names_out()

    print("\n--- SHAP Data ---")
    print("Processed training shape:", X_train_A_processed.shape)
    print("Number of feature names:", len(shap_feature_names))

    # =====================================================
    # SHAP EXPLAINER
    # =====================================================

    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(X_train_A_processed)

    print("\n--- SHAP Values ---")
    print("SHAP values shape:", shap_values.shape)

    # =====================================================
    # SELECT SHAP VALUES FOR PCOS CLASS
    # =====================================================

    # Class 1 = PCOS
    shap_values_pcos = shap_values[:, :, 1]

    print("\n--- PCOS SHAP Values ---")
    print("Shape:", shap_values_pcos.shape)

    # =====================================================
    # SHAP FEATURE IMPORTANCE
    # =====================================================

    shap_importance = pd.DataFrame({
        "Feature": shap_feature_names,
        "Mean |SHAP|": abs(shap_values_pcos).mean(axis=0)
    })

    shap_importance = shap_importance.sort_values(
        by="Mean |SHAP|",
        ascending=False
    )

    print("\n--- SHAP Feature Importance ---")
    print(shap_importance)

    # =====================================================
    # SHAP-BASED FEATURE SELECTION CANDIDATES
    # =====================================================

    print("\n--- Top 10 SHAP Features ---")
    print(shap_importance.head(10))

    print("\n--- Top 15 SHAP Features ---")
    print(shap_importance.head(15))

    print("\n--- Top 20 SHAP Features ---")
    print(shap_importance.head(20))

    # =====================================================
    # SHAP SUMMARY PLOT
    # =====================================================

    # shap.summary_plot(
    #     shap_values_pcos,
    #     X_train_A_processed,
    #     feature_names=shap_feature_names
    # )

    # =====================================================
    # INDIVIDUAL PATIENT SHAP EXPLANATION
    # =====================================================

    patient_index = 0

    patient_shap = shap_values_pcos[patient_index]
    patient_data = X_train_A_processed[patient_index]

    print("\n--- Individual Patient SHAP ---")
    print("Patient index:", patient_index)
    print("Actual PCOS:", y_train.iloc[patient_index])

    # Model prediction
    patient_prediction = rf_model.predict(
        patient_data.reshape(1, -1)
    )[0]

    patient_probability = rf_model.predict_proba(
        patient_data.reshape(1, -1)
    )[0, 1]

    print("Predicted PCOS:", patient_prediction)
    print("PCOS probability:", patient_probability)

    # =====================================================
    # SHAP WATERFALL PLOT
    # =====================================================

    shap.waterfall_plot(
        shap.Explanation(
            values=patient_shap,
            base_values=explainer.expected_value[1],
            data=patient_data,
            feature_names=shap_feature_names
        ),
        max_display=15
    )

    # =====================================================
    # FEATURE SELECTION EXPERIMENT
    # =====================================================

    from sklearn.feature_selection import SelectFromModel

    feature_counts = [10, 15, 20, 25]

    print("\n--- Feature Selection using Stratified 5-Fold CV ---")

    for k in feature_counts:

        # Random Forest used for feature selection
        selector_rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )

        # Final Random Forest
        final_rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )

        feature_selection_pipeline = Pipeline([
            ("preprocessing", rf_preprocessor),

            # Select top K features inside each CV fold
            ("feature_selection",
             SelectFromModel(
                 selector_rf,
                 threshold=-float("inf"),
                 max_features=k
             )),

            ("model", final_rf)
        ])

        scores = cross_validate(
            feature_selection_pipeline,
            X_train_A,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc"
            }
        )

        print(f"\nTop {k} Features:")
        print("Accuracy :", scores["test_accuracy"].mean())
        print("Precision:", scores["test_precision"].mean())
        print("Recall   :", scores["test_recall"].mean())
        print("F1 Score :", scores["test_f1"].mean())
        print("ROC-AUC  :", scores["test_roc_auc"].mean())

    # =====================================================
    # FINAL MODEL A / MODEL B FIT + EVALUATION ON TEST SET
    # (runs once, after the feature-selection loop above)
    # =====================================================

    # Fit preprocessing ONLY on Model A training data
    preprocessor.fit(X_train_A)

    # Transform training and testing data
    X_train_A_processed = preprocessor.transform(X_train_A)
    X_test_A_processed = preprocessor.transform(X_test_A)

    # Fit Model B preprocessing ONLY on Model B training data
    preprocessor_B.fit(X_train_B)

    X_train_B_processed = preprocessor_B.transform(X_train_B)
    X_test_B_processed = preprocessor_B.transform(X_test_B)

    print("\n--- Model B After Preprocessing ---")
    print("Training shape:", X_train_B_processed.shape)
    print("Testing shape :", X_test_B_processed.shape)

    # -----------------------------
    # MODEL A - LOGISTIC REGRESSION
    # -----------------------------

    model_A = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )

    model_A.fit(
        X_train_A_processed,
        y_train
    )

    y_pred_A = model_A.predict(
        X_test_A_processed
    )

    model_B = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )

    model_B.fit(
        X_train_B_processed,
        y_train
    )

    y_pred_B = model_B.predict(
        X_test_B_processed
    )

    print("\n--- Model B Predictions ---")
    print("Predictions:", y_pred_B[:10])
    print("Actual     :", y_test.iloc[:10].values)

    # -----------------------------
    # MODEL B - EVALUATION
    # -----------------------------

    accuracy_B = accuracy_score(y_test, y_pred_B)
    precision_B = precision_score(y_test, y_pred_B)
    recall_B = recall_score(y_test, y_pred_B)
    f1_B = f1_score(y_test, y_pred_B)

    y_prob_B = model_B.predict_proba(
        X_test_B_processed
    )[:, 1]

    roc_auc_B = roc_auc_score(
        y_test,
        y_prob_B
    )

    cm_B = confusion_matrix(
        y_test,
        y_pred_B
    )

    print("\n--- Model B Evaluation ---")
    print("Accuracy :", accuracy_B)
    print("Precision:", precision_B)
    print("Recall   :", recall_B)
    print("F1 Score :", f1_B)
    print("ROC-AUC  :", roc_auc_B)

    print("\nConfusion Matrix:")
    print(cm_B)

    # -----------------------------
    # MODEL A - EVALUATION
    # -----------------------------

    accuracy = accuracy_score(y_test, y_pred_A)
    precision = precision_score(y_test, y_pred_A)
    recall = recall_score(y_test, y_pred_A)
    f1 = f1_score(y_test, y_pred_A)

    y_prob_A = model_A.predict_proba(
        X_test_A_processed
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        y_prob_A
    )

    cm = confusion_matrix(
        y_test,
        y_pred_A
    )

    print("\n--- Model A Evaluation ---")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)

    print("\nConfusion Matrix:")
    print(cm)

    print("\n--- Model A Predictions ---")
    print("Predictions:", y_pred_A[:10])
    print("Actual     :", y_test.iloc[:10].values)

    # =====================================================
    # FINAL CROSS-VALIDATION COMPARISON
    # MEAN ± STANDARD DEVIATION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("FINAL CROSS-VALIDATION COMPARISON")
    print("MEAN ± STANDARD DEVIATION")
    print("=" * 70)



    


    # -----------------------------------------------------
    # 1. MODEL A - LOGISTIC REGRESSION
    # -----------------------------------------------------

    cv_A_LR = cross_validate(
    model_A_pipeline,
    X_train_A,
    y_train,
    cv=cv,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]
    )

    print("\n--- Model A - Logistic Regression ---")

    print(
    "Accuracy :",
    f"{cv_A_LR['test_accuracy'].mean():.4f} ± "
    f"{cv_A_LR['test_accuracy'].std():.4f}"
    )

    print(
    "Precision:",
    f"{cv_A_LR['test_precision'].mean():.4f} ± "
    f"{cv_A_LR['test_precision'].std():.4f}"
    )

    print(
    "Recall   :",
    f"{cv_A_LR['test_recall'].mean():.4f} ± "
    f"{cv_A_LR['test_recall'].std():.4f}"
    )

    print(
    "F1 Score :",
    f"{cv_A_LR['test_f1'].mean():.4f} ± "
    f"{cv_A_LR['test_f1'].std():.4f}"
    )

    print(
    "ROC-AUC  :",
    f"{cv_A_LR['test_roc_auc'].mean():.4f} ± "
    f"{cv_A_LR['test_roc_auc'].std():.4f}"
    )


    # -----------------------------------------------------
    # 2. MODEL A - TUNED RANDOM FOREST
    # -----------------------------------------------------

    cv_A_RF = cross_validate(
    best_rf_model,
    X_train_A,
    y_train,
    cv=cv,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]
    )

    print("\n--- Model A - Tuned Random Forest ---")

    print(
    "Accuracy :",
    f"{cv_A_RF['test_accuracy'].mean():.4f} ± "
    f"{cv_A_RF['test_accuracy'].std():.4f}"
    )

    print(
    "Precision:",
    f"{cv_A_RF['test_precision'].mean():.4f} ± "
    f"{cv_A_RF['test_precision'].std():.4f}"
    )

    print(
    "Recall   :",
    f"{cv_A_RF['test_recall'].mean():.4f} ± "
    f"{cv_A_RF['test_recall'].std():.4f}"
    )

    print(
    "F1 Score :",
    f"{cv_A_RF['test_f1'].mean():.4f} ± "
    f"{cv_A_RF['test_f1'].std():.4f}"
    )

    print(
    "ROC-AUC  :",
    f"{cv_A_RF['test_roc_auc'].mean():.4f} ± "
    f"{cv_A_RF['test_roc_auc'].std():.4f}"
    )


    # -----------------------------------------------------
    # 3. MODEL A - TUNED SVM
    # -----------------------------------------------------

    cv_A_SVM = cross_validate(
        svm_grid_search_A.best_estimator_,
        X_train_A,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    )

    print("\n--- Model A - Tuned SVM ---")

    print(
        "Accuracy :",
        f"{cv_A_SVM['test_accuracy'].mean():.4f} ± "
        f"{cv_A_SVM['test_accuracy'].std():.4f}"
    )

    print(
        "Precision:",
        f"{cv_A_SVM['test_precision'].mean():.4f} ± "
        f"{cv_A_SVM['test_precision'].std():.4f}"
    )

    print(
        "Recall   :",
        f"{cv_A_SVM['test_recall'].mean():.4f} ± "
        f"{cv_A_SVM['test_recall'].std():.4f}"
    )

    print(
        "F1 Score :",
        f"{cv_A_SVM['test_f1'].mean():.4f} ± "
        f"{cv_A_SVM['test_f1'].std():.4f}"
    )

    print(
        "ROC-AUC  :",
        f"{cv_A_SVM['test_roc_auc'].mean():.4f} ± "
        f"{cv_A_SVM['test_roc_auc'].std():.4f}"
    )


    # -----------------------------------------------------
    # 3. MODEL B - LOGISTIC REGRESSION
    # -----------------------------------------------------

    cv_B_LR = cross_validate(
    model_B_pipeline,
    X_train_B,
    y_train,
    cv=cv,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]
    )

    print("\n--- Model B - Logistic Regression ---")

    print(
    "Accuracy :",
    f"{cv_B_LR['test_accuracy'].mean():.4f} ± "
    f"{cv_B_LR['test_accuracy'].std():.4f}"
    )

    print(
    "Precision:",
    f"{cv_B_LR['test_precision'].mean():.4f} ± "
    f"{cv_B_LR['test_precision'].std():.4f}"
    )

    print(
    "Recall   :",
    f"{cv_B_LR['test_recall'].mean():.4f} ± "
    f"{cv_B_LR['test_recall'].std():.4f}"
    )

    print(
    "F1 Score :",
    f"{cv_B_LR['test_f1'].mean():.4f} ± "
    f"{cv_B_LR['test_f1'].std():.4f}"
    )

    print(
    "ROC-AUC  :",
    f"{cv_B_LR['test_roc_auc'].mean():.4f} ± "
    f"{cv_B_LR['test_roc_auc'].std():.4f}"
    )


    # -----------------------------------------------------
    # 4. MODEL B - TUNED RANDOM FOREST
    # -----------------------------------------------------

    cv_B_RF = cross_validate(
    best_rf_model_B,
    X_train_B,
    y_train,
    cv=cv,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]
    )

    print("\n--- Model B - Tuned Random Forest ---")

    print(
    "Accuracy :",
    f"{cv_B_RF['test_accuracy'].mean():.4f} ± "
    f"{cv_B_RF['test_accuracy'].std():.4f}"
    )

    print(
    "Precision:",
    f"{cv_B_RF['test_precision'].mean():.4f} ± "
    f"{cv_B_RF['test_precision'].std():.4f}"
    )

    print(
    "Recall   :",
    f"{cv_B_RF['test_recall'].mean():.4f} ± "
    f"{cv_B_RF['test_recall'].std():.4f}"
    )

    print(
    "F1 Score :",
    f"{cv_B_RF['test_f1'].mean():.4f} ± "
    f"{cv_B_RF['test_f1'].std():.4f}"
    )

    print(
    "ROC-AUC  :",
    f"{cv_B_RF['test_roc_auc'].mean():.4f} ± "
    f"{cv_B_RF['test_roc_auc'].std():.4f}"
    )


    

    # -----------------------------------------------------
    # 5. MODEL B - TUNED SVM
    # -----------------------------------------------------

    cv_B_SVM = cross_validate(
    svm_grid_search_B.best_estimator_,
    X_train_B,
    y_train,
    cv=cv,
    scoring=[
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
    ]
    )

    print("\n--- Model B - Tuned SVM ---")

    print(
    "Accuracy :",
    f"{cv_B_SVM['test_accuracy'].mean():.4f} ± "
    f"{cv_B_SVM['test_accuracy'].std():.4f}"
    )

    print(
    "Precision:",
    f"{cv_B_SVM['test_precision'].mean():.4f} ± "
    f"{cv_B_SVM['test_precision'].std():.4f}"
    )

    print(
    "Recall   :",
    f"{cv_B_SVM['test_recall'].mean():.4f} ± "
    f"{cv_B_SVM['test_recall'].std():.4f}"
    )

    print(
    "F1 Score :",
    f"{cv_B_SVM['test_f1'].mean():.4f} ± "
    f"{cv_B_SVM['test_f1'].std():.4f}"
    )

    print(
    "ROC-AUC  :",
    f"{cv_B_SVM['test_roc_auc'].mean():.4f} ± "
    f"{cv_B_SVM['test_roc_auc'].std():.4f}"
    )


    print("\n")
    print("=" * 70)
    print("END OF FINAL CROSS-VALIDATION COMPARISON")
    print("=" * 70)

    #########################################################
    print("\n--- Processed Data Check ---")
    print(
        "NaN in training:",
        np.isnan(X_train_A_processed).sum()
    )

    print(
        "NaN in testing:",
        np.isnan(X_test_A_processed).sum()
    )

    print(
        "Infinite values in training:",
        np.isinf(X_train_A_processed).sum()
    )

    print(
        "Infinite values in testing:",
        np.isinf(X_test_A_processed).sum()
    )

    print("\n--- Processed Feature Names ---")

    feature_names = preprocessor.get_feature_names_out()

    for i, name in enumerate(feature_names):
        print(i, ":", name)

    print("\nTotal processed features:", len(feature_names))

    print("\n--- After Preprocessing ---")
    print("Training shape:", X_train_A_processed.shape)
    print("Testing shape :", X_test_A_processed.shape)

    print("Dataset shape:", df.shape)

    print("\nModel A X shape:", X_A.shape)
    print("Model B X shape:", X_B.shape)
    print("y shape:", y.shape)

    print("\nModel A features:", len(MODEL_A_FEATURES))

    for feature in MODEL_A_FEATURES:
        print(feature)

    print("\nModel B features:", len(MODEL_B_FEATURES))

    for feature in MODEL_B_FEATURES:
        print(feature)

    print("\nTarget:", TARGET)

    print("\n--- Train/Test Split ---")
    print("Model A training:", X_train_A.shape)
    print("Model A testing :", X_test_A.shape)

    print("Model B training:", X_train_B.shape)
    print("Model B testing :", X_test_B.shape)

    print("Training target :", y_train.shape)
    print("Testing target  :", y_test.shape)

    print("\n--- Selected Feature Data Types ---")
    print(X_A.dtypes)

    print("\n--- Rows with missing selected values ---")
    print(
        df[
            df["AMH(ng/mL)"].isna() |
            df["Fast food (Y/N)"].isna()
        ][
            ["Patient File No.", "PCOS (Y/N)", "AMH(ng/mL)", "Fast food (Y/N)"]
        ]
    )