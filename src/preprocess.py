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