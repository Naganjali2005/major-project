"""
data_loader.py
====================================================================
Data loading and train/test splitting for the PCOS Screening Aid.

Provides:
  - load_data()      : reads and cleans the Excel dataset
  - load_split()     : returns the canonical 80/20 stratified split
  - audit_data()     : prints a quick dataset overview (replaces data_audit.py)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH, TARGET,
    MODEL_A_FEATURES, MODEL_B_FEATURES,
    RANDOM_STATE,
)


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
def load_data():
    """Load and clean the PCOS dataset from Excel."""
    df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    # AMH has mixed types in some exports – coerce to numeric
    df["AMH(ng/mL)"] = pd.to_numeric(df["AMH(ng/mL)"], errors="coerce")

    return df


# ------------------------------------------------------------------
# Split
# ------------------------------------------------------------------
def load_split():
    """
    Canonical 80/20 stratified train/test split (random_state=42).

    Returns a dict with keys:
        df, X_A, X_B, y,
        X_train_A, X_test_A,
        X_train_B, X_test_B,
        y_train, y_test
    """
    df = load_data()
    X_A = df[MODEL_A_FEATURES]
    X_B = df[MODEL_B_FEATURES]
    y = df[TARGET]

    X_train_A, X_test_A, y_train, y_test = train_test_split(
        X_A, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Use exactly the same patients for Model B
    X_train_B = X_B.loc[X_train_A.index]
    X_test_B = X_B.loc[X_test_A.index]

    print(f"Dataset: {df.shape}  | train {X_train_A.shape[0]} / "
          f"test {X_test_A.shape[0]}  | positives in test: "
          f"{int(y_test.sum())}")

    return dict(
        df=df,
        X_A=X_A, X_B=X_B, y=y,
        X_train_A=X_train_A, X_test_A=X_test_A,
        X_train_B=X_train_B, X_test_B=X_test_B,
        y_train=y_train, y_test=y_test,
    )


# ------------------------------------------------------------------
# Audit (replaces the old standalone data_audit.py)
# ------------------------------------------------------------------
def audit_data():
    """Print a quick dataset overview: shape, types, missing values,
    duplicates, target distribution, and key column value counts."""
    df = load_data()

    sep = "=" * 60

    print(sep)
    print("DATASET OVERVIEW")
    print(sep)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print(f"\n{sep}")
    print("COLUMN NAMES")
    print(sep)
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    print(f"\n{sep}")
    print("DATA TYPES")
    print(sep)
    print(df.dtypes)

    print(f"\n{sep}")
    print("MISSING VALUES")
    print(sep)
    print(df.isnull().sum())

    print(f"\n{sep}")
    print("DUPLICATE ROWS")
    print(sep)
    print("Number of duplicate rows:", df.duplicated().sum())

    print(f"\n{sep}")
    print("TARGET DISTRIBUTION")
    print(sep)
    print(df[TARGET].value_counts())
    print(df[TARGET].value_counts(normalize=True) * 100)

    columns_to_check = [
        "AMH(ng/mL)", "Fast food (Y/N)", "Cycle(R/I)",
        "Blood Group", "Pregnant(Y/N)",
        "Weight gain(Y/N)", "hair growth(Y/N)",
        "Skin darkening (Y/N)", "Hair loss(Y/N)",
        "Pimples(Y/N)", "Reg.Exercise(Y/N)",
    ]
    print(f"\n{sep}")
    print("KEY COLUMN VALUE COUNTS")
    print(sep)
    for col in columns_to_check:
        if col in df.columns:
            print(f"\n{col}")
            print(df[col].value_counts(dropna=False))

    print("\nRows where Cycle(R/I) is not 2 or 4:")
    print(df[~df["Cycle(R/I)"].isin([2, 4])][
        ["Patient File No.", "PCOS (Y/N)", "Age (yrs)", "Cycle(R/I)"]
    ])


if __name__ == "__main__":
    audit_data()
