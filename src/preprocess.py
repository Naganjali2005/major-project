"""
preprocess.py  —  Backward-compatibility shim
====================================================================
The original monolithic preprocess.py has been replaced by structured
modules under src/:

    config.py         - feature lists, constants, hyperparameters
    data_loader.py    - load_data(), load_split(), audit_data()
    preprocessing.py  - build_preprocessor(), make_pipeline(), cv

This shim re-exports everything that app.py and other code previously
imported from preprocess.py, so nothing breaks.
"""

# Re-export all public names from the new structured modules
from config import (          # noqa: F401
    DATA_PATH, TARGET,
    MODEL_A_FEATURES, MODEL_B_FEATURES,
    BINARY_FEATURES, CYCLE_FEATURE,
    NUMERIC_FEATURES, NUMERIC_FEATURES_B,
)

from data_loader import load_data   # noqa: F401

from preprocessing import (   # noqa: F401
    build_preprocessor,
    make_pipeline,
    make_clf,
    cv,
    preprocessor,
    preprocessor_B,
    rf_preprocessor,
    rf_preprocessor_B,
)
