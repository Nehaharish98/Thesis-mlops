# src/ml/common.py
from __future__ import annotations
import inspect
from typing import List, Tuple

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlflow.models.signature import infer_signature  # NEW

def pick_nonconstant_target(df: pd.DataFrame, candidates: List[str]) -> Tuple[pd.Series, str]:
    for col in candidates:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            print(f"✅ Using '{col}' as target (unique values: {df[col].nunique()})")
            return df[col], col
    raise ValueError(f"No non-constant target among: {candidates}")

def make_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric_features), ("cat", cat, categorical_features)])

def safe_cat_names(preprocessor: ColumnTransformer, cat_cols: List[str]) -> List[str]:
    enc = preprocessor.named_transformers_["cat"]
    if isinstance(enc, Pipeline):
        enc = enc.named_steps.get("encoder", enc)
    try:
        return list(enc.get_feature_names_out(cat_cols))
    except AttributeError:
        return list(enc.get_feature_names(cat_cols))

def summarise_target(y: pd.Series, name: str):
    print(f"\n🧪 Target '{name}' summary:")
    print("  nunique:", y.nunique(dropna=True))
    print("  describe:\n", y.describe())

def relative_errors(y_true: np.ndarray, y_pred: np.ndarray):
    eps = 1e-8
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
    nrmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()) / (np.mean(np.abs(y_true)) + eps))
    return mape, nrmse

# ===== NEW: MLflow signature helpers =====
def float_safe_input_example(df: pd.DataFrame) -> pd.DataFrame:
    """Cast integer columns to float64 to avoid MLflow schema enforcement issues."""
    ex = df.copy()
    int_cols = ex.select_dtypes(include=["int", "int64", "int32"]).columns
    if len(int_cols):
        ex[int_cols] = ex[int_cols].astype("float64")
    return ex

def infer_model_signature(pipe_or_model, X_sample: pd.DataFrame):
    """
    Build a realistic signature from a sample batch.
    Pass ~100-200 rows of the *post-split* features (not transformed).
    """
    Xs = float_safe_input_example(X_sample.copy())
    # Try to get predictions for signature inference; if not possible, fall back to input-only sig
    try:
        yp = pipe_or_model.predict(Xs)
        return infer_signature(Xs, yp)
    except Exception:
        return infer_signature(Xs)


def log_sklearn_model(model, name: str, **kwargs):
    """Compat helper that supports both `artifact_path` and newer `name` kwarg."""
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    key = "name" if "name" in params else "artifact_path"
    log_kwargs = dict(kwargs)
    log_kwargs[key] = name
    mlflow.sklearn.log_model(model, **log_kwargs)
