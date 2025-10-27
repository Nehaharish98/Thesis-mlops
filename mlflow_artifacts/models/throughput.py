"""
Align throughput.py with the simpler, regression-focused pattern used in
throughput_regression.py: load processed Parquet, engineer pairs, build a
one-hot + scaling pipeline, fit RandomForestRegressor, and log to MLflow.
"""

# --- repo-root import bootstrap ---
from pathlib import Path
import sys, os
# Go up to repo root (two levels) so `src/` is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# -----------------------------------

import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "nmops-throughput-regression")


def _latest_processed(default_fallback: str = "data/processed/cloud_network_performance_20251009_133228.parquet") -> str:
    d = REPO_ROOT / "data" / "processed"
    cand = sorted(d.glob("cloud_network_performance_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cand:
        return str(cand[0])
    return default_fallback


def _build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"region_snd", "region_rcv"}.issubset(df.columns):
        df["region_pair"] = df["region_snd"].astype(str) + "->" + df["region_rcv"].astype(str)
    if {"size_snd", "size_rcv"}.issubset(df.columns):
        df["vm_size_pair"] = df["size_snd"].astype(str) + "->" + df["size_rcv"].astype(str)
    return df


def main():
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Prefer EXP_PARQUET if provided, else latest processed dataset
    path = os.environ.get("EXP_PARQUET") or _latest_processed()
    df = pd.read_parquet(path)
    print(f"📊 Loaded {len(df)} network records from {path}")

    # Standardize/clean minimal set
    if "tput_mean" not in df.columns and "tput_median" in df.columns:
        df["tput_mean"] = df["tput_median"]
    for k in ["tput_mean", "delay_mean", "loss_mean"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    if "proto" in df.columns:
        df["proto"] = df["proto"].astype(str).str.lower()
    # derive hour/dow
    if "start_dt" in df.columns:
        ts = pd.to_datetime(df["start_dt"], errors="coerce")
        df["hour"] = ts.dt.hour
        df["dow"] = ts.dt.dayofweek
    elif "start_ts" in df.columns:
        ts = pd.to_datetime(df["start_ts"], unit="s", errors="coerce")
        df["hour"] = ts.dt.hour
        df["dow"] = ts.dt.dayofweek

    df = _build_pairs(df)

    # Target and filtering
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["tput_mean"])  # ensure target present
    df = df[df["tput_mean"] >= 0]

    # Features similar to throughput_regression.py (adapted to available columns)
    cat_cols = [c for c in ["provider", "region_pair", "vm_size_pair", "proto"] if c in df.columns]
    num_cols = [c for c in ["delay_mean", "loss_mean", "hour", "dow"] if c in df.columns]
    features = cat_cols + num_cols
    if not features:
        raise RuntimeError("No usable features detected. Please ensure provider/region/vm/proto or KPI columns exist.")

    X = df[features]
    y = df["tput_mean"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[("pre", pre), ("rf", model)])

    with mlflow.start_run(run_name="throughput_rf"):
        mlflow.log_params({"n_estimators": 300, "random_state": 42})

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))

        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(f"Metrics → RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        print(f"Artifacts URI: {mlflow.get_artifact_uri()}")

    return pipe


if __name__ == "__main__":
    main()
