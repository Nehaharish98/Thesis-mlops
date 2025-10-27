# --- repo-root import bootstrap ---
from pathlib import Path
import sys
# Go up to repo root (two levels) so `src/` is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# -----------------------------------

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.ml.common import float_safe_input_example, log_sklearn_model

def _pick_latency_target(df):
    """Pick a latency target with variance."""
    candidates = ["info_rcv_duration", "duration"]
    for col in candidates:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            print(f"✅ Using '{col}' as latency target (unique values: {df[col].nunique()})")
            return df[col], col
    raise ValueError("No suitable latency target found.")

def forecast_latency():
    """Forecast network latency using GradientBoostingRegressor."""

    # -------------------------
    # Load dataset
    # -------------------------
    df = pd.read_parquet("data/processed/cloud_network_performance_20251009_133228.parquet")
    print(f"📊 Loaded {len(df)} network records")

    # -------------------------
    # Target selection
    # -------------------------
    # Prefer measured delay from processed dataset
    if "delay_mean" in df.columns and df["delay_mean"].nunique(dropna=True) > 1:
        y, target_col = df["delay_mean"], "delay_mean"
    else:
        y, target_col = _pick_latency_target(df)

    # Drop rows with missing target
    mask = y.notna() & np.isfinite(y.values)
    if (~mask).any():
        print(f"⚠️ Dropping {int((~mask).sum())} rows with NaN/inf target '{target_col}'")
    df = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # -------------------------
    # Features
    # -------------------------
    numeric_features = ["duration_s", "granularity_snd", "granularity_rcv", "target_bwd"]
    categorical_features = ["provider", "proto", "region_snd", "region_rcv", "size_snd", "size_rcv", "port"]

    # Keep only existing cols
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    X = df[numeric_features + categorical_features].copy()

    # -------------------------
    # Preprocessing
    # -------------------------
    numeric_pre = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pre = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pre, numeric_features),
            ("cat", categorical_pre, categorical_features),
        ],
        remainder="drop"
    )

    # -------------------------
    # Train-test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # MLflow setup
    # -------------------------
    mlflow.set_experiment("latency_forecasting_experiment")

    with mlflow.start_run(run_name="latency_forecasting_real"):
        # Model
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

        # Log parameters
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("numeric_features", ",".join(numeric_features))
        mlflow.log_param("categorical_features", ",".join(categorical_features))

        # Log metrics
        mlflow.log_metrics({
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        })

        # Log model with example
        log_sklearn_model(
            pipe,
            "latency_forecasting_model",
            input_example=float_safe_input_example(X_test.iloc[:32])
        )

        # -------------------------
        # Output
        # -------------------------
        print("🔮 Latency Forecasting Results:")
        print(f"   Target: {target_col}")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAPE: {mape:.1f}%")

        # Feature importance
        try:
            model = pipe.named_steps["model"]
            importances = model.feature_importances_

            cat_names = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_features)
            feature_names = numeric_features + list(cat_names)

            feat_imp = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False)

            print("\n📈 Key Latency Drivers:")
            print(feat_imp.head(10).to_string(index=False))

        except Exception as e:
            print(f"(Could not compute feature importance: {e})")

        return pipe, (mae, rmse, mape)

if __name__ == "__main__":
    forecast_latency()
