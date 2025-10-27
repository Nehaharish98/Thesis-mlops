
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ml.utils import load_experiments, basic_clean_experiments, make_provider_region_pairs, ensure_columns

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "nmops-throughput-regression")

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_experiments()
    df = make_provider_region_pairs(df)
    df = basic_clean_experiments(df)

    ensure_columns(df, ["tput_mean","proto"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["tput_mean"])
    df = df[(df["tput_mean"] >= 0)]

    cat_cols = [c for c in ["provider_pair","region_pair","vm_size_pair","proto"] if c in df.columns]
    num_cols = [c for c in ["delay_mean","loss_mean","jitter_mean","hour","dow"] if c in df.columns]

    features = cat_cols + num_cols
    if not features:
        raise RuntimeError("No usable features detected. Please ensure provider/region/vm/proto or KPI columns exist.")

    X = df[features]
    y = df["tput_mean"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ],
        remainder="drop"
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("pre", pre), ("rf", model)])

    with mlflow.start_run(run_name="throughput_rf"):
        params = {"n_estimators": 300, "max_depth": None, "random_state": 42}
        mlflow.log_params(params)

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

if __name__ == "__main__":
    main()
