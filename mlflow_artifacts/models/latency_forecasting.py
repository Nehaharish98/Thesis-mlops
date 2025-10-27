
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.ml.utils import load_timeseries

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "nmops-latency-forecasting")

def build_supervised_lags(df, lags=12, horizon=1):
    ts = df.sort_values("ts").reset_index(drop=True)
    data = {"y": ts["value"].shift(-horizon)}
    for i in range(1, lags+1):
        data[f"lag_{i}"] = ts["value"].shift(i)
    out = pd.DataFrame(data)
    out["ts"] = ts["ts"]
    out = out.dropna().reset_index(drop=True)
    return out

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    ts = load_timeseries()
    ts = ts[ts["metric"].str.lower().eq("delay")].copy()
    if ts.empty:
        raise RuntimeError("No 'delay' metric found in timeseries parquet.")

    grp_sizes = ts.groupby("exp_id").size().sort_values(ascending=False)
    top_exp = grp_sizes.index[0]
    series = ts[ts["exp_id"] == top_exp][["ts","value"]].copy()
    series["ts"] = pd.to_datetime(series["ts"], errors="coerce")
    series = series.dropna()

    sup = build_supervised_lags(series, lags=12, horizon=1)

    X = sup[[c for c in sup.columns if c.startswith("lag_")]]
    y = sup["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    with mlflow.start_run(run_name="latency_rf_lags"):
        mlflow.log_params({"lags": 12, "horizon": 1, "n_estimators": 300})

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("rmse", float(rmse))
        mlflow.sklearn.log_model(model, artifact_path="model")

        prev = pd.DataFrame({"ts": sup.loc[X_test.index, "ts"], "y_true": y_test, "y_pred": y_pred})
        prev.to_csv("forecast_preview.csv", index=False)
        mlflow.log_artifact("forecast_preview.csv")

        print(f"Latency RF → MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        print(f"Artifacts URI: {mlflow.get_artifact_uri()}")

if __name__ == "__main__":
    main()
