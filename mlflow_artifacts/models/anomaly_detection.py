
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.ml.utils import load_experiments, basic_clean_experiments, ensure_columns, make_provider_region_pairs

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "nmops-anomaly-detection")

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_experiments()
    df = make_provider_region_pairs(df)
    df = basic_clean_experiments(df)

    ensure_columns(df, ["tput_mean","delay_mean","loss_mean"])
    feats = ["tput_mean","delay_mean","loss_mean"]
    if "jitter_mean" in df.columns:
        feats.append("jitter_mean")

    X = df[feats].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(n_estimators=400, contamination="auto", random_state=42, n_jobs=-1))
    ])

    with mlflow.start_run(run_name="anomaly_iforest"):
        mlflow.log_params({"n_estimators": 400, "contamination": "auto"})

        pipe.fit(X)
        scores = -pipe["iforest"].score_samples(pipe["scaler"].transform(X))
        preds = pipe["iforest"].predict(pipe["scaler"].transform(X))

        frac_anom = float((preds == -1).mean())
        mlflow.log_metric("fraction_anomalous", frac_anom)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        prev = (pd.DataFrame({"score": scores, "pred": preds})
                  .assign(idx=np.arange(len(scores)))
                  .sort_values("score", ascending=False)
                  .head(50))
        prev.to_csv("anomaly_top50.csv", index=False)
        mlflow.log_artifact("anomaly_top50.csv")

        print(f"IsolationForest flagged {frac_anom*100:.2f}% of rows as anomalous.")
        print(f"Artifacts URI: {mlflow.get_artifact_uri()}")

if __name__ == "__main__":
    main()
