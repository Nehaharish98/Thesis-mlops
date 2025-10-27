
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score

from src.ml.utils import load_experiments, basic_clean_experiments, make_provider_region_pairs, ensure_columns, define_slo_tiers

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT", "nmops-performance-classification")

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_experiments()
    df = make_provider_region_pairs(df)
    df = basic_clean_experiments(df)

    ensure_columns(df, ["tput_mean","delay_mean","loss_mean"])

    df["label"] = df.apply(define_slo_tiers, axis=1)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    cat_cols = [c for c in ["provider_pair","region_pair","vm_size_pair","proto"] if c in df.columns]
    num_cols = [c for c in ["tput_mean","delay_mean","loss_mean","jitter_mean","hour","dow"] if c in df.columns]

    X = df[cat_cols + num_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ],
        remainder="drop"
    )

    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[("pre", pre), ("rf", clf)])

    with mlflow.start_run(run_name="perf_classifier_rf"):
        mlflow.log_params({"n_estimators": 400, "class_weight": "balanced"})

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="macro")
        bacc = balanced_accuracy_score(y_test, y_pred)

        mlflow.log_metric("f1_macro", float(f1))
        mlflow.log_metric("balanced_accuracy", float(bacc))

        report = classification_report(y_test, y_pred, digits=3)
        with open("classification_report.txt","w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(report)
        print(f"Artifacts URI: {mlflow.get_artifact_uri()}")

if __name__ == "__main__":
    main()
