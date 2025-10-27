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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from src.ml.common import float_safe_input_example, log_sklearn_model

def classify_network_performance():
    """Classify network performance into categories like Excellent/Good/Fair/Poor."""

    # -------------------------
    # Load dataset
    # -------------------------
    df = pd.read_parquet("data/processed/cloud_network_performance_20251009_133228.parquet")
    print(f"📊 Loaded {len(df)} network records")

    # -------------------------
    # Create performance labels
    # -------------------------
    def _granularity(row):
        g1 = row.get("granularity_snd")
        g2 = row.get("granularity_rcv")
        vals = [v for v in [g1, g2] if pd.notna(v)]
        return min(vals) if vals else np.nan

    def assign_performance_class(row):
        # Lower delay is better; thresholds adapted to dataset stats
        delay = row.get("delay_mean", np.nan)
        duration_score = 4 if pd.notna(delay) and delay < 100 else 3 if pd.notna(delay) and delay < 200 else 2 if pd.notna(delay) and delay < 300 else 1

        bw = row.get("target_bwd", np.nan)
        bandwidth_score = 4 if pd.notna(bw) and bw > 1.5e9 else 3 if pd.notna(bw) and bw > 1e9 else 2 if pd.notna(bw) and bw > 5e8 else 1

        gran = _granularity(row)
        granularity_score = 4 if pd.notna(gran) and gran <= 0.5 else 3 if pd.notna(gran) and gran <= 0.8 else 2 if pd.notna(gran) and gran <= 1.0 else 1

        total_score = (duration_score + bandwidth_score + granularity_score) / 3
        if total_score >= 3.5:
            return "Excellent"
        elif total_score >= 2.5:
            return "Good"
        elif total_score >= 1.5:
            return "Fair"
        else:
            return "Poor"

    df = df.dropna(subset=["target_bwd"])  # allow delay/granularity to be missing; impute in pipeline
    df["performance_class"] = df.apply(assign_performance_class, axis=1)

    print("\n📈 Performance Distribution:")
    print(df["performance_class"].value_counts())

    # -------------------------
    # Features and Target
    # -------------------------
    numeric_features = ["duration_s", "delay_mean", "granularity_snd", "granularity_rcv", "target_bwd"]
    categorical_features = ["provider", "proto", "size_snd", "size_rcv", "region_snd", "region_rcv", "port"]

    # Defensive: keep only available columns
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    X = df[numeric_features + categorical_features]
    y = df["performance_class"].copy()

    # Handle rare classes to avoid stratify errors
    vc = y.value_counts()
    rare = vc[vc < 2].index.tolist()
    if rare:
        print(f"⚠️ Merging rare classes {rare} into 'Good' to enable stratification")
        y = y.replace({c: "Good" for c in rare})
        vc = y.value_counts()
    stratify_arg = y if (vc.min() >= 2 and len(vc) >= 2) else None

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
        ]
    )

    # -------------------------
    # Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    # -------------------------
    # MLflow Logging
    # -------------------------
    mlflow.set_experiment("performance_classification_experiment")

    with mlflow.start_run(run_name="performance_classification_real"):
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("accuracy", accuracy)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)

        log_sklearn_model(
            pipe,
            "performance_classifier_model",
            input_example=float_safe_input_example(X_test.iloc[:32])
        )

        print("\n🎯 Performance Classification Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        try:
            model = pipe.named_steps["model"]
            importances = model.feature_importances_
            cat_names = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_features)
            feat_names = numeric_features + list(cat_names)

            feat_imp = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
            print("\n📈 Most Important Features:")
            print(feat_imp.head(10).to_string(index=False))
        except Exception as e:
            print(f"(Could not extract feature importances: {e})")

        return pipe, accuracy, report

if __name__ == "__main__":
    classify_network_performance()
