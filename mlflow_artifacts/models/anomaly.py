# -*- coding: utf-8 -*-
"""
Anomaly Detection (IsolationForest + DBSCAN) with MLflow best practices:
- Experiment tracking (params, metrics, tags)
- Dataset lineage (hash, shape, columns) & input logging
- Model logging with signature & input example
- Model Registry registration and (optional) stage transition
- Reproducibility: seeds, code/package/env info
"""

# --- repo-root import bootstrap ---
from pathlib import Path
import sys, os, json, hashlib, datetime, subprocess, warnings
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# -----------------------------------

import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from src.ml.common import float_safe_input_example, log_sklearn_model  # keep your helpers

# ---------- Defaults ----------
PARQUET_PATH = os.environ.get("ANOMALY_PARQUET",
                               "data/processed/cloud_network_performance_20251009_133228.parquet")
EXPERIMENT = os.environ.get("ANOMALY_EXPERIMENT", "anomaly_detection_experiment")
REGISTERED_MODEL_NAME = os.environ.get("ANOMALY_REGISTERED_NAME", "anomaly_detector_isolation_forest")
TRANSITION_STAGE = os.environ.get("ANOMALY_TRANSITION_STAGE", "")  # "", "Staging", or "Production"

# Target anomaly windows & sweeps
ISO_CONTAM_CAND = [0.005, 0.01, 0.015, 0.02, 0.03]
DBSCAN_EPS_CAND = [0.3, 0.5, 0.7, 1.0]
DBSCAN_MS_CAND = [5, 10, 20]
TARGET_MIN, TARGET_MAX = 0.005, 0.03

warnings.filterwarnings("ignore", category=FutureWarning)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
    except Exception:
        return "unknown"


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    # robust, order-insensitive-ish fingerprint (schema + simple content hash)
    schema = json.dumps({"cols": list(df.columns), "dtypes": {c: str(t) for c, t in df.dtypes.items()}},
                        sort_keys=True).encode()
    # hash only first N rows for speed but still informative
    head_bytes = df.head(10000).to_csv(index=False).encode()
    return hashlib.sha256(schema + head_bytes).hexdigest()


def _select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    num_candidates = [
        "duration_s", "granularity_snd", "granularity_rcv", "target_bwd",
        "tput_mean", "tput_pctl95", "tput_pctl5",
        "delay_mean", "loss_mean",
        "tput_n", "delay_n", "loss_n"
    ]
    cols = [c for c in num_candidates if c in df.columns]
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop all-NaN columns
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X.drop(columns=all_nan_cols, inplace=True)
        cols = [c for c in cols if c not in all_nan_cols]

    # Median impute then zeros as last resort
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)
    return X


def _tune_isolation_forest(Xz: np.ndarray, rng_seed: int = 42):
    best_iso, best_rate, best_c = None, None, None
    for c in ISO_CONTAM_CAND:
        iso = IsolationForest(contamination=c, random_state=rng_seed)
        y = iso.fit_predict(Xz)
        rate = float((y == -1).mean())
        if (best_iso is None) or (TARGET_MIN <= rate <= TARGET_MAX and rate > (best_rate or 0)) or \
           (best_rate is not None and abs(rate - TARGET_MIN) < abs(best_rate - TARGET_MIN)):
            best_iso, best_rate, best_c = iso, rate, c
    return best_iso, best_rate, best_c


def _sweep_dbscan(Xz: np.ndarray):
    res = []
    for eps in DBSCAN_EPS_CAND:
        for ms in DBSCAN_MS_CAND:
            y = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xz)
            rate = float((y == -1).mean())
            res.append((eps, ms, rate))
    # prefer rates in (0, 5%], closest to 1%
    eps_best, ms_best, rate_best = sorted(
        res, key=lambda t: (0 if 0 < t[2] <= 0.05 else 1, abs(t[2] - 0.01))
    )[0]
    model = DBSCAN(eps=eps_best, min_samples=ms_best).fit(Xz)
    return model, eps_best, ms_best, float((model.labels_ == -1).mean())


def detect_network_anomalies(
    parquet_path: str = PARQUET_PATH,
    experiment: str = EXPERIMENT,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    transition_stage: str = TRANSITION_STAGE
):
    # ---------- Load ----------
    df = pd.read_parquet(parquet_path)
    print(f"📊 Loaded {len(df)} network records from {parquet_path}")

    # ---------- Features ----------
    Xnum = _select_numeric_features(df)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(Xnum)
    if np.isnan(Xz).any() or np.isinf(Xz).any():
        Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- MLflow setup ----------
    mlflow.set_experiment(experiment)
    run_name = f"anomaly_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tags = {
        "project": "network-monitoring-mlops",
        "task": "anomaly-detection",
        "algo.primary": "IsolationForest",
        "algo.secondary": "DBSCAN",
        "source.repo_root": str(REPO_ROOT),
        "git.commit": _git_commit(),
        "data.file": Path(parquet_path).name,
    }

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        run_id = run.info.run_id

        # ---------- Dataset lineage ----------
        ds_fp = _dataset_fingerprint(df)
        mlflow.log_params({
            "dataset_rows": int(len(df)),
            "dataset_cols": int(len(df.columns)),
            "features_used": ",".join(list(Xnum.columns)),
        })
        mlflow.set_tags({
            "dataset.hash": ds_fp,
            "dataset.path": parquet_path,
        })
        # Log the input dataset sample for reproducibility
        sample_preview = df.head(200)
        # MLflow >=2.4: log_input; keep a safe fallback
        try:
            mlflow.log_input(mlflow.data.from_pandas(sample_preview, source=parquet_path), context="inference")
        except Exception:
            sample_preview.to_parquet("dataset_sample.parquet", index=False)
            mlflow.log_artifact("dataset_sample.parquet", artifact_path="data_sample")

        # ---------- Params (sweeps) ----------
        mlflow.log_params({
            "iso.contam_candidates": json.dumps(ISO_CONTAM_CAND),
            "dbscan.eps_candidates": json.dumps(DBSCAN_EPS_CAND),
            "dbscan.min_samples_candidates": json.dumps(DBSCAN_MS_CAND),
            "target_rate_min": TARGET_MIN,
            "target_rate_max": TARGET_MAX,
            "scaler": "StandardScaler",
            "random_state": 42,
        })

        # ---------- Tune / fit ----------
        best_iso, best_rate, best_c = _tune_isolation_forest(Xz, rng_seed=42)
        # Log the sweep metrics
        for c in ISO_CONTAM_CAND:
            # re-fit quickly to get per-c point (cheap)
            tmp = IsolationForest(contamination=c, random_state=42).fit(Xz)
            mlflow.log_metric(f"iso.rate_c_{c}", float((tmp.predict(Xz) == -1).mean()))
        mlflow.log_param("iso.best_contamination", best_c)
        mlflow.log_metric("iso.rate_best", float(best_rate))

        db_model, eps_best, ms_best, db_rate_best = _sweep_dbscan(Xz)
        mlflow.log_param("dbscan.best_eps", eps_best)
        mlflow.log_param("dbscan.best_min_samples", ms_best)
        mlflow.log_metric("dbscan.rate_best", float(db_rate_best))

        y_iso = best_iso.fit_predict(Xz)
        y_db = np.where(db_model.labels_ == -1, -1, 1)
        combined = ((y_iso == -1) | (y_db == -1)).astype(int)

        n = len(df)
        n_iso = int((y_iso == -1).sum())
        n_db = int((y_db == -1).sum())
        n_comb = int(combined.sum())
        mlflow.log_metrics({
            "combined.rate": float(n_comb / n),
            "combined.count": int(n_comb),
            "iso.count": n_iso,
            "dbscan.count": n_db,
        })

        # ---------- Explain/inspect (quick distributions) ----------
        anomalies_df = df.copy()
        anomalies_df["anomaly_combined"] = combined
        anomalies = anomalies_df[anomalies_df["anomaly_combined"] == 1]
        if not anomalies.empty:
            # simple group counts as artifacts
            summary = {}
            for col in ["provider", "proto", "region_snd", "region_rcv"]:
                if col in anomalies.columns:
                    summary[col] = anomalies[col].value_counts().head(20).to_dict()
            Path("artifacts").mkdir(exist_ok=True)
            with open("artifacts/anomaly_group_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_artifact("artifacts/anomaly_group_summary.json", artifact_path="analysis")

            # store a small anomalies preview
            cols = [c for c in [
                "provider", "proto", "region_snd", "region_rcv",
                "duration_s", "delay_mean", "tput_mean", "loss_mean", "target_bwd",
                "anomaly_combined"
            ] if c in anomalies.columns]
            anomalies[cols].head(2000).to_parquet("artifacts/anomalies_preview.parquet", index=False)
            mlflow.log_artifact("artifacts/anomalies_preview.parquet", artifact_path="analysis")

        # ---------- Model logging (primary: IsolationForest) ----------
        # Build a pipeline-like dict with scaler info for clarity (you can swap to sklearn Pipeline if you prefer)
        # Signature uses numeric features -> anomaly label {-1, 1}
        input_example = float_safe_input_example(Xnum.iloc[:32])
        signature = infer_signature(Xnum.head(50), pd.Series([-1, 1], name="prediction").head(2))

        # Option A: use your helper (kept for compatibility)
        # log_sklearn_model(best_iso, "isolation_forest_model", input_example=input_example)

        # Option B: direct MLflow logging with registry
        mlflow.sklearn.log_model(
            sk_model=best_iso,
            artifact_path="isolation_forest_model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "scikit-learn>=1.2.0",
                "mlflow>=2.5.0",
                "pandas",
                "numpy"
            ],
            metadata={
                "scaler": "StandardScaler (fitted separately)",
                "features": list(Xnum.columns),
            }
        )

        # Also save & log the fitted scaler so you can wrap at serving time
        import joblib
        Path("artifacts").mkdir(exist_ok=True)
        joblib.dump(scaler, "artifacts/standard_scaler.joblib")
        mlflow.log_artifact("artifacts/standard_scaler.joblib", artifact_path="preprocessing")

        print("🚨 Anomaly Detection (tuned):")
        print(f"   Isolation Forest: {n_iso} ({100*n_iso/n:.2f}%)  [contam={best_c}]")
        print(f"   DBSCAN:          {n_db} ({100*n_db/n:.2f}%)  [eps={eps_best}, min_samples={ms_best}]")
        print(f"   Combined:        {n_comb} ({100*n_comb/n:.2f}%)")

        # ---------- Model Registry: optional stage transition ----------
        if transition_stage in {"Staging", "Production"}:
            client = MlflowClient()
            # Find the latest version that came from this run
            mv = None
            for v in client.get_latest_versions(registered_model_name):
                if v.run_id == run_id:
                    mv = v
                    break
            if mv is None:
                # Fallback: get last created version
                versions = client.search_model_versions(f"name='{registered_model_name}'")
                versions = sorted(versions, key=lambda m: int(m.version), reverse=True)
                if versions:
                    mv = versions[0]
            if mv is not None:
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=mv.version,
                    stage=transition_stage,
                    archive_existing_versions=False
                )
                print(f"✅ Transitioned {registered_model_name} v{mv.version} -> {transition_stage}")

        return {
            "run_id": run_id,
            "registered_model": registered_model_name,
            "iso_best_contamination": best_c,
            "iso_rate": float(best_rate),
            "dbscan_eps": eps_best,
            "dbscan_min_samples": ms_best,
            "dbscan_rate": float(db_rate_best),
            "combined_rate": float(n_comb / n),
            "dataset_hash": ds_fp,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Network anomaly detection with MLflow tracking & registry")
    parser.add_argument("--parquet", default=PARQUET_PATH, help="Path to parquet dataset")
    parser.add_argument("--experiment", default=EXPERIMENT, help="MLflow experiment name")
    parser.add_argument("--register", default=REGISTERED_MODEL_NAME,
                        help="MLflow Registered Model name (Model Registry)")
    parser.add_argument("--stage", default=TRANSITION_STAGE,
                        help="Optional stage to transition: '', 'Staging', or 'Production'")
    args = parser.parse_args()
    detect_network_anomalies(args.parquet, args.experiment, args.register, args.stage)
