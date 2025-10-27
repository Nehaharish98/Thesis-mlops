import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _to_float(v: Any) -> Optional[float]:
    """Best-effort float conversion; returns None on failure or None-like."""
    if v is None:
        return None
    try:
        if isinstance(v, str) and v.strip() == "":
            return None
        return float(v)
    except Exception:
        return None


def _flatten_synt(prefix: str, synt: Dict[str, Any], row: Dict[str, Any]) -> None:
    """Flatten a synthetic stats dict (mean, median, percentiles, etc.) into row with a prefix."""
    if not isinstance(synt, dict):
        return
    for k in [
        "mean",
        "median",
        "min",
        "max",
        "std",
        "pctl5",
        "pctl25",
        "pctl75",
        "pctl95",
    ]:
        row[f"{prefix}_{k}"] = _to_float(synt.get(k))


def flatten_record(rec: Dict[str, Any], provider: str, file_path: str) -> Dict[str, Any]:
    """Flatten one experiment record into an ML-friendly row."""
    row: Dict[str, Any] = {
        "provider": provider,
        "source_path": file_path,
        # top-level identifiers
        "proto": rec.get("proto") or rec.get("protocol"),
        "exp_name": rec.get("exp_name"),
        "start_ts": rec.get("start") or rec.get("start_time"),
        "duration_s": _to_float(rec.get("duration")),
    }

    # Campaign context
    camp = rec.get("camp", {}) or {}
    row["region_snd"] = camp.get("region_snd")
    row["region_rcv"] = camp.get("region_rcv")
    row["size_snd"] = camp.get("size_snd")
    row["size_rcv"] = camp.get("size_rcv")
    if row.get("region_snd") and row.get("region_rcv"):
        row["region_pair"] = f"{row['region_snd']}->{row['region_rcv']}"
    else:
        row["region_pair"] = None

    # Sender/receiver metadata
    info_snd = rec.get("info_snd", {}) or {}
    info_rcv = rec.get("info_rcv", {}) or {}
    row["tool_snd"] = info_snd.get("tool")
    row["tool_rcv"] = info_rcv.get("tool")
    row["ip_sender"] = info_snd.get("ip_sender") or info_rcv.get("ip_sender")
    row["ip_receiver"] = info_snd.get("ip_receiver") or info_rcv.get("ip_receiver")
    row["port"] = info_snd.get("port") or info_rcv.get("port")
    row["granularity_snd"] = _to_float(info_snd.get("granularity"))
    row["granularity_rcv"] = _to_float(info_rcv.get("granularity"))
    row["target_bwd"] = _to_float(info_snd.get("target_bwd") or info_rcv.get("target_bwd"))

    # Results: throughput, delay, loss, jitter (route excluded for ML summary)
    results = rec.get("results", {}) or {}
    for metric in ["tput", "delay", "loss", "jitter"]:
        m = results.get(metric)
        if not isinstance(m, dict):
            continue
        synt = m.get("synt")
        det = m.get("detailed")
        _flatten_synt(metric, synt, row)
        # Count of time-series points (useful ML feature)
        if isinstance(det, dict):
            row[f"{metric}_n"] = len(det)
        elif isinstance(det, list):
            row[f"{metric}_n"] = len(det)
        else:
            row[f"{metric}_n"] = None

    return row


def discover_result_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*_results.json")]


def infer_provider(root: Path, file_path: Path) -> str:
    """Infer provider as the first path segment under root (e.g., AWS/Azure)."""
    try:
        rel = file_path.relative_to(root)
        parts = rel.parts
        if parts:
            return parts[0]
    except Exception:
        pass
    return "Unknown"


def build_dataset(input_dir: str) -> pd.DataFrame:
    root = Path(input_dir).resolve()
    files = discover_result_files(root)
    if not files:
        raise FileNotFoundError(f"No *_results.json files found under {root}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        provider = infer_provider(root, fp)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict) or not rec.get("results"):
                continue
            rows.append(flatten_record(rec, provider, str(fp)))

    if not rows:
        raise RuntimeError("Found result files but could not extract any records.")

    df = pd.DataFrame(rows)

    # Type cleanup and ordering
    num_cols = [
        c
        for c in df.columns
        if any(k in c for k in ["_mean", "_median", "_min", "_max", "_std", "_pctl", "_n", "duration_s"])
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Add derived helpers
    if "start_ts" in df.columns:
        df["start_dt"] = pd.to_datetime(df["start_ts"], unit="s", errors="coerce")

    # Column order: identifiers -> context -> metrics
    id_cols = [
        "provider",
        "proto",
        "exp_name",
        "start_ts",
        "start_dt",
        "duration_s",
        "region_snd",
        "region_rcv",
        "region_pair",
        "size_snd",
        "size_rcv",
        "tool_snd",
        "tool_rcv",
        "ip_sender",
        "ip_receiver",
        "port",
        "granularity_snd",
        "granularity_rcv",
        "target_bwd",
        "source_path",
    ]
    metric_cols = sorted([c for c in df.columns if c not in id_cols])
    df = df[[c for c in id_cols if c in df.columns] + metric_cols]

    return df


def save_dataset(df: pd.DataFrame, output_dir: str, fmt: str = "parquet") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cloud_network_performance_{ts}.{('parquet' if fmt == 'parquet' else 'csv')}"

    if fmt == "parquet":
        try:
            df.to_parquet(out_path, index=False)
        except Exception as e:
            # Fallback to CSV if parquet engine not available
            alt = out_dir / f"cloud_network_performance_{ts}.csv"
            df.to_csv(alt, index=False)
            return alt
        return out_path
    else:
        df.to_csv(out_path, index=False)
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Build ML-friendly dataset from PaperDataset results")
    parser.add_argument(
        "--input",
        "-i",
        default="data/raw/PaperDataset",
        help="Root directory of the PaperDataset",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/processed",
        help="Directory to write the output dataset",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (parquet preferred)",
    )

    args = parser.parse_args()

    df = build_dataset(args.input)
    out = save_dataset(df, args.output_dir, args.format)
    print(f"✅ Dataset built: {out}")
    print(f"   Rows: {len(df):,}  Columns: {len(df.columns):,}")
    # Quick peek of important columns
    keep = [
        "provider",
        "proto",
        "region_pair",
        "size_snd",
        "size_rcv",
        "tput_mean",
        "delay_mean",
        "loss_mean",
    ]
    cols = [c for c in keep if c in df.columns]
    if cols:
        print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

