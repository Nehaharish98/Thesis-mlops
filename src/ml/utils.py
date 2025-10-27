
import os
import pandas as pd
import numpy as np
from pathlib import Path

DEFAULT_EXPERIMENTS = "data/processed/cloud_network_performance_20251009_133228.parquet"
DEFAULT_TIMESERIES  = "data/processed/cloud_network_performance_20251009_133228.parquet"

def load_experiments(path: str = None) -> pd.DataFrame:
    path = path or os.environ.get("EXP_PARQUET", DEFAULT_EXPERIMENTS)
    if not Path(path).exists():
        raise FileNotFoundError(f"Experiments parquet not found: {path}")
    return pd.read_parquet(path)

def load_timeseries(path: str = None) -> pd.DataFrame:
    path = path or os.environ.get("TS_PARQUET", DEFAULT_TIMESERIES)
    if not Path(path).exists():
        raise FileNotFoundError(f"Timeseries parquet not found: {path}")
    return pd.read_parquet(path)

def ensure_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return True

def make_provider_region_pairs(df: pd.DataFrame) -> pd.DataFrame:
    def _pair(a, b):
        a = str(a) if a is not None else "NA"
        b = str(b) if b is not None else "NA"
        return f"{a}->{b}"
    if {"camp.region_snd","camp.region_rcv"}.issubset(df.columns):
        df["region_pair"] = [ _pair(a,b) for a,b in zip(df["camp.region_snd"], df["camp.region_rcv"]) ]
    if {"camp.size_snd","camp.size_rcv"}.issubset(df.columns):
        df["vm_size_pair"] = [ _pair(a,b) for a,b in zip(df["camp.size_snd"], df["camp.size_rcv"]) ]
    if {"camp.provider_snd","camp.provider_rcv"}.issubset(df.columns):
        df["provider_pair"] = [ _pair(a,b) for a,b in zip(df["camp.provider_snd"], df["camp.provider_rcv"]) ]
    return df

def basic_clean_experiments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["tput_mean_bps","tput_mean","tput.synt.mean","tput_mean_mbps"]:
        if col in df.columns:
            df["tput_mean"] = df[col]
            break
    for col in ["delay_mean_ms","delay_mean","delay.synt.mean"]:
        if col in df.columns:
            df["delay_mean"] = df[col]
            break
    for col in ["loss_mean_pct","loss_mean","loss.synt.mean"]:
        if col in df.columns:
            df["loss_mean"] = df[col]
            break
    for col in ["jitter_mean_ms","jitter_mean","jitter.synt.mean"]:
        if col in df.columns:
            df["jitter_mean"] = df[col]
            break
    for k in ["tput_mean","delay_mean","loss_mean","jitter_mean"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    if "proto" in df.columns:
        df["proto"] = df["proto"].astype(str).str.lower()
    for cand in ["start","start_ts","info_snd.start_time","start_time"]:
        if cand in df.columns:
            ts = pd.to_datetime(df[cand], errors="coerce", unit="s")
            if ts.notna().any():
                df["start_ts"] = ts
                df["hour"] = df["start_ts"].dt.hour
                df["dow"] = df["start_ts"].dt.dayofweek
            break
    return df

def define_slo_tiers(row, dl_ms=150.0, tp_mbps=100.0, loss_pct=1.0):
    dl = row.get("delay_mean", np.nan)
    tp = row.get("tput_mean", np.nan)
    ls = row.get("loss_mean", np.nan)
    if pd.isna(dl) or pd.isna(tp) or pd.isna(ls):
        return np.nan
    if (dl < 0.5*dl_ms) and (tp >= 1.5*tp_mbps) and (ls <= 0.25*loss_pct):
        return 3
    if (dl <= dl_ms) and (tp >= tp_mbps) and (ls <= loss_pct):
        return 2
    if (dl <= 1.5*dl_ms) and (tp >= 0.5*tp_mbps) and (ls <= 2.0*loss_pct):
        return 1
    return 0
