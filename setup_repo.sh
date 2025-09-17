#!/bin/bash
set -e

# Repo name
REPO_NAME="interdc-mlops"

# Create base structure
mkdir -p $REPO_NAME/{data/raw,artifacts/eda,src/{data,eda},notebooks}

# Create requirements (if you want pip fallback)
cat > $REPO_NAME/requirements.txt <<EOL
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
pyarrow==17.0.0
requests==2.32.3
loguru==0.7.2
EOL

# Create environment.yml for conda
cat > $REPO_NAME/environment.yml <<EOL
name: interdc-mlops
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pandas=2.2.2
  - numpy=1.26.4
  - matplotlib=3.9.2
  - seaborn=0.13.2
  - pyarrow=17.0.0
  - requests=2.32.3
  - pip
  - pip:
      - loguru==0.7.2
EOL

# Create README
cat > $REPO_NAME/README.md <<EOL
# InterDC MLOps

Steps:
1. Create environment:
   \`\`\`
   conda env create -f environment.yml
   conda activate interdc-mlops
   \`\`\`

2. Download dataset:
   \`\`\`
   python -m src.data.download_mendeley --url <Mendeley ZIP URL> --out data/raw
   \`\`\`

3. Run EDA:
   \`\`\`
   python -m src.eda.eda_report --input data/raw/dataset.parquet --out artifacts/eda
   \`\`\`
EOL

# Download script
cat > $REPO_NAME/src/data/download_mendeley.py <<'EOL'
import argparse, requests, zipfile, io, os, pandas as pd
from loguru import logger

def main(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Downloading dataset from {url}")
    r = requests.get(url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(out_dir)
    logger.success(f"Extracted dataset to {out_dir}")

    dfs = []
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(root,f)))
    if dfs:
        pd.concat(dfs, ignore_index=True).to_parquet(os.path.join(out_dir,"dataset.parquet"))
        logger.success("Saved merged parquet file.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    main(args.url, args.out)
EOL

# EDA script
cat > $REPO_NAME/src/eda/eda_report.py <<'EOL'
import argparse, os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from loguru import logger

def load_data(path):
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix == ".csv":
        return pd.read_csv(p)
    elif p.is_dir():
        csvs = list(p.glob("*.csv"))
        if csvs: return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    raise ValueError(f"No valid data found at {path}")

def main(inp, out):
    os.makedirs(out, exist_ok=True)
    df = load_data(inp)
    logger.info(f"Loaded dataset with {df.shape[0]} rows, {df.shape[1]} columns")

    df.describe().to_csv(os.path.join(out,"summary.csv"))
    logger.info("Saved summary statistics")

    for col in ["throughput_mbps","latency_ms","jitter_ms","packet_loss_pct"]:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df[col], bins=50, kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"hist_{col}.png"))
            plt.close()
    logger.success(f"EDA report saved to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/dataset.parquet")
    ap.add_argument("--out", default="artifacts/eda")
    args = ap.parse_args()
    main(args.input, args.out)
EOL

# Init Git repo
cd $REPO_NAME
git init
git add .
git commit -m "Initial repo with dataset download + EDA"
echo "✅ Repo $REPO_NAME created. Now create GitHub repo and push."