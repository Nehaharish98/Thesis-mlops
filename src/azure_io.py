"""
Azure Blob Storage I/O helper functions for Thesis-mlops pipeline.
- Auto-loads .env so AZURE_STORAGE_CONNECTION_STRING is always available.
- Provides simple helpers to list, read, and write blobs.
"""

import os, io, json, pandas as pd, yaml, logging
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# --------------------------------------------------------------------
# Load environment variables automatically
# --------------------------------------------------------------------
load_dotenv()  # looks for .env in repo root

# --------------------------------------------------------------------
# Read config
# --------------------------------------------------------------------
_cfg = yaml.safe_load(open("config/data.yaml"))
_conn = os.getenv(_cfg["azure"]["connection_string_env"])

if not _conn:
    raise RuntimeError(
        f"⚠️ Connection string env var '{_cfg['azure']['connection_string_env']}' not set. "
        f"Did you create a .env with AZURE_STORAGE_CONNECTION_STRING?"
    )

_blob = BlobServiceClient.from_connection_string(_conn)

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def list_blobs(prefix: str = ""):
    """List blob names in raw container (optionally with prefix)."""
    c = _blob.get_container_client(_cfg["azure"]["container_raw"])
    return [b.name for b in c.list_blobs(name_starts_with=prefix)]

def read_json_blob(blob_name: str):
    """
    Download a blob from 'raw' container and parse JSON safely.
    - Returns parsed JSON (list or dict) if valid.
    - Returns None if blob is empty or not valid JSON.
    """
    import logging, json

    bc = _blob.get_blob_client(_cfg["azure"]["container_raw"], blob_name)

    try:
        raw = bc.download_blob().readall()
        if not raw:
            logging.warning(f"⚠️ Empty blob: {blob_name}")
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logging.error(f"❌ Blob is not valid JSON: {blob_name} ({e})")
            return None

    except Exception as e:
        logging.error(f"❌ Failed to download blob {blob_name}: {e}")
        return None


def write_csv_processed(df: pd.DataFrame, name: str):
    """Upload DataFrame as CSV to processed container."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode('utf-8')
    
    bc = _blob.get_blob_client(_cfg["azure"]["container_processed"], name)
    bc.upload_blob(csv_data, overwrite=True)
    logging.info(f"✅ CSV Upload complete: {name}")
