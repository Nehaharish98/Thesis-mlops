"""
Ingest raw JSON from Azure Blob -> DataFrame -> Parquet.
- Always saves locally (data/processed/all_records.parquet).
- If AZURE_STORAGE_CONNECTION_STRING is set, also uploads to Azure 'processed' container.
"""
import pandas as pd
import logging, time, os
from pathlib import Path
from src.azure_io import list_blobs, read_json_blob, write_parquet_processed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def parse_vm_proto(x: str):
    if "_" in x:
        vm, proto = x.split("_", 1)
    else:
        vm, proto = x, "UNKNOWN"
    return vm, proto

def main():
    start = time.time()
    rows = []
    blobs = list_blobs()
    logging.info(f"Found {len(blobs)} blobs to process")

    for i, blob in enumerate(blobs, 1):
        logging.info(f"[{i}/{len(blobs)}] Processing blob: {blob}")
        if not blob.endswith(".json"):
            logging.warning(f"⏭️ Skipping non-JSON blob: {blob}")
            continue

        parts = blob.split("/")
        if len(parts) < 3:
            logging.warning(f"⏭️ Skipping blob with unexpected path: {blob}")
            continue

        provider, vmproto, region = parts[0], parts[1], parts[2]
        az = parts[3] if len(parts) > 3 and parts[3].endswith(".json") is False else None

        try:
            data = read_json_blob(blob)
        except Exception as e:
            logging.error(f"❌ Failed to read {blob}: {e}")
            continue

        if not data:
            logging.warning(f"⚠️ Empty JSON blob: {blob}")
            continue

        vm_size, l4_protocol = parse_vm_proto(vmproto)
        recs = data if isinstance(data, list) else [data]

        for r in recs:
            if not isinstance(r, dict):
                logging.warning(f"⚠️ Unexpected record format in {blob}: {type(r)}")
                continue
            r.update(dict(provider=provider, region=region, az=az,
                      vm_size=vm_size, l4_protocol=l4_protocol))
            rows.append(r)



    df = pd.DataFrame(rows)
    logging.info(f"Parsed {len(df):,} records with {df.shape[1]} columns")

    # --- always save locally ---
    out_path = Path("data/processed/all_records.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logging.info(f"💾 Saved locally: {out_path} ({len(df):,} rows)")

    # --- optional: upload to Azure ---
    if os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        try:
            write_parquet_processed(df, "all_records.parquet")
        except Exception as e:
            logging.error(f"⚠️ Upload to Azure failed: {e}")
    else:
        logging.warning("No AZURE_STORAGE_CONNECTION_STRING set. Skipping Azure upload.")

    elapsed = time.time() - start
    logging.info(f"✅ Ingest complete in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
