"""One-shot script: partition and upload gold_2022_2025.parquet to S3.

Reads data/gold_2022_2025.parquet, splits by (year, round_number, session),
and writes each partition to the Hive layout the loader expects:

    s3://<bucket>/gold/year=<YYYY>/round=<NN>/session=<S>/laps.parquet

Usage:
    uv run python scripts/upload_gold_to_s3.py

AWS credentials are sourced from the standard chain (env vars, ~/.aws/credentials,
instance profile) — no keys are hardcoded here.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_LOCAL_FILE = Path("data/gold_2022_2025.parquet")
_BUCKET = os.getenv("PITWALL_AWS_S3_BUCKET", "pitwall-ai-prod")
_REGION = os.getenv("PITWALL_AWS_REGION", "us-east-1")


def main() -> None:
    if not _LOCAL_FILE.exists():
        print(f"ERROR: {_LOCAL_FILE} not found. Run scripts/export_for_deploy.py first.")
        sys.exit(1)

    print(f"Reading {_LOCAL_FILE} …")
    df = pd.read_parquet(_LOCAL_FILE)
    years = sorted(df["year"].unique().tolist())
    print(f"Loaded {len(df):,} rows  |  seasons: {years}")

    client = boto3.client("s3", region_name=_REGION)

    # Verify the bucket is reachable before starting the upload.
    try:
        client.head_bucket(Bucket=_BUCKET)
    except Exception as exc:
        print(f"ERROR: cannot reach s3://{_BUCKET} — {exc}")
        sys.exit(1)

    groups = list(df.groupby(["year", "round_number", "session"]))
    total = len(groups)
    print(f"Uploading {total} partitions to s3://{_BUCKET}/gold/ …\n")

    for i, ((year, rnd, session), part_df) in enumerate(groups, 1):
        key = f"gold/year={int(year)}/round={int(rnd):02d}/session={session}/laps.parquet"
        buf = io.BytesIO()
        part_df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        buf.seek(0)
        client.put_object(
            Bucket=_BUCKET,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        print(f"  [{i:>3}/{total}] {key}  ({len(part_df)} rows)")

    print(f"\nDone — {total} partitions uploaded to s3://{_BUCKET}/gold/")
    print("Set PITWALL_STORAGE_BACKEND=s3 in .env and restart Streamlit.")


if __name__ == "__main__":
    main()
