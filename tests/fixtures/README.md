# Test fixtures

Small Parquet files used by integration and data-quality tests.

## Generating fixtures

Run the ingestion pipeline against a small session (FP1 is fastest to download)
and then copy the output from MinIO:

```bash
make up
make ingest SEASON=2024 ROUND=1 SESSION=FP1
# Then export from MinIO to tests/fixtures/
```

## What's here (gitignored except this README)

- `silver_2024_R01_sample.parquet` — 50 laps from 2024 Bahrain GP (silver layer)
- `gold_2024_R01_sample.parquet` — same laps after gold transform

Full-season Parquet files live in MinIO, not in the repo.
