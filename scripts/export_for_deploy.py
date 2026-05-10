"""Export trained models and gold data for baked Docker/Streamlit Cloud deployment.

Run this after any retraining to refresh the artifacts in models_baked/ and data/.
The Dockerfile COPY instructions pick these up automatically on the next build.

Usage:
    uv run python scripts/export_for_deploy.py
"""

from __future__ import annotations

import logging
import os
import pathlib

import mlflow
import mlflow.tracking
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("PITWALL_MLFLOW_TRACKING_URI", "mlruns"))
    client = mlflow.tracking.MlflowClient()

    root = pathlib.Path(__file__).parent.parent
    models_out = root / "models_baked"
    data_out = root / "data"
    models_out.mkdir(exist_ok=True)
    data_out.mkdir(exist_ok=True)

    # Tire degradation model
    logger.info("Exporting tire model...")
    mlflow.artifacts.download_artifacts(
        artifact_uri="models:/pitwall-tire-degradation/latest",
        dst_path=str(models_out / "tire"),
    )
    logger.info("  -> models_baked/tire/")

    # Safety car model
    logger.info("Exporting SC model...")
    mlflow.artifacts.download_artifacts(
        artifact_uri="models:/pitwall-safety-car/latest",
        dst_path=str(models_out / "sc"),
    )
    logger.info("  -> models_baked/sc/")

    # PPO strategy policy (latest run in strategy-policy experiment)
    logger.info("Exporting policy...")
    exp = client.get_experiment_by_name("strategy-policy")
    if exp is None:
        raise RuntimeError("MLflow experiment 'strategy-policy' not found. Train policy first.")
    runs = client.search_runs(
        [exp.experiment_id], order_by=["start_time DESC"], max_results=1
    )
    if not runs:
        raise RuntimeError("No policy training runs found. Run make train-policy first.")
    run_id = runs[0].info.run_id
    mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="model", dst_path=str(models_out / "policy")
    )
    logger.info("  -> models_baked/policy/  (run %s)", run_id[:8])

    # 2024 gold data
    logger.info("Exporting gold data...")
    from ml.models._loader import load_gold_seasons  # noqa: PLC0415

    # Temporarily ensure we read from S3/MinIO, not local (avoid circular)
    orig = os.environ.pop("PITWALL_STORAGE_BACKEND", None)
    try:
        df = load_gold_seasons([2024])
    finally:
        if orig:
            os.environ["PITWALL_STORAGE_BACKEND"] = orig

    out_path = data_out / "gold_2024.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("  -> data/gold_2024.parquet  (%d rows, %.1f MB)", len(df), out_path.stat().st_size / 1e6)

    logger.info("Export complete. Run `docker build -t pitwall-ai .` to rebuild the image.")


if __name__ == "__main__":
    main()
