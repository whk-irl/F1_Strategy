# Pitwall AI — Streamlit demo image
# Bakes in all model artifacts and 2024 gold data so the container is fully
# self-contained: no MLflow server, no MinIO, no AWS credentials required.
#
# Build:  docker build -t pitwall-ai .
# Run:    docker run -p 8501:8501 pitwall-ai
# Open:   http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# System deps for LightGBM and PyTorch CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv --no-cache-dir

# Copy lockfile and project metadata first (layer-cached until deps change)
COPY pyproject.toml uv.lock ./

# Install only runtime deps (no dev/test extras); skip torch CUDA index
RUN uv sync --no-dev --frozen --no-install-project

# Copy source packages
COPY services/ services/
COPY ml/ ml/

# Copy baked model artifacts and gold data
COPY models_baked/ models_baked/
COPY data/ data/

# Streamlit server config
COPY .streamlit/ .streamlit/

# Environment: switch all loaders to local-file mode
ENV PITWALL_STORAGE_BACKEND=local
ENV PITWALL_LOCAL_DATA_PATH=/app/data/gold_2024.parquet
ENV PITWALL_TIRE_MODEL_URI=/app/models_baked/tire
ENV PITWALL_SC_MODEL_URI=/app/models_baked/sc
ENV PITWALL_POLICY_PATH=/app/models_baked/policy/model/policy.zip

EXPOSE 8501

ENTRYPOINT ["uv", "run", "streamlit", "run", "services/frontend/app.py", \
            "--server.port=8501", \
            "--server.headless=true", \
            "--server.address=0.0.0.0"]
