-- Creates the additional database used by Prefect server.
-- The primary `mlflow` database is created by the POSTGRES_DB env var.
CREATE DATABASE prefect;
GRANT ALL PRIVILEGES ON DATABASE prefect TO pitwall;
