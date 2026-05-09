# Architecture

## Overview

Pitwall AI is a single-car race strategy engine. Given the current race state (lap number, tire age, compound, gap to leader, track status), it outputs a strategic action: **pit now**, **stay out**, or **change compound on next stop**.

The system is built around a learned race simulator (OpenAI Gym environment) and a Proximal Policy Optimization (PPO) agent trained against it. The simulator is validated against historical races before any RL training begins, preventing the agent from learning to exploit a broken environment.

## Data flow

```
FastF1 API (open historical data, 2018+)
    │
    ▼  services/ingestion — Prefect flow, runs on schedule or on-demand
┌──────────────────────────────────────────────────────┐
│  Bronze  │  Raw FastF1 dump (per-lap + telemetry)    │
│          │  Parquet on MinIO / S3                    │
└──────────┬───────────────────────────────────────────┘
           │  pandera schema enforcement + cleaning
           ▼
┌──────────────────────────────────────────────────────┐
│  Silver  │  Per-lap canonical records                │
│          │  Typed, deduplicated, race-ID stamped     │
└──────────┬───────────────────────────────────────────┘
           │  feature engineering (lags, deltas, rates)
           ▼
┌──────────────────────────────────────────────────────┐
│  Gold    │  ML-ready feature tables                  │
│          │  Registered in Feast                      │
└──────────┬───────────────────────────────────────────┘
           │
     ┌─────┴──────┬─────────────────┐
     ▼            ▼                 ▼
 Tire deg.   Safety-car prob.   Baseline pace
 (LightGBM)  (LightGBM)        (LightGBM)
     │            │                 │
     └─────┬──────┴─────────────────┘
           ▼
   Race Simulator (Gym env)
   Spearman ρ ≥ 0.7 guardrail before RL training
           │
           ▼
   PPO Strategy Policy
           │
           ▼
   BentoML → FastAPI  →  Streamlit frontend
```

## Component descriptions

### services/ingestion

A Prefect flow that:
1. Calls FastF1 to download session data (laps, car data, weather).
2. Writes the raw dump to MinIO as **bronze** Parquet, partitioned by `year/round/session`.
3. Cleans, renames, and type-coerces columns → validates with a pandera schema → writes **silver**.
4. Engineers ML features (tire degradation rate, lap-time delta, safety-car flags) → validates → writes **gold** and registers the feature view in Feast.

### ml/models/tire_degradation

A per-compound LightGBM model that predicts lap-time increase per lap of tire age. Features: compound, tyre_life_laps, track_temp, air_temp, fuel_load_est, driver_pace_baseline. Target: `lap_time_s` (or delta from stint median).

### ml/models/safety_car

A LightGBM classifier that predicts P(safety car in next N laps). Features: lap_number, race_progress, incidents_count (from track status changes), circuit_id. Binary cross-entropy loss; decision threshold tuned for recall.

### ml/models/strategy_policy

A PPO agent (Stable-Baselines3) trained against the race simulator. Observation space: lap number, tyre_life, compound, gap_to_leader, sc_probability, estimated laps on current tyre, position. Action space: Discrete(3) — stay out / pit (same compound) / pit (alternate compound).

### services/simulator

An OpenAI Gym environment that simulates lap-by-lap race progression using the three LightGBM models. Receives a strategy action each step and returns the next race state and a reward shaped around final race position.

### services/api

A BentoML service wrapping the PPO policy. Accepts a JSON race state and returns the recommended action with a probability distribution over actions (for confidence display).

### services/frontend

A Streamlit app that lets a user scrub through a historical race and watch the strategy recommendation update each lap.

## Key design decisions

- **Cloud-agnostic:** Terraform provisions EKS or AKS; application code sees only an S3-compatible API and standard SQL. See `docs/decisions/0001-cloud-agnostic-mlops.md`.
- **Simulator-first RL:** The Spearman ρ ≥ 0.7 guardrail (on held-out 2024 races) must pass before any RL training. This is a hard gate in the training pipeline.
- **Single-car scope:** Multi-agent team strategy is an explicit v1 non-goal. See `docs/scope.md`.
- **No LLM layer:** Natural-language strategy queries dilute the project's ML story. The frontend is a lap-by-lap scrubber, not a chatbot.
