"""Compare LightGBM, TCN+GRU, and PatchTST tire models on held-out validation rounds.

Run:
    PITWALL_STORAGE_BACKEND=local python scripts/validate_tire_models.py
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PITWALL_STORAGE_BACKEND", "local")

import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pyfunc
from sklearn.metrics import mean_absolute_error, r2_score

from ml.models._loader import load_gold_seasons
from ml.models.tire_degradation.sequence_dataset import (
    FEATURE_COLS,
    StintSequenceDataset,
)
from ml.models.tire_degradation.predict_sequence import load_sequence_model

VAL_ROUNDS = [5, 10, 15, 20]
SEASONS = [2022, 2023, 2024, 2025]
LGBM_FEATURES = [
    "tyre_life_laps",
    "compound_encoded",
    "race_progress",
    "track_status_encoded",
    "is_fresh_tyre",
    "lap_delta_to_field_median_s",
]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

print("Loading gold data...")
df = load_gold_seasons(SEASONS)
print(f"  {len(df):,} total laps")

# Load training norm stats from the baked TCN+GRU artifact so the validation
# dataset uses the same feature scale the model was trained on.
_, train_norm_stats, _ = load_sequence_model("tcn_gru")

# Build validation dataset using the same held-out rounds as training.
val_ds = StintSequenceDataset(
    df, norm_stats=train_norm_stats, val_rounds=VAL_ROUNDS, is_val=True
)
print(f"  {len(val_ds):,} validation windows  (rounds {VAL_ROUNDS}, all seasons)")

all_x: np.ndarray = torch.stack(val_ds._xs).numpy()   # (N, seq_len, 8)
all_y: np.ndarray = torch.stack(val_ds._ys).numpy()   # (N,)
norm = val_ds.norm_stats

print(f"\n  target stats — mean={all_y.mean():.3f}s  std={all_y.std():.3f}s\n")

results: dict[str, tuple[float, float]] = {}

# ---------------------------------------------------------------------------
# LightGBM — point-in-time scalar prediction
# ---------------------------------------------------------------------------
print("Evaluating LightGBM...")
mlflow.set_tracking_uri("mlruns")
lgbm = mlflow.pyfunc.load_model("models_baked/tire")

# Un-normalise the last step in each window to get original-scale features.
last_raw: np.ndarray = all_x[:, -1, :] * (norm.std + 1e-8) + norm.mean
lgbm_input = pd.DataFrame(last_raw, columns=FEATURE_COLS)[LGBM_FEATURES]
lgbm_preds: np.ndarray = lgbm.predict(lgbm_input)
results["LightGBM (baseline)"] = (
    float(mean_absolute_error(all_y, lgbm_preds)),
    float(r2_score(all_y, lgbm_preds)),
)
print(f"  MAE={results['LightGBM (baseline)'][0]:.4f}  R2={results['LightGBM (baseline)'][1]:.4f}")

# ---------------------------------------------------------------------------
# Sequence models — full 10-lap window context
# ---------------------------------------------------------------------------
x_t = torch.from_numpy(all_x)

for mt, label in [("tcn_gru", "TCN+GRU"), ("patch_tst", "PatchTST")]:
    print(f"Evaluating {label}...")
    model, _ns, _ = load_sequence_model(mt)
    model.eval()
    with torch.no_grad():
        preds: np.ndarray = model(x_t).numpy()
    results[label] = (
        float(mean_absolute_error(all_y, preds)),
        float(r2_score(all_y, preds)),
    )
    print(f"  MAE={results[label][0]:.4f}  R2={results[label][1]:.4f}")

# ---------------------------------------------------------------------------
# Per-compound breakdown (sequence models have full context advantage here)
# ---------------------------------------------------------------------------
print("\n--- Per-compound breakdown (TCN+GRU) ---")
model_tcn, ns_tcn, _ = load_sequence_model("tcn_gru")
model_tcn.eval()
with torch.no_grad():
    tcn_preds_all: np.ndarray = model_tcn(x_t).numpy()

# Recover compound from last window step
last_compound_norm: np.ndarray = all_x[:, -1, FEATURE_COLS.index("compound_encoded")]
last_compound_raw: np.ndarray = (
    last_compound_norm * (norm.std[FEATURE_COLS.index("compound_encoded")] + 1e-8)
    + norm.mean[FEATURE_COLS.index("compound_encoded")]
)
compound_names = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
for enc, name in compound_names.items():
    mask = np.round(last_compound_raw).astype(int) == enc
    if mask.sum() < 10:
        continue
    mae_c = float(mean_absolute_error(all_y[mask], tcn_preds_all[mask]))
    r2_c = float(r2_score(all_y[mask], tcn_preds_all[mask]))
    print(f"  {name:<8}  n={mask.sum():5d}  MAE={mae_c:.4f}  R2={r2_c:.4f}")

# ---------------------------------------------------------------------------
# Per-tyre-age bucket (early stint vs mid vs late)
# ---------------------------------------------------------------------------
print("\n--- Per-tyre-age bucket (TCN+GRU) ---")
last_age_norm: np.ndarray = all_x[:, -1, FEATURE_COLS.index("tyre_life_laps")]
last_age_raw: np.ndarray = (
    last_age_norm * (norm.std[FEATURE_COLS.index("tyre_life_laps")] + 1e-8)
    + norm.mean[FEATURE_COLS.index("tyre_life_laps")]
)
buckets = [("early (1-10 laps)", (1, 10)), ("mid (11-25)", (11, 25)), ("late (26+)", (26, 99))]
for label, (lo, hi) in buckets:
    mask = (last_age_raw >= lo) & (last_age_raw <= hi)
    if mask.sum() < 10:
        continue
    mae_b = float(mean_absolute_error(all_y[mask], tcn_preds_all[mask]))
    print(f"  {label:<20}  n={mask.sum():5d}  MAE={mae_b:.4f}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print()
print("=" * 52)
print(f"{'Model':<22}  {'Val MAE (s)':>12}  {'Val R2':>8}")
print("-" * 52)
best_mae = min(v[0] for v in results.values())
for name, (mae, r2) in results.items():
    star = " *" if mae == best_mae else "  "
    print(f"{name:<22}  {mae:12.4f}  {r2:8.4f}{star}")
print("=" * 52)
print("  * best MAE")
