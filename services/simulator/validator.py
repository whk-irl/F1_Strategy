"""Simulator fidelity validation — Spearman ρ gate before RL training.

For each race in the gold data:
1. Replay every driver's *actual* pit decisions through the tyre model.
2. Sum simulated lap times → total race time per driver.
3. Rank drivers by simulated total time.
4. Compare to actual finishing positions (gold ``position`` column).
5. Compute Spearman ρ between simulated rank and actual finishing position.

Pass criterion (from CLAUDE.md): mean Spearman ρ ≥ 0.7 across all 2024 races.

Entry point:
    python -m services.simulator.validator
    make validate-simulator
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
from ml.models._loader import load_gold_seasons
from scipy.stats import spearmanr

from services.simulator.lap_simulator import (
    PIT_LOSS_S,
    field_median_by_lap,
    simulate_driver_race,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Per-race helpers
# ---------------------------------------------------------------------------


def _actual_finishing_positions(race_df: pd.DataFrame) -> dict[Any, int]:
    """Extract each driver's actual finishing position from the final lap."""
    final_lap = int(race_df["lap_number"].max())
    final = race_df[race_df["lap_number"] == final_lap][["driver_number", "position"]].dropna(
        subset=["position"]
    )
    return {row["driver_number"]: int(row["position"]) for _, row in final.iterrows()}


def replay_race_standings(
    race_df: pd.DataFrame,
    tire_model: Any,
    pit_loss_s: float = PIT_LOSS_S,
) -> dict[Any, float]:
    """Simulate total race time for every driver using their actual pit strategy.

    Args:
        race_df: Gold rows for a single race (all drivers).
        tire_model: Loaded MLflow pyfunc tyre-degradation model.
        pit_loss_s: Seconds added on each pit-in lap.

    Returns:
        Dict mapping driver_number → total simulated race time in seconds.
        Drivers with fewer than 5 recorded laps are excluded.
    """
    field_medians = field_median_by_lap(race_df)
    standings: dict[Any, float] = {}

    for drv, drv_df in race_df.groupby("driver_number"):
        if drv_df["lap_time_s"].notna().sum() < 5:
            continue
        result = simulate_driver_race(drv_df, field_medians, tire_model, pit_loss_s)
        standings[drv] = float(result["simulated_lap_time_s"].sum())

    return standings


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_simulator(
    gold_df: pd.DataFrame,
    tire_model: Any,
    min_spearman: float = 0.70,
    pit_loss_s: float = PIT_LOSS_S,
    verbose: bool = True,
) -> dict[str, Any]:
    """Validate simulator fidelity via Spearman ρ on held-out race outcomes.

    Args:
        gold_df: Full gold DataFrame (multiple races; session == "R" rows used).
        tire_model: Loaded MLflow pyfunc tyre-degradation model.
        min_spearman: Pass threshold for mean Spearman ρ (default 0.70).
        pit_loss_s: Seconds added on each pit-in lap.
        verbose: Log per-race Spearman ρ if True.

    Returns:
        Dict with keys:
            ``rho_per_race``  list of per-race Spearman ρ values
            ``mean_rho``      float — average across races
            ``passed``        bool — True if mean_rho ≥ min_spearman
    """
    races = gold_df[gold_df["session"] == "R"]
    rho_scores: list[float] = []

    for (year, rnd), race_df in races.groupby(["year", "round_number"]):
        actual_positions = _actual_finishing_positions(race_df)
        sim_standings = replay_race_standings(race_df, tire_model, pit_loss_s)

        common_drivers = [d for d in sim_standings if d in actual_positions]
        if len(common_drivers) < 3:
            logger.warning(
                "Race %d Rd %02d: only %d comparable drivers — skipping.",
                year,
                rnd,
                len(common_drivers),
            )
            continue

        actual_pos = [actual_positions[d] for d in common_drivers]
        sim_times = pd.Series({d: sim_standings[d] for d in common_drivers})
        sim_ranks = sim_times.rank().tolist()

        rho, _ = spearmanr(actual_pos, sim_ranks)
        rho_scores.append(float(rho))

        if verbose:
            logger.info(
                "Race %d Rd %02d: Spearman ρ = %+.3f  (n=%d drivers)",
                year,
                rnd,
                rho,
                len(common_drivers),
            )

    mean_rho = float(np.mean(rho_scores)) if rho_scores else 0.0
    passed = mean_rho >= min_spearman
    status = "✓ PASS" if passed else "✗ FAIL"

    logger.info(
        "Simulator validation: mean ρ = %.3f across %d races → %s (threshold %.2f)",
        mean_rho,
        len(rho_scores),
        status,
        min_spearman,
    )

    return {
        "rho_per_race": rho_scores,
        "mean_rho": mean_rho,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    seasons: Annotated[
        str,
        typer.Option("--seasons", help="Comma-separated seasons to validate, e.g. 2024."),
    ] = "2024",
    min_rho: Annotated[
        float,
        typer.Option("--min-rho", help="Minimum mean Spearman ρ to pass."),
    ] = 0.70,
) -> None:
    """Validate simulator fidelity and print Spearman ρ per race."""
    import mlflow

    tracking_uri = os.getenv("PITWALL_MLFLOW_TRACKING_URI", "mlruns")
    tire_model_uri = os.getenv("PITWALL_TIRE_MODEL_URI", "models:/pitwall-tire-degradation/latest")
    mlflow.set_tracking_uri(tracking_uri)
    tire_model = mlflow.pyfunc.load_model(tire_model_uri)

    season_list = [int(s.strip()) for s in seasons.split(",")]
    gold_df = load_gold_seasons(season_list)

    result = validate_simulator(gold_df, tire_model, min_rho)

    if not result["passed"]:
        raise SystemExit(
            f"Simulator validation FAILED: mean ρ={result['mean_rho']:.3f} < {min_rho:.2f}. "
            "Fix the simulator before starting RL training."
        )


if __name__ == "__main__":
    app()
