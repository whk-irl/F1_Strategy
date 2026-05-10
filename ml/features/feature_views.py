"""Feature view re-exports — thin wrapper over definitions.py."""

try:
    from definitions import (  # Feast context
        lap_pace_fv,
        lap_race_context_fv,
        lap_tire_fv,
    )
except ImportError:
    from ml.features.definitions import (  # normal package import
        lap_pace_fv,
        lap_race_context_fv,
        lap_tire_fv,
    )

__all__ = ["lap_pace_fv", "lap_race_context_fv", "lap_tire_fv"]
