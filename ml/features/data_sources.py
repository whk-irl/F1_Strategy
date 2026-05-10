"""Data source re-exports — thin wrapper over definitions.py."""

try:
    from definitions import gold_laps_source  # Feast context
except ImportError:
    from ml.features.definitions import gold_laps_source  # normal package import

__all__ = ["gold_laps_source"]
