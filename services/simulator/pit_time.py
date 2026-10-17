"""Portable measured pit-time prediction shared by training and Race Center."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models_baked" / "pit_time.json"


@lru_cache(maxsize=4)
def load_pit_time_model(path: str | Path = DEFAULT_MODEL) -> dict[str, Any]:
    """Load a published fit; missing artifacts are an explicit error to callers."""
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("format_version") != 1:
        raise ValueError("Unsupported pit-time model format")
    return cast(dict[str, Any], model)


def predict_pit_time(model: dict[str, Any], reference_round: int) -> dict[str, Any]:
    """Return venue estimate, falling back to the measured global distribution."""
    values = model.get("venues", {}).get(str(reference_round), model["global"])
    return {
        "lane_s": float(values["lane_s"]),
        "lane_low_s": float(values["lane_low_s"]),
        "lane_high_s": float(values["lane_high_s"]),
        "net_loss_s": float(values["net_loss_s"]),
        "net_low_s": float(values["net_low_s"]),
        "net_high_s": float(values["net_high_s"]),
        "sample_count": int(values["sample_count"]),
        "scope": "venue" if str(reference_round) in model.get("venues", {}) else "global",
    }
