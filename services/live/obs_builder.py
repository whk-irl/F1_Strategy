"""Maps live OpenF1 data to the 21-dim observation vector expected by the PPO policy.

The obs vector layout mirrors ``services/simulator/env.py`` exactly (see
``OBS_DIM = 21``).  This module is intentionally stateless — callers hold a
``LiveRaceState`` object and pass it in on every update.

Compound mapping (OpenF1 → compound_encoded):
    "SOFT"     → 0
    "MEDIUM"   → 1
    "HARD"     → 2
    "INTERMEDIATE" → 3
    "WET"      → 4
    unknown    → 5
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Matches env.py OBS_DIM
OBS_DIM: int = 21

_COMPOUND_MAP: dict[str, int] = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
}

_WET_COMPOUNDS: frozenset[int] = frozenset({3, 4})

# Mirrors env.py synthetic degradation baselines
_BASE_DEGRADE_RATE: dict[int, float] = {0: 0.08, 1: 0.05, 2: 0.03, 3: 0.01, 4: 0.01, 5: 0.05}
_BASE_CLIFF_LAP: dict[int, int] = {0: 22, 1: 32, 2: 42, 3: 50, 4: 50, 5: 35}


def encode_compound(compound_str: str | None) -> int:
    """Convert a raw OpenF1 compound string to compound_encoded (0–5)."""
    if compound_str is None:
        return 5
    return _COMPOUND_MAP.get(compound_str.upper(), 5)


@dataclass
class DriverLiveState:
    """Rolling state for a single driver, updated each lap."""

    driver_number: int
    total_laps: int

    # Current lap being tracked (1-indexed, matches OpenF1)
    current_lap: int = 0
    compound_encoded: int = 5
    tyre_life: int = 0
    position: int = 10
    last_lap_time_s: float | None = None
    pit_stops: int = 0
    sc_laps_since_last: int = 50

    # Running gap info (seconds)
    gap_ahead_s: float = 5.0
    gap_behind_s: float = 5.0
    tyre_age_ahead: int = 10
    tyre_age_behind: int = 10

    # Field compound distribution (fraction on wet tyres this lap)
    wet_fraction: float = 0.0
    # Synthetic degradation accumulator (mirrors env._synthetic_deg)
    synthetic_deg: float = 0.0
    # Track whether last lap saw a SC/VSC (for free stop bonus obs)
    sc_active: bool = False
    # Overcut flag: car ahead pitted last lap
    car_ahead_pitted_last_lap: bool = False
    # SC onset probability (from SC model or 0 if unavailable)
    sc_probability: float = 0.0

    # Pit loss (seconds) for this circuit
    pit_loss_s: float = 22.0
    # Number of field drivers
    n_drivers: int = 20

    # Lap history for gap estimation (last 3 lap times)
    lap_times_s: list[float] = field(default_factory=list)

    def update_tyre_deg(self) -> None:
        """Accumulate synthetic degradation for the current lap."""
        rate = _BASE_DEGRADE_RATE.get(self.compound_encoded, 0.05)
        self.synthetic_deg += rate

    def cliff_lap(self) -> int:
        """Return estimated cliff lap for the current compound."""
        return _BASE_CLIFF_LAP.get(self.compound_encoded, 35)

    def build_obs(self) -> np.ndarray:
        """Build the 21-dim observation vector from the current state.

        Returns:
            float32 array of shape (21,).
        """
        total = max(self.total_laps, 1)
        laps_done = max(self.current_lap - 1, 0)
        laps_left = max(total - laps_done, 0)

        race_progress = laps_done / total
        laps_remaining_norm = laps_left / total
        tyre_life_norm = min(self.tyre_life / 60.0, 1.0)
        compound_norm = self.compound_encoded / 5.0
        is_fresh = 1.0 if self.tyre_life <= 1 else 0.0
        syn_deg_norm = float(np.clip(self.synthetic_deg / 5.0, -1.0, 1.0))
        pos_norm = self.position / max(self.n_drivers, 1)
        gap_ahead_norm = float(np.clip(self.gap_ahead_s / 60.0, 0.0, 1.0))
        gap_behind_norm = float(np.clip(self.gap_behind_s / 60.0, 0.0, 1.0))
        tyre_age_ahead_norm = min(self.tyre_age_ahead / 60.0, 1.0)
        tyre_age_behind_norm = min(self.tyre_age_behind / 60.0, 1.0)

        undercut_threat = (
            1.0 if (self.gap_behind_s < 3.0 and self.tyre_age_behind < self.tyre_life) else 0.0
        )
        overcut_opp = 1.0 if self.car_ahead_pitted_last_lap else 0.0
        free_stop = 1.0 if self.sc_active else 0.0

        # Simplified: cars in pit window ≈ cars within 2s ahead (unknown from OpenF1 alone)
        cars_lose_norm = 0.0  # conservative — OpenF1 doesn't give pit window directly
        pit_loss_norm = min(self.pit_loss_s / 30.0, 1.0)
        sc_laps_norm = min(self.sc_laps_since_last / 50.0, 1.0)
        is_wet = 1.0 if self.wet_fraction >= 0.3 else 0.0
        stops_norm = min(self.pit_stops / 3.0, 1.0)

        obs = np.array(
            [
                race_progress,  # 0
                laps_remaining_norm,  # 1
                tyre_life_norm,  # 2
                compound_norm,  # 3
                is_fresh,  # 4
                syn_deg_norm,  # 5
                pos_norm,  # 6
                gap_ahead_norm,  # 7
                gap_behind_norm,  # 8
                tyre_age_ahead_norm,  # 9
                tyre_age_behind_norm,  # 10
                undercut_threat,  # 11
                overcut_opp,  # 12
                free_stop,  # 13
                cars_lose_norm,  # 14
                pit_loss_norm,  # 15
                self.sc_probability,  # 16
                sc_laps_norm,  # 17
                is_wet,  # 18
                self.wet_fraction,  # 19
                stops_norm,  # 20
            ],
            dtype=np.float32,
        )
        return obs


def update_from_openf1(
    state: DriverLiveState,
    lap_record: dict[str, Any] | None,
    stint_record: dict[str, Any] | None,
    position: int | None,
    pit_count: int,
    sc_active: bool,
    field_stints: list[dict[str, Any]],
    weather: dict[str, Any] | None,
) -> None:
    """Update a DriverLiveState from fresh OpenF1 API records in-place.

    Args:
        state: The driver's rolling state to update.
        lap_record: Latest lap record from ``/laps`` endpoint (may be None).
        stint_record: Current stint from ``/stints`` (may be None).
        position: Current on-track position (1-based), or None if unknown.
        pit_count: Total pit stops made so far.
        sc_active: Whether a SC/VSC is currently active.
        field_stints: All stint records for the field (for wet-fraction calc).
        weather: Latest weather record from ``/weather``, or None.
    """
    if lap_record is not None:
        state.current_lap = int(lap_record.get("lap_number", state.current_lap))
        raw_lt = lap_record.get("lap_duration")
        if raw_lt is not None:
            try:
                state.last_lap_time_s = float(raw_lt)
                state.lap_times_s.append(state.last_lap_time_s)
                if len(state.lap_times_s) > 5:
                    state.lap_times_s.pop(0)
            except (TypeError, ValueError):
                pass

    if stint_record is not None:
        state.compound_encoded = encode_compound(stint_record.get("compound"))
        lap_start = stint_record.get("lap_start", 1) or 1
        state.tyre_life = max(state.current_lap - int(lap_start) + 1, 0)

    if position is not None:
        state.position = position
    state.pit_stops = pit_count
    state.sc_active = sc_active

    if sc_active:
        state.sc_laps_since_last = 0
    else:
        state.sc_laps_since_last = min(state.sc_laps_since_last + 1, 50)

    # Wet fraction: proportion of field on INTER/WET in current lap area
    wet_count = sum(
        1
        for s in field_stints
        if encode_compound(s.get("compound")) in _WET_COMPOUNDS
        and (s.get("lap_end") is None or int(s.get("lap_end") or 0) >= state.current_lap - 1)
        and int(s.get("lap_start") or 0) <= state.current_lap
    )
    total_stints = max(len({s["driver_number"] for s in field_stints}), 1)
    state.wet_fraction = min(wet_count / total_stints, 1.0)

    # Accumulate synthetic degradation for this lap
    state.update_tyre_deg()

    # Track temperature adjustment via weather
    if weather is not None:
        track_temp = weather.get("track_temperature")
        if track_temp is not None:
            with contextlib.suppress(TypeError, ValueError):
                _apply_temp_adjustment(state, float(track_temp))


def _apply_temp_adjustment(state: DriverLiveState, track_temp_c: float) -> None:
    """Nudge synthetic_deg upward for very hot conditions (>40 °C track temp)."""
    ref_temp = 35.0
    coeff = 0.018
    if track_temp_c > ref_temp:
        extra = (
            (track_temp_c - ref_temp) * coeff * _BASE_DEGRADE_RATE.get(state.compound_encoded, 0.05)
        )
        state.synthetic_deg += extra
