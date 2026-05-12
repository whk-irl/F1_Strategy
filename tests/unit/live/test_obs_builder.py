"""Unit tests for services/live/obs_builder.py.

All tests are pure Python — no network calls, no MLflow, no FastF1.
"""

from __future__ import annotations

import numpy as np
import pytest
from services.live.obs_builder import (
    OBS_DIM,
    DriverLiveState,
    encode_compound,
    update_from_openf1,
)

# ---------------------------------------------------------------------------
# encode_compound
# ---------------------------------------------------------------------------


class TestEncodeCompound:
    def test_known_compounds(self) -> None:
        assert encode_compound("SOFT") == 0
        assert encode_compound("MEDIUM") == 1
        assert encode_compound("HARD") == 2
        assert encode_compound("INTERMEDIATE") == 3
        assert encode_compound("WET") == 4

    def test_case_insensitive(self) -> None:
        assert encode_compound("soft") == 0
        assert encode_compound("Medium") == 1

    def test_unknown_returns_5(self) -> None:
        assert encode_compound("HYPERSOFT") == 5
        assert encode_compound("") == 5

    def test_none_returns_5(self) -> None:
        assert encode_compound(None) == 5


# ---------------------------------------------------------------------------
# DriverLiveState.build_obs
# ---------------------------------------------------------------------------


class TestBuildObs:
    def _make_state(self, **kwargs: object) -> DriverLiveState:
        defaults = {"driver_number": 44, "total_laps": 50}
        defaults.update(kwargs)  # type: ignore[arg-type]
        return DriverLiveState(**defaults)  # type: ignore[arg-type]

    def test_obs_shape_and_dtype(self) -> None:
        obs = self._make_state().build_obs()
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_all_values_finite(self) -> None:
        state = self._make_state(current_lap=10, tyre_life=8, compound_encoded=0, position=5)
        obs = state.build_obs()
        assert np.all(np.isfinite(obs))

    def test_race_progress_at_start(self) -> None:
        state = self._make_state(current_lap=1, total_laps=50)
        obs = state.build_obs()
        assert obs[0] == pytest.approx(0.0)  # race_progress = 0/50

    def test_race_progress_at_midpoint(self) -> None:
        state = self._make_state(current_lap=26, total_laps=50)
        obs = state.build_obs()
        assert obs[0] == pytest.approx(25 / 50)

    def test_laps_remaining_norm(self) -> None:
        state = self._make_state(current_lap=26, total_laps=50)
        obs = state.build_obs()
        assert obs[1] == pytest.approx(25 / 50)

    def test_tyre_life_clipped_at_1(self) -> None:
        state = self._make_state(tyre_life=90)  # > 60 → clipped
        obs = state.build_obs()
        assert obs[2] == pytest.approx(1.0)

    def test_compound_norm(self) -> None:
        state = self._make_state(compound_encoded=2)  # HARD
        obs = state.build_obs()
        assert obs[3] == pytest.approx(2 / 5)

    def test_is_fresh_tyre_on_lap_1(self) -> None:
        state = self._make_state(tyre_life=1)
        obs = state.build_obs()
        assert obs[4] == pytest.approx(1.0)

    def test_is_fresh_tyre_false_after_several_laps(self) -> None:
        state = self._make_state(tyre_life=10)
        obs = state.build_obs()
        assert obs[4] == pytest.approx(0.0)

    def test_undercut_threat_when_close_with_fresher_car_behind(self) -> None:
        state = self._make_state(gap_behind_s=2.0, tyre_age_behind=5, tyre_life=15)
        obs = state.build_obs()
        assert obs[11] == pytest.approx(1.0)  # undercut_threat

    def test_no_undercut_threat_when_far_behind(self) -> None:
        state = self._make_state(gap_behind_s=10.0, tyre_age_behind=5, tyre_life=15)
        obs = state.build_obs()
        assert obs[11] == pytest.approx(0.0)

    def test_free_stop_when_sc_active(self) -> None:
        state = self._make_state(sc_active=True)
        obs = state.build_obs()
        assert obs[13] == pytest.approx(1.0)  # free_stop

    def test_no_free_stop_outside_sc(self) -> None:
        state = self._make_state(sc_active=False)
        obs = state.build_obs()
        assert obs[13] == pytest.approx(0.0)

    def test_wet_flag_above_threshold(self) -> None:
        state = self._make_state(wet_fraction=0.5)
        obs = state.build_obs()
        assert obs[18] == pytest.approx(1.0)  # is_wet

    def test_wet_flag_below_threshold(self) -> None:
        state = self._make_state(wet_fraction=0.1)
        obs = state.build_obs()
        assert obs[18] == pytest.approx(0.0)

    def test_stops_norm_clipped_at_1(self) -> None:
        state = self._make_state(pit_stops=10)
        obs = state.build_obs()
        assert obs[20] == pytest.approx(1.0)

    def test_position_norm(self) -> None:
        state = self._make_state(position=5, n_drivers=20)
        obs = state.build_obs()
        assert obs[6] == pytest.approx(5 / 20)

    def test_overcut_opportunity(self) -> None:
        state = self._make_state(car_ahead_pitted_last_lap=True)
        obs = state.build_obs()
        assert obs[12] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DriverLiveState.update_tyre_deg / cliff_lap
# ---------------------------------------------------------------------------


class TestTyreDeg:
    def test_deg_accumulates_per_lap(self) -> None:
        state = DriverLiveState(driver_number=1, total_laps=50, compound_encoded=0)  # SOFT
        assert state.synthetic_deg == pytest.approx(0.0)
        state.update_tyre_deg()
        assert state.synthetic_deg == pytest.approx(0.08)
        state.update_tyre_deg()
        assert state.synthetic_deg == pytest.approx(0.16)

    def test_cliff_lap_by_compound(self) -> None:
        for compound, expected in [(0, 22), (1, 32), (2, 42), (3, 50), (4, 50)]:
            state = DriverLiveState(driver_number=1, total_laps=50, compound_encoded=compound)
            assert state.cliff_lap() == expected

    def test_cliff_lap_unknown_compound(self) -> None:
        state = DriverLiveState(driver_number=1, total_laps=50, compound_encoded=5)
        assert state.cliff_lap() == 35


# ---------------------------------------------------------------------------
# update_from_openf1
# ---------------------------------------------------------------------------


class TestUpdateFromOpenF1:
    def _base_state(self) -> DriverLiveState:
        return DriverLiveState(driver_number=44, total_laps=50)

    def test_lap_number_updated(self) -> None:
        state = self._base_state()
        update_from_openf1(state, {"lap_number": 12}, None, None, 0, False, [], None)
        assert state.current_lap == 12

    def test_lap_time_recorded(self) -> None:
        state = self._base_state()
        update_from_openf1(
            state, {"lap_number": 5, "lap_duration": 92.3}, None, None, 0, False, [], None
        )
        assert state.last_lap_time_s == pytest.approx(92.3)
        assert 92.3 in state.lap_times_s

    def test_invalid_lap_time_ignored(self) -> None:
        state = self._base_state()
        update_from_openf1(
            state, {"lap_number": 5, "lap_duration": None}, None, None, 0, False, [], None
        )
        assert state.last_lap_time_s is None

    def test_stint_updates_compound_and_tyre_life(self) -> None:
        state = self._base_state()
        state.current_lap = 10
        stint = {"compound": "MEDIUM", "lap_start": 6}
        update_from_openf1(state, None, stint, None, 0, False, [], None)
        assert state.compound_encoded == 1  # MEDIUM
        assert state.tyre_life == 5  # 10 - 6 + 1

    def test_position_updated(self) -> None:
        state = self._base_state()
        update_from_openf1(state, None, None, 3, 0, False, [], None)
        assert state.position == 3

    def test_position_unchanged_when_none(self) -> None:
        state = self._base_state()
        state.position = 7
        update_from_openf1(state, None, None, None, 0, False, [], None)
        assert state.position == 7

    def test_pit_count_updated(self) -> None:
        state = self._base_state()
        update_from_openf1(state, None, None, None, 2, False, [], None)
        assert state.pit_stops == 2

    def test_sc_active_resets_laps_since_last(self) -> None:
        state = self._base_state()
        state.sc_laps_since_last = 20
        update_from_openf1(state, None, None, None, 0, True, [], None)
        assert state.sc_laps_since_last == 0

    def test_no_sc_increments_laps_since_last(self) -> None:
        state = self._base_state()
        state.sc_laps_since_last = 5
        update_from_openf1(state, None, None, None, 0, False, [], None)
        assert state.sc_laps_since_last == 6

    def test_laps_since_last_capped_at_50(self) -> None:
        state = self._base_state()
        state.sc_laps_since_last = 50
        update_from_openf1(state, None, None, None, 0, False, [], None)
        assert state.sc_laps_since_last == 50

    def test_wet_fraction_from_field_stints(self) -> None:
        state = self._base_state()
        state.current_lap = 10
        field_stints = [
            {"driver_number": 1, "compound": "INTERMEDIATE", "lap_start": 8, "lap_end": None},
            {"driver_number": 2, "compound": "INTERMEDIATE", "lap_start": 8, "lap_end": None},
            {"driver_number": 3, "compound": "SOFT", "lap_start": 1, "lap_end": None},
            {"driver_number": 4, "compound": "MEDIUM", "lap_start": 1, "lap_end": None},
        ]
        update_from_openf1(state, None, None, None, 0, False, field_stints, None)
        # 2 wet out of 4 drivers = 0.5
        assert state.wet_fraction == pytest.approx(0.5)

    def test_dry_field_gives_zero_wet_fraction(self) -> None:
        state = self._base_state()
        state.current_lap = 5
        field_stints = [
            {"driver_number": 1, "compound": "SOFT", "lap_start": 1, "lap_end": None},
            {"driver_number": 2, "compound": "HARD", "lap_start": 1, "lap_end": None},
        ]
        update_from_openf1(state, None, None, None, 0, False, field_stints, None)
        assert state.wet_fraction == pytest.approx(0.0)

    def test_tyre_deg_accumulates_on_update(self) -> None:
        state = self._base_state()
        state.compound_encoded = 1  # MEDIUM
        update_from_openf1(state, None, None, None, 0, False, [], None)
        assert state.synthetic_deg == pytest.approx(0.05)

    def test_lap_times_buffer_capped_at_5(self) -> None:
        state = self._base_state()
        for lap in range(1, 8):
            update_from_openf1(
                state,
                {"lap_number": lap, "lap_duration": 90.0 + lap},
                None,
                None,
                0,
                False,
                [],
                None,
            )
        assert len(state.lap_times_s) == 5
