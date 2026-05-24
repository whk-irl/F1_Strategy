"""Unit tests for services/live/race_log.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from services.live.obs_builder import DriverLiveState
from services.live.race_log import (
    ENDPOINTS,
    LiveTickContext,
    append_live_tick,
    endpoint_key,
    list_sessions,
    load_endpoint,
    reset_storage,
)


@pytest.fixture
def mock_storage(monkeypatch: pytest.MonkeyPatch) -> dict[str, pd.DataFrame]:
    """In-memory parquet store keyed by S3 object key."""
    store: dict[str, pd.DataFrame] = {}

    class FakeStorage:
        def read_parquet(self, key: str) -> pd.DataFrame:
            if key not in store:
                raise KeyError(key)
            return store[key].copy()

        def write_parquet(self, df: pd.DataFrame, key: str) -> None:
            store[key] = df.copy()

        def list_keys(self, prefix: str) -> list[str]:
            return [k for k in store if k.startswith(prefix)]

    fake = FakeStorage()
    monkeypatch.setattr("services.live.race_log.get_storage", lambda: fake)
    reset_storage()
    return store


class TestEndpointKey:
    def test_openf1_layout(self) -> None:
        assert endpoint_key("laps", 9839) == "pitwall_live/laps/session_key=9839/data.parquet"

    def test_unknown_endpoint_raises(self) -> None:
        with pytest.raises(ValueError):
            endpoint_key("unknown", 1)


class TestAppendLiveTick:
    def _ctx(self, store: dict[str, pd.DataFrame]) -> LiveTickContext:
        client = MagicMock()
        client.get_latest_lap.return_value = {"lap_number": 5, "lap_duration": 90.1}
        client.get_current_stint.return_value = {
            "compound": "MEDIUM",
            "lap_start": 1,
            "lap_end": None,
        }
        client.get_latest_position.return_value = 3
        client.get_pit_stops.return_value = []

        policy = MagicMock()

        def recommend(obs: np.ndarray, _policy: MagicMock, _mt: str) -> tuple:
            return 0, "Stay out", {"Stay out": 0.9, "Pit — SOFT": 0.05, "Pit — MEDIUM": 0.03, "Pit — HARD": 0.02}, None

        return LiveTickContext(
            session_meta={
                "session_key": 9839,
                "meeting_key": 1276,
                "session_name": "Race",
                "session_type": "Race",
                "year": 2026,
                "country_name": "Canada",
                "circuit_short_name": "Montreal",
                "meeting_name": "Canadian Grand Prix",
                "date_start": "2026-05-24T18:00:00Z",
                "total_laps": 70,
            },
            client=client,
            drivers=[{"driver_number": 1, "name_acronym": "NOR"}],
            policy=policy,
            model_type="ppo",
            model_key="default",
            sc_active=False,
            field_stints=[],
            weather={"track_temperature": 30.0, "rainfall": 0.0},
            pit_loss_s=22.0,
            recommend_fn=recommend,
            driver_states={},
        )

    def test_writes_multiple_endpoints(self, mock_storage: dict[str, pd.DataFrame]) -> None:
        ctx = self._ctx(mock_storage)
        written = append_live_tick(ctx)
        assert len(written) >= 4
        laps = load_endpoint("laps", 9839)
        assert len(laps) == 1
        assert laps.iloc[0]["lap_number"] == 5
        assert laps.iloc[0]["session_key"] == 9839
        preds = load_endpoint("strategy_predictions", 9839)
        assert preds.iloc[0]["recommended_label"] == "Stay out"

    def test_dedupes_same_lap(self, mock_storage: dict[str, pd.DataFrame]) -> None:
        ctx = self._ctx(mock_storage)
        append_live_tick(ctx)
        append_live_tick(ctx)
        laps = load_endpoint("laps", 9839)
        assert len(laps) == 1

    def test_list_sessions(self, mock_storage: dict[str, pd.DataFrame]) -> None:
        append_live_tick(self._ctx(mock_storage))
        sessions = list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_key"] == 9839
