"""Measured pit-time targets and prediction contracts."""

from __future__ import annotations

import json

from ml.pipelines.pit_time import build_pit_frame, fit_pit_model, score_pit_model
from services.simulator.pit_time import predict_pit_time


def test_pit_loss_uses_in_and_out_laps(tmp_path):
    raw = tmp_path / "openf1_raw"
    raw.mkdir()
    (tmp_path / "pit_sessions.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "session_key": 7,
                        "year": 2025,
                        "round_number": 16,
                        "circuit_short_name": "Monza",
                    }
                ]
            }
        )
    )
    (raw / "pit_session_key-7.json").write_text(
        json.dumps([{"driver_number": 44, "lap_number": 5, "lane_duration": 24.0}])
    )
    (raw / "laps_session_key-7.json").write_text(
        json.dumps(
            [
                {
                    "driver_number": 44,
                    "lap_number": lap,
                    "lap_duration": 105.0 if lap == 4 else 99.0 if lap == 5 else 90.0,
                    "is_pit_out_lap": lap == 5,
                }
                for lap in range(1, 10)
            ]
        )
    )
    frame = build_pit_frame(tmp_path)
    assert frame.iloc[0]["lane_s"] == 24.0
    assert frame.iloc[0]["net_loss_s"] == 24.0
    model = fit_pit_model(frame)
    assert predict_pit_time(model, 16)["net_loss_s"] == 24.0
    assert predict_pit_time(model, 0)["scope"] == "global"
    assert score_pit_model(model, frame)["lane_mae_s"] == 0.0
