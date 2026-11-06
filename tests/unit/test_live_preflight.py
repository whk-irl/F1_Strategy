"""Preflight tests use provider doubles and never contact OpenF1."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from services.live import preflight

NOW = datetime(2026, 10, 25, 19, 10, tzinfo=UTC)
CREDENTIALS = {"OPENF1_USERNAME": "private-user", "OPENF1_PASSWORD": "private-password"}
SESSION = {
    "session_key": 1400,
    "meeting_key": 1200,
    "session_name": "Race",
    "circuit_short_name": "Austin",
    "date_start": "2026-10-25T19:00:00+00:00",
    "date_end": "2026-10-25T22:00:00+00:00",
    "unexpected_provider_field": "private-provider-payload",
}


def active_feed():
    feed = MagicMock()

    def get(endpoint, params, **kwargs):
        if endpoint == "sessions":
            return [dict(SESSION)]
        if endpoint == "drivers":
            return [{"driver_number": 1, "full_name": "Not part of report"}]
        if endpoint in {"pit", "race_control", "stints"}:
            return []
        return [{"date": (NOW - timedelta(seconds=5)).isoformat(), "secret_body": "hidden"}]

    feed.get.side_effect = get
    return feed


def test_ready_report_is_safe_and_bounds_current_samples():
    feed = active_feed()
    report = preflight.run_preflight(feed, now=NOW, driver=44, environ=CREDENTIALS)
    assert report["status"] == "ready"
    assert report["session_active"] is True
    assert report["credentials"]["automatic_renewal_available"] is True
    assert report["endpoints"]["location"]["sample_age_s"] == 5
    assert report["endpoints"]["pit"]["sample_count"] == 0
    assert "unexpected_provider_field" not in report["latest_session"]
    assert all(secret not in str(report) for secret in [*CREDENTIALS.values(), "hidden"])
    for call in feed.get.call_args_list:
        endpoint = call.args[0]
        if endpoint in preflight.REQUIRED_FRESH:
            assert call.kwargs["after"] == "2026-10-25T19:08:00+00:00"
            assert call.kwargs["before"] == NOW.isoformat()
        if endpoint == "car_data":
            assert call.args[1]["driver_number"] == 44


@pytest.mark.parametrize("sessions", [[], [{**SESSION, "date_end": "2026-10-25T19:05:00Z"}]])
def test_no_active_session_is_waiting_not_a_connectivity_failure(sessions):
    feed = MagicMock()
    feed.get.return_value = sessions
    report = preflight.run_preflight(feed, now=NOW, environ=CREDENTIALS)
    assert report["status"] == "waiting"
    assert report["failures"] == []
    assert report["session_active"] is False
    feed.get.assert_called_once()


def test_missing_live_credentials_fail_without_polling_active_telemetry():
    feed = active_feed()
    report = preflight.run_preflight(feed, now=NOW, environ={})
    assert report["status"] == "failed"
    assert report["failures"] == ["live_credentials_missing"]
    feed.get.assert_called_once()


def test_offline_only_checks_configuration_and_warns_about_renewal():
    feed = MagicMock()
    report = preflight.run_preflight(
        feed, now=NOW, offline=True, environ={"OPENF1_TOKEN": "private-token"}
    )
    assert report["status"] == "offline"
    assert report["session_active"] is None
    assert "token_only_no_automatic_renewal" in report["notes"]
    assert "private-token" not in str(report)
    feed.get.assert_not_called()
    assert preflight.run_preflight(feed, offline=True, environ={})["status"] == "failed"


def test_provider_errors_never_expose_provider_bodies_or_exception_text():
    feed = MagicMock()
    feed.get.side_effect = RuntimeError("request failed: private-token; private-user")
    report = preflight.run_preflight(feed, now=NOW, environ=CREDENTIALS)
    assert report["status"] == "failed"
    assert report["failures"] == ["sessions_unavailable"]
    assert "private-token" not in str(report)
    assert "private-user" not in str(report)


def test_endpoint_errors_are_recorded_and_other_checks_continue():
    feed = active_feed()
    fallback = feed.get.side_effect

    def get(endpoint, params, **kwargs):
        if endpoint == "weather":
            raise ValueError("private response body")
        return fallback(endpoint, params, **kwargs)

    feed.get.side_effect = get
    report = preflight.run_preflight(feed, now=NOW, environ=CREDENTIALS)
    assert report["status"] == "failed"
    assert report["failures"] == ["weather_unavailable"]
    assert report["endpoints"]["car_data"]["fresh"] is True
    assert "private response" not in str(report)


@pytest.mark.parametrize("date", [None, "not-a-date", "2026-10-25T19:08:59Z"])
def test_absent_malformed_or_stale_timing_fails(date):
    feed = active_feed()
    fallback = feed.get.side_effect

    def get(endpoint, params, **kwargs):
        if endpoint == "location":
            return [{"date": date}]
        return fallback(endpoint, params, **kwargs)

    feed.get.side_effect = get
    report = preflight.run_preflight(feed, now=NOW, environ=CREDENTIALS)
    assert report["status"] == "failed"
    assert report["failures"] == ["location_missing_or_stale"]


@pytest.mark.parametrize("session", [{"session_key": 1}, {**SESSION, "date_end": "NaT"}])
def test_invalid_session_metadata_is_a_failed_check(session):
    feed = MagicMock()
    feed.get.return_value = [session]
    report = preflight.run_preflight(feed, now=NOW, environ=CREDENTIALS)
    assert report["failures"] == ["invalid_session_metadata"]


def test_cli_loads_environment_at_entry_and_writes_safe_report(tmp_path, monkeypatch, capsys):
    feed = MagicMock()
    factory = MagicMock(return_value=feed)
    load = MagicMock()
    monkeypatch.setattr("dotenv.load_dotenv", load)
    monkeypatch.setattr(preflight, "RaceFeed", factory)
    monkeypatch.setenv("OPENF1_TOKEN", "private-token")
    monkeypatch.delenv("OPENF1_USERNAME", raising=False)
    monkeypatch.delenv("OPENF1_PASSWORD", raising=False)
    output = tmp_path / "safe" / "preflight.json"
    assert preflight.main(["--provider", "openf1", "--offline", "--output", str(output)]) == 0
    load.assert_called_once()
    feed.get.assert_not_called()
    stdout = capsys.readouterr().out
    assert output.read_text() == stdout
    assert '"status": "offline"' in stdout
    assert "private-token" not in stdout
    monkeypatch.delenv("OPENF1_TOKEN")
    assert preflight.main(["--provider", "openf1", "--offline"]) == 1


def test_cli_rejects_invalid_car_number():
    with pytest.raises(SystemExit) as error:
        preflight.main(["--driver", "0", "--offline"])
    assert error.value.code == 2


def test_free_offline_needs_no_credentials_or_provider_access():
    probe = MagicMock()
    report = preflight.run_free_preflight(offline=True, probe=probe)
    assert report["status"] == "offline"
    assert report["credentials"]["required"] is False
    probe.assert_not_called()


@pytest.mark.parametrize("status", ["ready", "waiting", "failed"])
def test_free_probe_reports_status_and_drops_any_raw_provider_content(status):
    probe = MagicMock(
        return_value={
            "status": status,
            "connected": status != "failed",
            "sample_count": 5,
            "sample_age_s": 2.0,
            "latest_sample_at": NOW.isoformat(),
            "raw": "private-provider-body",
            "error": "private-token",
        }
    )
    report = preflight.run_free_preflight(probe_seconds=15, probe=probe)
    assert report["status"] == status
    assert report["endpoints"]["signalr"]["sample_count"] == 5
    assert "private" not in str(report)
    probe.assert_called_once_with(15)


def test_free_probe_failure_and_nonfinite_age_are_safe():
    probe = MagicMock(side_effect=RuntimeError("private-provider-body"))
    report = preflight.run_free_preflight(probe=probe)
    assert report["status"] == "failed"
    assert report["failures"] == ["free_timing_unavailable"]
    assert "private" not in str(report)
    probe.side_effect = None
    probe.return_value = {"connected": True, "status": "waiting", "sample_age_s": float("inf")}
    report = preflight.run_free_preflight(probe=probe)
    assert report["endpoints"]["signalr"]["sample_age_s"] is None
    probe.return_value = {"connected": True, "status": "ready", "sample_count": 0}
    report = preflight.run_free_preflight(probe=probe)
    assert report["status"] == "failed"
    assert report["failures"] == ["free_timing_missing_or_stale"]


def test_cli_defaults_to_the_free_provider_without_constructing_paid_client(monkeypatch, capsys):
    free = MagicMock(return_value={"status": "waiting"})
    factory = MagicMock()
    monkeypatch.setattr(preflight, "run_free_preflight", free)
    monkeypatch.setattr(preflight, "RaceFeed", factory)
    assert preflight.main(["--probe-seconds", "12"]) == 0
    free.assert_called_once_with(offline=False, probe_seconds=12)
    factory.assert_not_called()
    assert '"status": "waiting"' in capsys.readouterr().out
