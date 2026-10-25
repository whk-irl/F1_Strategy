"""Safe, bounded readiness checks for the delayed OpenF1 live race view.

Run with ``python -m services.live.preflight --output runs/live/preflight.json``.
The report deliberately excludes credentials, HTTP responses and exception text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from services.live.race_data import RaceFeed, timestamp

POLL_INTERVAL_S = 15
PLAYBACK_BUFFER_S = 20
MAX_SAMPLE_AGE_S = 60
ENDPOINTS = (
    "drivers",
    "laps",
    "position",
    "intervals",
    "stints",
    "pit",
    "weather",
    "race_control",
    "location",
    "car_data",
)
REQUIRED_FRESH = frozenset({"position", "location", "car_data"})
SESSION_FIELDS = (
    "session_key",
    "meeting_key",
    "session_name",
    "circuit_short_name",
    "country_name",
    "location",
    "date_start",
    "date_end",
)


def credential_status(environ: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Report only presence, never environment values or account identifiers."""
    values = os.environ if environ is None else environ
    return {
        "token_present": bool(values.get("OPENF1_TOKEN", "").strip()),
        "username_present": bool(values.get("OPENF1_USERNAME", "").strip()),
        "password_present": bool(values.get("OPENF1_PASSWORD", "").strip()),
    }


def _sample_summary(rows: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    dates = []
    for row in rows:
        value = row.get("date", row.get("date_start"))
        if value:
            try:
                parsed = timestamp(value)
                if not pd.isna(parsed):
                    dates.append(parsed)
            except (ValueError, TypeError, OverflowError):
                continue
    newest = max(dates, default=None)
    age = (timestamp(now) - newest).total_seconds() if newest is not None else None
    return {
        "sample_count": len(rows),
        "latest_sample_at": newest.isoformat() if newest is not None else None,
        "sample_age_s": round(max(0.0, age), 2) if age is not None else None,
    }


def run_preflight(
    feed: RaceFeed,
    *,
    now: datetime | None = None,
    offline: bool = False,
    driver: int = 1,
    max_sample_age_s: int = MAX_SAMPLE_AGE_S,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Check configuration and active-session data without storing raw responses.

    A connected provider with no active session is ``waiting`` (exit success).
    Offline checks cannot confirm subscription access, connectivity or freshness.
    Endpoint failures, missing live credentials and absent/stale active timing fail.
    """
    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=UTC)
    checked_at = checked_at.astimezone(UTC)
    credentials = credential_status(environ)
    renewal = credentials["username_present"] and credentials["password_present"]
    configured = credentials["token_present"] or renewal
    report: dict[str, Any] = {
        "checked_at": checked_at.isoformat(),
        "status": "offline" if offline else "waiting",
        "provider": "openf1",
        "mode": "delayed_polling",
        "poll_interval_s": POLL_INTERVAL_S,
        "playback_buffer_s": PLAYBACK_BUFFER_S,
        "freshness_limit_s": max_sample_age_s,
        "credentials": {**credentials, "automatic_renewal_available": renewal},
        "latest_session": None,
        "session_active": None if offline else False,
        "endpoints": {},
        "failures": [] if configured else ["live_credentials_missing"],
        "notes": [],
    }
    failures: list[str] = report["failures"]
    if credentials["token_present"] and not renewal:
        report["notes"].append("token_only_no_automatic_renewal")
    if offline:
        report["notes"].append("network_and_subscription_access_not_checked")
        if failures:
            report["status"] = "failed"
        return report
    try:
        sessions = feed.get("sessions", {"session_key": "latest"}, refresh=True)
        report["endpoints"]["sessions"] = {"connected": True, "sample_count": len(sessions)}
    except Exception:
        # Do not include exception messages: they may contain request URLs or secrets.
        report["endpoints"]["sessions"] = {"connected": False}
        failures.append("sessions_unavailable")
        report["status"] = "failed"
        return report
    if not sessions:
        report["notes"].append("no_session_published")
        report["status"] = "failed" if failures else "waiting"
        return report
    try:
        latest = max(sessions, key=lambda session: timestamp(session["date_start"]))
        start, end = timestamp(latest["date_start"]), timestamp(latest["date_end"])
        key = int(latest["session_key"])
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError("Invalid session dates")
        report["latest_session"] = {
            field: latest[field] for field in SESSION_FIELDS if field in latest
        }
        active = start <= timestamp(checked_at) <= end and not latest.get("is_cancelled", False)
        report["session_active"] = bool(active)
    except (KeyError, TypeError, ValueError, OverflowError):
        failures.append("invalid_session_metadata")
        report["status"] = "failed"
        return report
    if not active:
        report["notes"].append("no_active_session")
        report["status"] = "failed" if failures else "waiting"
        return report
    if not configured:
        report["status"] = "failed"
        return report
    window = max(120, max_sample_age_s * 2)
    after = (checked_at - timedelta(seconds=window)).isoformat()
    before = checked_at.isoformat()
    for endpoint in ENDPOINTS:
        params = {"session_key": key}
        if endpoint == "car_data":
            params["driver_number"] = driver
        try:
            bounded = endpoint in REQUIRED_FRESH
            rows = feed.get(
                endpoint,
                params,
                after=after if bounded else None,
                before=before if bounded else None,
                refresh=True,
            )
            summary = {"connected": True, **_sample_summary(rows, checked_at)}
            report["endpoints"][endpoint] = summary
            if endpoint == "drivers" and not rows:
                failures.append("drivers_missing")
            if endpoint in REQUIRED_FRESH:
                age = summary["sample_age_s"]
                summary["fresh"] = age is not None and age <= max_sample_age_s
                if not summary["fresh"]:
                    failures.append(f"{endpoint}_missing_or_stale")
        except Exception:
            report["endpoints"][endpoint] = {"connected": False}
            failures.append(f"{endpoint}_unavailable")
    if not report["endpoints"].get("laps", {}).get("sample_count"):
        report["notes"].append("circuit_map_waiting_for_first_recorded_lap")
    report["status"] = "failed" if failures else "ready"
    return report


def run_free_preflight(
    *,
    offline: bool = False,
    probe_seconds: int = 30,
    probe: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Probe the experimental free SignalR adapter without requiring paid secrets."""
    report: dict[str, Any] = {
        "checked_at": datetime.now(UTC).isoformat(),
        "status": "offline" if offline else "waiting",
        "provider": "free_signalr",
        "mode": "experimental_incremental_timing",
        "poll_interval_s": POLL_INTERVAL_S,
        "playback_buffer_s": PLAYBACK_BUFFER_S,
        "credentials": {"required": False, "automatic_renewal_available": False},
        "latest_session": None,
        "endpoints": {},
        "failures": [],
        "notes": ["gps_and_full_telemetry_not_guaranteed", "temporal_iql_not_promoted"],
    }
    if offline:
        report["notes"].append("network_and_active_timing_not_checked")
        return report
    try:
        if probe is None:
            from services.live.free_timing import probe_free_timing

            probe = probe_free_timing
        result = probe(probe_seconds)
        connected = result.get("connected") is True
        count = result.get("sample_count", 0)
        age = result.get("sample_age_s")
        summary: dict[str, Any] = {
            "connected": connected,
            "sample_count": count if isinstance(count, int) and count >= 0 else 0,
            "sample_age_s": age
            if isinstance(age, (int, float)) and math.isfinite(age) and age >= 0
            else None,
        }
        latest = result.get("latest_sample_at")
        if isinstance(latest, str):
            try:
                parsed = timestamp(latest)
                if not pd.isna(parsed):
                    summary["latest_sample_at"] = parsed.isoformat()
            except (TypeError, ValueError, OverflowError):
                pass
        report["endpoints"]["signalr"] = summary
        state = result.get("status")
        if not connected or state == "failed":
            report["status"] = "failed"
            report["failures"].append("free_timing_unavailable")
        elif state == "ready":
            age = summary["sample_age_s"]
            if summary["sample_count"] and age is not None and age <= MAX_SAMPLE_AGE_S:
                report["status"] = "ready"
            else:
                report["status"] = "failed"
                report["failures"].append("free_timing_missing_or_stale")
        else:
            report["notes"].append("waiting_for_active_timing")
        return report
    except Exception:
        report["status"] = "failed"
        report["endpoints"]["signalr"] = {"connected": False}
        report["failures"].append("free_timing_unavailable")
        return report


def main(argv: Sequence[str] | None = None) -> int:
    """Load local secrets only at CLI entry, then print/write a sanitised report."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("free", "openf1"), default="free")
    parser.add_argument("--offline", action="store_true", help="Check configuration without HTTP")
    parser.add_argument("--probe-seconds", type=int, default=30, help="Free SignalR probe duration")
    parser.add_argument("--driver", type=int, default=1, help="Car number for telemetry check")
    parser.add_argument("--output", type=Path, help="Save the same safe JSON report to this path")
    parser.add_argument("--max-sample-age", type=int, default=MAX_SAMPLE_AGE_S, metavar="SECONDS")
    args = parser.parse_args(argv)
    if args.driver < 1 or args.max_sample_age < 1 or args.probe_seconds < 1:
        parser.error("--driver, --max-sample-age and --probe-seconds must be positive")
    report = (
        run_free_preflight(offline=args.offline, probe_seconds=args.probe_seconds)
        if args.provider == "free"
        else run_preflight(
            RaceFeed(),
            offline=args.offline,
            driver=args.driver,
            max_sample_age_s=args.max_sample_age,
        )
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
