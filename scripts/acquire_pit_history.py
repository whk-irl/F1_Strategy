"""Cache completed OpenF1 race pit and lap records for pit-time modelling.

Run ``uv run python scripts/acquire_pit_history.py --years 2023,2024,2025,2026``.
The existing ArchiveClient stores each immutable response under data/lake/openf1_raw.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from services.ingestion.acquire import write_json
from services.ingestion.openf1_archive import ArchiveClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2023,2024,2025,2026")
    parser.add_argument("--root", type=Path, default=Path("data/lake"))
    args = parser.parse_args()
    client = ArchiveClient(args.root / "openf1_raw")
    sessions = []
    acquisition_manifest = args.root / "manifest.json"
    acquired = (
        json.loads(acquisition_manifest.read_text(encoding="utf-8")).get("sessions", {})
        if acquisition_manifest.exists()
        else {}
    )
    known_rounds = {
        int(item["session_key"]): int(key.split("/")[1])
        for key, item in acquired.items()
        if item.get("session_key") is not None and item.get("status") == "complete"
    }
    cutoff = datetime.now(UTC) - timedelta(days=1)
    for year in [int(value) for value in args.years.split(",")]:
        calendar = sorted(
            client.get("sessions", year=year, session_name="Race"),
            key=lambda item: item["date_start"],
        )
        for round_number, session in enumerate(calendar, start=1):
            if (
                session.get("is_cancelled")
                or datetime.fromisoformat(session["date_start"]) > cutoff
            ):
                continue
            key = int(session["session_key"])
            try:
                pits = client.get("pit", session_key=key)
                laps = client.get("laps", session_key=key)
            except Exception as error:
                print(f"{year} {key}: unavailable ({error})", flush=True)
                continue
            if not pits or not laps:
                continue
            sessions.append(
                {
                    "session_key": key,
                    "year": year,
                    "round_number": known_rounds.get(key, round_number),
                    "circuit_short_name": session["circuit_short_name"],
                    "date_start": session["date_start"],
                    "pit_rows": len(pits),
                    "lap_rows": len(laps),
                }
            )
            print(f"{year} {session['circuit_short_name']}: {len(pits)} stops", flush=True)
            write_json(args.root / "pit_sessions.json", {"sessions": sessions})
    print(f"Cached {len(sessions)} races", flush=True)


if __name__ == "__main__":
    main()
