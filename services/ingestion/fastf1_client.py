"""Thin wrapper around the FastF1 library.

Responsibilities:
- Enable and configure the disk cache once, at construction time.
- Expose a narrow interface that returns plain DataFrames (no FastF1 objects
  leak out of this module).
- Raise a consistent ``FastF1ClientError`` for any FastF1 / network failures
  so callers don't need to know FastF1's exception hierarchy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd

from .config import IngestionSettings

logger = logging.getLogger(__name__)

# Track status codes from the F1 timing feed.
TRACK_STATUS_CODES: dict[str, str] = {
    "1": "clear",
    "2": "yellow",
    "4": "safety_car",
    "5": "red_flag",
    "6": "virtual_safety_car",
    "7": "vsc_ending",
}


class FastF1ClientError(RuntimeError):
    """Raised when FastF1 cannot retrieve or parse session data."""


class FastF1Client:
    """Fetches F1 session data via the FastF1 library.

    Args:
        settings: Ingestion service configuration.  The FastF1 cache is
            enabled once at construction time using ``settings.fastf1_cache_path``.
    """

    def __init__(self, settings: IngestionSettings) -> None:
        cache_path = Path(settings.fastf1_cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_path))
        logger.info("FastF1 cache enabled at %s", cache_path)

    def get_session_laps(
        self,
        year: int,
        round_number: int,
        session_type: str = "R",
    ) -> pd.DataFrame:
        """Return all laps for the requested session.

        Args:
            year: Championship year (2018+).
            round_number: Round number within the season (1-based).
            session_type: One of ``'FP1'``, ``'FP2'``, ``'FP3'``, ``'Q'``,
                ``'SQ'``, ``'S'``, or ``'R'``.

        Returns:
            A copy of ``session.laps`` as a plain ``pd.DataFrame``.

        Raises:
            FastF1ClientError: If FastF1 cannot load the session.
        """
        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception as exc:
            raise FastF1ClientError(
                f"Failed to load session {year} R{round_number} {session_type}: {exc}"
            ) from exc

        laps: pd.DataFrame = session.laps.copy()
        logger.info(
            "Loaded %d laps for %d R%d %s",
            len(laps),
            year,
            round_number,
            session_type,
        )
        return laps

    def get_session_weather(
        self,
        year: int,
        round_number: int,
        session_type: str = "R",
    ) -> pd.DataFrame:
        """Return weather data for the requested session.

        Args:
            year: Championship year.
            round_number: Round number.
            session_type: Session identifier.

        Returns:
            A copy of ``session.weather_data`` as a plain ``pd.DataFrame``.

        Raises:
            FastF1ClientError: If FastF1 cannot load the session.
        """
        try:
            session = fastf1.get_session(year, round_number, session_type)
            session.load(laps=False, telemetry=False, weather=True, messages=False)
        except Exception as exc:
            raise FastF1ClientError(
                f"Failed to load weather for {year} R{round_number} {session_type}: {exc}"
            ) from exc

        weather: pd.DataFrame = session.weather_data.copy()
        return weather
