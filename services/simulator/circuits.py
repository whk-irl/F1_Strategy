"""Map calendar identities to the existing 2024 circuit calibration table."""

from __future__ import annotations

LOCATION_REFERENCE: dict[str, int] = {
    "Sakhir": 1,
    "Jeddah": 2,
    "Melbourne": 3,
    "Suzuka": 4,
    "Shanghai": 5,
    "Miami": 6,
    "Miami Gardens": 6,
    "Imola": 7,
    "Monaco": 8,
    "Monte Carlo": 8,
    "Montréal": 9,
    "Montreal": 9,
    "Barcelona": 10,
    "Catalunya": 10,
    "Spielberg": 11,
    "Silverstone": 12,
    "Budapest": 13,
    "Hungaroring": 13,
    "Spa-Francorchamps": 14,
    "Zandvoort": 15,
    "Monza": 16,
    "Baku": 17,
    "Marina Bay": 18,
    "Singapore": 18,
    "Austin": 19,
    "Mexico City": 20,
    "São Paulo": 21,
    "Sao Paulo": 21,
    "Interlagos": 21,
    "Las Vegas": 22,
    "Lusail": 23,
    "Yas Island": 24,
    "Yas Marina": 24,
    "Yas Marina Circuit": 24,
    "Abu Dhabi": 24,
}

LEGACY_REFERENCES: dict[int, list[int]] = {
    2022: [1, 2, 3, 7, 6, 10, 8, 17, 9, 12, 11, 0, 13, 14, 15, 16, 18, 4, 19, 20, 21, 24],
    2023: [1, 2, 3, 17, 6, 8, 10, 9, 11, 12, 13, 14, 15, 16, 18, 4, 23, 19, 20, 21, 22, 24],
    2024: list(range(1, 25)),
    2025: [3, 5, 4, 1, 2, 6, 7, 8, 10, 9, 11, 12, 14, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
}


def legacy_reference(year: int, round_number: int) -> int:
    """Use neutral calibration for unknown circuits instead of another circuit's values."""
    rounds = LEGACY_REFERENCES.get(year, [])
    return rounds[round_number - 1] if 1 <= round_number <= len(rounds) else 0
