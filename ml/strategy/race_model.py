"""Race-time model for single-car dry pit strategy.

Lap time on a clean green lap for driver ``d`` on lap ``l`` running compound ``c``
at tyre age ``a`` is modelled as::

    t = base_d + fuel * l + offset[c] + deg[c] * a + curv[c] * a**2

``fuel`` absorbs fuel burn and track evolution; it is identified separately from
tyre age because age resets at every stop while lap number does not. This is the
term the old simulator's per-stint LightGBM delta could not separate.

Only the strategy-dependent part (offset/deg/curv and pit losses) matters for
decisions; ``base_d`` and ``fuel`` are identical across strategies for one car.

Pit losses are measured relative to the cars that did not stop on the same laps,
separately for green and neutralised (SC/VSC) laps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

DRY = (0, 1, 2)  # SOFT, MEDIUM, HARD (compound_encoded)
N_TYRE_PARAMS = 8  # off_S, off_M, deg_S, deg_M, deg_H, curv_S, curv_M, curv_H
_SEASONS = {
    2022: """bahrain saudi australia imola miami spain monaco baku canada britain austria
        france hungary belgium netherlands monza singapore japan usa mexico brazil abudhabi""",
    2023: """bahrain saudi australia baku miami monaco spain canada austria britain hungary
        belgium netherlands monza singapore japan qatar usa mexico brazil lasvegas abudhabi""",
    2024: """bahrain saudi australia japan china miami imola monaco canada spain austria britain
        hungary belgium netherlands monza baku singapore usa mexico brazil lasvegas qatar
        abudhabi""",
    2025: """australia china japan bahrain saudi miami imola monaco spain canada austria britain
        belgium hungary netherlands monza baku singapore usa mexico brazil lasvegas qatar
        abudhabi""",
}
CALENDAR: dict[int, list[str]] = {year: names.split() for year, names in _SEASONS.items()}


def circuit_of(year: int, rnd: int) -> str:
    """Circuit key for a season/round (rounds are re-ordered across seasons)."""
    names = CALENDAR.get(int(year))
    if names is None or not 1 <= int(rnd) <= len(names):
        return f"{year}-{rnd}"
    return names[int(rnd) - 1]


@dataclass(frozen=True)
class TyreParams:
    """Strategy-relevant race parameters. Seconds; HARD offset is the reference 0."""

    off: tuple[float, float, float] = (-0.6, -0.3, 0.0)
    deg: tuple[float, float, float] = (0.08, 0.05, 0.03)
    curv: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pit_green: float = 22.0
    pit_neutral: float = 12.0
    p_neutral: float = 0.08

    def as_vector(self) -> np.ndarray:
        return np.array([self.off[0], self.off[1], *self.deg, *self.curv], dtype=float)

    @staticmethod
    def from_vector(v: np.ndarray, **kw: float) -> TyreParams:
        return TyreParams(
            off=(float(v[0]), float(v[1]), 0.0),
            deg=(float(v[2]), float(v[3]), float(v[4])),
            curv=(float(v[5]), float(v[6]), float(v[7])),
            **kw,
        )

    def stint_cost(self, compound: int, age: np.ndarray | int) -> np.ndarray:
        a = np.asarray(age, dtype=float)
        c = int(compound)
        return self.off[c] + self.deg[c] * a + self.curv[c] * a * a


# ---------------------------------------------------------------------------
# Race preparation
# ---------------------------------------------------------------------------


def _flags(status: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = status.astype(str)
    neutral = s.str.contains("4") | s.str.contains("6") | s.str.contains("7")
    red = s.str.contains("5")
    return neutral, red


@dataclass
class Race:
    """One race's laps with derived race-level timeline."""

    year: int
    rnd: int
    laps: pd.DataFrame
    n_laps: int
    neutral_lap: np.ndarray  # index 0..n_laps; True if lap neutralised for the field
    red_flag: bool
    wet: bool
    circuit: str = field(default="")

    @property
    def key(self) -> tuple[int, int]:
        return (self.year, self.rnd)


def prepare_race(df: pd.DataFrame, year: int, rnd: int) -> Race:
    laps = df[(df["year"] == year) & (df["round_number"] == rnd)].copy()
    laps["neutral"], laps["red"] = _flags(laps["track_status"])
    laps["pit_in"] = laps["pit_in_this_lap"].astype(bool)
    laps["pit_out"] = laps["pit_out_this_lap"].astype(bool)
    n = int(laps["lap_number"].max())
    share = laps.groupby("lap_number")["neutral"].mean().reindex(range(n + 1), fill_value=0.0)
    neutral_lap = (share >= 0.3).to_numpy().copy()
    neutral_lap[0] = False
    wet = bool(laps["compound_encoded"].isin([3, 4]).mean() > 0.02)
    return Race(
        year=year,
        rnd=rnd,
        laps=laps,
        n_laps=n,
        neutral_lap=neutral_lap,
        red_flag=bool(laps["red"].any()),
        wet=wet,
        circuit=str(laps["circuit_key"].iloc[0])
        if "circuit_key" in laps.columns and len(laps)
        else circuit_of(year, rnd),
    )


def clean_laps(race: Race, through_lap: int | None = None) -> pd.DataFrame:
    """Green, representative, dry laps usable for fitting tyre parameters."""
    x = race.laps
    if through_lap is not None:
        x = x[x["lap_number"] <= through_lap]
    after_neutral = np.zeros(len(x), dtype=bool)
    lap_idx = x["lap_number"].to_numpy()
    prev = np.clip(lap_idx - 1, 0, race.n_laps)
    after_neutral = race.neutral_lap[prev] | race.neutral_lap[np.clip(lap_idx, 0, race.n_laps)]
    keep = (
        x["compound_encoded"].isin(DRY)
        & x["lap_time_s"].notna()
        & x["tyre_life_laps"].notna()
        & ~x["pit_in"]
        & ~x["pit_out"]
        & ~x["neutral"]
        & ~x["red"]
        & (x["lap_number"] > 1)
        & ~after_neutral
        & x["is_accurate"].fillna(False).astype(bool)
    )
    x = x[keep]
    if x.empty:
        return x
    # Drop traffic/mistake outliers per lap relative to the field.
    med = x.groupby("lap_number")["lap_time_s"].transform("median")
    return x[(x["lap_time_s"] < med * 1.04) & (x["lap_time_s"] > med * 0.96)]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_tyre_params(
    laps: pd.DataFrame,
    prior: np.ndarray,
    prior_precision: np.ndarray,
    max_age: int = 45,
    iters: int = 6,
) -> np.ndarray:
    """Huber-IRLS ridge fit of the 8 tyre parameters toward ``prior``.

    Driver/car levels and the lap (fuel/evolution) slope are free nuisance terms.
    """
    if len(laps) < 30:
        return prior.copy()
    x = laps[laps["tyre_life_laps"] <= max_age]
    drivers = pd.Categorical(x["driver_number"].astype(str))
    nd = len(drivers.categories)
    rows = len(x)
    comp = x["compound_encoded"].to_numpy().astype(int)
    age = x["tyre_life_laps"].to_numpy().astype(float)
    lap = x["lap_number"].to_numpy().astype(float)
    y = x["lap_time_s"].to_numpy().astype(float)

    nuis = np.zeros((rows, nd + 1))
    nuis[np.arange(rows), drivers.codes] = 1.0
    nuis[:, nd] = lap
    tyre = np.zeros((rows, N_TYRE_PARAMS))
    tyre[:, 0] = comp == 0
    tyre[:, 1] = comp == 1
    for c in DRY:
        m = comp == c
        tyre[m, 2 + c] = age[m]
        tyre[m, 5 + c] = age[m] ** 2 / 10.0  # scaled for conditioning
    X = np.hstack([nuis, tyre])
    p0 = np.concatenate([np.zeros(nd + 1), prior * np.r_[np.ones(5), np.full(3, 10.0)]])
    lam = np.concatenate(
        [np.full(nd + 1, 1e-6), prior_precision / np.r_[np.ones(5), np.full(3, 100.0)]]
    )
    w = np.ones(rows)
    beta = p0.copy()
    for _ in range(iters):
        Xw = X * w[:, None]
        A = X.T @ Xw + np.diag(lam)
        b = Xw.T @ y + lam * p0
        beta = np.linalg.solve(A, b)
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        k = 1.345 * s
        w = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
    out = beta[nd + 1 :].copy()
    out[5:] /= 10.0
    return project_physical(out)


def project_physical(v: np.ndarray) -> np.ndarray:
    """Physical constraints: wear never makes a tyre faster; softer is not slower new."""
    v = v.copy()
    v[2:5] = np.maximum(v[2:5], 0.0)
    v[5:8] = np.clip(v[5:8], 0.0, 0.01)
    v[1] = min(v[1], 0.0)
    v[0] = min(v[0], v[1])
    return v


def measure_pit_losses(race: Race) -> tuple[list[float], list[float]]:
    """Net loss of each stop versus non-stopping cars on the same two laps."""
    x = race.laps
    moving = x[~x["pit_in"] & ~x["pit_out"] & x["lap_time_s"].notna()]
    ref = moving.groupby("lap_number")["lap_time_s"].median()
    green: list[float] = []
    neutral: list[float] = []
    for _, drv in x.groupby("driver_number"):
        drv = drv.set_index("lap_number").sort_index()
        resid = (drv["lap_time_s"] - ref.reindex(drv.index)).where(
            ~drv["pit_in"] & ~drv["pit_out"] & ~drv["neutral"]
        )
        offset = float(resid.median()) if resid.notna().sum() >= 5 else 0.0
        for lap in drv.index[drv["pit_in"]]:
            if lap + 1 not in drv.index or lap not in ref.index or lap + 1 not in ref.index:
                continue
            t_in, t_out = drv.at[lap, "lap_time_s"], drv.at[lap + 1, "lap_time_s"]
            if not (np.isfinite(t_in) and np.isfinite(t_out)) or drv.at[lap, "red"]:
                continue
            loss = t_in + t_out - ref[lap] - ref[lap + 1] - 2 * offset
            if not 3.0 < loss < 45.0:
                continue
            (neutral if race.neutral_lap[int(lap)] else green).append(float(loss))
    return green, neutral


def race_truth(race: Race, prior: TyreParams, strength: float = 0.25) -> TyreParams:
    """Hindsight parameters from the complete race (evaluation only).

    ``prior`` should be the pooled (empirical-Bayes) fit across races; a moderate
    pull toward it stabilises compounds a race barely used.
    """
    vec = fit_tyre_params(clean_laps(race), prior.as_vector(), _precision(weak=False) * strength)
    g, n = measure_pit_losses(race)
    pg = float(np.median(g)) if len(g) >= 3 else prior.pit_green
    pn = float(np.median(n)) if len(n) >= 3 else min(prior.pit_neutral, pg)
    pneu = float(race.neutral_lap[2:].mean()) if race.n_laps > 2 else prior.p_neutral
    return TyreParams.from_vector(vec, pit_green=pg, pit_neutral=min(pn, pg), p_neutral=pneu)


def empirical_bayes_truths(
    races: dict[tuple[int, int], Race],
    pool_keys: set[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], TyreParams]:
    """Two-pass hindsight fits: weak per-race fits -> pooled prior -> regularised fits.

    ``pool_keys`` restricts which races form the pooled prior (pass the training
    split so validation/test races cannot shape how they themselves are scored).
    """
    first = {k: race_truth(r, TyreParams(), strength=0.05) for k, r in races.items()}
    pool = [v for k, v in first.items() if pool_keys is None or k in pool_keys]
    pooled = pool_priors(pool)
    return {k: race_truth(r, replace(pooled, p_neutral=pooled.p_neutral)) for k, r in races.items()}


def _precision(weak: bool) -> np.ndarray:
    # Offsets/deg in s and s/lap; curvature in s/lap^2. Larger = trust prior more.
    base = np.array([2.0, 2.0, 400.0, 400.0, 400.0, 4e5, 4e5, 4e5])
    return base * (0.05 if weak else 1.0)


def pool_priors(fits: list[TyreParams]) -> TyreParams:
    if not fits:
        return TyreParams()
    v = np.median(np.stack([f.as_vector() for f in fits]), axis=0)
    return TyreParams.from_vector(
        v,
        pit_green=float(np.median([f.pit_green for f in fits])),
        pit_neutral=float(np.median([f.pit_neutral for f in fits])),
        p_neutral=float(np.mean([f.p_neutral for f in fits])),
    )


def prerace_prior(
    truths: dict[tuple[int, int], TyreParams],
    races: dict[tuple[int, int], Race],
    target: Race,
) -> TyreParams:
    """Chronological prior: only races strictly before ``target``."""
    earlier = [k for k in truths if k < target.key]
    glob = pool_priors([truths[k] for k in earlier])
    same = [truths[k] for k in earlier if races[k].circuit == target.circuit]
    if not same:
        return glob
    circ = pool_priors(same)
    w = len(same) / (len(same) + 1.0)
    v = w * circ.as_vector() + (1 - w) * glob.as_vector()
    return TyreParams.from_vector(
        v,
        pit_green=w * circ.pit_green + (1 - w) * glob.pit_green,
        pit_neutral=w * circ.pit_neutral + (1 - w) * glob.pit_neutral,
        p_neutral=w * circ.p_neutral + (1 - w) * glob.p_neutral,
    )


def inrace_update(race: Race, prior: TyreParams, through_lap: int) -> TyreParams:
    """Posterior using only field laps completed through ``through_lap``."""
    laps = clean_laps(race, through_lap)
    vec = fit_tyre_params(laps, prior.as_vector(), _precision(weak=False))
    return replace(
        TyreParams.from_vector(vec),
        pit_green=prior.pit_green,
        pit_neutral=prior.pit_neutral,
        p_neutral=prior.p_neutral,
    )
