"""Exact dynamic-programming pit planner for one car.

State at the start of lap ``l``: compound ``c``, tyre age ``a`` during that lap and
``f`` = whether two different dry compounds have already been run. Each future
lap is independently neutralised (SC/VSC) with probability ``p_neutral``; the
current lap's status is observed. A stop is made at the end of a lap.

The value is expected remaining strategy-dependent race time (seconds). Finishing
without two dry compounds costs ``ILLEGAL`` (the sporting rule the old reward and
the IQL reward both omitted, which made "never pit" nearly optimal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.strategy.race_model import DRY, TyreParams

ILLEGAL = 1e4
STAY = 0  # actions: 0 stay, 1 SOFT, 2 MEDIUM, 3 HARD (matches the legacy env)


@dataclass
class Plan:
    """DP tables for one parameter set and race length."""

    params: TyreParams
    n_laps: int
    max_age: int
    V: np.ndarray  # [n_laps + 2, 3, max_age + 2, 2]

    def q_values(
        self, lap: int, compound: int, age: int, two_used: bool, neutral: bool
    ) -> np.ndarray:
        """Expected remaining time after deciding at the END of ``lap`` (4 actions)."""
        p, n, A = self.params, self.n_laps, self.max_age
        a = min(age, A)
        lap_cost = 0.0 if neutral else float(p.stint_cost(compound, a))
        q = np.full(4, np.inf)
        if lap >= n:
            q[STAY] = lap_cost + (0.0 if two_used else ILLEGAL)
            return q
        nxt = self.V[lap + 1]
        f = int(two_used)
        q[STAY] = lap_cost + (nxt[compound, min(a + 1, A + 1), f])
        loss = p.pit_neutral if neutral else p.pit_green
        for c2 in DRY:
            f2 = int(two_used or c2 != compound)
            q[1 + c2] = lap_cost + loss + nxt[c2, 1, f2]
        return q

    def act(self, lap: int, compound: int, age: int, two_used: bool, neutral: bool) -> int:
        return int(np.argmin(self.q_values(lap, compound, age, two_used, neutral)))


def solve(
    params: TyreParams, n_laps: int, max_age: int = 70, p_neutral: float | None = None
) -> Plan:
    """Backward induction. O(laps * compounds * age) — milliseconds."""
    p = params.p_neutral if p_neutral is None else p_neutral
    A = max_age
    ages = np.arange(A + 2, dtype=float)
    cost = np.stack([params.stint_cost(c, ages) for c in DRY])  # [3, A+2]
    cost[:, A + 1] = ILLEGAL  # exceeding the age cap is not allowed
    V = np.zeros((n_laps + 2, 3, A + 2, 2))
    V[n_laps + 1, :, :, 0] = ILLEGAL
    V[n_laps + 1, :, :, 1] = 0.0
    for lap in range(n_laps, 0, -1):
        if lap == n_laps:
            cont_stay = V[n_laps + 1]
            q_g = cost[:, :, None] + cont_stay
            q_n = cont_stay.copy()
            V[lap] = p * q_n + (1 - p) * q_g
            continue
        nxt = V[lap + 1]
        stay = np.concatenate([nxt[:, 1:, :], nxt[:, -1:, :]], axis=1)  # age+1
        # pit to c2: flag becomes 1 if c2 != c
        pit_best = np.full((3, 2), np.inf)
        for c in DRY:
            for f in (0, 1):
                pit_best[c, f] = min(nxt[c2, 1, int(f or c2 != c)] for c2 in DRY)
        cont_g = np.minimum(stay, params.pit_green + pit_best[:, None, :])
        cont_n = np.minimum(stay, params.pit_neutral + pit_best[:, None, :])
        V[lap] = p * cont_n + (1 - p) * (cost[:, :, None] + cont_g)
    return Plan(params=params, n_laps=n_laps, max_age=A, V=V)


def solve_known_timeline(
    params: TyreParams, neutral: np.ndarray, n_laps: int, max_age: int = 70
) -> Plan:
    """Oracle: neutral laps known in advance (hindsight lower bound)."""
    A = max_age
    ages = np.arange(A + 2, dtype=float)
    cost = np.stack([params.stint_cost(c, ages) for c in DRY])
    cost[:, A + 1] = ILLEGAL
    V = np.zeros((n_laps + 2, 3, A + 2, 2))
    V[n_laps + 1, :, :, 0] = ILLEGAL
    for lap in range(n_laps, 0, -1):
        s = bool(neutral[lap]) if lap < len(neutral) else False
        lc = 0.0 if s else cost[:, :, None]
        nxt = V[lap + 1]
        if lap == n_laps:
            V[lap] = lc + nxt
            continue
        stay = np.concatenate([nxt[:, 1:, :], nxt[:, -1:, :]], axis=1)
        pit_best = np.array(
            [[min(nxt[c2, 1, int(f or c2 != c)] for c2 in DRY) for f in (0, 1)] for c in DRY]
        )
        loss = params.pit_neutral if s else params.pit_green
        V[lap] = lc + np.minimum(stay, loss + pit_best[:, None, :])
    plan = Plan(params=params, n_laps=n_laps, max_age=A, V=V)
    return plan
