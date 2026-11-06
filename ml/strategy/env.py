"""Gymnasium environment for learning pit strategy on the validated race model.

Fixes relative to ``services.simulator.env.F1RaceEnv`` (kept unchanged):

* Reward is strategy race time only: ``-(tyre cost + pit loss) / REWARD_SCALE``
  per lap, no position/heuristic shaping, discount 1.0, so return ranks
  strategies exactly by race time (the legacy reward misranked 45 pairs).
* The two-dry-compound rule is enforced at the finish.
* Tyre wear is the calibrated fuel-corrected model, not a 70% hand-written curve.
* SC/VSC laps are drawn from a Markov timeline per circuit, so the agent learns
  the value of waiting for a cheap neutralised stop.
* The agent observes *estimated* race parameters (truth plus estimation noise),
  matching what is knowable in a race; it never sees the future SC timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ml.strategy.race_model import DRY, TyreParams, project_physical

REWARD_SCALE = 10.0
ILLEGAL_PENALTY = 50.0  # = 500 s; more than any legal strategy difference
MAX_AGE = 60
OBS_DIM = 18
OBS_DIM_V2 = OBS_DIM + 3
EVIDENCE_SCALE = 300.0  # clean field laps per compound at which evidence saturates
START_COMPOUND_P = (0.3, 0.55, 0.15)  # roughly the observed S/M/H grid mix


@dataclass(frozen=True)
class Scenario:
    params: TyreParams
    n_laps: int


@dataclass(frozen=True)
class ErrorTrajectory:
    """One real race's in-race estimation error, lap by lap (estimate minus hindsight).

    Row i is the estimate available when deciding at the end of lap i+1 (data through
    lap i); ``progress`` is that lap / race length. ``evidence`` counts clean field
    laps per dry compound behind the estimate.
    """

    progress: np.ndarray  # [L]
    tyre: np.ndarray  # [L, 8] in TyreParams.as_vector order
    evidence: np.ndarray  # [L, 3]
    pit_green: float
    pit_neutral: float
    p_neutral: float

    def at(self, progress: float) -> int:
        return int(min(np.searchsorted(self.progress, progress - 1e-9), len(self.progress) - 1))


def observation_v2(
    lap: int,
    n: int,
    c: int,
    a: int,
    two_used: bool,
    neutral: bool,
    est: TyreParams,
    evidence: np.ndarray,
) -> np.ndarray:
    """v1 plus how much evidence stands behind each compound's estimate."""
    o = np.zeros(OBS_DIM_V2, dtype=np.float32)
    o[:OBS_DIM] = observation(lap, n, c, a, two_used, neutral, est)
    o[OBS_DIM:] = np.minimum(np.asarray(evidence, dtype=np.float32) / EVIDENCE_SCALE, 1.0)
    return o


def observation(
    lap: int, n: int, c: int, a: int, two_used: bool, neutral: bool, est: TyreParams
) -> np.ndarray:
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[0] = lap / n
    o[1] = (n - lap) / 80.0
    o[2] = n / 80.0
    o[3 + c] = 1.0
    o[6] = a / 50.0
    o[7] = float(two_used)
    o[8] = float(neutral)
    o[9], o[10] = est.off[0], est.off[1]
    o[11:14] = np.asarray(est.deg) * 10.0
    o[14:17] = np.asarray(est.curv) * 100.0
    o[17] = (est.pit_green - est.pit_neutral) / 20.0
    return o


class StrategyEnv(gym.Env[np.ndarray, int]):  # type: ignore[misc]
    """One car, one dry race. Actions: 0 stay, 1 SOFT, 2 MEDIUM, 3 HARD."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        scenarios: list[Scenario],
        randomize: float = 0.2,
        estimate_noise: float = 0.15,
        mean_neutral_len: float = 4.0,
        errors: list[ErrorTrajectory] | None = None,
        obs_version: str = "v1",
    ) -> None:
        """``errors``: replay measured estimation-error trajectories instead of the
        fixed lognormal ``estimate_noise`` (P1 calibration)."""
        super().__init__()
        if not scenarios:
            raise ValueError("Need at least one training scenario")
        if obs_version not in ("v1", "v2"):
            raise ValueError(f"Unknown observation version {obs_version}")
        self.scenarios = scenarios
        self.randomize = randomize
        self.estimate_noise = estimate_noise
        self.mean_neutral_len = mean_neutral_len
        self.errors = errors
        self.obs_version = obs_version
        self.traj: ErrorTrajectory | None = None
        self.action_space = spaces.Discrete(4)
        dim = OBS_DIM_V2 if obs_version == "v2" else OBS_DIM
        self.obs_dim = dim
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(dim,), dtype=np.float32)

    def _perturb(self, p: TyreParams, scale: float) -> TyreParams:
        rng: np.random.Generator = self.np_random
        k = np.exp(rng.normal(0.0, scale, size=11))
        return TyreParams(
            off=(p.off[0] * k[0], p.off[1] * k[1], 0.0),
            deg=(p.deg[0] * k[2], p.deg[1] * k[3], p.deg[2] * k[4]),
            curv=(p.curv[0] * k[5], p.curv[1] * k[6], p.curv[2] * k[7]),
            pit_green=float(p.pit_green * k[8] ** 0.5),
            pit_neutral=float(min(p.pit_neutral * k[9] ** 0.5, p.pit_green)),
            p_neutral=float(np.clip(p.p_neutral * k[10], 0.0, 0.4)),
        )

    def _timeline(self, n: int, p_neutral: float) -> np.ndarray:
        """Markov SC/VSC laps with the scenario's neutralised share."""
        stay = 1.0 - 1.0 / self.mean_neutral_len
        start = p_neutral / (self.mean_neutral_len * max(1e-6, 1.0 - p_neutral))
        out = np.zeros(n + 2, dtype=bool)
        on = False
        for lap in range(2, n + 1):
            u = self.np_random.random()
            on = (u < stay) if on else (u < start)
            out[lap] = on
        return out

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        sc = self.scenarios[int(self.np_random.integers(len(self.scenarios)))]
        self.truth = self._perturb(sc.params, self.randomize)
        self.est = self._perturb(self.truth, self.estimate_noise)
        if self.errors:
            self.traj = self.errors[int(self.np_random.integers(len(self.errors)))]
        self.n = sc.n_laps
        self.neutral = self._timeline(self.n, self.truth.p_neutral)
        self.lap = 1
        self.c = int(self.np_random.choice(3, p=START_COMPOUND_P))
        self.a = int(self.np_random.integers(1, 5))
        self.used = {self.c}
        return self._obs(), {}

    def estimate(self) -> tuple[TyreParams, np.ndarray]:
        """What the car knows at this decision: truth plus a replayed real error."""
        if self.traj is None:
            return self.est, np.full(3, EVIDENCE_SCALE)
        i = self.traj.at(self.lap / self.n)
        vec = project_physical(self.truth.as_vector() + self.traj.tyre[i])
        green = max(5.0, self.truth.pit_green + self.traj.pit_green)
        est = TyreParams.from_vector(
            vec,
            pit_green=green,
            pit_neutral=float(np.clip(self.truth.pit_neutral + self.traj.pit_neutral, 3.0, green)),
            p_neutral=float(np.clip(self.truth.p_neutral + self.traj.p_neutral, 0.0, 0.4)),
        )
        return est, self.traj.evidence[i]

    def _obs(self) -> np.ndarray:
        est, evidence = self.estimate()
        args = (self.lap, self.n, self.c, self.a, len(self.used) >= 2, bool(self.neutral[self.lap]))
        if self.obs_version == "v2":
            return observation_v2(*args, est, evidence)
        return observation(*args, est)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        neutral = bool(self.neutral[self.lap])
        cost = 0.0 if (neutral or self.lap == 1) else float(self.truth.stint_cost(self.c, self.a))
        if self.lap >= self.n:
            action = 0
        if self.a >= MAX_AGE and action == 0:
            action = 1 + (2 if self.c != 2 else 1)  # forced stop; counted as the agent's
        if action:
            cost += self.truth.pit_neutral if neutral else self.truth.pit_green
            self.c, self.a = int(action) - 1, 1
            self.used.add(self.c)
        else:
            self.a += 1
        reward = -cost / REWARD_SCALE
        terminated = self.lap >= self.n
        if terminated and len(self.used) < 2:
            reward -= ILLEGAL_PENALTY
        self.lap += 1
        obs = self._obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, reward, terminated, False, {"stops": len(self.used)}


def scenarios_from(
    truths: dict[tuple[int, int], TyreParams], laps: dict[tuple[int, int], int]
) -> list[Scenario]:
    return [Scenario(params=truths[k], n_laps=laps[k]) for k in sorted(truths)]


__all__ = [
    "DRY",
    "ErrorTrajectory",
    "Scenario",
    "StrategyEnv",
    "observation",
    "observation_v2",
    "scenarios_from",
]
