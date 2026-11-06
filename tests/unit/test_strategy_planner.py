"""Tests for the race-time model, DP planner, evaluator and RL environment."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from ml.strategy import evaluate as ev
from ml.strategy.env import REWARD_SCALE, Scenario, StrategyEnv
from ml.strategy.planner import ILLEGAL, solve, solve_known_timeline
from ml.strategy.race_model import (
    TyreParams,
    clean_laps,
    fit_tyre_params,
    inrace_update,
    prepare_race,
)

P = TyreParams(
    off=(-0.6, -0.3, 0.0),
    deg=(0.12, 0.07, 0.04),
    curv=(0.002, 0.0, 0.0),
    pit_green=22.0,
    pit_neutral=11.0,
    p_neutral=0.1,
)


class _Race:
    """Minimal stand-in for evaluate.score (only the SC timeline is needed)."""

    def __init__(self, neutral: np.ndarray) -> None:
        self.neutral_lap = neutral


def _brute_force(params: TyreParams, n: int, neutral: np.ndarray, start: tuple[int, int]) -> float:
    best = np.inf
    race = _Race(neutral)
    for k in (0, 1, 2):
        for laps in itertools.combinations(range(1, n), k):
            for comps in itertools.product(range(3), repeat=k):
                sched = dict(zip(laps, comps, strict=True))
                best = min(best, ev.score(params, race, n, start, ev.from_schedule(sched))[0])
    return best


@pytest.mark.parametrize("sc_lap", [None, 6])
def test_dp_matches_brute_force_with_known_timeline(sc_lap: int | None) -> None:
    n = 12
    neutral = np.zeros(n + 2, dtype=bool)
    if sc_lap:
        neutral[sc_lap] = True
    plan = solve_known_timeline(P, neutral, n)
    got = ev.score(P, _Race(neutral), n, (0, 2), plan.act)[0]
    assert got == pytest.approx(_brute_force(P, n, neutral, (0, 2)), abs=1e-9)


def test_compound_rule_forces_a_stop_even_without_wear() -> None:
    flat = TyreParams(off=(0.0, 0.0, 0.0), deg=(0, 0, 0), curv=(0, 0, 0), p_neutral=0.0)
    plan = solve(flat, 30)
    total, stops, _ = ev.score(flat, _Race(np.zeros(32, bool)), 30, (1, 1), plan.act)
    assert stops == 1
    assert total < ILLEGAL


def test_planner_never_finishes_illegal_and_prefers_neutral_stop() -> None:
    n = 50
    neutral = np.zeros(n + 2, dtype=bool)
    neutral[20:23] = True
    plan = solve(P, n)
    total, stops, sched = ev.score(P, _Race(neutral), n, (1, 3), plan.act)
    assert total < ILLEGAL
    assert stops >= 1
    assert any(20 <= lap <= 22 for lap, _ in sched)


def test_env_return_equals_evaluator_time() -> None:
    env = StrategyEnv([Scenario(P, 40)], randomize=0.0, estimate_noise=0.0)
    env.reset(seed=3)
    neutral, start = env.neutral.copy(), (env.c, env.a)
    sched = {18: 2}
    if start[0] == 2:
        sched = {18: 1}
    ret, done = 0.0, False
    while not done:
        _, r, done, _, _ = env.step(sched[env.lap] + 1 if env.lap in sched else 0)
        ret += r
    expected = ev.score(env.truth, _Race(neutral), 40, start, ev.from_schedule(sched))[0]
    assert -ret * REWARD_SCALE == pytest.approx(expected, abs=1e-6)


def _synthetic_race(truth: TyreParams, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = 50
    for d in range(16):
        pit = int(rng.integers(15, 35))
        c1, c2 = (1, 2) if d % 2 else (0, 1)
        base = 90 + 0.1 * d
        for lap in range(1, n + 1):
            c, age = (c1, lap + 2) if lap <= pit else (c2, lap - pit)
            t = base - 0.05 * lap + float(truth.stint_cost(c, age)) + rng.normal(0, 0.15)
            rows.append(
                {
                    "year": 2030,
                    "round_number": 1,
                    "driver_number": d,
                    "lap_number": lap,
                    "lap_time_s": t + (20 if lap in (pit, pit + 1) else 0),
                    "compound_encoded": c,
                    "tyre_life_laps": age,
                    "track_status": "1",
                    "pit_in_this_lap": lap == pit,
                    "pit_out_this_lap": lap == pit + 1,
                    "is_accurate": True,
                }
            )
    return pd.DataFrame(rows)


def test_fit_recovers_degradation_separately_from_fuel() -> None:
    truth = TyreParams(off=(-0.5, -0.2, 0.0), deg=(0.10, 0.06, 0.03), curv=(0, 0, 0))
    race = prepare_race(_synthetic_race(truth), 2030, 1)
    v = fit_tyre_params(clean_laps(race), np.zeros(8), np.full(8, 1e-6))
    np.testing.assert_allclose(v[2:5], truth.deg, atol=0.015)


def test_inrace_update_is_causal() -> None:
    truth = TyreParams(deg=(0.10, 0.06, 0.03))
    df = _synthetic_race(truth)
    a = inrace_update(prepare_race(df, 2030, 1), P, 20)
    df.loc[df["lap_number"] > 20, "lap_time_s"] += 5.0 * df["tyre_life_laps"]
    b = inrace_update(prepare_race(df, 2030, 1), P, 20)
    assert a == b


def test_ensemble_averages_probabilities_and_summary_avoids_test_selection(tmp_path) -> None:
    import json

    from ml.strategy.ensemble import EnsemblePolicy
    from ml.strategy.summarize_ppo import summarize
    from stable_baselines3 import PPO

    env = StrategyEnv([Scenario(P, 20)])
    models = [
        PPO("MlpPolicy", env, n_steps=64, batch_size=64, seed=s, device="cpu") for s in (1, 2)
    ]
    ens = EnsemblePolicy(models)
    obs, _ = env.reset(seed=0)
    p = ens.probabilities(obs)
    assert p.shape == (4,)
    assert p.sum() == pytest.approx(1.0, abs=1e-5)
    assert ens.act(obs) == int(np.argmax(p))
    for seed, gap in ((1, -2.0), (2, 3.0)):  # best seed beats planner, the mean does not
        d = tmp_path / f"seed-{seed}"
        d.mkdir()
        (d / "benchmark.json").write_text(
            json.dumps({"seed": seed, "ppo_illegal_share": 0.0, "ppo_minus_planner_s": gap})
        )
    assert summarize(tmp_path, tolerance=5.0)["serve"] == "planner"


def _trajectory(n: int = 40, err: float = 0.05):
    from ml.strategy.env import ErrorTrajectory

    tyre = np.zeros((n, 8))
    tyre[:, 3] = np.linspace(err, 0.0, n)  # medium wear error shrinks over the race
    return ErrorTrajectory(
        progress=np.arange(1, n + 1) / n,
        tyre=tyre,
        evidence=np.tile([0.0, 100.0, 50.0], (n, 1)),
        pit_green=1.5,
        pit_neutral=-1.0,
        p_neutral=0.0,
    )


def test_error_replay_shapes_estimates_not_costs() -> None:
    from ml.strategy.env import OBS_DIM_V2

    env = StrategyEnv([Scenario(P, 40)], randomize=0.0, errors=[_trajectory()], obs_version="v2")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM_V2,)
    est, _ = env.estimate()
    assert est.deg[1] == pytest.approx(P.deg[1] + 0.05, abs=2e-3)  # early: large error
    assert est.pit_green == pytest.approx(P.pit_green + 1.5)
    np.testing.assert_allclose(obs[-3:], [0.0, 100 / 300, 50 / 300], atol=1e-6)
    while env.lap < 40:
        env.step(0)
    late, _ = env.estimate()
    assert late.deg[1] == pytest.approx(P.deg[1], abs=2e-3)  # late: error replayed as ~0
    assert env.truth == P  # costs always use the truth


def test_planner_warm_start_learns_planner_actions() -> None:
    from ml.strategy.train_ppo import planner_demos, warm_start
    from stable_baselines3 import PPO

    env = StrategyEnv([Scenario(P, 30)], randomize=0.1, errors=[_trajectory(30)])
    obs, acts = planner_demos(env, episodes=20, seed=0)
    assert len(obs) == len(acts) == 20 * 30
    assert (acts > 0).any()  # the planner pits
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, seed=0, device="cpu")
    assert warm_start(model, obs, acts, epochs=40) > 0.9
