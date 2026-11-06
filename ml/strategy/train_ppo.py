"""Train one PPO seed for one training-loop arm (configs/strategy-training-loop-v1.json).

Arms (docs/guides/strategy-training-loop-plan-20261105.md):
    A  control: fixed lognormal estimate noise, v1 observations
    B  replay measured in-race estimation-error trajectories (P1)
    C  B + warm start by imitating the DP planner acting on the same noisy estimates
    D  B + v2 observations (evidence behind each compound's estimate)

Only training-split races (2021-23) build scenarios and error trajectories.
Scoring is done per arm by ``ml.strategy.protocol`` on the whole seed ensemble.

    python -m ml.strategy.train_ppo --arm B --seed 42 --out runs/loop/arm-B/seed-42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ml.strategy import evaluate as ev
from ml.strategy.env import StrategyEnv, scenarios_from
from ml.strategy.planner import Plan, solve
from ml.strategy.race_model import Race, TyreParams, inrace_update

ARMS: dict[str, dict[str, Any]] = {
    "A": {"errors": False, "obs": "v1", "warm_start": False},
    "B": {"errors": True, "obs": "v1", "warm_start": False},
    "C": {"errors": True, "obs": "v1", "warm_start": True},
    "D": {"errors": True, "obs": "v2", "warm_start": False},
}


def planner_demos(env: StrategyEnv, episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Observations and DP-planner actions, the planner seeing the env's estimates."""
    obs_out, act_out = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        cache: dict[tuple[int, ...], Plan] = {}
        done = False
        while not done:
            est, _ = env.estimate()
            key = (*tuple(np.round(est.as_vector(), 5)), env.n)
            if key not in cache:
                cache[key] = solve(est, env.n)
            act = cache[key].act(
                env.lap, env.c, env.a, len(env.used) >= 2, bool(env.neutral[env.lap])
            )
            obs_out.append(obs)
            act_out.append(act)
            obs, _, done, _, _ = env.step(act)
    return np.asarray(obs_out, dtype=np.float32), np.asarray(act_out, dtype=np.int64)


def warm_start(model: PPO, obs: np.ndarray, acts: np.ndarray, epochs: int = 20) -> float:
    """Behaviour-clone the planner into the PPO actor (cross-entropy on its actions)."""
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    x, y = torch.as_tensor(obs), torch.as_tensor(acts)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(1024):
            dist = pol.get_distribution(x[idx])
            loss = -dist.log_prob(y[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        agree = (pol.get_distribution(x).distribution.probs.argmax(1) == y).float().mean()
    return float(agree)


def main() -> None:
    from ml.strategy.estimation_error import load
    from ml.strategy.protocol import CONFIG, load_universe

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--data", type=Path, default=Path("data/strategy_gold_2021_2026.parquet"))
    ap.add_argument("--errors", type=Path, default=Path("data/estimation_error_train.npz"))
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo-episodes", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    arm = ARMS[args.arm]
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    u = load_universe(args.data, json.loads(CONFIG.read_text()))
    train = {k: u.truths[k] for k in u.splits["train"]}
    scen = scenarios_from(train, {k: u.races[k].n_laps for k in train})
    errors = load(args.errors) if arm["errors"] else None
    kwargs: dict[str, Any] = {"errors": errors, "obs_version": arm["obs"]}
    env = make_vec_env(lambda: StrategyEnv(scen, **kwargs), n_envs=args.n_envs, seed=args.seed)
    model = PPO(
        "MlpPolicy",
        env,
        gamma=1.0,
        gae_lambda=0.95,
        n_steps=1024,
        batch_size=1024,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    t0 = time.time()
    record: dict[str, Any] = {"arm": args.arm, "seed": args.seed, "timesteps": args.timesteps}
    if arm["warm_start"]:
        demo_env = StrategyEnv(scen, **kwargs)
        obs, acts = planner_demos(demo_env, args.demo_episodes, args.seed)
        record["warm_start"] = {
            "samples": len(acts),
            "planner_agreement": warm_start(model, obs, acts),
        }
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out / "policy.zip")
    record["train_seconds"] = round(time.time() - t0, 1)
    record["train_races"] = len(train)
    (args.out / "train.json").write_text(json.dumps(record, indent=2))
    (args.out.parent / "candidate.json").write_text(
        json.dumps({"arm": args.arm, "observation": arm["obs"]})
    )
    print(json.dumps(record))


def rl_policy(
    model: PPO, race: Race, prior: TyreParams, n: int, cache: dict[int, TyreParams]
) -> ev.Policy:
    """Legacy v1 adapter used by earlier analyses (kept for reproducibility)."""
    from ml.strategy.env import observation

    def act(lap: int, c: int, a: int, f: bool, s: bool) -> int:
        if lap not in cache:
            cache[lap] = inrace_update(race, prior, lap - 1) if lap > 5 else prior
        return int(
            model.predict(observation(lap, n, c, a, f, s, cache[lap]), deterministic=True)[0]
        )

    return act


if __name__ == "__main__":
    main()
