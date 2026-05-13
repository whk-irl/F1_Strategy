# ADR 0003 — DQN + Prioritized Experience Replay for strategy policy

**Status:** Accepted
**Date:** 2026-W19

## Context

The initial PPO policy (`ml/models/strategy_policy/train.py`) collapsed to a
degenerate "pit at lap 1" strategy across all races.  Root cause analysis:

1. **Reward imbalance** — driving past the cliff lap costs up to −0.20 per
   lap, but pitting early costs only ~0.25 extra in lap-time penalty.  An
   agent that pits at lap 1 gets fresh tyres for the whole race and avoids
   *all* future cliff penalties, which outweighs the one-lap cost.

2. **On-policy credit assignment** — PPO with GAE(λ=0.95) struggles to
   assign credit across a 50-lap episode.  The causal link between "pit
   early" and "lose position later from a second-stop deficit" is too
   attenuated over the horizon.

3. **Sparse rare events** — Safety-car windows appear in a minority of
   laps.  PPO collects rollouts uniformly, so the policy rarely updates on
   these high-value transitions.

Two fixes are applied together:

- **Reward fix (env.py):** an underutilisation penalty deters pits with
  fewer than `_MIN_STINT_LAPS = 8` laps on the current set (−0.5 per
  unused lap, exempt during SC/VSC).  This addresses the root cause and
  benefits the original PPO as well.

- **Algorithm switch:** replace PPO with DQN + Prioritized Experience Replay.

## Decision

Use **DQN with Proportional Prioritized Experience Replay** (Schaul et al.,
*Prioritized Experience Replay*, ICLR 2016, arXiv:1511.05952).

Key properties:

| Property | Benefit for this problem |
|---|---|
| Off-policy replay | Past transitions are re-used; rare SC windows are revisited more often |
| Explicit Q-values | Directly shows the expected value of each compound at each race state — inspectable |
| PER priority weighting | High-TD transitions (miscalibrated pit windows) are sampled more, focusing learning where the policy is most wrong |
| IS weight correction | Counters the bias from non-uniform sampling; prevents over-fitting to a single class of transition |
| Smaller env count | DQN uses one env; no vectorisation complexity |

The implementation lives in:

- `ml/models/strategy_policy/per_buffer.py` — `_SumTree` + `PrioritizedReplayBuffer`
- `ml/models/strategy_policy/train_dqn.py` — `DQNPER(DQN)` + training CLI
- `ml/models/strategy_policy/predict_dqn.py` — inference API

The original PPO files (`train.py`, `predict.py`) are preserved so both
policies can be evaluated side-by-side and the PPO run can serve as a
baseline in MLflow.

## Hyper-parameter rationale

| Parameter | Value | Reasoning |
|---|---|---|
| `buffer_size` | 100 000 | ~2 000 episodes × 50 laps; fits in RAM |
| `learning_starts` | 10 000 | Fill buffer before any gradient update |
| `target_update_interval` | 10 000 | Hard update (τ=1) every 10 k gradient steps |
| `exploration_fraction` | 0.30 | Explore for first 600 k steps of 2 M total |
| `gamma` | 0.995 | Matches PPO; necessary for 50-lap horizon |
| `per_alpha` | 0.6 | Standard value from paper |
| `per_beta` | 0.4 → 1.0 | Annealed over training to full IS correction |
| `net_arch` | [256, 128] | Same as PPO baseline — controls for architecture |

## Alternatives considered

1. **PPO + action masking** — prevents early pits via `MaskablePPO`.  Fast
   to implement, but the constraint is structural, not learned; the agent
   never develops intuition about *why* early pits are bad and cannot
   generalise to corner cases (e.g. lap-2 puncture).  Deferred to a
   possible future experiment.

2. **Behavioral Cloning → PPO fine-tune** — pre-train on historical race
   strategies; likely higher sample efficiency.  Requires a separate BC
   data pipeline (obs-action pairs extracted from silver/gold data) which
   would add ~200 LOC and a new training stage.  Candidate for a follow-on
   ADR if DQN+PER underperforms.

3. **Offline RL (IQL)** — train entirely from static replay buffer.  Would
   need `d3rlpy` (new dependency) and an offline dataset construction
   pipeline.  Conservative offline policies may under-explore relative to
   DQN.

## Consequences

**Positive:**
- PER re-samples safety-car and undercut transitions more often, directly
  improving the situations that matter most for strategy.
- Q-values are readable numbers that can be shown in the Streamlit UI
  alongside the recommended action.
- DQN+PER is a well-understood baseline with extensive literature support.

**Negative / risks:**
- DQN trains on a single environment (no `n_envs` parallelism); wall-clock
  time per timestep is higher than PPO with 4 envs.  Compensated by lower
  `total_timesteps` (2 M vs 3 M) due to better sample efficiency.
- The replay buffer stores 100 k transitions in RAM; with 21 float32 obs
  features the overhead is ~17 MB — negligible.
- PER introduces two extra hyperparameters (`alpha`, `beta`); default
  values from the paper are used and logged in MLflow for tracking.
