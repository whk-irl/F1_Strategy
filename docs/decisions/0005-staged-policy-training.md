# ADR 0005 — Historical pretraining, history-aware PPO and chronological adaptation

Status: Accepted for experiments; production promotion requires measured validation.
Date: 2026-10-15

## Problem

Independent model scripts did not provide a reproducible acquisition-to-fine-tuning
experiment. The old feature builder used full-stint medians and slopes, allowing
future laps into historical inputs. Policy scripts also mixed training and evaluation
events and did not enforce the simulator gate themselves.

## Research and choice

The [RSRL study](https://link.springer.com/article/10.1007/s10994-026-07081-3)
compares DQN, constrained DQN, DRQN and A3C in its proprietary simulator. It selects
DRQN, motivating memory by partial observability and evolving opponent gaps. That
result supports testing temporal context, but does not establish the best algorithm
for this project's different simulator and data.

[Fieni et al.](https://arxiv.org/abs/2512.21570) compare a learning-based policy with
an optimization formulation of the same race problem. Their hybrid energy-management
action space differs from our four discrete pit actions. We adopt the principle of
an optimization baseline, rather than copying their SAC formulation.

The [Mendez/Thraves author repository](https://github.com/Felipe-mendezp/f1-pit-stop-drl)
provides F1 experiments with DQN, A2C, TRPO, PPO and recurrent PPO. It reinforces the
need to evaluate actual outcomes rather than choose an algorithm from its name.

Decision: use **behavioral cloning → PPO on older events → lower-rate PPO fine-tuning
on recent events** as the trainable candidate. Four lap frames provide short-term
memory while retaining Stable-Baselines3 PPO's tested rollout and checkpoint code.
Three compound-use flags augment the original 21-value observation, yielding 96
values. This is not DRQN or recurrent PPO; those remain meaningful future ablations.

Behavioral cloning is supervised imitation of recorded actions, not proof that those
actions were optimal. Pit labels may use the next recorded compound; observations
only use preceding replay history. Capped square-root class balancing helps expose
rare pit decisions. Validation selects the initial or improved PPO checkpoint, so
fine-tuning is allowed to retain its initial checkpoint when adaptation worsens it.

We considered [IQL](https://arxiv.org/abs/2110.06169), a principled offline RL method.
The public lap table lacks observed counterfactual returns, tire inventory and several
team decision inputs. Reconstructing rewards from this simulator would still inherit
its assumptions. IQL is deferred until an explicit offline transition dataset and
appropriate offline-policy evaluation are available.

## Acquisition and validation

- FastF1 is primary; historical OpenF1 is an explicit, recorded fallback. OpenF1 raw
  responses are retained. Sector-sum consistency is an inferred quality flag, not
  FastF1's authoritative accuracy/deleted-lap status.
- Resume verifies partition and checkpoint hashes. Sessions are atomic publications.
- Gold v2 uses expanding, past-only stint references. Windows cannot bridge missing,
  pit or safety-car laps. Actual UTC completion timestamps are retained when available;
  bundled legacy data is explicitly marked as having synthetic timestamps.
- Source years are 2022–23. The entire 2024 season is excluded from fitting and used
  for simulator gates. Adaptation uses 2025 and 2026 through round 10; rounds 11–13
  of 2026 select checkpoints; later 2026 rounds are reserved for final evaluation.
- Both the legacy replay validator and the actual policy environment must achieve
  mean Spearman rho >= 0.7 before policy training. Reports name the eligible race scope.
- Policy evaluation compares BC, source PPO, fine-tuned PPO, a tyre-age heuristic and
  deterministic dynamic programming on identical race/driver/seed episodes. Event-level
  bootstrap intervals avoid counting drivers in the same race as independent events.

## Consequences and limits

Local runs need no MinIO or tracking server. SQLite-backed MLflow records model files
and evidence at completion; configuration may point to an existing tracking server.
The original S3/Prefect entry point also derives distance and keeps timing/weather.

The new policy has its own manifest and inference adapter. It must not be loaded as
an old 21-input baked PPO. Generated weights and raw data stay outside Git.

This remains a retrospective recorded-field simulator. Opponents do not respond to
the agent; initial pace estimates use recorded early laps; future field/weather
conditions are fixed by the replay. The policy scope excludes wet and red-flag events.
Tire inventory and exceptional event-specific sporting rules are not modeled. The
dynamic-programming baseline optimizes nominal synthetic tyre cost, not a full stochastic
traffic model. Passing the benchmark is not evidence of superiority to real strategists.
The release bundle stays unpromoted and exposes all test outcomes, including failures.
