# Working pit policy: root cause and replacement (November 5, 2026)

## Why every policy since V9 failed

Measured on `data/gold_2022_2025.parquet` (74 dry, red-flag-free races):

1. **The IQL actor was behaviour cloning.** Rewards are `-lap_time/100`, so the
   value gap between pitting and staying at a real decision point is ~0.10
   reward units (median 9.9 s; p90 19.5 s). `ml/pipelines/iql.py` weights the
   actor by `exp(advantage / 3.0)` → weights 1.03–1.07, i.e. essentially plain
   imitation. Imitating a 3.42% pit-lap base rate gives a near-constant ~3.4%
   pit probability (V9 reported "about 3.3%"), and argmax never chooses pit.
   That is the whole "zero pit calls" failure; no data provenance repair could
   fix it.
2. **Wrong target afterwards.** Later cycles scored models by F1 against the laps
   real teams pitted. That measures imitation of decisions made with private
   information. It caps out around 30–35% and says nothing about strategy quality.
3. **Legacy simulator reward** (`services/simulator/env.py`) mixes lap time with
   position/heuristic bonuses (45 misranked strategy pairs, per
   `ppo-repair-20261104.md`), uses a 70% hand-written wear curve and has no
   two-compound rule.

## Replacement: `ml/strategy`

| Module | Role |
|---|---|
| `race_model.py` | Fuel-corrected tyre model `off[c] + deg[c]·age + curv[c]·age²` with Huber-IRLS, driver levels and a lap slope as nuisance terms; pit loss measured against non-stopping cars, separately for green and SC/VSC laps; empirical-Bayes priors by circuit, using only earlier races. |
| `planner.py` | Exact backward-induction DP over (lap, compound, age, two-compounds-used) with a per-lap SC/VSC probability and the finishing rule. Milliseconds per solve. |
| `evaluate.py` | Chronological held-out benchmark: hindsight truth per race, real SC timeline, the same scorer for team / planner / heuristics / oracle. |
| `env.py`, `train_ppo.py` | Gymnasium env on the same model: race-time reward only, discount 1, compound rule, Markov SC timeline, noisy parameter estimates in the observation. SB3 PPO plus a benchmark against the planner. |

The planner only sees priors from strictly earlier races, field laps completed
before the decision, and whether the current lap is neutralised.

## Results (2024–2025 held out, 40 races, 692 driver-races)

Mean strategy time versus the team's real strategy, scored in the same model
(`runs/strategy-planner-20261105/summary.json`):

| Policy | Gain vs team | Regret vs hindsight oracle |
|---|---:|---:|
| DP planner (in-race updates) | **+12.4 s** (72% of cars ≥ team) | **9.9 s** |
| DP planner, pre-race prior only | +8.2 s | 14.1 s |
| One stop at half distance | +5.2 s | 17.1 s |
| One stop, take SC in 30–70% window | +4.4 s | 17.9 s |
| Real team strategy | 0 | 22.3 s |

The planner made zero "never pit" decisions (mean 1.54 stops vs 1.70 for teams).
Robustness (`robustness.json`): with true degradation ±30% or pit loss ±20% that
the planner does not know, it still beats the one-stop heuristic by 2.6–9.1 s and
the team by 9.1–10.7 s. Model check (`model_check.json`): on drivers held out of
the fit, the tyre terms cut lap error from 0.481 to 0.409 s MAE.

PPO (`ppo-local-seed11.json`, 2M steps, 4 min on 2 CPUs): 1.57 stops, 0% illegal
finishes, +6.6 s vs team, 3.3 s behind the planner (4.4 s behind at 400k steps).

**Read the "vs team" column with care.** The model has no traffic, track position
or tyre-set inventory. Teams optimise for those, so any time-only policy looks
better than them here; even the naive one-stop gains 5 s. The defensible claims
are the ranking among policies and the regret to the oracle.

## Commands

```bash
python -m pytest -q tests/unit/test_strategy_planner.py   # DP == brute force, causality, env/scorer parity
python -m ml.strategy.evaluate --years 2024,2025 --out runs/strategy-planner-eval
python -m ml.strategy.train_ppo --timesteps 2000000 --seed 11 --out runs/strategy-ppo-local/seed-11
```

Cluster (CPU only, 5 seeds × 20M steps, `defq`):

```bash
module load slurm/slurm/21.08.8 && mkdir -p logs
sbatch infra/slurm/train_strategy_ppo.sbatch            # TIMESTEPS=40000000 sbatch ... to go longer
```

Each seed writes `runs/strategy-ppo-<job>/seed-<s>/benchmark.json`. Accept a PPO
policy only if `ppo_illegal_share == 0` and `ppo_minus_planner_s` is near 0. The
planner is the reference and also the fallback serving policy.

`configs/pipeline-control.json` still says `hpc_submission_allowed: false` for the
legacy IQL pipeline. This job does not read that file. Update it when you release
the hold.

## Next steps

1. Track position: score finishing position against the recorded rival paths
   (undercut/overcut) once the time-only policy is accepted.
2. Wet races and red flags (18 of 92 races excluded) need intermediate/wet
   compounds and free tyre changes.
3. Wire `planner.solve(...).act` into the API/Race Center in place of the IQL
   checkpoint.

## Improvement investigation (November 5, 00:45)

Regret decomposition on 2024–25 (507 full-distance driver-races): deployed
planner 10.5 s; same planner given the hindsight parameters 3.1 s. So 7.4 s is
parameter estimation, ~3.1 s is unknowable SC timing. Swapping in true pit losses
only saves 0.3 s; true tyre parameters save 5.8 s.

Two estimation fixes were tested on 2023 only (priors from 2022), keeping
2024–25 untouched:

- In-race prior strength ×0.03…×30: 12.7–26.0 s vs 14.0 s current; no robust
  winner (noise across settings exceeds gains). Kept ×1.
- Posterior-averaged decisions (12 parameter draws, covariance ×1/×4/×16):
  14.3–15.2 s vs 14.0 s. Rejected.

The residual looks like systematic disagreement between two estimates of the same
race (the hindsight "truth" is also a fit), not sampling noise. The planner is
unchanged.

HPC job fixes found by a local dry run of the exact sbatch: the repo's pytest
`addopts` enforce `--cov-fail-under=80`, which fails a single-file run (now
overridden with `-o addopts=""`). DummyVecEnv steps envs in one process, so
CPUs were cut from 16 to 4. Bundle: `runs/strategy-planner-20261105/
f1-strategy-planner-20261105.tar.gz` (sha256 278691e1…7efd65). Submit with
`submit_hpc.ps1`; when done, `summary.json` in the run directory says
`serve: planner|ppo`.

## Trained PPO policy (November 5, 01:55)

The cluster alias is only in the Windows SSH config, which this agent cannot use,
so two of the five planned seeds were trained to the full 20M steps on the
agent's own CPU workspace (~37 min each, identical hyperparameters to
`ml.strategy.train_ppo`). Evidence: `runs/strategy-ppo-local-20261105/`.

| Seed | PPO − planner (s/race) | 95% CI, race bootstrap | Illegal | Stops |
|---|---:|---|---:|---:|
| 42 | −1.23 | [−3.26, +0.77] | 0% | 1.53 |
| 23 | −0.92 | [−3.10, +1.25] | 0% | 1.64 |

Both final (pre-declared 20M) policies sit slightly ahead of the DP planner on
held-out 2024–25 (regret to oracle 9.26 / 9.57 s vs 10.49 s; +11.1 / +10.8 s
vs team), but the interval includes zero: PPO **matches** the planner; it is not
shown to beat it. A 6M checkpoint was looked at for monitoring only and not
selected. Load with `PPO.load("runs/strategy-ppo-local-20261105/policy-seed42.zip")`
and drive it through `ml.strategy.train_ppo.rl_policy`. The 5-seed HPC bundle
(`submit_hpc.ps1`) would tighten the interval.

## HPC submission (November 5, 13:40 UTC)

Submitted by the user with `submit_hpc.ps1`: array job **186536** (seeds 42, 11,
23, 67, 89; 20M steps each, defq, 4 CPUs), summary job **186537**
(`afterany:186536`). The cluster's pre-submit check passed (bundle checksums, 7/7 tests,
V9 venv `/users/s252161/f1-strategy-policy-v9-20261027/.venv`). Remote project:
`~/f1-strategy-planner-20261105`; results land in `runs/strategy-ppo-186536/`
with `summary.json` from the summary job. Record: `runs/strategy-planner-20261105/submission.log`.

## HPC results and served model (November 5, 08:55)

Jobs 186536/186537 finished (5 seeds × 20M steps, ~43 min each, 0% illegal).
Per seed vs planner on 2024–25: 42 −1.76, 23 −1.71, 67 −1.32, 11 +0.76,
89 +0.82 s/race. Seed spread (2.6 s) exceeds the mean gain, so do not serve a single
seed picked on these races. The cluster's original `summary.json` did exactly
that ("best_seed") and is kept as `summary.hpc-original.json`; `summarize_ppo` now
decides on the seed mean.

Choices made without the test races (`runs/strategy-ppo-186536/final_comparison.json`):

| Policy | 2023 validation, PPO − planner | 2024–25 test | 95% CI (race bootstrap) | Races better |
|---|---:|---:|---|---:|
| 5-seed ensemble (`ml.strategy.ensemble`) | −0.56 | **−1.68** | [−3.49, +0.17] | 62% |
| Seed chosen on 2023 (seed 23) | −1.21 | −1.71 | [−4.08, +0.63] | 53% |

Served model: the 5-seed ensemble. It consistently edges the planner by ~1.7 s
per race (regret to oracle ≈ 8.8 s vs 10.5 s; team 20.4 s), but the 95% interval
still touches zero. Keep the DP planner as the reference and fallback.
