# Model card: Pitwall AI strategy advisor v1

**Released mode: DP planner (default).** **Opt-in mode: PPO ensemble `7e8800ab93b81c68`.**
Decision record: `docs/guides/strategy-training-loop-plan-20261105.md`.

## What it does
For one car in a dry race, at the end of each lap: **stay out** or **pit for
SOFT / MEDIUM / HARD**, with the planner's estimate of the race time the best
stop saves now (`pit_saves_s`) and a one-line explanation. Inputs are the car's
state and the field's public timing so far; nothing about the future is used.

| Interface | Command |
|---|---|
| Python | `StrategyAdvisor.from_config().recommend(CarState, [FieldLap], mode)` (`ml/strategy_serving/advisor.py`) |
| HTTP | `uvicorn services.api.strategy:app` → `POST /strategy/recommend`, `GET /strategy/health` |
| Race replay | `python -m ml.strategy_serving.replay --year 2024 --round 16 --driver 16 [--mode ensemble]` |
| Race Center | Strategy command → Strategy engine: **Strategy advisor** or **Ensemble · opt-in** (recorded races) |
| Live shadow (not shown to users) | `python -m services.live.advisor_shadow watch` / `review`; gate `configs/strategy-live-shadow-v1.json` |
| Rebuild priors | `python -m ml.strategy_serving.priors` (after new races are ingested) |

## Models
- **Planner (default):** race-time model (fuel-corrected wear per compound, pit
  loss measured against non-stopping cars, SC/VSC rate per circuit) with priors
  from earlier races at the circuit, updated with field laps through the
  previous lap; exact dynamic programming over stops, compounds and the
  two-compound rule. Serving priors: `models_baked/strategy_priors.json`
  (105 races, 28 circuits, through 2026 r18).
- **Ensemble (opt-in):** 5 PPO seeds trained on 2021–23 scenarios with replayed
  real in-race estimation errors (arm B); actions = argmax of mean probabilities.
  Served weights are checked against `FROZEN.json` on load.

## Evaluation (frozen protocol `configs/strategy-training-loop-v1.json`)
Strategy-dependent race time per car, in a per-race hindsight model, against the
real SC/VSC timeline; race-cluster bootstrap 95% CIs.

| | Validation 2024–25 (40 races) | 2026 test r1–18 (17 races, used once) |
|---|---|---|
| Ensemble − planner | −1.58 s [−4.38, +0.93] | −3.49 s [−10.58, +5.04] |
| Stress (perturbed wear/pit/SC) | −1.38 s [−3.71, +0.82] | — |
| SC/VSC races only | −2.96 s [−6.35, −0.17] | — |
| Gap to hindsight optimum: ensemble / planner / real teams | 8.8 / 10.4 / 20.4 s | 16.5 / 20.0 / 29.5 s |
| Illegal finishes (ensemble) | 0% | 0% |

**Why the ensemble is not the default:** the pre-registered rule required the
upper CI on the 2026 test to be below +1.0 s; with 17 races it is +5.04.
Every seed was faster on the test, so this is a power failure, not evidence
of harm. A single promotion decision is pre-registered in
`configs/strategy-prospective-v1.json`: 2026 test races pooled with rounds
19..end, decided once after the final round
(`python -m ml.strategy_serving.prospective decide --final-round N`).

**Validation gate amendment:** the planned behaviour gate "share of stops under
SC/VSC ≥ planner" was replaced, with the user's approval, after validation and
before the test, by the outcome it protects (no worse on SC/VSC races). The
ensemble makes fewer SC stops (12% vs 27%) but is faster in those races.

## Limits: do not use outside these
- Dry, red-flag-free races only. Wet tyres and red-flag tyre changes are not modelled.
- No traffic or track position: undercut/overcut and position battles are ignored,
  which is why every model "beats" real teams on time; compare policies to each other.
- New sets are assumed fresh; tyre inventory is not modelled.
- 2026 regulations: all policies are ~2× further from the optimum than on 2024–25;
  priors from older seasons transfer imperfectly.
- Single car; no team or multi-car strategy.
- Advice for analysis and replay; not a race-operations system.
