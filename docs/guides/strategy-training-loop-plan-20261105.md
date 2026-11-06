# Plan: training loop to a usable pit-strategy model (v1)

Starting point (November 5): 5-seed PPO ensemble, −1.68 s/race vs the DP planner
on 2024–25 (95% CI [−3.49, +0.17]), 0% illegal finishes. Machine-readable
protocol: `configs/strategy-training-loop-v1.json`. Phases run in order; each has
an exit check. Nothing here touches the legacy IQL pipeline or its holds.

## What limits us now (measured, not assumed)

| Finding | Evidence | Consequence for the plan |
|---|---|---|
| 7.4 of 10.5 s planner regret is in-race parameter estimation; SC timing ≈3 s | regret decomposition | Train against realistic estimation error, not a fixed 15% guess |
| Seeds differ by 2.6 s/race | HPC 186536 | Reduce variance (regularize toward planner, ensembles); report seed spread |
| 40 test races → CI half-width ≈1.8 s | race bootstrap | A 1–2 s gain cannot be *proven* on real races alone; gate on non-inferiority, report superiority |
| 2024–25 has now been looked at many times | this session | Demote to validation; use untouched 2026 r1–18 as the single-use final test |
| Model has no traffic, used sets, or SC persistence | `race_model.py` | Extensions are later, gated work (P3), not prerequisites |

## Data splits (frozen now)

- **Train scenarios:** dry, red-flag-free races from 2021–2023 (adds 22 races from
  `data/lake/gold/year=2021` to the 2022–23 set).
- **Validation (repeat use allowed, all selection happens here):** 2024–2025, plus a
  *stress validation* that replays every validation race 20× with resampled SC
  timelines and ±30% degradation / ±20% pit-loss perturbations. It reuses the same
  races, so it adds stability, not independent evidence.
- **Final test (single use):** 2026 rounds 1–18 from the lake, dry and red-flag-free. The
  2026 regulation change is a real generalization test; report it separately.
  Rounds 19+ stay sealed for prospective checks. Every test access is logged with
  the candidate's code/weight hashes; a second access requires a new candidate.

## Phases

### P0 — Evaluation harness and data (no training)
- Loader for lake gold partitions (2021, 2026) with the same filters; 2026 circuit
  keys from lake metadata (do not hand-write the 2026 calendar).
- `ml/strategy/protocol.py`: one entry point `evaluate --candidate <dir> --split
  val|stress|test` → per-race table, paired race-bootstrap CI vs planner, seed
  spread, gates, report JSON. Test split refuses unless the candidate is frozen.
- Unit tests for split leakage (no 2024+ race in training scenarios), bootstrap,
  and the single-use test log.
- **Exit:** current ensemble reproduces −1.68 s on validation through the new harness.

### P1 — Calibrate the training environment to real estimation error
- On training races only, measure deployed-estimator error vs hindsight truth per
  parameter as a function of race progress and laps of data per compound.
- Replace `StrategyEnv`'s fixed lognormal noise with this empirical error model
  (error shrinks as the race progresses; compounds not yet run stay at prior error).
- Add observation features: estimate uncertainty and laps of evidence per compound.
- **Exit:** simulated estimation errors match measured ones (distribution check on
  training races); env/scorer parity test still passes.

### P2 — Training loop (HPC, one array per arm, 5 seeds × 20M steps, ~45 min)
| Arm | Change | Why |
|---|---|---|
| A | Current recipe, retrained on 2021–23 | Control for the new data |
| B | A + calibrated estimation error (P1) | Targets the 7.4 s regret source |
| C | B + planner-guided auxiliary loss (KL to DP action, annealed to 0) | Cut seed variance |
| D | B + uncertainty observations (P1) | Let the policy hedge when estimates are weak |

- `ml/strategy/loop.py` builds scenarios, writes one bundle per arm, and produces
  the submit script and watcher (the user runs submission; SSH stays with the user).
- Selection on validation + stress only: ensemble mean vs planner, seed spread,
  illegal rate. **Exit:** one arm chosen with seed spread < 1.0 s and validation
  mean ≤ current ensemble; otherwise iterate once on the losing mechanism.

### P3 — Fidelity extensions (only after P2; each a separate arm)
Used-set tyre penalty (`is_fresh_tyre`), Markov SC/VSC with lap-1 hazard,
uncapped wear curvature (cliff). Track position / undercut is a v2 milestone,
not v1.

### P4 — Release candidate and single test
Freeze code, data and weight hashes, then run the 2026 test once. Release gates:

| Gate | Threshold |
|---|---|
| Legality | 0 illegal finishes on validation, stress and test |
| Non-inferiority vs planner | Upper 95% CI of (ensemble − planner) < +1.0 s/race on validation and test |
| Robustness | Under deg ±30% / pit ±20%, ensemble not worse than planner by > 1.0 s |
| Behaviour sanity | Mean stops within [1, 3]; no pits on the final lap; stops during SC/VSC at least as often as the planner |
| Serving | Reload parity on all saved decisions; < 50 ms per decision on CPU |

Superiority (upper CI < 0) is reported, not required. A test failure is
reported as-is; no tuning on 2026. The planner is then served.

### P5 — Usable model delivery
- `ml/strategy/serve.py`: `recommend(race_state)` → action, ensemble
  probabilities, and the planner's seconds-saved estimate as the explanation
  ("pit now: −3.2 s vs staying out").
- FastAPI endpoint and Race Center replay integration; CLI
  `python -m ml.strategy.replay --year --round --driver` prints per-lap calls.
- Model card `docs/model-cards/strategy-ensemble-v1.md`: data, splits, gates,
  known limits (dry only, no traffic model).

## Defaults taken (change any before P2)
1. Primary release claim is **non-inferiority** to the planner; superiority is reported.
2. **2026 r1–18 is the single-use final test**; 2024–25 becomes validation.
3. Wet and red-flag races stay out of v1.

## Progress log

### P0 complete (November 5, ~10:30)
- `ml/strategy/data.py` builds `data/strategy_gold_2021_2026.parquet` (132 races; circuit
  keys from the lake manifest's `circuit_reference_round`). Usable after dry/no-red
  filters: train 48 (2021–23), validation 40 (2024–25), test 17 (2026 r1–18).
- `ml/strategy/protocol.py`: val/stress/test scoring, race-clustered bootstrap,
  gates, `freeze` + single-use test ledger (`runs/strategy-test-ledger.jsonl`).
- **Leak fix:** truths are now regularised toward a prior pooled over training
  races only. Legacy mode reproduces the old −1.679 s [−3.492, +0.166] exactly, so
  the harness mechanics are unchanged; the change in the number is the leak.
- **Baseline (HPC 186536 ensemble) under the frozen protocol:** validation −0.785 s
  [−2.833, +1.415], stress −0.108 s [−2.031, +1.762]; legal ✓, non-inferior ✗,
  robust ✓. Reports: `runs/strategy-loop-p2/baseline-hpc186536-*.json`.

### P1 complete
- Measured on training races (`runs/strategy-loop-p2/p1_estimation_error_train.parquet`):
  wear-rate estimate RMS error 0.043–0.055 s/lap in the first quarter, 0.015–0.026
  in the last quarter (env previously assumed ±15%); wear is under-estimated by
  0.005–0.023 s/lap; pit-loss prior RMS 2.2 s green, 5.5 s neutralised.
- Bias-correcting the planner with the training-race bias: −0.06 s [−1.37, +1.22]
  on validation. Rejected; variance, not bias, is the problem.
- `StrategyEnv(errors=..., obs_version=...)` replays the 48 measured trajectories
  (`data/estimation_error_train.npz`); v2 observations add evidence per compound.

### P2 packaged
`runs/strategy-loop-p2/f1-strategy-loop-p2-20261105.tar.gz` (sha256 5dabbb57…1c2d):
arms A–D × seeds 42/11/23/67/89 (20 tasks, 20M steps each) + dependent per-arm
val/stress evaluation. Local checks: 16/16 tests, all four arms train, evaluation
runs for v1 and v2 observations, bundle verified from a fresh extraction.
Submit with `submit_p2.ps1`, watch with `watch_p2.ps1 -Loop`.

### P2 results (HPC 186837/186838, November 5)
| Arm | Val vs planner | 95% CI | Stress | Seed spread | Gates |
|---|---:|---|---:|---:|---|
| baseline 186536 | −0.79 | [−2.83, +1.42] | −0.11 | 2.77 | non-inferior ✗ |
| A control (+2021) | −0.55 | [−2.46, +1.66] | −0.38 | 2.74 | non-inferior ✗ |
| **B error replay** | **−1.58** | **[−4.38, +0.93]** | **−1.38** | **0.76** | **all ✓** |
| C B + planner warm start | −1.25 | [−4.17, +1.35] | −0.96 | 2.36 | non-inferior ✗ |
| D B + evidence obs | −0.70 | [−3.24, +1.73] | −0.46 | 0.99 | non-inferior ✗ |

All five B seeds beat the planner (−0.86…−1.63); B wins every stress condition.
A ≈ baseline, so the gain is the calibrated estimation error, not the extra data.
C and D are rejected. Arm B selected (P2 exit met).

### P4: release checks, amendment, single test
- Release checks (`ml/strategy/release.py`, validation): mean stops 1.45 (planner
  1.49), no late stops, reload parity on 5,000 decisions, 0.42 ms/decision.
- **Gate amendment (user-approved, after validation, before test):** "SC stop
  share ≥ planner" failed (12.4% vs 27.2%) while B was fastest in SC/VSC races.
  Replaced by an outcome gate: SC/VSC validation races −2.96 s [−6.35, −0.17] ✓.
  Recorded in `configs/strategy-training-loop-v1.json` → `amendments`.
- Frozen candidate `7e8800ab93b81c68` (`runs/strategy-loop-p2/loop/arm-B/FROZEN.json`).
  Test opened once at 2026-11-05T19:39Z (`runs/strategy-test-ledger.jsonl`).
- **2026 test (17 races):** ensemble −3.49 s/race vs planner, CI [−10.58, +5.04];
  all five seeds −2.0…−4.5; 0% illegal; regret 16.5 s vs planner 20.0 s, teams 29.5 s.
  **Non-inferiority gate fails (upper +5.04 > +1.0): an underpowered test, not
  evidence of inferiority.** Post hoc val+test pooled: −2.04 s [−4.69, +0.84].
- Per the frozen rule the planner remains the default release; B is the leading
  candidate. Promotion route without reusing the test: a pre-registered
  prospective check of the same frozen candidate on 2026 r19+ as those races arrive.

### Decision and P5 delivered (November 5, ~14:45)
User decision: **keep to the rule**. Planner is the released default; the frozen
ensemble ships as opt-in only.
- Serving (`ml/strategy_serving/`, outside the frozen `ml/strategy` so the
  fingerprint stays valid): `advisor.py` (`CarState`/`FieldLap` → `Recommendation`
  with `pit_saves_s` and an explanation; ensemble weights verified against
  `FROZEN.json` on load), `priors.py` → `models_baked/strategy_priors.json`
  (105 races, 28 circuits, through 2026 r18), `replay.py` CLI.
- API: `services/api/strategy.py` (`POST /strategy/recommend`, `GET /strategy/health`).
- Config: `configs/strategy-serving.json`; served ensemble copy in
  `models_baked/strategy-ensemble-armB` (matches the freeze).
- **Pre-registered promotion check:** `configs/strategy-prospective-v1.json`,
  registered before any 2026 r19+ data existed. One decision after the final 2026
  round: pool test r1–18 with r19..N; promote if 0 illegal and upper CI < +1.0 s.
  `python -m ml.strategy_serving.prospective status|decide --final-round N`; guards
  verify weights, frozen code, config, history rows and saved test rows; premature
  `decide` refuses without consuming the decision (bug found and fixed in dry run).
- Model card: `docs/model-cards/strategy-advisor-v1.md`. Tests: 24 pass.
- Still to do when data arrives: ingest 2026 r19+ (`services.ingestion.postrace`),
  rebuild `data/strategy_gold_2021_2026.parquet` with `ml.strategy.data`, run
  `prospective status` for monitoring and `decide` once after the final round.

### P5 Race Center integration (November 5, ~21:10)
- Race Center gains **Strategy advisor** (released planner) and **Ensemble · opt-in**
  engines (`services/live/advisor_strategy.py`, `services/frontend/race_center.py`,
  component `app.js`). Recorded races only; live mode holds. Default engine unchanged.
- Calls built from Race Center's own OpenF1 bundle match the evaluated-dataset advisor
  on 2026 US GP: 99.83% of 1,163 car-laps, median Δ pit-saving 0.02 s. Engines
  disagree on 65 of 1,163 calls (5.6%).
- Checks: 13 existing Race Center tests + 8 new + 8 serving tests pass; headless
  browser render of the real component for both engines (screenshots in docs/assets).
- Advisor explanation now says "lap N on this set" and labels gaps < 0.5 s a near tie.

### Release readiness (November 5, ~22:30)
- **Full unit regression:** 4,056 pass; the same 14 failures occur with and without
  tonight's changes (missing run files, newer FastF1 parser, pandas version), so zero
  regressions. Run in an offline copy; re-run in the locked env before release.
- **Real Streamlit:** Race Center runs offline from cache; both engines selected
  through the live UI render server-computed calls. First-switch latency in the
  offline copy is dominated by blocked OpenF1 refresh attempts; advisor load is
  0.5 s (planner engine no longer imports PyTorch).
- **Live shadow:** `services/live/advisor_shadow.py` + gate
  `configs/strategy-live-shadow-v1.json`. Found and fixed on the US GP capture: stint
  records with lap_start 0 (compound mismatch 19% -> 0.7%) and stint-feed lag after
  stops (12 pit calls on fresh tyres -> 0, via a 45 s in-lap hold). Development
  result (in-sample): clear-call agreement 98.85%. Live mode stays held until two
  untouched races pass.
