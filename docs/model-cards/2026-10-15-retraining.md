# October 15, 2026 retraining result

**Decision:** retain dynamic programming as the strongest reference in this benchmark.
The new history-aware PPO candidate completed pretraining and fine-tuning, but did
not beat that reference on held-out replay events. Its artifact remains unpromoted.

## Reproduction

- Configuration: [`configs/training.local.json`](../../configs/training.local.json).
- Method and sources: [ADR 0005](../decisions/0005-staged-policy-training.md).
- Commands and artifact layout: [training guide](../guides/staged-training.md).
- Seed: 42. Python 3.11.4, PyTorch 2.5.1+cu121, SB3 2.8.0, LightGBM 4.6.0.
- Local hardware: RTX 3060 Laptop GPU, 6 GB; sequence training on CUDA, PPO on CPU.
- Run fingerprint: `bf5285cf45cdcdfe346e1aab6105fa535234f89fd67a1b7058c74cc05b192a91`.
- Training-source digest: `51d18967fc249fc6fe7a907d150a2cc298adfb06ab32441be286a98bbbe7ebcb`.

Weights, raw data, full per-episode evidence and MLflow records are generated
artifacts under ignored local directories. This card records the result without
committing those artifacts.

## Data and stages

The dataset contains 120,935 lap records across 109 events. Bundled 2022–2025 records
were rebuilt with causal features. OpenF1 supplied 19,645 additional lap records
from 17 completed 2026 races after FastF1 returned HTTP 401.

| Split | Events | Lap records |
|---|---:|---:|
| Source training, 2022–2023 | 38 | 41,536 |
| Source validation | 6 | 6,461 |
| Simulator gate, 2024 | 24 | 26,604 |
| Adaptation training, 2025 through 2026 round 10 | 34 | 38,001 |
| Recent validation, 2026 rounds 11–13 | 3 | 3,856 |
| Final test, completed 2026 rounds 14 onward | 4 | 4,477 |

Policy training/evaluation excludes wet and red-flag events. Final policy evaluation
uses 40 paired driver/seed episodes across three eligible test events. All four test
events remain available to the supporting-model evaluation with its own quality filters.

TCN/GRU training stopped after 16 source epochs and completed 15 adaptation epochs.
Behavioral cloning used 21,337 historical observations including 699 pit decisions.
PPO executed 200,704 source steps and 100,352 adaptation steps. Validation selected
the 150,000-step source checkpoint; the final selected policy reports 250,352 cumulative
steps because fine-tuning continues from that selected checkpoint.

Both simulator gates passed: mean Spearman rho **0.827** on the legacy validator
and **0.817** on the actual policy environment's eligible dry races. These are
conditional replay fidelity checks, not independent race-result forecasting scores.

## Held-out outcomes

| Policy | Mean position | Mean race time (s) | Mean stops | Two-compound compliance |
|---|---:|---:|---:|---:|
| Behavioral cloning | 8.575 | 5985.67 | 1.35 | 100% |
| Source PPO | 8.575 | 5986.80 | 1.70 | 100% |
| Fine-tuned PPO | 8.575 | 5985.04 | 1.60 | 100% |
| Tyre-age heuristic | 8.600 | 5977.63 | 1.00 | 100% |
| Dynamic programming | **8.425** | **5973.64** | 1.00 | 100% |

Fine-tuned PPO is 1.75 seconds faster than source PPO on the episode-weighted mean,
but the dynamic-programming reference remains stronger. The event-weighted mean
time difference against DP is **+11.28 seconds**, with an event-bootstrap interval
of **+7.11 to +16.11 seconds**. Only three events and one training seed are represented;
that interval does not cover training-seed or simulator uncertainty.

Supporting-model test results:

- Tabular tire MAE: **3.082 s**, versus **3.199 s** for a zero-delta prediction.
- Sequence next-lap MAE: **0.729 s**, versus **0.731 s** for repeating the prior delta.
- Safety-car average precision: **0.320**, versus positive-label prevalence **0.124**.

The tabular and sequence errors use different eligible rows and prediction inputs;
their absolute MAEs are not a matched architecture comparison. The sequence gain
over persistence is small.

## Scope and verification

Opponents replay recorded trajectories rather than responding to the policy.
Early recorded pace and fixed scenario conditions inform the simulator. Tire inventory,
exceptional event-specific sporting rules and wet/red-flag decisions are not modeled.
The result does not establish superiority in a live race.

Verification: **154 tests passed, 86.44% configured coverage**, repository-wide Ruff
lint/format checks passed, and mypy passed all 59 source files. A completed-stage
resume verified every saved stage digest; final model hashes and inference loading
were checked. The run and artifacts were recorded in local MLflow.

The October 16 Race Center changes postdate this training-source snapshot. The
saved provenance remains unchanged. A new training run against the updated checkout
requires a new output directory under the pipeline's conservative source-change guard.
After the UI additions, 166 tests passed with 89.08% configured coverage; Ruff and
mypy passed. Recorded replay, playback controls, driver selection and responsive
layouts were checked in the browser. Authenticated active-session timing remains
to be verified during a live session.
