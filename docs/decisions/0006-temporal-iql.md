# ADR 0006 — Temporal implicit Q-learning for pit strategy

## Status

Proposed research candidate. Promotion requires the held-out benchmark in this
document; completing training does not promote the policy.

## Problem

The staged behavioral-cloning/PPO candidate was 13.73 seconds slower than the
dynamic-programming baseline on its three held-out events. PPO learns mainly from
the project simulator, while real pit actions are sparse: only 3,469 of 101,290
legacy rows are pit rows. This combination encourages poorly supported actions and
makes a short recent-season fine-tune unstable.

## Decision

Train a discrete **temporal implicit Q-learning (IQL)** policy. Each decision uses
12 causal observation frames. A small causal Transformer represents tyre history,
traffic, undercut threat, pit loss, safety-car state, and compound-use flags.

IQL fits two action-value functions only at actions present in historical races.
It fits a state value by expectile regression and trains the actor with
advantage-weighted behavior cloning. This avoids the out-of-distribution action
maximization that makes offline Q-learning unsafe while still preferring better
logged decisions. The method follows the IQL objective; the temporal encoder is a
project-specific addition motivated by the partial observability of stint state.
At inference, a sporting-rule shield forces an unused dry compound when 15% of the
race remains if the policy has not complied, and blocks consecutive stops on a
nearly new set. The learned policy still controls the strategic pit window before
that deadline.

The corpus expands to completed races from 2021–2026. FastF1's timing archive did
not expose complete lap timing for the attempted 2018–2020 sessions, so those years
are excluded rather than filling core race features with synthetic values. Splits are chronological and
event-disjoint:

- source training: 2021–2022, with rounds 5, 10, and 15 reserved for source validation;
- simulator gate: 2023;
- recent adaptation: 2024 through 2026 round 10;
- adaptation validation: 2026 rounds 11–13;
- final test: 2026 rounds 14 onward.

Only complete dry races without red flags enter this policy version. Wet strategy
and tire allocation remain separate future work.

## Evidence and alternatives

- Kostrikov, Nair, and Levine, *Offline Reinforcement Learning with Implicit
  Q-Learning* (ICLR 2022), defines the expectile value and advantage-weighted
  extraction used here: <https://arxiv.org/abs/2110.06169>.
- Kumar et al., *Conservative Q-Learning for Offline Reinforcement Learning*
  (NeurIPS 2020), is the principal conservative-Q alternative:
  <https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html>.
- Chen et al., *Decision Transformer* (NeurIPS 2021), supports sequence modeling
  but needs careful return coverage in this highly imbalanced action dataset:
  <https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html>.
- Thomas et al., *Race Strategy Reinforcement Learning* (Machine Learning, 2026),
  shows the value of recurrent state and evaluation across unseen tracks, but its
  Mercedes simulator and data are proprietary:
  <https://link.springer.com/article/10.1007/s10994-026-07081-3>.

## Acceptance

The final test set is opened once. The candidate must achieve 100% two-compound
compliance, a non-positive paired event mean time difference versus dynamic
programming, and a bootstrap 95% upper bound at or below zero. Multiple HPC seeds
should also be reported before deployment; the single configured seed is an
end-to-end pipeline run, not final statistical evidence.
