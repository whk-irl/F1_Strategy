# ADR 0004 — Sequence tire degradation models (TCN+GRU and PatchTST)

**Status:** Accepted  
**Date:** 2026-W20  
**Supersedes:** Nothing (additive alongside LightGBM baseline in ADR 0001)

## Context

The LightGBM tire degradation model (see `ml/models/tire_degradation/train.py`) treats
each lap as an independent observation.  It predicts `lap_time_delta_s` from a snapshot
of tyre state features, ignoring the *sequence* of prior laps in the same stint.  This
is a deliberate simplification that works well for the batch tire table inside the Gym
simulator, but it misses:

1. **Degradation trajectory curvature** — tyres often degrade non-linearly; the rate
   of increase in lap time can itself accelerate or plateau depending on compound and
   temperature history.
2. **Contextual anomalies** — a slow sector behind a backmarker, a brief yellow-flag
   period, or a fuel-rich opening phase all show up as outliers in the feature snapshot
   but are interpretable in context.

Two sequence architectures were evaluated:

- **TCN + GRU** — Temporal Convolutional Network front-end (causal dilated convolutions)
  followed by a Gated Recurrent Unit.  The TCN captures multi-scale local patterns;
  the GRU integrates longer sequential dependencies.  Dilations up to `2^(n_tcn_layers-1)`
  give a receptive field that covers the full `SEQ_LEN=10` window.

- **PatchTST** — inspired by Nie et al. "A Time Series is Worth 64 Words" (ICLR 2023).
  The input sequence is split into non-overlapping patches (default: 5 laps each), each
  embedded as a single token.  A lightweight Transformer encoder attends across the
  patch tokens.  With `SEQ_LEN=10` and `patch_len=5` this gives only 2 tokens, keeping
  the attention cost negligible and avoiding the quadratic scaling concern for longer
  sequences.

## Decision

Both models are implemented and trained in parallel, registered as separate MLflow
experiments (`tire-sequence-models`).  Selection for production serving is based on
offline validation MAE on held-out rounds 5, 10, 15, 20.

### Scalar interface compatibility

The simulator's tire table calls `predict_lap_time_delta(tyre_life_laps, …)` in
a tight batch loop and cannot maintain per-driver stint state.  To satisfy this
constraint the sequence models expose the **same scalar signature** as the LightGBM
predictor, with the single input step repeated `SEQ_LEN` times — a steady-state
approximation that performs best mid-stint when degradation is roughly linear.  The
interface limitation is documented in `predict_sequence.py`.

The more accurate `predict_stint_from_history` interface is intended for the
Streamlit visualisation layer where real stint history is available.

## Consequences

**Positive:**

- Sequence models capture degradation curvature that LightGBM misses, which should
  improve strategy recommendations for long stints.
- PatchTST's patch tokenisation naturally aggregates lap-level noise, reducing the
  impact of single-lap anomalies.
- Both models are tracked in MLflow alongside the LightGBM baseline, enabling
  side-by-side comparison of validation MAE and Spearman ρ on held-out races.

**Negative / trade-offs:**

- The steady-state scalar approximation degrades accuracy for laps 1–`SEQ_LEN` of
  a new stint where the model sees a repeated identical row rather than real history.
- PyTorch inference adds latency versus a LightGBM `predict` call; acceptable for the
  Streamlit demo but would need benchmarking before real-time serving.
- Training requires PyTorch and a full gold dataset (`SEQ_LEN+1` laps per stint); very
  short stints produce no windows, so the training set is slightly smaller than the
  LightGBM dataset.

## Future work

- **Stateful Gym environment extension:** Replace the batch tire table with a per-driver
  stint state buffer so the simulator can call `predict_stint_from_history` instead of
  the scalar approximation.  This is deferred to v2 (see `docs/scope.md`).
- **Longer context:** Increase `SEQ_LEN` to 20 once more multi-season data is ingested;
  the TCN's exponential dilation schedule scales without architecture changes.
- **Channel independence:** Evaluate PatchTST with per-feature patch streams (Nie et al.
  §3.2) once compound-specific sub-models are considered.
