"""Pit-strategy advisor: the usable model.

Given one car's state and the field's laps so far, return STAY or PIT(compound),
the planner's estimate of what pitting now saves, and (in ensemble mode) the
frozen PPO ensemble's decision. Semantics match evaluation exactly: the decision
at the end of lap ``lap`` uses field laps through ``lap - 1``.

Modes (configs/strategy-serving.json):
  * ``planner``  default release: exact DP on in-race estimates.
  * ``ensemble`` opt-in: frozen arm-B ensemble 7e8800ab93b81c68. It failed the
    pre-registered 2026 non-inferiority gate on power (−3.49 s/race, CI
    [−10.58, +5.04]); promotion depends on the pre-registered prospective check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from ml.strategy.env import observation, observation_v2
from ml.strategy.planner import ILLEGAL, Plan, solve
from ml.strategy.race_model import TyreParams, clean_laps, inrace_update, prepare_race
from ml.strategy_serving.priors import GLOBAL, load

if TYPE_CHECKING:  # PyTorch loads only when an ensemble is configured
    from ml.strategy.ensemble import EnsemblePolicy

Compound = Literal["SOFT", "MEDIUM", "HARD"]
COMPOUNDS: tuple[Compound, ...] = ("SOFT", "MEDIUM", "HARD")
Action = Literal["STAY", "PIT_SOFT", "PIT_MEDIUM", "PIT_HARD"]
ACTIONS: tuple[Action, ...] = ("STAY", "PIT_SOFT", "PIT_MEDIUM", "PIT_HARD")
SERVING_CONFIG = Path("configs/strategy-serving.json")


class CarState(BaseModel):
    """The car being advised, at the end of lap ``lap`` (stop happens at end of this lap)."""

    circuit: str = Field(description="Circuit key, e.g. 'monza'; unknown keys use the global prior")
    total_laps: int = Field(gt=1)
    lap: int = Field(ge=1)
    compound: Compound
    tyre_age: int = Field(ge=1, description="Laps on this set including the current lap")
    compounds_used: list[Compound] = Field(description="Dry compounds already run, incl. current")
    neutral: bool = Field(default=False, description="Current lap is under SC/VSC")

    @model_validator(mode="after")
    def _check(self) -> CarState:
        if self.lap > self.total_laps:
            raise ValueError("lap exceeds total_laps")
        if self.compound not in self.compounds_used:
            raise ValueError("compounds_used must include the current compound")
        return self


class FieldLap(BaseModel):
    """One completed lap by any car (public timing)."""

    driver_number: str
    lap_number: int
    lap_time_s: float | None
    compound: Compound | Literal["INTERMEDIATE", "WET"]
    tyre_life_laps: int | None
    track_status: str = "1"
    pit_in: bool = False
    pit_out: bool = False
    is_accurate: bool = True


class Recommendation(BaseModel):
    mode: Literal["planner", "ensemble"]
    action: Literal["STAY", "PIT_SOFT", "PIT_MEDIUM", "PIT_HARD"]
    planner_action: Literal["STAY", "PIT_SOFT", "PIT_MEDIUM", "PIT_HARD"]
    pit_saves_s: float = Field(
        description="Planner: expected race time saved by the best stop now vs staying out"
    )
    ensemble_pit_probability: float | None = None
    explanation: str
    estimates: dict[str, float]
    model: dict[str, str]
    option_costs_s: dict[str, float] = Field(
        default_factory=dict,
        description="Planner: expected remaining strategy time of each legal option vs the best",
    )


def _field_frame(laps: list[FieldLap], circuit: str) -> pd.DataFrame:
    enc = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
    rows = [
        {
            "year": 9999,
            "round_number": 1,
            "driver_number": x.driver_number,
            "lap_number": x.lap_number,
            "lap_time_s": np.nan if x.lap_time_s is None else x.lap_time_s,
            "compound_encoded": enc[x.compound],
            "tyre_life_laps": np.nan if x.tyre_life_laps is None else x.tyre_life_laps,
            "track_status": x.track_status,
            "pit_in_this_lap": x.pit_in,
            "pit_out_this_lap": x.pit_out,
            "is_accurate": x.is_accurate,
            "circuit_key": circuit,
        }
        for x in laps
    ]
    return pd.DataFrame(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StrategyAdvisor:
    """Loads priors once; ensemble policies only if the ensemble mode is configured."""

    def __init__(self, priors: Path, ensemble_dir: Path | None = None) -> None:
        self.priors = load(priors)
        self.priors_sha = _sha(priors)[:16]
        self.ensemble: EnsemblePolicy | None = None
        self.ensemble_obs = "v1"
        self.candidate_id = ""
        if ensemble_dir is not None:
            self._load_ensemble(ensemble_dir)

    @classmethod
    def from_config(cls, config: Path = SERVING_CONFIG) -> StrategyAdvisor:
        cfg = json.loads(config.read_text())
        ens = cfg.get("ensemble", {}).get("path")
        return cls(Path(cfg["priors"]), Path(ens) if ens else None)

    def _load_ensemble(self, root: Path) -> None:
        from ml.strategy.ensemble import EnsemblePolicy

        frozen = json.loads((root / "FROZEN.json").read_text())
        for rel, digest in frozen["policies"].items():
            if _sha(root / rel) != digest:
                raise ValueError(
                    f"{root / rel} does not match the frozen candidate; refusing to serve it"
                )
        meta = json.loads((root / "candidate.json").read_text())
        self.ensemble = EnsemblePolicy.load(sorted(root.glob("seed-*/policy.zip")))
        self.ensemble_obs = meta.get("observation", "v1")
        self.candidate_id = str(frozen["candidate_id"])

    def estimate(self, state: CarState, laps: list[FieldLap]) -> tuple[TyreParams, np.ndarray]:
        prior = self.priors.get(state.circuit, self.priors[GLOBAL])
        df = _field_frame(laps, state.circuit)
        if df.empty or state.lap <= 5:
            return prior, np.zeros(3)
        race = prepare_race(df, 9999, 1)
        seen = clean_laps(race, state.lap - 1)
        evidence = np.array([(seen["compound_encoded"] == k).sum() for k in (0, 1, 2)], float)
        return inrace_update(race, prior, state.lap - 1), evidence

    def recommend(
        self,
        state: CarState,
        laps: list[FieldLap],
        mode: Literal["planner", "ensemble"] = "planner",
    ) -> Recommendation:
        if mode == "ensemble" and self.ensemble is None:
            raise ValueError("Ensemble mode is not configured on this server")
        est, evidence = self.estimate(state, laps)
        return self.decide(state, est, evidence, solve(est, state.total_laps), mode)

    def decide(
        self,
        state: CarState,
        est: TyreParams,
        evidence: np.ndarray,
        plan: Plan,
        mode: Literal["planner", "ensemble"] = "planner",
    ) -> Recommendation:
        """Decision from precomputed estimates and DP plan (shared by every car on a lap)."""
        if mode == "ensemble" and self.ensemble is None:
            raise ValueError("Ensemble mode is not configured on this server")
        c = COMPOUNDS.index(state.compound)
        two_used = len(set(state.compounds_used)) >= 2
        n, lap, a, s = state.total_laps, state.lap, state.tyre_age, state.neutral
        q = plan.q_values(lap, c, a, two_used, s)
        planner_act = int(np.argmin(q))
        saves = float(q[0] - np.min(q[1:])) if lap < n else 0.0
        act, p_pit = planner_act, None
        if self.ensemble is not None:
            obs = (
                observation_v2(lap, n, c, a, two_used, s, est, evidence)
                if self.ensemble_obs == "v2"
                else observation(lap, n, c, a, two_used, s, est)
            )
            probs = self.ensemble.probabilities(obs)
            p_pit = float(1.0 - probs[0])
            if mode == "ensemble":
                act = int(np.argmax(probs)) if lap < n else 0
        costs = {
            ACTIONS[i]: float(q[i]) for i in range(4) if np.isfinite(q[i]) and q[i] < ILLEGAL / 2
        }
        return Recommendation(
            mode=mode,
            action=ACTIONS[act],
            planner_action=ACTIONS[planner_act],
            pit_saves_s=round(saves, 2),
            ensemble_pit_probability=None if p_pit is None else round(p_pit, 3),
            explanation=_explain(ACTIONS[act], ACTIONS[planner_act], saves, state, est),
            estimates={
                **{
                    f"deg_{k.lower()}_s_per_lap": round(est.deg[i], 4)
                    for i, k in enumerate(COMPOUNDS)
                },
                "pit_loss_green_s": round(est.pit_green, 2),
                "pit_loss_neutral_s": round(est.pit_neutral, 2),
            },
            model={"priors": self.priors_sha, "ensemble_candidate": self.candidate_id or "none"},
            option_costs_s={k: round(v - min(costs.values()), 2) for k, v in costs.items()},
        )


def _explain(action: str, planner: str, saves: float, state: CarState, est: TyreParams) -> str:
    wear = est.deg[COMPOUNDS.index(state.compound)]
    head = f"{state.compound}, lap {state.tyre_age} on this set, wearing ~{wear:.3f} s/lap"
    if abs(saves) < 0.5:
        best = "staying out" if planner == "STAY" else planner.replace("_", " ").lower() + " now"
        why = f"a near tie: {best} is best by {abs(saves):.2f} s"
    elif planner == "STAY":
        why = f"staying out is {-saves:.1f} s better than the best stop now"
    else:
        why = f"{planner.replace('_', ' ').lower()} now saves ~{saves:.1f} s vs staying out"
    cheap = " Neutralised lap: the stop costs less now." if state.neutral else ""
    note = "" if action == planner else f" Ensemble chooses {action} (planner: {planner})."
    return f"{head}; {why}.{cheap}{note}"
