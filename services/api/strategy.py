"""FastAPI service for pit-strategy advice.

    uvicorn services.api.strategy:app --port 8000
    POST /strategy/recommend   {"state": CarState, "field_laps": [FieldLap], "mode": "planner"}
    GET  /strategy/health

The planner is the default and released mode. ``mode="ensemble"`` serves the frozen
opt-in candidate named in configs/strategy-serving.json, if configured.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from ml.strategy_serving.advisor import (
    SERVING_CONFIG,
    CarState,
    FieldLap,
    Recommendation,
    StrategyAdvisor,
)
from pydantic import BaseModel, Field

app = FastAPI(title="Pitwall AI strategy advisor", version="1.0.0")


class RecommendRequest(BaseModel):
    state: CarState
    field_laps: list[FieldLap] = Field(
        default_factory=list, description="All cars' completed laps so far"
    )
    mode: Literal["planner", "ensemble"] = "planner"


@lru_cache(maxsize=1)
def advisor() -> StrategyAdvisor:
    return StrategyAdvisor.from_config(SERVING_CONFIG)


@app.get("/strategy/health")
def health() -> dict[str, object]:
    adv = advisor()
    modes = ["planner"] + (["ensemble"] if adv.ensemble is not None else [])
    return {
        "status": "ok",
        "default_mode": "planner",
        "modes": modes,
        "priors": adv.priors_sha,
        "ensemble_candidate": adv.candidate_id or None,
    }


@app.post("/strategy/recommend", response_model=Recommendation)
def recommend(req: RecommendRequest) -> Recommendation:
    try:
        return advisor().recommend(req.state, req.field_laps, req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
