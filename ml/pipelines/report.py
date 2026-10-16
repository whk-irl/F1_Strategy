"""Human-readable run evidence and paired, event-level uncertainty estimates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def paired_event_difference(
    candidate: dict[str, Any], baseline: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Bootstrap event means rather than treating drivers in one race as independent."""
    grouped: dict[tuple[int, int], list[float]] = {}
    for first, second in zip(candidate["episodes"], baseline["episodes"], strict=True):
        identity = ("year", "round", "driver", "seed")
        if any(first[key] != second[key] for key in identity):
            raise ValueError("Policy comparison episodes are not paired")
        key = (first["year"], first["round"])
        grouped.setdefault(key, []).append(first["time_s"] - second["time_s"])
    means = np.asarray([np.mean(values) for values in grouped.values()])
    samples = (
        np.random.default_rng(seed)
        .choice(means, size=(2000, len(means)), replace=True)
        .mean(axis=1)
    )
    return {
        "mean_event_time_difference_s": float(means.mean()),
        "bootstrap_95_ci_s": np.quantile(samples, [0.025, 0.975]).tolist(),
        "events": len(means),
        "sign": "negative means candidate is faster",
        "note": "Replay-event uncertainty only; does not include training-seed or simulator error",
    }


def write_report(directory: Path, report: dict[str, Any], gate: dict[str, Any]) -> None:
    """Write a compact model card with metrics and the limitations of the evidence."""
    lines = [
        "# Staged training run",
        "",
        report["algorithm"],
        "",
        f"Run fingerprint: `{report['fingerprint']}`",
        "",
        f"Simulator gate: legacy replay rho={gate['legacy_replay']['mean_rho']:.3f}; "
        f"RL environment rho={gate['policy_environment']['mean_rho']:.3f}.",
        "",
        "## Held-out policy evaluation",
        "",
        "| Policy | Mean position | Mean time (s) | Stops | Two-compound compliance | Reward |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["policy"].items():
        lines.append(
            f"| {name} | {metrics['mean_position']:.2f} | {metrics['mean_time_s']:.2f} | "
            f"{metrics['mean_stops']:.2f} | {metrics['compound_compliance']:.0%} | "
            f"{metrics['mean_reward']:.2f} |"
        )
    lines += [
        "",
        "The selected fine-tuning checkpoint is chosen on validation reward. "
        "Test results are reported once and never used to select a checkpoint.",
        "",
        "## Supporting models",
        "",
        "```json",
        json.dumps(report["world_model_test"], indent=2),
        "```",
        "",
        "## Evidence limits",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines += [
        "",
        "This bundle is an unpromoted research candidate. Benchmark acceptance: "
        f"**{report['candidate_passes_benchmark']}**. Detailed paired episodes, event-bootstrap "
        "intervals, source checksums and split membership are saved beside this report.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
