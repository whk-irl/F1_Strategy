"""Quick smoke-test for the trained PPO strategy policy.

Run from the F1_Strategy directory:
    uv run python test_policy.py
"""

import numpy as np
from ml.models.strategy_policy.predict import action_probabilities, load_policy, recommend_action

print("Loading policy...")
model = load_policy()
print("Policy loaded.\n")

# Scenario 1: worn softs, undercut threat active
obs_undercut = np.array(
    [
        0.40,  # race_progress
        0.60,  # laps_remaining_norm
        0.55,  # own_tyre_life_norm  (33 laps on tyre)
        0.00,  # own_compound_norm   (SOFT)
        0.00,  # is_fresh_tyre
        0.55,  # own_synthetic_deg_norm (heavy deg)
        0.35,  # position_norm       (P7 of 20)
        0.05,  # gap_ahead_s_norm    (3s to car ahead)
        0.04,  # gap_behind_s_norm   (2.4s — close!)
        0.25,  # tyre_age_ahead_norm
        0.08,  # tyre_age_behind_norm (fresher tyres behind)
        1.00,  # undercut_threat     ACTIVE
        0.00,  # overcut_opportunity
        0.00,  # free_stop_available
        0.05,  # cars_lose_if_pit_norm
        0.77,  # pit_loss_norm       (23s)
        0.10,  # sc_probability
        0.60,  # sc_laps_since_last_norm
        0.00,  # is_wet
        0.00,  # wet_fraction
    ],
    dtype="float32",
)

# Scenario 2: safety car out, fresh tyres opportunity
obs_sc = np.array(
    [
        0.55,  # race_progress
        0.45,  # laps_remaining_norm
        0.30,  # own_tyre_life_norm  (18 laps on tyre)
        0.20,  # own_compound_norm   (MEDIUM)
        0.00,  # is_fresh_tyre
        0.25,  # own_synthetic_deg_norm (moderate deg)
        0.50,  # position_norm       (P10 of 20)
        0.20,  # gap_ahead_s_norm    (12s to car ahead)
        0.20,  # gap_behind_s_norm   (12s behind)
        0.30,  # tyre_age_ahead_norm
        0.30,  # tyre_age_behind_norm
        0.00,  # undercut_threat
        0.00,  # overcut_opportunity
        1.00,  # free_stop_available  SC ACTIVE
        0.10,  # cars_lose_if_pit_norm
        0.77,  # pit_loss_norm
        0.00,  # sc_probability
        0.00,  # sc_laps_since_last_norm
        0.00,  # is_wet
        0.00,  # wet_fraction
    ],
    dtype="float32",
)

# Scenario 3: early race, fresh tyres, no threat — should stay out
obs_stay = np.array(
    [
        0.10,  # race_progress       (early)
        0.90,  # laps_remaining_norm
        0.10,  # own_tyre_life_norm  (6 laps, very fresh)
        0.00,  # own_compound_norm   (SOFT)
        0.00,  # is_fresh_tyre
        0.05,  # own_synthetic_deg_norm (minimal deg)
        0.30,  # position_norm       (P6)
        0.10,  # gap_ahead_s_norm
        0.15,  # gap_behind_s_norm
        0.10,  # tyre_age_ahead_norm
        0.10,  # tyre_age_behind_norm
        0.00,  # undercut_threat
        0.00,  # overcut_opportunity
        0.00,  # free_stop_available
        0.00,  # cars_lose_if_pit_norm
        0.77,  # pit_loss_norm
        0.05,  # sc_probability
        0.60,  # sc_laps_since_last_norm
        0.00,  # is_wet
        0.00,  # wet_fraction
    ],
    dtype="float32",
)

scenarios = [
    ("Undercut threat (worn softs, car 2.4s behind on fresher rubber)", obs_undercut),
    ("Safety car out — free stop available", obs_sc),
    ("Early race, 6 laps on tyre, no threat", obs_stay),
]

for name, obs in scenarios:
    action, label = recommend_action(obs, model)
    probs = action_probabilities(obs, model)
    print(f"Scenario: {name}")
    print(f"  Recommendation: {label}")
    print("  Probabilities:")
    for act, p in probs.items():
        bar = "#" * int(p * 40)
        print(f"    {act:<20} {p:.1%}  {bar}")
    print()
