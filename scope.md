# Scope & Non-Goals

Being explicit about what this project is *not* prevents scope creep and lets reviewers evaluate it on the right terms.

## In scope

- Recommending strategic actions (pit / stay / compound choice) for a single car given race state.
- Modeling tire degradation, lap-time evolution, and safety-car probability from historical F1 data.
- A learned race simulator validated against real race outcomes.
- A reinforcement-learning policy trained against that simulator.
- Production-grade deployment on Kubernetes with full MLOps tooling (registry, monitoring, CI/CD, IaC).

## Explicit non-goals

- **Real-time live race timing.** F1's live timing feed is not freely redistributable; this project simulates "live" decisions over recorded race state. The user-facing demo scrubs through historical races.
- **Multi-car coordinated strategy.** Real teams optimize across both cars (e.g., split strategy). This project models a single car. Extending to multi-agent is a future-work item, not a v1 promise.
- **Aerodynamic / driver-input modeling.** No CFD, no driver style modeling beyond per-driver pace baselines.
- **Predicting race results from outside the race.** This is not a betting / pre-race outcome prediction tool.
- **Beating real F1 strategists.** Real teams have private data this project doesn't (car setup, fuel maps, driver feedback). The benchmark is "beats heuristic baselines and explains its reasoning" — not "beats Mercedes."

## Validation criteria

The simulator must achieve **Spearman ρ ≥ 0.7** between predicted and actual finishing positions on a held-out test set of 2024 races, when seeded with real driver actions, before the RL agent is trained on top of it. If it doesn't, the simulator gets fixed before any RL work happens. This guardrail prevents the classic RL failure mode of an agent learning to exploit a broken environment.
