# US GP preparation snapshot — October 24, 2026

The public OpenF1 import completed for Austin meeting **1297**. The configured race
is session **11396**, Sunday October 25 at **15:00 CDT / 20:00 UTC**, over **56 laps**.
The schedule/distance is checked against the [official race hub](https://www.formula1.com/en/racing/2026/united-states).

| Session | OpenF1 key | Acquired laps | First three |
|---|---:|---:|---|
| FP1 | 11389 | 507 | Leclerc, Verstappen, Norris |
| FP2 | 11390 | 589 | Leclerc, Antonelli, Verstappen |
| FP3 | 11391 | 423 | Hamilton, Verstappen, Leclerc |
| Qualifying | 11392 | 280 | Antonelli, Verstappen, Hadjar |

The **1,799 laps** are held in `data/weekends/2026-united-states`, separate from the
race-policy training lake. Raw JSON, transformed Parquet, grid and circuit-trace
artifacts have SHA256 provenance. The deployable summary is
`configs/weekends/us-2026.snapshot.json`.

Qualifying and the published starting grid are separate records. OpenF1 attaches
this grid to the qualifying session, so acquisition queries by meeting key. The
first three remain Antonelli, Verstappen and Hadjar, while some lower positions
change. Refresh before lights out for any subsequent penalties or withdrawals.

Practice estimates use within-driver/session/stint slopes, removing pit laps,
flagged laps, out laps and obvious cooldown laps. They are shrunk toward Austin
priors because fuel loads and traffic are unknown:

| Compound | Usable stints | Clean laps | Blended slope (s/lap of age) |
|---|---:|---:|---:|
| Soft | 10 | 83 | 0.11107 |
| Medium | 10 | 85 | 0.11317 |
| Hard | 5 | 42 | 0.07963 |

These are weekend reference-model inputs. Cliff ages remain priors, and no IQL
network was retrained or promoted using these sessions.

The actual unauthenticated SignalR probe connected and received nine initial
topics with 22 qualifying drivers. It reported **waiting**, with no fresh race
timestamp. That verifies connection and snapshot decoding; it does not establish
race-day delivery, latency or prediction quality. The free adapter requests core
timing, tyres, weather and race control, with GPS/car telemetry excluded. The
preloaded map uses 156 actual qualifying GPS points as a static circuit trace.

Follow the [live test guide](live-race-test.md) and start
`./scripts/start_usgp_live.ps1` around 14:30–14:40 CDT. Verify advancing race
timestamps, current lap, tyre state and rival gaps, then retain the raw stream and
server snapshot/call logs for retrospective scoring.
