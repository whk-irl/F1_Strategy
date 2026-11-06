# United States GP live dress rehearsal — 25 October 2026

The verified race start is **Sunday 25 October, 15:00 CDT (20:00 UTC)** at Austin,
over **56 laps**. The project now contains **1,799 pre-race laps** from FP1, FP2,
FP3 and qualifying. The published race starting grid is fetched by `meeting_key`
and retained separately from qualifying classification; their orders differ.
Practice calibration uses only clean FP laps and keeps fuel/traffic uncertainty
explicit. These sessions stay outside the race-policy training lake.

Run this weekend's **shadow test on this computer**. The free FastF1 SignalR adapter
can supply core timing without an OpenF1 subscription. Its live incremental path is
experimental: verify it during an active session before relying on the display.
GPS and full car telemetry are not guaranteed on this route. The adapter cannot
promise every OpenF1 race-center feature from free timing alone.

The first free-timing connectivity probe failed. **Live access has not yet been
confirmed on this computer**; rerun the bounded probe during an active session
before treating the race-day setup as ready.

The Temporal IQL candidate remains unpromoted after its smoke run. Assess the feed
and reference strategy desk prospectively; a running screen does not validate the
new policy's racing performance.

## Before race day

1. Use this computer with a stable connection, outbound HTTPS/WebSocket access, the
   current project dependencies and persistent storage for capture logs. Disable
   sleep and connect power. Start capture well before the 15:00 CDT race start.
2. Select the free provider in preflight. It does not require paid OpenF1 secrets:

   ```powershell
   uv run python -m services.live.preflight --provider free --offline --output runs/live/usgp-offline.json
   ```

3. Run the bounded SignalR probe while a session is active:

   ```powershell
   uv run python -m services.live.preflight --provider free --probe-seconds 30 --output runs/live/usgp-preflight.json
   ```

   `waiting` means the connection is available but active timing has not arrived.
   An inactive session cannot demonstrate that race timing will flow. `failed`
   exits with code 1 and gives a safe failure code. Retain the safe JSON report.
4. Start the local launcher at about **14:30–14:40 CDT** and keep this computer on:

   ```powershell
   .\scripts\start_usgp_live.ps1
   ```

   It refreshes the pre-race context, starts the free receiver and local Streamlit
   app, and writes logs plus process IDs under `runs/live/usgp-*`. Open
   `http://localhost:8501/` and choose Live. A pre-existing app on that port is reused;
   choose another `-Port` if that listener is a different application.

   The official FastF1 SignalR recorder normally writes a recording for
   **post-session** loading; this project's custom incremental adapter is the
   experimental part that makes arriving timing usable during the session.
5. Select the car to follow when timing arrives. The free adapter deliberately
   omits GPS and full telemetry; do not interpret their absence as a car stopped
   on track. A working launcher does not establish that the remote feed is ready.

## During the race

- Check that session identity, current lap, positions, compounds and timestamps
  advance. Use the timing panel for this free-source test; car motion on the map
  cannot be validated without a GPS feed.
- The race center's polling path refreshes every **15 seconds** and plays a
  **20-second buffered** view. Provider and network latency add to that delay.
  Measure the observed delay; this is a delayed shadow test, not instantaneous
  pit-wall execution.
- At each recommendation or material change, record UTC time, displayed race lap,
  displayed sample age/delay, car/compound/tyre age, recommendation and alternate
  window, pit-loss estimate/range, projected rejoin, named rival and risk.
  Record the outcome after the relevant pit cycle and any missing inputs.
  The app also appends bounded inputs to `data/live/11396/snapshots.jsonl` and
  structured prospective calls to `data/live/11396/recommendations.jsonl`. The
  launcher saves the raw timing stream and process logs under its timestamped
  `runs/live/usgp-*` folder. Keep these for scoring after the race.
- If timestamps stop advancing or the adapter reports stale data, treat the
  recommendation as unavailable and record the outage. Rerun the probe after
  recovery. Do not score a frozen recommendation as a new live call.

## Optional OpenF1 provider

The existing paid live route is still supported with `--provider openf1`. It needs
a subscription with real-time access and server-side `OPENF1_USERNAME` plus
`OPENF1_PASSWORD` for automatic token renewal, or `OPENF1_TOKEN` until it expires.
There is no need to buy this for the free-timing dress rehearsal. Never put secrets
in browser code, Git, screenshots or shared run reports.

```powershell
uv run python -m services.live.preflight --provider openf1 --driver 1 --output runs/live/usgp-openf1.json
```

Change the car number to one present in this weekend's field. OpenF1 checks
bounded location, telemetry and position samples (freshness limit: 60 seconds)
plus connectivity for the other race-center endpoints. Empty sparse pit and
race-control streams are allowed. No active session is `waiting`, not an outage.
Preflight only reports credential presence and never emits account values or HTTP
response bodies.

## After the chequered flag

Archive the completed session using the normal acquisition workflow. Compare the
shadow calls with measured pit-lane times, observed rejoin positions, undercuts and
actual stops. Separate data delays from model errors, and report missing-data periods.
Keep the full race as prospective evaluation data; do not train on the test race and
then report its replay as independent real-time validation.

Schedule reference: [official United States GP weekend](https://www.formula1.com/en/racing/2026/united-states).
Implementation reference: [FastF1 live timing documentation](https://docs.fastf1.dev/livetiming.html).
