# Race Center

Run `uv run streamlit run services/frontend/app.py` from the repository root.
The default workspace is a race broadcast view; **Strategy Lab** opens the existing
model and simulation explorer. The launcher keeps heavy model imports out of Race Center.

## Recorded race controls

- Choose a season (2023 onwards) and completed race in the header.
- Click a driver in the classification tower to load that driver's onboard telemetry.
- Play/pause, restart, seek or change playback speed from 1× to 60×.
- **Aa** toggles driver labels. **⌖** follows the selected car; **⤢** resets the map.
- **About the data** explains the sources and strategy limitations.

The first visit downloads GPS in eight-minute chunks and one driver's telemetry.
These responses and a compact replay bundle are cached under `data/race_center/`
(ignored by Git). Downloading a new race may take a minute or more. No paid connection
is needed for historical data. Upstream outages are reported with retry controls.

The map follows actual OpenF1 x/y samples with interpolation only across nearby
samples, never across gaps longer than three seconds. A complete recorded lap supplies
the circuit trace. Sector markers come from that lap's recorded sector timestamps.
Coordinates are approximate, and the map does not establish true lane position.
The view may rotate to fit the available screen; it is not a geographic north-up map.

Timing, gaps, telemetry, tyre stints, weather and messages follow the replay cursor.
Lap and sector times appear only once a lap is completed. Gaps older than 45 seconds
and onboard samples older than three seconds display as unavailable. Cars with no
recent lap data are marked **NO DATA**, rather than inferring a formal retirement.
Playback pauses on initial load when the browser requests reduced motion.
Every race selection and fresh page load starts at lights out; playback position is
not restored from browser storage. Display preferences such as speed and map labels
are retained.
The browser animates locally; it does not call the provider for every frame.

## Live sessions

[OpenF1](https://openf1.org/docs/) offers historical data for free from 2023 and
requires a subscription for real-time data. Configure its access token as the
`OPENF1_TOKEN` environment variable on the **server**, then restart Streamlit.
The token is used in the server's Authorization header and never included in
component payloads, cache files, error messages or browser requests.

Tokens [expire after one hour](https://openf1.org/auth.html). For automatic renewal,
configure `OPENF1_USERNAME` and `OPENF1_PASSWORD` on the server instead (or alongside
the initial token). The client obtains a token from OpenF1's documented OAuth endpoint,
renews before expiry and retries an expired-token response. With only `OPENF1_TOKEN`,
replace an expired token and restart the server. No credentials are entered in the UI.

The app checks whether the latest session is currently active. When none is active,
it displays a waiting screen and a return-to-replay button. With access configured,
live timing refreshes every 15 seconds, with a 20-second playback buffer so cars can
move smoothly between polls. The interface shows current delay and marks stale data.
This is a delayed timing display with network/provider latency.
Live sessions are not saved as immutable completed replays. A recent recorded lap
is needed before a circuit trace can be drawn. The authenticated feed must be
verified during an active session; the local verification used recorded data and
mocked live-feed contracts, with no active-session subscription available.

## Strategy panel

The race center's desk is a **nominal planning reference**, separate from the trained
PPO policy. Its pit-lane estimate comes from [measured OpenF1 stops](../model-cards/2026-10-16-pit-time-retraining.md)
at the circuit. As earlier routine stops arrive in the selected race, their median
updates the estimate without reading future replay records. The panel also shows the
last recorded lane time for the selected driver. Extended stops remain visible but
are excluded from the routine forecast.

The estimated race-time loss uses the difference between observed in/out laps and
nearby clean laps; it is less certain than lane time. During a safety car or VSC,
the nominal planner applies a 0.6 adjustment to that loss. It compares estimated
loss with current same-lap gaps to project a rejoin position and uncertainty range.
The command panel compares boxing now, the model window, and a three-lap extension.
Its rival battle tracks up to three cars within 15 seconds, including their compound,
tyre age, relative gap and recent-stop state. It distinguishes a fresh-tyre undercut
threat, an undercut already in progress, an overcut opportunity, and a nearby attack
target. This is a timing estimate, not a guarantee: traffic, tyre warm-up, unusual pit
service and changing gaps can alter the result.

The desk runs finite-horizon dynamic programming over stay/soft/medium/hard actions
with measured-history pit cost, base degradation rates
0.08/0.05/0.03 seconds per lap of age, cliff ages 22/32/42 and compound offsets
0/0.25/0.55 seconds. It enforces two dry compounds; a stop changes tyres on the
following lap. The degradation parameters are illustrative and independent of the
calibrated simulator's evaluation DP. The desk ignores tyre inventory and future weather,
and holds the call during red flags. Rain and wet-compound states switch the panel to
a weather crossover call rather than applying the dry optimization.
Live timing also suspends the nominal plan because the feed does not confirm the
scheduled race distance; it does not treat the last observed lap as the final lap.

The [retraining model card](../model-cards/2026-10-16-pit-time-retraining.md) records the actual
BC → PPO → fine-tuning result and why the new checkpoint was not promoted.
Strategy Lab continues to expose the baked model experiments separately.

## Implementation and checks

`services/live/race_data.py` provides cached, bounded requests and the shared replay/live
schema. It preserves mixed numeric/string gap values (`+1 LAP`) and missing telemetry.
`services/frontend/race_center.py` bridges Streamlit and the local component in
`services/frontend/race_center_component/`. The component uses no CDN or build step.
Streamlit 1.37 or newer is required for live fragments.

Unit tests cover URL comparison syntax, chronological bounds, race-start alignment,
sector locations, mixed gap types, archive reuse, retries and safe errors, completed
race selection, live timestamps and track reuse. Browser checks cover playback,
driver selection, map controls, waiting state and desktop/mobile layouts.
