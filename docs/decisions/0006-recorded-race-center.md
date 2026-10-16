# ADR 0006: Recorded-position Race Center

**Status:** accepted, 2026-10-16.

## Context

The user requested a circuit map, moving cars and race-style panels alongside the
data and model lifecycle update. The prior Streamlit screen primarily showed model
charts and lap-by-lap recommendations. Browser animation should not rerun the model
or download a full field of GPS for every frame.

## Decision

Keep Streamlit and add a local HTML/CSS/Canvas component using the standard Streamlit
component protocol. A lightweight launcher opens Race Center; Strategy Lab retains
the legacy model explorer. No JavaScript build system or CDN is required.

OpenF1 supplies actual location, timing, telemetry, tyre, weather and control records.
The server requests historical GPS in bounded chunks, caches responses, and emits a
compact timeline. Sector positions come from recorded sector timestamps on the
reference lap. The browser owns playback and interpolates adjacent GPS samples.
Missing data remains missing, and completed-lap details follow the replay cursor.

The live fragment polls every 15 seconds; a 20-second browser playback buffer allows
movement between requests. Authentication and token renewal remain on the server.
Waiting, stale and failed connections are explicit. Current live availability requires
an OpenF1 subscription, and authenticated active-session behavior still needs field
verification. Replay has been checked using recorded data.

The strategy desk uses a clearly labelled nominal dry-race DP reference. It is
independent of the trained model and the calibrated simulator's benchmark DP.
The newly trained PPO candidate was not promoted after evaluation. Live planning is
unavailable without a confirmed race distance. Research-mode model comparison
remains in Strategy Lab.

## Consequences

The app gains fluid playback without loading ML dependencies for Race Center.
Historical GPS requires network access on first use, with potentially large local
cache files. Provider positions are approximate and timing may lag. Streamlit 1.37+
is needed for fragments. Component code has a browser verification workflow in
addition to Python data-contract tests. The [guide](../guides/race-center.md) records
controls, assumptions, coverage and limitations.
