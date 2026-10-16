# Hosted demo deployment

The public demo is [f1strategy-forza.streamlit.app](https://f1strategy-forza.streamlit.app/).
Its existing Streamlit Community Cloud configuration watches the GitHub repository
`whk-irl/F1_Strategy`, branch **`test_branch`**, with entry file
**`services/frontend/app.py`**. The repository's default branch is **`main`**.

Updating a feature branch alone does not update the hosted app or the default GitHub
README. Publish the reviewed release to `main`, then merge it into `test_branch`
without rewriting that branch's history. Resolve overlapping frontend changes in
favor of the release's Race Center launcher while retaining the existing backend
modules. Streamlit automatically redeploys when its watched branch changes.

## Release checks

1. Run Ruff, mypy and pytest, and confirm GitHub CI passes.
2. Open a PR against `main`, inspect the final diff, and squash-merge the release.
3. Merge the release into `test_branch`; run checks against the combined code and
   push the resulting merge. Preserve unrelated changes already on that branch.
4. Open the public app, allow dependencies and recorded data to load, and verify
   the circuit map, moving cars, timing tower, playback controls and Strategy Lab.

The hosted app starts with recorded replay. Historical GPS is downloaded on first
use and cached in the app filesystem. This cache is ephemeral and may need rebuilding
after a reboot or redeployment. No subscription or credentials are needed for replay.

The latest published training model card and chart are committed under `docs/`, so
Strategy Lab can show the result on a clean deployment. If `PITWALL_TRAINING_RUN`
points to an available local run, that run's full report takes precedence.
Model weights, raw acquisition caches and MLflow databases are not added to Git.

## Optional live access

Configure OpenF1 credentials through the hosting provider's server-side secret store.
Use `OPENF1_TOKEN`, or `OPENF1_USERNAME` and `OPENF1_PASSWORD` for hourly token renewal.
Root-level Streamlit secrets are available as environment variables. Restart the app
after changing connection settings. Do not put credentials in the README, source,
component payloads or browser configuration.

Current live data requires an OpenF1 subscription. The UI checks session activity,
uses a 20-second playback buffer and refreshes every 15 seconds. It displays waiting,
stale or failed feed states explicitly. See the [race-center guide](race-center.md)
for the feed semantics and the remaining active-session verification requirement.

References: [Streamlit automatic app updates](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/edit-your-app),
[Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
