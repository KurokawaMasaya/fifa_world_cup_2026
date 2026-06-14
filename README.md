# FIFAproject2026

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-read--only-009688)](https://fastapi.tiangolo.com/)
![Status](https://img.shields.io/badge/status-live%20demo-brightgreen)

A 2026 FIFA World Cup forecasting system that combines calibrated match prediction, roster-aware adjustments, tournament simulation, live result evaluation, live tournament simulation, and a read-only FastAPI backend.

This is not just a one-time prediction table. CupCast is built as a live forecasting system: it produces pre-match probabilities, tracks completed results, evaluates calibration as matches finish, re-simulates the tournament from the current state, and exposes the latest outputs through an API.

## What It Does

- Predicts group-stage win/draw/loss probabilities.
- Displays likely scorelines using a separate scoreline layer.
- Simulates tournament advancement and champion probabilities.
- Fetches live match results from ESPN JSON endpoints.
- Evaluates predictions automatically after completed matches.
- Runs live tournament simulation conditioned on real completed results.
- Exposes outputs through a read-only FastAPI backend.

## Why This Project Is Interesting

Most sports prediction projects stop at pre-match probabilities. This project treats forecasting as a live system:

1. Predictions are generated before matches.
2. Completed matches are fetched automatically.
3. Accuracy and calibration are evaluated live.
4. Tournament probabilities are re-simulated based on actual results.
5. Outputs are exposed through a deployed API-ready backend.

## System Architecture

```text
Historical Results + Ratings + Squad Data
        |
        v
Calibrated W/D/L Model
        |
        v
Roster-Aware Adjustment
        |
        v
Scoreline Display Layer
        |
        v
Tournament Simulation
        |
        v
ESPN Live Results -> Live Evaluation -> Live Tournament Simulation
        |
        v
Read-only FastAPI Backend
```

## Quick Demo

Run the local API:

```bash
PYTHONPATH=. python3 -m uvicorn src.api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Example read-only endpoints:

- `GET /health`
- `GET /predictions/group-stage`
- `GET /predictions/matches`
- `GET /live/evaluation`
- `GET /live/tournament-simulation`
- `GET /simulation/tournament`

## Current Status

- V1 calibrated team-strength model: completed.
- V2 player-informed / roster-aware probability layer: completed and current.
- Production scoreline display layer: V2.4 ABCD no-E gated blowout mixture.
- Read-only FastAPI backend: completed.
- ESPN live updater: prepared.
- Live evaluation: handles pre-tournament state with zero completed matches.
- Live tournament simulation: completed as a state-conditioned simulator.
- Large datasets and heavy generated outputs: moved outside GitHub and documented separately.

## Model Architecture

### V1: Calibrated Team-Strength Baseline

V1 is the calibrated team-level forecasting backbone. It uses team-level strength inputs, weighted Elo-style ratings, and external anchors to estimate each national team's baseline quality.

The V1 flow is:

```text
historical international matches
  -> team strength / weighted Elo
  -> strength difference
  -> expected goals
  -> calibrated win/draw/loss probabilities
```

V1 is evaluated with:

- accuracy
- Brier score
- log loss
- mean probability assigned to the actual outcome
- expected-goals error
- draw calibration error

### V2: Player-Informed Roster-Aware Model

V2 is now the current player-informed model layer. It is no longer treated as merely experimental.

V2 builds on the calibrated V1 probability backbone. It does not blindly replace team-strength ratings or overwrite calibrated probabilities. Instead, it applies small roster-aware probability adjustments after V1 has produced the base win/draw/loss probabilities.

V2 accounts for:

- squad depth
- player value concentration
- star-player dependence
- roster availability
- manual squad overrides
- player-informed uncertainty

This design keeps the calibrated team-strength model stable while allowing roster changes and player availability to affect match probabilities.

V2 also keeps outcome probability and scoreline display separated. Official W/D/L probabilities come from the calibrated V2 probability stack, while scorelines are display diagnostics.

## Prediction Output Design

The prediction table does not force a single winner pick. This is intentional.

Football draws can have meaningful probability even when the highest single outcome is a team win. A forced argmax label can hide that uncertainty, so the project exposes:

- team A win probability
- draw probability
- team B win probability
- top-5 displayed scorelines
- top-5 scoreline probabilities

The clean frontend/API prediction file is:

```text
output/predictions/group_stage_predictions_clean.csv
```

This file is designed for presentation and API use. It avoids internal diagnostics, model parameters, and debug fields.

The current production scoreline version is `v24_abcd_no_e`. W/D/L probabilities remain produced by the calibrated V2 probability stack; the V2.4 ABCD no-E promotion only changes the displayed scoreline layer. The presentation output now shows top-5 scoreline candidates instead of a single exact top-1 scoreline.

## Tournament Simulation

The tournament simulator estimates advancement and championship probabilities by repeatedly simulating the full World Cup path.

It estimates probabilities for stages such as:

- group-stage qualification
- Round of 32
- Round of 16
- quarterfinal
- semifinal
- final
- champion

The simulator uses the latest adjusted team inputs and should be rerun after meaningful roster, model, or rating-input changes.

Simulation outputs are stored locally under:

```text
output/simulations/
output/brackets/
```

Large simulation and bracket outputs are not tracked in GitHub.

The project also includes a live state-conditioned tournament simulator. This is separate from the pre-tournament simulator. It does not update team strength, retrain the model, or change W/D/L probabilities. It only conditions the tournament state on completed real match results, then simulates the remaining group-stage and knockout path.

Live simulation output is stored at:

```text
output/live/live_tournament_simulation.csv
```

## Live Data and Evaluation

The live pipeline uses ESPN JSON endpoints to prepare fixture and result files. It does not scrape ESPN HTML.

Live evaluation compares final match results against locked pre-match predictions. It only evaluates rows where:

```text
status == final
```

Before the tournament starts, it is expected that:

```text
completed_matches = 0
status = waiting_for_matches
```

Scheduled, not-started, postponed, cancelled, unknown, and in-progress matches are not treated as completed evaluation data.

The live pipeline runs in this order:

```text
1. Fetch ESPN live fixtures/results
2. Export the daily Beijing-time prediction table
3. Evaluate completed matches against locked predictions
4. Run live tournament simulation from the current tournament state
5. Write live outputs under output/live/ and output/predictions/
```

The live tournament simulation uses completed real scores from ESPN and existing model probabilities for remaining matches. It is a state-conditioned simulation, not a dynamic strength update model.

Example commands:

```bash
python3 src/live/update_live_pipeline.py
python3 src/live/update_live_pipeline.py --live-sim-n-sims 50000
python3 src/live/update_live_pipeline.py --skip-live-simulation
python3 src/live/export_daily_beijing_predictions.py --date 2026-06-14
```

The daily Beijing-time table defaults to the current date in `Asia/Shanghai` and writes:

```text
output/predictions/group_stage_games_today_beijing_top5.csv
output/reports/group_stage_games_today_beijing_top5.md
```

For local cron testing, the project uses:

```bash
python3 src/live/update_live_pipeline.py --live-sim-n-sims 10000
```

If the FastAPI package is outside this source project, set the API package live
output directory before running the live pipeline:

```bash
export CUPCAST_API_PACKAGE_LIVE_DIR=/Users/joe/Desktop/FIFAproject2026/cupcast_api_package/output/live
python3 src/live/update_live_pipeline.py --live-sim-n-sims 10000
```

## API Backend

The API backend is a read-only FastAPI service. It reads existing CSV outputs and returns JSON for frontend or gadget use.

When running an extracted API package from another folder, point it at this
project's generated output directory:

```bash
export CUPCAST_OUTPUT_DIR=/Users/joe/Desktop/McGill/projects/FIFAproject2026/output
uvicorn src.api.main:app --reload
```

The API does not:

- train models
- tune parameters
- fetch ESPN data
- run live evaluation
- generate predictions
- run tournament simulations
- modify files

Frontends should call list/index endpoints first, then call detail endpoints using the returned `match_id` or team name.

Key endpoints:

```text
GET /health
GET /predictions/group-stage
GET /predictions/matches
GET /predictions/teams
GET /predictions/group-stage/{match_id}
GET /simulation/tournament
GET /simulation/teams
GET /simulation/team/{team}
GET /live/results
GET /live/evaluation
GET /live/tournament-simulation
```

The `/live/tournament-simulation` endpoint only reads the latest `output/live/live_tournament_simulation.csv`. It does not run simulations on request.

## Early Live Validation

The first completed match was correctly predicted with an exact 2-0 displayed scoreline. This single match is not statistically meaningful evidence by itself, but it confirms that the prediction output, live result update, evaluation pipeline, and API layer are working end to end.

## Data Availability

Full raw datasets and large generated outputs are not included in this GitHub repository because of file size limits. The full data package is stored separately on Google Drive.

Google Drive data link:

```text
https://drive.google.com/file/d/1bJnvtBXX1PASxo_KnYHa7Js0-cYG1v1I/view?usp=drive_link
```

To restore the full local dataset:

1. Download the data package from the Google Drive link above.
2. Extract the downloaded package.
3. Copy the extracted folders back into the project using the paths below.

```text
data/raw/
data/external/
output/simulations/
output/brackets/
output/diagnostics/
```

If sample data is included in the downloaded package, it should be placed under:

```text
data/sample/
```

See `docs/DATA.md` for the external data handoff instructions.

Sample and demo outputs are included only for schema inspection, API testing, and lightweight frontend integration. They are not a replacement for the full local data package.

## Key Issues Encountered and Fixes

| Issue | Cause | Fix |
|---|---|---|
| GitHub push failed because large datasets exceeded file size limits. | Raw datasets and generated outputs were too large for normal GitHub tracking. | Moved full datasets to Google Drive and kept GitHub focused on code, configs, docs, and small demo outputs. |
| Draw prediction issue. | Draw probability can be meaningful, but an argmax pick rarely chooses draw. | Removed forced winner picks and exposed full W/D/L probabilities. |
| Scoreline concentration issue. | A Poisson single most-likely scoreline can over-concentrate around common results such as 1-1. | Separated scoreline display from W/D/L probabilities and introduced Negative Binomial top-scoreline display. |
| V2 vs tournament simulation mismatch. | The tournament simulator originally read team-level weighted Elo and did not automatically reflect player or roster updates. | Introduced an adjusted team input / rating bridge before rerunning tournament simulation. |
| Live evaluation before tournament start. | No final matches exist before kickoff. | Evaluation handles zero completed matches gracefully with `waiting_for_matches` status. |
| API usability issue. | Detail endpoints require `match_id` or exact team name. | Added list/index endpoints for matches and teams so frontend users do not manually type IDs. |
| Local cron limitation. | Local cron only runs when the computer is awake. | Recommend server-side cron for deployed API and live updates. |

## Repository Safety / Public Release Notes

GitHub tracks:

- source code
- configuration files
- documentation
- API code
- small demo outputs

GitHub does not track:

- full raw datasets
- local data package files
- heavy generated simulation outputs
- full diagnostics outputs
- local archive files

The public repository is configured to exclude:

- private keys
- `.env` files
- raw large datasets
- local deployment scripts with credentials
- log files
- IDE metadata

Do not commit large raw data or simulation outputs. Restore full data locally from the Google Drive package when reproducing the complete workflow.
