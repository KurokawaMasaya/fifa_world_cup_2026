# FIFAproject2026

## Project Overview

FIFAproject2026 is a forecasting system for the 2026 FIFA World Cup. It produces match-level probabilities, scoreline displays, tournament simulation probabilities, live result tracking, live evaluation outputs, and a read-only API for frontend or desktop gadget integration.

The project is designed to answer questions such as:

- What are the group-stage win/draw/loss probabilities for each fixture?
- What is the most likely displayed scoreline for a match?
- Which teams are most likely to advance through each tournament stage?
- How do live results compare with locked pre-match predictions?
- How can a frontend read predictions and live outputs without rerunning model code?

## Current Status

- V1 calibrated team-strength model: completed.
- V2 player-informed / roster-aware probability layer: completed and current.
- Read-only FastAPI backend: completed.
- ESPN live updater: prepared.
- Live evaluation: handles pre-tournament state with zero completed matches.
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
- displayed scoreline
- scoreline probability

The clean frontend/API prediction file is:

```text
output/predictions/group_stage_predictions_clean.csv
```

This file is designed for presentation and API use. It avoids internal diagnostics, model parameters, and debug fields.

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
2. Evaluate completed matches against locked predictions
3. Run live tournament simulation from the current tournament state
4. Write live outputs under output/live/
```

The live tournament simulation uses completed real scores from ESPN and existing model probabilities for remaining matches. It is a state-conditioned simulation, not a dynamic strength update model.

Example commands:

```bash
python3 src/live/update_live_pipeline.py
python3 src/live/update_live_pipeline.py --live-sim-n-sims 50000
python3 src/live/update_live_pipeline.py --skip-live-simulation
```

For local cron testing, the project uses:

```bash
python3 src/live/update_live_pipeline.py --live-sim-n-sims 10000
```

## API Backend

The API backend is a read-only FastAPI service. It reads existing CSV outputs and returns JSON for frontend or gadget use.

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

## Local API Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
PYTHONPATH=. python3 -m uvicorn src.api.main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Repository Policy

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

Do not commit large raw data or simulation outputs. Restore full data locally from the Google Drive package when reproducing the complete workflow.
