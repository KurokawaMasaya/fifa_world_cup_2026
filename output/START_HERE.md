# CupCast Output: Start Here

If you only need the current files, use these:

## Public Prediction/API Files

- `output/predictions/group_stage_predictions_clean.csv` - current public group-stage predictions. Uses V2.4 ABCD no-E production scoreline display and top-5 scoreline candidates.
- `output/predictions/group_stage_predictions_clean__metadata.json` - metadata for the current public prediction file.
- `output/predictions/group_stage_games_today_beijing_top5.csv` - today's Beijing-time match table with top-5 scorelines.
- `output/reports/group_stage_games_today_beijing_top5.md` - markdown version of today's Beijing-time match table.

## Live Files

- `output/live/fixtures_results.csv` - ESPN fixtures/results normalized CSV.
- `output/live/group_standings.csv` - ESPN standings if available.
- `output/live/worldcup_group_stage_live_summary.csv` - live evaluation summary.
- `output/live/live_tournament_simulation.csv` - state-conditioned live tournament simulation.

## API/Model Summary

- `output/diagnostics/model_performance_summary.csv` - model performance summary used by API.
- `output/diagnostics/team_strength_default.csv` - current team-strength diagnostics used by simulation outputs.

## Current Simulation/Bracket

- `output/simulations/tournament_simulation_results_default.csv` - main pre-tournament simulation output.
- `output/simulations/group_stage_monte_results_default.csv` - main group-stage Monte Carlo output.
- `output/brackets/sample_knockout_bracket_default.csv` - sample bracket output.

Everything else is grouped into archive or diagnostics subfolders for traceability.
