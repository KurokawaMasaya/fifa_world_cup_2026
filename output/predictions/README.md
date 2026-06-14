# Predictions Folder

Current files to use:

- `group_stage_predictions_clean.csv` - public/API prediction table.
- `group_stage_predictions_clean__metadata.json` - metadata for the public table.
- `group_stage_predictions_v2_uncertainty_tuned.csv` - internal detailed prediction file used by diagnostics/API health.
- `group_stage_predictions_v2_uncertainty.csv` - internal prediction output used by older/live lock scripts.

Daily Beijing tables:

- `daily_beijing/` contains dated Beijing-time game tables.
- `group_stage_games_today_beijing_top5.csv` is also kept at the top level for automation/API convenience.

Versioned scoreline files:

- `versioned_scoreline/` contains V2.4 production/shadow copies and freeze checks.

Archives:

- `archive/legacy_prediction_versions/` contains old V1/V2 prediction variants and V2.0 baseline output.
- `archive/evaluation/` contains prediction-level evaluation outputs.
