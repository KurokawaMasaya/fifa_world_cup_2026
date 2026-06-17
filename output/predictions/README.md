# Predictions Folder

Current production models:

- W/D/L model: frozen calibrated probability model.
- Scoreline model: V2.4 ABCD no-E.

Use these files first:

- `live/group_stage_round1_predictions__v24_abcd_no_e.csv` - live production output for current Round 1 publishing.
- `live/group_stage_round1_predictions__v24_abcd_no_e__metadata.json` - metadata for the live Round 1 output.
- `static_snapshots/group_stage_full_72_static_initial__v24_abcd_no_e.csv` - full 72-match static snapshot for research and rollback.
- `group_stage_predictions_clean.csv` - public/API clean table retained for compatibility.
- `group_stage_predictions_clean__metadata.json` - metadata for the public/API clean table.

Internal W/D/L/detail files:

- `group_stage_predictions_v2_uncertainty_tuned.csv` - detailed probability backbone used by diagnostics/API health.
- `group_stage_predictions_v2_uncertainty.csv` - older internal prediction output used by legacy/live-lock paths.

Daily Beijing tables:

- `daily_beijing/` contains dated Beijing-time game tables.
- `group_stage_games_today_beijing_top5.csv` is also kept at the top level for automation/API convenience.

Versioned scoreline files:

- `versioned_scoreline/group_stage_predictions__scoreline_v24_abcd_no_e_production.csv` - V2.4 ABCD no-E production copy.
- `versioned_scoreline/*freeze_check.csv` - W/D/L freeze checks.

Archives:

- `archive/legacy_prediction_versions/` contains old V1/V2 prediction variants and V2.0 rollback baseline output.
