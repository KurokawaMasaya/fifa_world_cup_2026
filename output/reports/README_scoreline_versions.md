# Scoreline Version Notes

Production group-stage scoreline display has been promoted to `v24_abcd_no_e`.

- `v24_abcd_no_e` is the production scoreline display layer.
- `v20_stable` remains preserved as the previous stable production baseline.
- V2.1 and V2.2 scoreline calibration work should be treated as research and diagnostic output only.
- V2.2 is not deployed because it improved total-goals and over-2.5 calibration but degraded exact/top-N accuracy, two-plus goal win rate, and margin-bucket stability.
- W/D/L outcome probabilities remain frozen and are produced by the calibrated V2 probability model. The scoreline layer only affects the displayed scoreline columns.
- V2.4 ABCD no-E was promoted from shadow to production by user decision after live shadow evaluation. This promotion did not change W/D/L probabilities.
- The public clean prediction output now presents top-5 scoreline candidates instead of a single exact top-1 scoreline.

Current production files:

- `output/predictions/group_stage_predictions_clean.csv`
- `output/predictions/group_stage_predictions__scoreline_v24_abcd_no_e_production.csv`
- `output/predictions/group_stage_predictions_clean__metadata.json`
- `output/predictions/group_stage_predictions__v24_abcd_no_e_promotion_wdl_freeze_check.csv`
- `output/predictions/group_stage_predictions__top5_presentation_wdl_freeze_check.csv`

Preserved baseline:

- `output/predictions/group_stage_predictions__scoreline_v20_stable.csv`

Beijing-time game tables:

- `output/predictions/group_stage_games_2026_06_14_beijing_top5.csv`
- `output/reports/group_stage_games_2026_06_14_beijing_top5.md`
- `output/predictions/group_stage_games_today_beijing_top5.csv`
- `output/reports/group_stage_games_today_beijing_top5.md`

Automation:

- `src/live/export_daily_beijing_predictions.py` exports the table for a specific Beijing date or defaults to today in `Asia/Shanghai`.
- `src/live/update_live_pipeline.py` now runs this exporter after fetching ESPN live fixtures/results, so the server cron updates the daily Beijing-time table automatically.
