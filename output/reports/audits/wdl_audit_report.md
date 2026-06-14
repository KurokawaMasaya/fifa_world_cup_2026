# W/D/L Model Read-Only Audit

This audit reads existing prediction/evaluation outputs only. It does not modify model code, ratings, probabilities, scoreline logic, or production prediction files.

## Sources

- Group-stage W/D/L source: `/Users/joe/Desktop/McGill/projects/FIFAproject2026/output/predictions/group_stage_predictions_v2_uncertainty_tuned.csv`
- Historical calibration source: `/Users/joe/Desktop/McGill/projects/FIFAproject2026/output/diagnostics/v2_uncertainty_match_level_evaluation.csv`

## Probability Validity

- Valid rows: 72
- Invalid rows: 0

## Overall Distribution

- Mean draw probability: 0.213
- Mean favorite win probability: 0.621

## Historical Backtest

Backtest accuracy 0.607, Brier 0.520, log loss 0.884, actual outcome probability 0.483.

## Draw Behavior

Draw behavior by favorite/rating/lambda bucket is saved in `wdl_draw_behavior.csv`. Review monotonicity there; this report does not apply parameter changes.

## Scoreline Consistency

Scoreline consistency status counts: `{'partial_top_scorelines_only': 72}`

Full scoreline grids are not present in the current prediction CSV. The consistency file therefore marks rows as partial top-scoreline diagnostics rather than full-grid W/D/L consistency errors.

## Team-Level Bias

Largest team-level calibration errors are saved in `wdl_team_bias_audit.csv`.

            team  matches  calibration_error
    Saint Martin        2           0.339944
        Barbados        2           0.332559
Papua New Guinea        1           0.329838
           Aruba        4           0.290106
          Belize        1           0.286051
       Mauritius        2           0.280266
      Montserrat        3           0.276447
        Pakistan        4           0.274513
 Solomon Islands        2           0.255671
          Tahiti        3           0.254140

## Structural Issues Worth Investigating

- Draw is never the argmax pick in existing historical summaries, even though mean draw probability is meaningful. This may be acceptable for argmax classification but is important for presentation.
- Full scoreline-grid consistency cannot be audited from current prediction files because full grids are not saved.
- Extreme cases are listed in `wdl_extreme_cases.csv` for manual review.

## Confirmation

- No W/D/L model files were modified.
- No prediction probabilities were modified.
- No scoreline logic was modified.
- No production prediction files were overwritten.
