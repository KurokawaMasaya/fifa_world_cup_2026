# CupCast Model Versioning

CupCast keeps production, experimental, and test parameter sets separate so
forecast outputs can be compared without silently changing the default model.

## Modes

- `default`: production/non-test forecasts. Loads `config/model_params_default.json`.
- `experimental`: comparison forecasts. Loads `config/model_params_experimental.json`.
- `test`: validation/tuning reproduction. Loads `config/model_params_tuned_validation.json`.

If no mode is specified, scripts use `default`.

## Model Lineage

V1 was the initial team-level model. It used the first custom Elo strength
pipeline and the early Poisson expected-goals mapping.

V1.1 introduced anchored team strength and tuned Poisson/draw calibration:
anchored final strength blends the external FIFA/Elo anchor with the internal
model rating, then applies the configured confederation pool correction. The
Poisson model uses the `total_share` expected-goals mapping and a small tuned
draw boost for closer matchups.

`v2_tuned_default` is the current config `model_version` and is the default
non-test model. This is the promoted V1.1 tuned production setup described in
the project notes.

## Default Parameters

The default model is defined in `config/model_params_default.json`. Main entry
points load this file unless another mode is requested:

- single-match prediction: `python src/models/poisson_match_model.py`
- group-stage Monte Carlo: `python src/simulation/group_stage_simulator.py`
- tournament Monte Carlo: `python src/simulation/tournament_simulator.py`
- historical evaluation: `python src/evaluation/evaluate_match_predictions.py`

Production outputs use `_default.csv` filenames, for example
`data/processed/tournament_simulation_results_default.csv` and
`data/processed/evaluation_summary_default.csv`. Test and experimental modes
write `_test.csv` or `_experimental.csv` outputs.

Major output files include model metadata columns:

- `model_version`
- `model_status`
- `parameter_config_path`
- `rating_col`
- `bracket_source`
- `uses_random_pairing`
- `random_seed_used_for`

The knockout bracket is resolved from `data/raw/matches.csv`. Random seeds are
used only for scoreline sampling and drawn-match tiebreakers, not bracket
construction, R32 pairing, or team ordering.
