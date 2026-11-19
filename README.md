# CupCast | 2026 FIFA World Cup forecasts

CupCast issued a pre-match forecast for each of the **104 matches** at the 2026 World Cup. It estimates the probabilities of a win, draw and loss in regulation time, ranks five candidate scorelines, and uses Monte Carlo simulation to estimate how far each team will advance.

Spain led the pre-tournament simulation with a **22.47%** chance of winning the title. By the final, with Argentina as the opponent, CupCast gave Spain **36.6%** to win in 90 minutes, a draw **29.6%**, and Argentina **33.8%**. The model’s selected match scenario took Spain through on penalties. The title probability includes uncertainty about qualification and future opponents. The final-match probabilities are conditional on the two finalists.

## Forecast archive

| Stage | Coverage | Original CSV |
|---|---|---|
| Pre-tournament simulation | 48 teams, 100,000 runs | [Group qualification and title probabilities](output/simulations/tournament_simulation_results_default.csv) |
| Group stage | 24 matches per round | [Round 1](output/predictions/live/group_stage_round1_predictions__v24_abcd_no_e.csv) · [Round 2](output/predictions/live/group_stage_round2_predictions_clean.csv) · [Round 3](output/predictions/live/group_stage_round3_predictions_clean.csv) |
| Round of 32 | 16 matches | [R32, including extra time and penalties](output/predictions/round_of_32_predictions_after_group_stage_clean_with_et_penalties.csv) |
| Round of 16 | 8 matches | [R16](output/predictions/round_of_16_predictions_after_r32_clean.csv) |
| Quarterfinals | 4 matches | [Quarterfinals](output/predictions/R8_predictions_clean.csv) |
| Semifinals | 2 matches | [Semifinals](output/predictions/semifinal_predictions_clean.csv) |
| Third-place match and final | 2 matches in one table | [Third place and final](output/predictions/third_place_and_final_predictions_clean.csv) |

The simulation stores probabilities as fractions from 0 to 1. The match tables use percentages in columns ending in `_pct`. In the knockout tables, `final_winner` and `match_resolution` give the model’s selected winner and how the match was predicted to end.

Earlier public versions can be inspected alongside the archive. The June 12 release included a 72-match group-stage baseline. The July 4 snapshot includes the full R16 table, with Morocco at 63.3% to beat Canada in regulation in the round’s opening fixture. Each forecast was issued before its own kickoff. A later CSV export or repository cleanup has its own timestamp. [June 12 release](https://github.com/KurokawaMasaya/fifa_world_cup_2026/releases/tag/v1) · [July 4 R16 snapshot](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c21c729f54eb5130509dffed8cbf0ec5c2a3254e/output/predictions/round_of_16_predictions_after_r32_clean.csv)

## How the model works

**Team strength and expected goals (V1).** Historical international results feed a weighted Elo rating. The update accounts for match importance, home advantage and the margin of victory, with a cap on the influence of large wins against weaker opponents. The final strength estimate blends that rating with an external anchor. The Poisson match model first estimates total goals, then allocates each team a share based on the strength gap. Summing the score probabilities gives regulation-time win, draw and loss probabilities. [Poisson match model](src/models/poisson_match_model.py)

**Squad adjustments (V2).** CupCast makes small adjustments to those probabilities using squad depth, concentration of player market value, dependence on star players and club form. Applying these adjustments after the team model limits double-counting of player quality already reflected in national-team results. [V2 probability model](src/models/v2_probability_stack.py) · [Default configuration](config/model_params_default.json)

**Scoreline display (V2.4).** A historical check found that the single most likely score contained three or more goals in only 1.5% of matches. Across the full distribution, the model assigned 56.2% probability on average to three or more goals, and 49.0% of matches actually reached that total. This gap between a single score and the range of plausible outcomes led CupCast to display five candidates. The production model, `v24_abcd_no_e`, mixes a baseline negative binomial score distribution with a second distribution for a possible large win. The mixture weight uses the favorite’s win probability, the expected-goal imbalance, the favorite’s scoring capacity and the underdog’s expected goals. A proposed motivation factor was omitted because reliable inputs were unavailable. The June 14 release check compared all 72 group fixtures and found identical win/draw/loss probabilities before and after the scoreline change. [Scoreline model](src/models/favorite_blowout_mixture_scoreline.py) · [V2.4 release](https://github.com/KurokawaMasaya/fifa_world_cup_2026/releases/tag/v2.4.0)

**Tournament and round updates.** The pre-tournament simulator advances the top two teams in each of the 12 groups and the eight best third-place teams. Its saved run used a fixed approximation of the knockout bracket. Once the actual R32 field was known, the match forecasts used the confirmed pairings. Between rounds, completed matches supplied updated team ratings. For example, all 16 R32 results went into a capped rating update before the R16 forecasts. The live simulation could also hold completed results fixed and resample the remaining tournament with the supplied ratings. [Tournament simulator](src/simulation/tournament_simulator.py) · [R32 bracket handling](src/simulation/r32_bracket_source.py)

## Historical evaluation

The V2 win/draw/loss backtest covered **1,989 historical international matches**. Choosing the highest-probability outcome gave **60.68% accuracy**, with a multiclass **Brier score of 0.5202** and **log loss of 0.8841**. These metrics describe the historical evaluation sample. [Evaluation configuration](config/model_params_default.json)

A separate scoreline backtest covered **5,414 matches**. An exact-score hit means the observed score appeared among the displayed candidates. Moving from V2.0 to V2.4 slightly increased top-five coverage while top-three coverage edged down.

| Historical exact-score hit rate | V2.0 | V2.4 |
|---|---:|---:|
| First candidate | 12.65% | 12.87% |
| First three candidates | 34.54% | 34.48% |
| First five candidates | 51.57% | 51.85% |

The [research model card](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c21c729f54eb5130509dffed8cbf0ec5c2a3254e/output/reports/scoreline_research/v24_abcd_no_e__model_card.md) records the historical comparison before V2.4 entered production.

**Reading the original Round 2 and Round 3 files.** During the group stage, a display update substituted some fourth- and fifth-ranked scores to include plausible higher-scoring outcomes. In 20 positions in Round 2 and 14 in Round 3, the percentage beside the new score still belongs to the previous candidate. Those 34 percentages should be excluded from per-score probability analysis. The win/draw/loss columns were unaffected by this display update.

## Run the local API

The easiest way to inspect a forecast is to open a CSV above. For JSON output, start the read-only FastAPI service from the repository root.

```bash
python -m pip install -r requirements.txt
python -m uvicorn src.api.main:app
```

Open `http://127.0.0.1:8000/docs`. The current checkout serves the three group-stage rounds at `/predictions/live/round1`, `/predictions/live/round2` and `/predictions/live/round3`. The 48-team simulation is available at `/simulation/tournament`. Some historical live and evaluation routes require additional runtime CSVs, so the health check reports `degraded` with the public archive alone.

During the tournament, the pipeline fetched completed results from ESPN for evaluation and simulation updates. A hosted API was verified in June 2026. The local commands above work with the published forecast files. The separate June 11 data package contains raw, sample and processed data for the modeling workflow. [Input data package](https://drive.google.com/file/d/1bJnvtBXX1PASxo_KnYHa7Js0-cYG1v1I/view?usp=drive_link)
