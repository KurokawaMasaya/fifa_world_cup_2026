# CupCast — 2026 FIFA World Cup Forecasting Engine

CupCast is a probabilistic forecasting system for the 2026 FIFA World Cup. The project models international team strength, converts team strength into match-level win/draw/loss probabilities, simulates the full tournament bracket, and evaluates prediction quality through historical backtesting.

The current default version is V2: a player-informed probability adjustment model built on the V1 anchored team-strength baseline. V1 still provides calibrated team strength through anchored_final_strength, while V2 applies squad uncertainty, superstar, and club-form layers to the final win/draw/loss probabilities before prediction display and tournament simulation.

---

## 1. Project Motivation

Most sports prediction projects stop at single-match prediction. CupCast is designed as a full tournament forecasting engine.

The goal is not only to predict one match, but to answer questions such as:

- Which teams are most likely to win the 2026 World Cup?
- How does group-stage performance affect knockout-stage probability?
- How do team strength, expected goals, and bracket path interact?
- How reliable are the model’s predicted probabilities?
- Does the model outperform simple baselines?

This project combines applied probability, sports analytics, data cleaning, model calibration, and tournament simulation.

---

## 2. Current Version

### V1: Team-Level Forecasting Baseline

V1 models each national team as a team-level entity using anchored_final_strength. It remains available as the baseline layer and comparison model.

### V2: Player-Impact Probability Layer

V2 is the current default model. It does not replace anchored_final_strength or directly add player market value to team strength. Instead, it applies small probability post-processing layers using squad depth, superstar dependence, and recent club-form signals. This keeps the team-strength model stable while allowing squad changes, such as player replacements, to affect prediction and simulation probabilities.

The model pipeline is:

text Historical international match data         ↓ Team-name standardization and match cleaning         ↓ Anchored team strength rating         ↓ Expected-goals estimation         ↓ Poisson scoreline model         ↓ Match win/draw/loss probabilities         ↓ World Cup group-stage and knockout simulation         ↓ Monte Carlo tournament probabilities         ↓ Historical model evaluation 

---

## 3. Data Sources

The project uses structured international football data, including:

- Historical international match results
- Shootout results
- Former team names and standardized team-name mappings
- 2026 World Cup teams
- 2026 World Cup match schedule and bracket structure
- External team-strength anchors such as FIFA/Elo-style ratings

Core raw files include:

text data/raw/results.csv data/raw/shootouts.csv data/raw/former_names.csv data/raw/teams.csv data/raw/matches.csv 

Processed files include:

text data/processed/matches_clean.csv data/processed/team_ratings_world_cup_elo.csv data/processed/team_strength_v2_player_impacted.csv output/tournament_simulation_results_default.csv 

### Data Availability

Full raw datasets and large generated outputs are not tracked in this GitHub repository due to file size limits. They are stored separately on Google Drive. See [docs/DATA.md](docs/DATA.md) for download instructions and expected local paths.

---

## 4. Data Cleaning

The data-cleaning pipeline standardizes historical international match records before modeling.

Key steps include:

1. Team-name standardization

   Historical names are mapped to current names using former_names.csv.

   Example:

   text    Former name → Current standardized team name    

2. Match result cleaning

   The model extracts:

   - match date
   - home team
   - away team
   - home score
   - away score
   - tournament
   - neutral-site indicator
   - regulation result

3. Shootout handling

   Shootouts are handled separately from regulation results.

   If a match is tied after regulation:

   - the regulation result remains a draw
   - the shootout winner is stored separately
   - shootout wins are not treated as regulation wins for Elo-style team-strength updates

4. World Cup team filtering

   The model can filter matches involving 2026 World Cup teams while still preserving historical matches against non-qualified teams when useful for rating estimation.

---

## 5. Team Strength Model

V1 uses an anchored team-strength model.

The model combines:

text custom weighted team rating + external FIFA/Elo-style anchor + calibration adjustments = anchored final team strength 

This was necessary because a pure custom Elo model can overrate teams that perform strongly within regional competition pools and underrate historically elite teams facing stronger schedules.

The final team strength is used as the input to the match prediction model.

---

## 6. Match Prediction Model

CupCast uses a Poisson-based scoreline model.

The model first converts team-strength difference into expected goals:

text Team A strength Team B strength         ↓ strength difference         ↓ expected goals for Team A expected goals for Team B 

Then, for each possible scoreline, the model calculates:

text P(Team A scores x goals) × P(Team B scores y goals) 

This produces a full scoreline probability distribution.

From the scoreline distribution, the model aggregates:

text P(Team A win) P(draw) P(Team B win) 

The model also returns:

- expected goals for both teams
- top scoreline probabilities
- most likely scoreline
- win/draw/loss probabilities

---

## 7. Draw Calibration

Historical backtesting showed that the raw model slightly underpredicted draws.

To address this, V1 includes a lightweight draw calibration layer for close-strength matches. The adjustment increases draw probability more for evenly matched games and has little effect on mismatched games.

This improves draw calibration while preserving the relative win/loss probability between the two teams.

---

## 8. Tournament Simulator

The tournament simulator uses the match prediction model to simulate the full 2026 World Cup.

The simulator includes:

### Group Stage

Each group-stage match is simulated by sampling from the model’s scoreline probability distribution.

Group standings are updated using:

text Win = 3 points Draw = 1 point Loss = 0 points 

Teams are ranked by:

1. points
2. goal difference
3. goals scored
4. random tiebreaker for unresolved ties in the MVP version

### Qualification

For the 2026 format:

text 12 groups × 4 teams = 48 teams Top 2 from each group qualify = 24 teams Best 8 third-place teams qualify = 8 teams Total knockout teams = 32 teams 

### Knockout Stage

The knockout stage is resolved from matches.csv bracket placeholders.

Examples:

text 1A = winner of Group A 2B = runner-up of Group B 3CEFHI = one qualified third-place team from Groups C/E/F/H/I W73 = winner of Match 73 RU101 = runner-up/loser of Match 101 

The simulator does not randomly generate knockout pairings. Randomness is used only for:

- scoreline sampling
- tiebreakers
- knockout advancement after drawn scorelines

The bracket itself is deterministic and based on the schedule structure.

---

## 9. Monte Carlo Simulation

The tournament is simulated many times to estimate stage advancement probabilities.

For each team, the model estimates:

- probability of reaching the Round of 32
- probability of reaching the Round of 16
- probability of reaching the quarterfinals
- probability of reaching the semifinals
- probability of reaching the final
- probability of winning the World Cup

Example output:

text Team        Champion Probability Spain       21.32% Argentina   17.46% France      15.77% England      9.30% Portugal     4.61% Brazil       4.55% Germany      4.42% Morocco      3.79% Netherlands  3.33% Belgium      3.14% 

---

## 10. Model Evaluation

The model is evaluated through historical backtesting.

For each test match, the model generates pre-match probabilities using only information available before the match. The predicted probability distribution is compared against the actual result.

Evaluation metrics include:

- accuracy
- Brier score
- log loss
- mean probability assigned to the actual outcome
- goal MAE
- goal RMSE
- draw calibration error

### Tuned V1 Evaluation

The tuned model achieved:

text n_matches: 1123 accuracy: 0.6100 mean_brier_score: 0.5135 mean_log_loss: 0.8735 mean_actual_outcome_probability: 0.4893 goal_mae: 0.9942 goal_rmse: 1.2950 predicted_draw_rate: 0.2166 actual_draw_rate: 0.2378 draw_calibration_error: 0.0212 

### Baseline Comparison

The model was compared against simple baselines:

text random_uniform        accuracy 0.4837 | Brier 0.6667 | log loss 1.0986 historical_frequency  accuracy 0.4837 | Brier 0.6315 | log loss 1.0481 elo_only              accuracy 0.6058 | Brier 0.5322 | log loss 0.9055 final_model           accuracy 0.6058 | Brier 0.5221 | log loss 0.8889 

The final model matches the Elo-only model in hard prediction accuracy but improves probability quality, reducing both Brier score and log loss.

This means the final model does not simply predict more winners correctly; it produces better-calibrated probability distributions.

---

## 11. Current Limitations

V1 is a team-level model. It does not yet fully account for:

- individual player quality
- squad market value
- injuries
- suspensions
- player availability
- starting lineup uncertainty
- squad depth
- tactical matchups
- goalkeeper-specific effects
- player aging and national-team turnover

This is especially important for national-team football because the same country can have very different squad strength across tournament cycles.

Examples:

text France with key players available vs. France with injuries Belgium 2018 vs. Belgium 2026 Argentina with or without Messi Brazil with or without key attackers 

---

## 12. V2 Player-Informed Post-Processing

V2 is an experimental player-informed probability post-processing layer.

V2 does not directly change team strength. It does not replace `anchored_final_strength`, and it does not add player market value onto the rating. Instead, V1 first produces the base win/draw/loss probabilities, then V2 makes small post-processing adjustments based on player market-value-derived squad depth features.

V2 currently has two player-informed post-processing layers:

1. Squad uncertainty layer

   This uses top-five value concentration to estimate squad volatility and depth.

text top_5_value_eur / squad_market_value_eur = depth_concentration 

   Teams with high top-five concentration are treated as more star-dependent and potentially more volatile. Deeper squads are treated as more stable.

2. Superstar impact layer

   This captures teams with one exceptional match-winning player whose value is unusually high relative to their own squad. It uses the top player's value, top-player share, and top-player-to-squad median ratio to make a small probability adjustment after the uncertainty layer.

This design avoids double-counting team strength already captured by the V1 anchored rating. Player features only make small probability adjustments after the calibrated V1 prediction.

Model status:

- V1 remains the default model unless V2 uncertainty improves probability quality.
- V2 uncertainty is experimental unless it improves log loss or Brier score.
- V2 is intended to adjust confidence and volatility, not to make strong teams automatically stronger.

---

## 13. Repository Structure

text FIFAproject2026/   config/   data/     raw/     processed/   docs/   notebooks/   output/     live/     predictions/     simulations/     diagnostics/     brackets/     reports/   src/     app/     data/     evaluation/     live/     models/     ratings/     simulation/     tournament/   tests/   README.md   requirements.txt   summary.txt 

Generated Outputs:

- `output/live/`: locked predictions, actual results, and live evaluation files
- `output/predictions/`: deterministic match-level predictions
- `output/simulations/`: Monte Carlo tournament simulation outputs
- `output/diagnostics/`: model comparison, tuning, and data-quality reports
- `output/brackets/`: generated tournament bracket outputs
- `output/reports/`: markdown, HTML, PDF, and text reports

---

## 14. How to Run

Install dependencies:

bash pip install -r requirements.txt 

Run data cleaning:

bash python src/data/clean_results.py 

Run rating generation:

bash python src/ratings/build_team_ratings.py 

Run match prediction:

bash python src/tournament/predict_group_stage_results.py 

Run V2 uncertainty group-stage prediction:

bash python src/tournament/predict_group_stage_results_v2_uncertainty.py 

Run tournament simulation:

bash python src/tournament/monte_carlo.py --mode default 

Run model evaluation:

bash python src/evaluation/evaluate_match_predictions.py --mode default 

Run V2 uncertainty evaluation:

bash python src/evaluation/evaluate_v2_uncertainty.py 

Run parameter tuning:

bash python src/evaluation/tune_parameters.py 

---

## 15. Model Versioning

Current default model:

text model_version: v1 model_status: default 

Earlier versions are preserved for comparison:

text v1: anchored team-level model with tuned Poisson/draw calibration v2_uncertainty: experimental probability post-processing layer using squad depth volatility 

---

## 16. Project Status

Current status:

text Data cleaning: complete Team-strength model: complete Poisson match model: complete Tournament simulator: complete Historical evaluation: complete Parameter tuning: complete V2 uncertainty layer: experimental Dashboard: planned 

---

## 17. Summary

CupCast V1 demonstrates a complete probabilistic forecasting pipeline for the 2026 FIFA World Cup.

The system:

- cleans and standardizes historical international football data
- estimates team strength using anchored ratings
- converts team strength into expected goals
- generates scoreline and win/draw/loss probabilities
- simulates the full World Cup tournament
- evaluates predictive performance against baselines
- identifies calibration issues and improves probability quality through tuning

The next major development step is validating whether V2 uncertainty improves probability quality enough to promote beyond experimental status.


---

## Final Update — API and Current Default Model

Current default model:

```text
V1 anchored_final_strength
  ↓
calibrated Poisson win/draw/loss probabilities
  ↓
V2 player-impact probability stack
  ↓
squad uncertainty + superstar impact + club form
  ↓
frontend predictions and tournament simulation
```

The active default config is:

```text
config/model_params_default.json
model_version = v2_uncertainty_superstar_club_form
rating_col = anchored_final_strength
use_v2_probability_stack = true
player_impact_layers = squad_uncertainty, superstar, club_form
```

Read-only API:

```bash
uvicorn src.api.main:app --reload
```

Main frontend endpoints:

```text
GET /health
GET /predictions/group-stage
GET /predictions/matches
GET /predictions/teams
GET /simulation/tournament
GET /simulation/teams
GET /simulation/team/{team}
GET /live/results
GET /live/standings
GET /bracket/sample
```

A tutor handoff zip is available at:

```text
deliverables/cupcast_api_package.zip
```

Generated outputs are organized under:

```text
output/predictions/
output/simulations/
output/diagnostics/
output/brackets/
output/live/
output/reports/
```
