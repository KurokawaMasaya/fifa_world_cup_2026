# CupCast — 2026 FIFA World Cup Forecasting Engine

CupCast is a probabilistic forecasting system for the 2026 FIFA World Cup. The project models international team strength, converts team strength into match-level win/draw/loss probabilities, simulates the full tournament bracket, and evaluates prediction quality through historical backtesting.

The current version, V1, is a team-level forecasting model. It uses historical international match results, anchored team ratings, Poisson goal modeling, and Monte Carlo simulation to estimate each team’s probability of advancing through the tournament and winning the World Cup.

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

### V1: Team-Level Forecasting Model

V1 models each national team as a team-level entity. It does not yet explicitly include individual player quality, injuries, squad depth, or lineup availability.

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

text data/processed/matches_clean.csv data/processed/team_ratings_world_cup_elo.csv data/processed/tournament_simulation_results_default.csv data/processed/evaluation_summary_default.csv 

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

## 12. V2 Roadmap

V2 will extend the model with a squad-adjusted strength layer.

Planned structure:

text anchored team strength + squad quality adjustment + star power adjustment + squad depth adjustment + player availability adjustment = final V2 tournament strength 

Potential player/squad features include:

- squad market value
- top 5 player market value
- average player value
- squad age
- position balance
- number of players in top European leagues
- key-player injury status
- availability of star players
- squad depth concentration

The goal of V2 is not to replace the team-level model, but to adjust it using current squad information.

---

## 13. Repository Structure

text CupCast/   data/     raw/     processed/    src/     data/     ratings/     models/     tournament/     evaluation/     features/    config/     model_params_default.json     model_params_v1.json     model_params_experimental.json    docs/     model_versioning.md    README.md   requirements.txt 

---

## 14. How to Run

Install dependencies:

bash pip install -r requirements.txt 

Run data cleaning:

bash python src/data/clean_results.py 

Run rating generation:

bash python src/ratings/build_team_ratings.py 

Run match prediction:

bash python src/models/predict_match.py 

Run tournament simulation:

bash python src/tournament/monte_carlo.py --mode default 

Run model evaluation:

bash python src/evaluation/evaluate_match_predictions.py --mode default 

Run parameter tuning:

bash python src/evaluation/tune_parameters.py 

---

## 15. Model Versioning

Current default model:

text model_version: v1 model_status: default 

Earlier versions are preserved for comparison:

text v1: anchored team-level model with tuned Poisson/draw calibration experimental: future squad-adjusted V2 models 

---

## 16. Project Status

Current status:

text Data cleaning: complete Team-strength model: complete Poisson match model: complete Tournament simulator: complete Historical evaluation: complete Parameter tuning: complete Squad-adjusted V2: in progress Dashboard: planned 

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

The next major development step is V2, which will incorporate squad and player-level adjustments.
