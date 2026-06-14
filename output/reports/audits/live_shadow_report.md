# Live Shadow Evaluation: V2.0 vs V2.4 ABCD no-E

This is an evaluation-only report. It reads frozen predictions and final live results.
It does not modify W/D/L probabilities, scoreline model parameters, production files, or simulation logic.

## Sample Size

- Live matches evaluated: 4

## Metric Definitions

- W/D/L direction evaluates the frozen W/D/L model: argmax(team_a_win_pct, draw_pct, team_b_win_pct).
- Scoreline direction evaluates the outcome implied by the displayed top-1 scoreline.
- Exact scoreline hit implies scoreline direction hit.
- W/D/L direction and scoreline direction may differ when the top-1 scoreline is draw but W/D/L argmax is a win, or vice versa.

## Overall Metrics

| Model | W/D/L direction | Exact top-1 | Top-5 hit | Scoreline direction |
|---|---:|---:|---:|---:|
| V2.0 stable | 0.750 | 0.250 | 0.250 | 0.250 |
| V2.4 ABCD no-E shadow | 0.750 | 0.250 | 0.750 | 0.250 |

## Delta vs V2.0

- Exact delta: 0.000
- Top-5 delta: 0.500
- Scoreline-direction delta: 0.000

## V2.4 Tail Calibration

- Predicted favorite 4+ mean: 0.132
- Actual favorite 4+ rate: 0.250
- Predicted favorite 5+ mean: 0.055
- Actual favorite 5+ rate: 0.000
- Predicted margin 4+ mean: 0.073
- Actual margin 4+ rate: 0.000
- Predicted total goals 5+ mean: 0.184
- Actual total goals 5+ rate: 0.250

## Decision

Sample is too small for promotion decisions; keep V2.4 in shadow.

Promotion requires a larger live sample and no major tail overshoot.

## Confirmation

- W/D/L probabilities were untouched.
- V2.4 ABCD no-E parameters were untouched.
- Production predictions were not overwritten.
