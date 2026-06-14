# Model card — V2.4 ABCD no-E gated blowout mixture (`v24_abcd_no_e`)

**Production status:** shadow / research only (NOT promoted to production).
**W/D/L frozen:** CONFIRMED (freeze check PASS).

## Purpose
Improve the displayed-scoreline *tail* for structurally mismatched games (a
strong favorite vs a weak underdog) without globally inflating scorelines and
without touching calibrated W/D/L probabilities. The stable V2.0 Negative
Binomial grid systematically under-represents plausible blowout/collapse tail
events (4-0, 5-0, 5-1) for clear mismatches; this layer adds a small, gated,
pre-match-structural blowout component.

## Formula
```
P(scoreline) = (1 - p_blowout_final) * P_normal(scoreline)
             +      p_blowout_final  * P_blowout(scoreline)

p_blowout_final = clamp(p_blowout_raw * blowout_gate * blowout_k, 0, bucket_cap)
blowout_gate    = A * B * C * D            # E (motivation) disabled => neutral
```
- `P_normal` = stable V2.0 Negative Binomial scoreline grid (unchanged).
- `P_blowout` = NB grid with the favorite's lambda thickened and the underdog's
  suppressed; affects only the favorite's right tail.
- `bucket_cap` caps blowout mass per favorite bucket so blowouts can never be
  force-promoted to the top display.

## Gates (all pre-match structural features only)
- **A — favorite_dominance_gate:** rises with the favorite's W/D/L win
  probability (soft ramp above ~0.58). Near zero for even matchups.
- **B — lambda_imbalance_gate:** rises with |lambda_fav - lambda_dog| / total.
  Near zero when expected goals are balanced.
- **C — favorite_scoring_capacity_gate:** rises with the favorite's own base
  lambda (it must actually be able to score a lot).
- **D — underdog_suppression_gate:** rises as the underdog's base lambda falls
  (a real collapse needs the underdog kept quiet).
- **E — motivation_gate:** **DISABLED / neutral.** Real pre-match motivation
  data (dead rubbers, rotation, must-win) is not currently available, so an
  estimated motivation gate would add noise, not signal. It is held at 1.0 and
  excluded from the product.

## Why V2.4 improves over V2.3
V2.3 applied a bucket-capped blowout mixture but its blowout probability was
driven mainly by favorite win-prob and imbalance, so it could still raise tail
mass in games where the favorite's own scoring capacity was modest or the
underdog was not actually suppressed. V2.4 multiplies in the **A·B·C·D** gate,
which zeroes out blowout mass unless *all four* structural conditions hold
together. This concentrates the tail adjustment on genuine mismatches and keeps
balanced/slight-favorite games essentially identical to V2.0.

## Backtest summary (2014+ historical, V2.0 stable -> V2.4 ABCD no-E)
- exact_top1: 0.1265 -> 0.1287
- top3: 0.3454 -> 0.3448
- top5: 0.5157 -> 0.5185
- winner_direction: 0.4985 -> 0.4985
- displayed_draw_rate: 0.4693 -> 0.4693
- mean_goals_pred: 1.7505 -> 1.7658 (actual 2.6928)
- favorite_scores_4_plus (pred): 0.1519 -> 0.1638 (actual 0.1365)
- favorite_scores_5_plus (pred): 0.0679 -> 0.0797 (actual 0.0707)
- margin_4_plus (pred): 0.0929 -> 0.1049 (actual 0.1051)
- total_goals_5_plus (pred): 0.1910 -> 0.2021 (actual 0.1500)
- combined_tail_error: 0.0187 -> 0.0204

## Known limitation
The top-1 *displayed* scoreline remains naturally conservative: the exact-score
mode of a low-to-moderate lambda match is a low-scoring line (1-0 / 1-1 / 2-1),
so the headline displayed scoreline rarely becomes a blowout even when blowout
*tail probability* rises correctly. V2.4 is therefore best read through its tail
probabilities and full distribution, not the single top-1 cell. This is an
inherent property of single-cell display, not a model defect.

## W/D/L freeze confirmation
The shadow output copies `team_a_win_pct`, `draw_pct`, `team_b_win_pct` verbatim
from the production clean file. The freeze check
(`v24_abcd_no_e__wdl_freeze_check.csv`) reports
PASS with 0 rows changed.

## Production status & next step
Shadow only. **Recommended next step: live shadow evaluation** — log V2.4 ABCD
no-E predictions alongside production V2.0 as real 2026 results arrive and score
the tail calibration on live data. Do **not** make further architecture changes
(no Markov, no market odds, no V2.6 representative selector, no agent context)
until live shadow evidence justifies it.

## Parameters
```
GatedBlowoutParams(normal_k=12.0, blowout_k=6.0, blowout_lambda_multiplier=1.9, blowout_lambda_add=0.9, max_blowout_lambda_fav=6.2, underdog_blowout_lambda_multiplier=0.8, min_underdog_blowout_lambda=0.15, p_favorite_weight=0.28, p_imbalance_weight=0.1, p_rating_weight=0.04, p_multiplier=1.15, blowout_k_factor=1.0, use_favorite_dominance_gate=True, use_lambda_imbalance_gate=True, use_favorite_scoring_capacity_gate=True, use_underdog_suppression_gate=True, use_motivation_gate=False, motivation_factor=1.0, favorite_win_prob_threshold=0.58, favorite_lambda_threshold=1.65, lambda_imbalance_threshold=0.25, favorite_scoring_capacity_power=1.0)
```
