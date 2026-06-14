from __future__ import annotations

"""Finalize the V2.4 ABCD no-E gated blowout mixture as a clean shadow profile.

This script is a RESEARCH / SHADOW finalizer. It does NOT touch production:
  * It NEVER overwrites output/predictions/group_stage_predictions_clean.csv.
  * It NEVER recomputes or modifies W/D/L probabilities. The displayed W/D/L
    percentages in the shadow output are copied verbatim from the frozen
    production clean file; only the scoreline layer changes.

Model (display/research only):
    P(scoreline) = (1 - p_blowout_final) * P_normal(scoreline)
                 +      p_blowout_final  * P_blowout(scoreline)

    p_blowout_final = clamp(p_blowout_raw * blowout_gate * blowout_k, 0, bucket_cap)
    blowout_gate    = A * B * C * D            (E / motivation disabled => neutral)

      A = favorite_dominance_gate
      B = lambda_imbalance_gate
      C = favorite_scoring_capacity_gate
      D = underdog_suppression_gate
      E = motivation_gate            (disabled: no real motivation data available)

P_normal is the stable V2.0 Negative Binomial grid. P_blowout thickens only the
favorite's right tail. Everything here is gated on PRE-MATCH structure
(favorite win prob, lambda imbalance, scoring capacity); no live/post-match data.

Outputs:
  1. output/predictions/versioned_scoreline/group_stage_predictions__scoreline_v24_abcd_no_e_shadow.csv
  2. output/predictions/versioned_scoreline/group_stage_predictions__scoreline_v24_abcd_no_e_shadow__metadata.json
  3. output/diagnostics/scoreline_research/v24_abcd_no_e_final/
       - v24_abcd_no_e__comparison_vs_v20_summary.csv
       - v24_abcd_no_e__comparison_vs_v20_by_bucket.csv
       - v24_abcd_no_e__tail_calibration.csv
       - v24_abcd_no_e__gate_diagnostics.csv
       - v24_abcd_no_e__example_matches.csv
       - v24_abcd_no_e__wdl_freeze_check.csv
       - v24_abcd_no_e__model_card.md
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.model_config import load_model_config  # noqa: E402
from src.diagnostics.backtest_favorite_blowout_mixture import (  # noqa: E402
    DEFAULT_INPUT,
    build_predictions,
    load_matches,
)
from src.diagnostics.backtest_v24_gated_blowout_mixture import (  # noqa: E402
    base_params,
    summarize_grid_for_match,
)
from src.models.favorite_blowout_mixture_scoreline import (  # noqa: E402
    BlowoutMixtureParams,
    GatedBlowoutParams,
    gated_favorite_blowout_mixture_scoreline_grid,
    tail_metrics,
    top_scorelines,
)
from src.models.negative_binomial_scoreline import (  # noqa: E402
    get_top_scorelines,
    negative_binomial_scoreline_grid,
)


PROFILE_NAME = "v24_abcd_no_e"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_PATH = PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_clean.csv"
TUNED_PATH = (
    PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_v2_uncertainty_tuned.csv"
)
V20_STABLE_PATH = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "archive"
    / "legacy_prediction_versions"
    / "group_stage_predictions__scoreline_v20_stable.csv"
)
SHADOW_PATH = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "versioned_scoreline"
    / "group_stage_predictions__scoreline_v24_abcd_no_e_shadow.csv"
)
SHADOW_METADATA_PATH = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "versioned_scoreline"
    / "group_stage_predictions__scoreline_v24_abcd_no_e_shadow__metadata.json"
)
RESEARCH_DIR = (
    PROJECT_ROOT / "output" / "diagnostics" / "scoreline_research" / "v24_abcd_no_e_final"
)

NORMAL_K = 12.0
MAX_GOALS = 10
WDL_PCT_COLUMNS = ["team_a_win_pct", "draw_pct", "team_b_win_pct"]
FREEZE_TOLERANCE = 1e-9

# Shared blowout shaping (V2.4 research-tuned values) used for the mixture's
# blowout component; matches the historical V2.4 backtest defaults.
MIXTURE_PARAMS = BlowoutMixtureParams(
    normal_k=12.0,
    blowout_k=6.0,
    blowout_lambda_multiplier=1.90,
    blowout_lambda_add=0.90,
    max_blowout_lambda_fav=6.2,
    underdog_blowout_lambda_multiplier=0.80,
    p_favorite_weight=0.28,
    p_imbalance_weight=0.10,
    p_rating_weight=0.04,
    p_multiplier=1.15,
)


def v24_abcd_no_e_params() -> GatedBlowoutParams:
    """Canonical V2.4 ABCD no-E parameters (E/motivation gate disabled)."""
    return base_params(use_motivation_gate=False, motivation_factor=1.0)


def parse_scoreline(scoreline: str) -> tuple[int, int]:
    goals_a, goals_b = scoreline.split("-")
    return int(goals_a), int(goals_b)


# ---------------------------------------------------------------------------
# Part 1/2/4: shadow group-stage output + diagnostics
# ---------------------------------------------------------------------------
def generate_shadow_predictions(params: GatedBlowoutParams) -> pd.DataFrame:
    """Build the V2.4 ABCD no-E shadow predictions for the group stage.

    W/D/L percentages are copied verbatim from the frozen production clean file;
    only the scoreline layer is recomputed. Raw model inputs (lambdas, raw V2
    win probabilities) come from the tuned prediction file.
    """
    clean = pd.read_csv(CLEAN_PATH)
    tuned = pd.read_csv(TUNED_PATH)
    raw_mode_lookup = {}
    if V20_STABLE_PATH.exists():
        v20 = pd.read_csv(V20_STABLE_PATH)
        raw_mode_lookup = dict(zip(v20["match_id"], v20["predicted_scoreline"]))

    merged = clean.merge(
        tuned[
            [
                "match_id",
                "lambda_a",
                "lambda_b",
                "v2_p_team_a_win",
                "v2_p_draw",
                "v2_p_team_b_win",
                "strength_diff",
            ]
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    rows = []
    for record in merged.itertuples(index=False):
        lambda_a = float(record.lambda_a)
        lambda_b = float(record.lambda_b)
        grid, meta = gated_favorite_blowout_mixture_scoreline_grid(
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            p_team_a_win=float(record.v2_p_team_a_win),
            p_draw=float(record.v2_p_draw),
            p_team_b_win=float(record.v2_p_team_b_win),
            rating_gap=float(record.strength_diff),
            max_goals=MAX_GOALS,
            params=params,
        )
        top5 = top_scorelines(grid, top_n=5)
        selected = str(top5[0]["scoreline"])
        favorite_side = str(meta["favorite_side"])
        tail = tail_metrics(grid, favorite_side=favorite_side)
        favorite_team = record.team_a if favorite_side == "team_a" else record.team_b

        # Raw mode = stable V2.0 NB display mode (authoritative production v20).
        raw_mode = raw_mode_lookup.get(record.match_id)
        if raw_mode is None:
            v20_grid = negative_binomial_scoreline_grid(
                lambda_a=lambda_a, lambda_b=lambda_b, dispersion_k=NORMAL_K, max_goals=MAX_GOALS
            )
            raw_mode = get_top_scorelines(v20_grid, top_n=1, mode="mode")[0]["scoreline"]

        rows.append(
            {
                # --- public schema (W/D/L FROZEN, copied from production) ---
                "match_id": int(record.match_id),
                "group": record.group,
                "team_a": record.team_a,
                "team_b": record.team_b,
                "team_a_win_pct": int(record.team_a_win_pct),
                "draw_pct": int(record.draw_pct),
                "team_b_win_pct": int(record.team_b_win_pct),
                "predicted_scoreline": selected,
                "scoreline_probability_pct": int(round(float(top5[0]["probability"]) * 100)),
                # --- scoreline diagnostics (display/research only) ---
                "scoreline_version": PROFILE_NAME,
                "raw_mode_scoreline": str(raw_mode),
                "selected_scoreline": selected,
                "top_5_scorelines": json.dumps([str(item["scoreline"]) for item in top5]),
                "top_5_scoreline_probs": json.dumps([float(item["probability"]) for item in top5]),
                "favorite_team": favorite_team,
                "favorite_win_prob": float(meta["favorite_win_prob"]),
                "favorite_bucket": str(meta["favorite_bucket"]),
                "p_blowout_raw": float(meta["p_blowout_raw"]),
                "blowout_gate": float(meta["blowout_gate"]),
                "blowout_k": float(meta["blowout_k"]),
                "p_blowout_final": float(meta["p_blowout_final"]),
                "bucket_cap": float(meta["bucket_cap"]),
                "A_favorite_dominance_gate": float(meta["favorite_dominance_gate"]),
                "B_lambda_imbalance_gate": float(meta["lambda_imbalance_gate"]),
                "C_favorite_scoring_capacity_gate": float(meta["favorite_scoring_capacity_gate"]),
                "D_underdog_suppression_gate": float(meta["underdog_suppression_gate"]),
                "E_motivation_gate_disabled": True,
                "p_favorite_scores_4_plus": float(tail["p_favorite_scores_4_plus"]),
                "p_favorite_scores_5_plus": float(tail["p_favorite_scores_5_plus"]),
                "p_margin_4_plus": float(tail["p_margin_4_plus"]),
                "p_total_goals_5_plus": float(tail["p_total_goals_5_plus"]),
                "tail_risk_index": float(tail["tail_risk_index"]),
            }
        )
    return pd.DataFrame(rows)


def write_metadata(shadow_df: pd.DataFrame, source_script: str) -> dict:
    metadata = {
        "scoreline_version": PROFILE_NAME,
        "production_status": "shadow",
        "WDL_frozen": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_script": source_script,
        "rows": int(len(shadow_df)),
        "wdl_source": "output/predictions/group_stage_predictions_clean.csv (copied verbatim)",
        "scoreline_model": "src/models/favorite_blowout_mixture_scoreline.py "
        "gated_favorite_blowout_mixture_scoreline_grid",
        "normal_state": "V2.0 stable Negative Binomial scoreline grid",
        "blowout_gate_components": ["A_favorite_dominance", "B_lambda_imbalance",
                                    "C_favorite_scoring_capacity", "D_underdog_suppression"],
        "motivation_gate_E": "disabled (no real pre-match motivation data available)",
        "shadow_output": str(SHADOW_PATH.relative_to(PROJECT_ROOT)),
        "production_output_not_modified": str(CLEAN_PATH.relative_to(PROJECT_ROOT)),
        "notes": (
            "V2.4 ABCD no-E is a gated blowout-mixture scoreline layer built on the "
            "stable V2.0 Negative Binomial grid. P(scoreline) = (1 - p_blowout_final) * "
            "P_normal + p_blowout_final * P_blowout, with p_blowout_final = clamp("
            "p_blowout_raw * (A*B*C*D) * blowout_k, 0, bucket_cap). It only thickens the "
            "favorite's right tail for structurally mismatched, pre-match-favored games. "
            "It does not modify calibrated W/D/L probabilities and is shadow/research only."
        ),
    }
    SHADOW_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


# ---------------------------------------------------------------------------
# Part 7: W/D/L freeze check
# ---------------------------------------------------------------------------
def wdl_freeze_check(shadow_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    clean = pd.read_csv(CLEAN_PATH)
    merged = clean.merge(shadow_df, on="match_id", suffixes=("_prod", "_shadow"))
    diffs = {
        col: float((merged[f"{col}_prod"] - merged[f"{col}_shadow"]).abs().max())
        for col in WDL_PCT_COLUMNS
    }
    changed_mask = (
        ((merged["team_a_win_pct_prod"] - merged["team_a_win_pct_shadow"]).abs() > FREEZE_TOLERANCE)
        | ((merged["draw_pct_prod"] - merged["draw_pct_shadow"]).abs() > FREEZE_TOLERANCE)
        | ((merged["team_b_win_pct_prod"] - merged["team_b_win_pct_shadow"]).abs() > FREEZE_TOLERANCE)
    )
    rows_changed = int(changed_mask.sum())
    passed = (
        all(v <= FREEZE_TOLERANCE for v in diffs.values()) and rows_changed == 0
    )
    report = pd.DataFrame(
        [
            {
                "scoreline_version": PROFILE_NAME,
                "rows_compared": int(len(merged)),
                "max_abs_diff_team_a_win_pct": diffs["team_a_win_pct"],
                "max_abs_diff_draw_pct": diffs["draw_pct"],
                "max_abs_diff_team_b_win_pct": diffs["team_b_win_pct"],
                "rows_wdl_changed": rows_changed,
                "status": "PASS" if passed else "FAIL",
            }
        ]
    )
    return report, passed


# ---------------------------------------------------------------------------
# Part 5/6: historical comparison V2.0 stable vs V2.4 ABCD no-E
# ---------------------------------------------------------------------------
def _v20_match_frame(base: pd.DataFrame) -> pd.DataFrame:
    """Normalize V2.0 baseline columns to the metric frame schema."""
    return pd.DataFrame(
        {
            "favorite_bucket": base["favorite_bucket"],
            "exact_top1_hit": base["v20_exact_scoreline_hit"],
            "top3_hit": base["v20_top3_scoreline_hit"],
            "top5_hit": base["v20_top5_scoreline_hit"],
            "winner_direction_correct": base["v20_winner_direction_correct"],
            "predicted_scoreline": base["normal_top_scoreline"],
            "mean_goals_pred": base["v20_predicted_total_goals"],
            "mean_goals_actual": base["actual_total_goals"],
            "favorite_scores_4_plus_pred": base["v20_p_favorite_scores_4_plus"],
            "favorite_scores_4_plus_actual": base["v20_actual_favorite_scores_4_plus"],
            "favorite_scores_5_plus_pred": base["v20_p_favorite_scores_5_plus"],
            "favorite_scores_5_plus_actual": base["v20_actual_favorite_scores_5_plus"],
            "margin_4_plus_pred": base["v20_p_margin_4_plus"],
            "margin_4_plus_actual": base["v20_actual_margin_4_plus"],
            "total_goals_5_plus_pred": base["v20_p_total_goals_5_plus"],
            "total_goals_5_plus_actual": base["actual_total_goals"] >= 5,
            "over_2_5_pred": base["v20_predicted_over_2_5"],
            "over_2_5_actual": base["actual_over_2_5"],
        }
    )


def _metric_row(df: pd.DataFrame, profile: str) -> dict[str, object]:
    """Compute the comparison metric set from a normalized per-match frame."""
    draws = df["predicted_scoreline"].map(lambda s: parse_scoreline(s)[0] == parse_scoreline(s)[1])
    fav4_err = abs(df["favorite_scores_4_plus_pred"].mean() - df["favorite_scores_4_plus_actual"].mean())
    fav5_err = abs(df["favorite_scores_5_plus_pred"].mean() - df["favorite_scores_5_plus_actual"].mean())
    margin4_err = abs(df["margin_4_plus_pred"].mean() - df["margin_4_plus_actual"].mean())
    total5_err = abs(df["total_goals_5_plus_pred"].mean() - df["total_goals_5_plus_actual"].mean())
    over25_err = abs(df["over_2_5_pred"].mean() - df["over_2_5_actual"].mean())
    return {
        "profile": profile,
        "sample_size": int(len(df)),
        "exact_top1": float(df["exact_top1_hit"].mean()),
        "top3": float(df["top3_hit"].mean()),
        "top5": float(df["top5_hit"].mean()),
        "winner_direction": float(df["winner_direction_correct"].mean()),
        "mean_goals_pred": float(df["mean_goals_pred"].mean()),
        "mean_goals_actual": float(df["mean_goals_actual"].mean()),
        "displayed_draw_rate": float(draws.mean()),
        "low_score_rate": float((df["mean_goals_pred"] <= 1).mean()),
        "favorite_scores_4_plus_pred": float(df["favorite_scores_4_plus_pred"].mean()),
        "favorite_scores_4_plus_actual": float(df["favorite_scores_4_plus_actual"].mean()),
        "favorite_scores_5_plus_pred": float(df["favorite_scores_5_plus_pred"].mean()),
        "favorite_scores_5_plus_actual": float(df["favorite_scores_5_plus_actual"].mean()),
        "margin_4_plus_pred": float(df["margin_4_plus_pred"].mean()),
        "margin_4_plus_actual": float(df["margin_4_plus_actual"].mean()),
        "total_goals_5_plus_pred": float(df["total_goals_5_plus_pred"].mean()),
        "total_goals_5_plus_actual": float(df["total_goals_5_plus_actual"].mean()),
        "over_2_5_pred": float(df["over_2_5_pred"].mean()),
        "over_2_5_actual": float(df["over_2_5_actual"].mean()),
        "favorite_scores_4_plus_abs_error": fav4_err,
        "favorite_scores_5_plus_abs_error": fav5_err,
        "margin_4_plus_abs_error": margin4_err,
        "total_goals_5_plus_abs_error": total5_err,
        "over_2_5_abs_error": over25_err,
        "tail_error": (fav4_err + fav5_err + margin4_err + total5_err) / 4.0,
        "combined_tail_error": (fav5_err + margin4_err + total5_err) / 3.0,
    }


def run_historical_comparison(start_date: str, input_path: Path) -> dict[str, pd.DataFrame]:
    matches = load_matches(input_path, start_date=start_date)
    model_config = load_model_config("default")
    base = build_predictions(
        matches=matches,
        model_config=model_config,
        max_goals=MAX_GOALS,
        dispersion_k=12.0,
        mixture_params=MIXTURE_PARAMS,
    )

    params = v24_abcd_no_e_params()
    v24_rows = [summarize_grid_for_match(row, PROFILE_NAME, params) for _, row in base.iterrows()]
    v24_ml = pd.DataFrame(v24_rows)

    # build_predictions already carries the (pre-match) favorite bucket, and the
    # V2.4 grid uses the same bucket function, so they match. Use it directly.
    v20_frame = _v20_match_frame(base)

    # ----- summary (overall) -----
    v20_summary = _metric_row(v20_frame, "v20_stable")
    v24_summary = _metric_row(v24_ml, PROFILE_NAME)
    delta = {"profile": "delta_v24_minus_v20", "sample_size": v24_summary["sample_size"]}
    for key, value in v24_summary.items():
        if key in ("profile", "sample_size"):
            continue
        delta[key] = value - v20_summary[key]
    summary = pd.DataFrame([v20_summary, v24_summary, delta])

    # ----- by bucket -----
    by_bucket_rows = []
    for bucket in sorted(v24_ml["favorite_bucket"].unique()):
        v20_b = _metric_row(v20_frame[v20_frame["favorite_bucket"] == bucket], "v20_stable")
        v24_b = _metric_row(v24_ml[v24_ml["favorite_bucket"] == bucket], PROFILE_NAME)
        v20_b["favorite_bucket"] = bucket
        v24_b["favorite_bucket"] = bucket
        by_bucket_rows.extend([v20_b, v24_b])
    by_bucket = pd.DataFrame(by_bucket_rows)

    # ----- tail calibration (per bucket, both profiles) -----
    tail_rows = []
    for bucket in sorted(v24_ml["favorite_bucket"].unique()):
        for profile, frame in (("v20_stable", v20_frame), (PROFILE_NAME, v24_ml)):
            sub = frame[frame["favorite_bucket"] == bucket]
            tail_rows.append(
                {
                    "profile": profile,
                    "favorite_bucket": bucket,
                    "sample_size": int(len(sub)),
                    "favorite_scores_4_plus_pred": float(sub["favorite_scores_4_plus_pred"].mean()),
                    "favorite_scores_4_plus_actual": float(sub["favorite_scores_4_plus_actual"].mean()),
                    "favorite_scores_5_plus_pred": float(sub["favorite_scores_5_plus_pred"].mean()),
                    "favorite_scores_5_plus_actual": float(sub["favorite_scores_5_plus_actual"].mean()),
                    "margin_4_plus_pred": float(sub["margin_4_plus_pred"].mean()),
                    "margin_4_plus_actual": float(sub["margin_4_plus_actual"].mean()),
                    "total_goals_5_plus_pred": float(sub["total_goals_5_plus_pred"].mean()),
                    "total_goals_5_plus_actual": float(sub["total_goals_5_plus_actual"].mean()),
                }
            )
    tail = pd.DataFrame(tail_rows)

    # ----- gate diagnostics (V2.4 only; overall + per bucket) -----
    gate_rows = []
    for label, sub in [("overall", v24_ml)] + [
        (bucket, v24_ml[v24_ml["favorite_bucket"] == bucket])
        for bucket in sorted(v24_ml["favorite_bucket"].unique())
    ]:
        gate_rows.append(
            {
                "favorite_bucket": label,
                "sample_size": int(len(sub)),
                "avg_A_favorite_dominance_gate": float(sub["favorite_dominance_gate"].mean()),
                "avg_B_lambda_imbalance_gate": float(sub["lambda_imbalance_gate"].mean()),
                "avg_C_favorite_scoring_capacity_gate": float(sub["favorite_scoring_capacity_gate"].mean()),
                "avg_D_underdog_suppression_gate": float(sub["underdog_suppression_gate"].mean()),
                "avg_blowout_gate": float(sub["blowout_gate"].mean()),
                "avg_p_blowout_raw": float(sub["p_blowout_raw"].mean()),
                "avg_p_blowout_final": float(sub["p_blowout_final"].mean()),
                "avg_bucket_cap": float(sub["bucket_cap"].mean()),
                "pct_p_blowout_final_gt_0_01": float((sub["p_blowout_final"] > 0.01).mean()),
                "pct_p_blowout_final_gt_0_05": float((sub["p_blowout_final"] > 0.05).mean()),
                "pct_p_blowout_final_gt_0_10": float((sub["p_blowout_final"] > 0.10).mean()),
                "E_motivation_gate": "disabled_neutral_no_data",
            }
        )
    gates = pd.DataFrame(gate_rows)

    # ----- example matches (most blowout-shaped, plus exact hits) -----
    examples_cols = [
        "match_id", "date", "team_a", "team_b", "favorite_team", "favorite_bucket",
        "favorite_win_prob", "predicted_scoreline", "top_5_scorelines",
        "p_blowout_raw", "blowout_gate", "p_blowout_final", "bucket_cap",
        "favorite_dominance_gate", "lambda_imbalance_gate",
        "favorite_scoring_capacity_gate", "underdog_suppression_gate",
        "p_favorite_scores_4_plus_pred", "p_margin_4_plus_pred",
        "exact_top1_hit",
    ]
    v24_ml = v24_ml.rename(
        columns={
            "favorite_scores_4_plus_pred": "p_favorite_scores_4_plus_pred",
            "margin_4_plus_pred": "p_margin_4_plus_pred",
        }
    )
    actual_lookup = base.set_index("match_id")["actual_scoreline"]
    top_blowout = v24_ml.sort_values("p_blowout_final", ascending=False).head(40)
    exact_hits = v24_ml[v24_ml["exact_top1_hit"]].head(20)
    examples = pd.concat([top_blowout, exact_hits]).drop_duplicates("match_id")
    examples = examples[[c for c in examples_cols if c in examples.columns]].copy()
    examples["actual_scoreline"] = examples["match_id"].map(actual_lookup)

    return {
        "summary": summary,
        "by_bucket": by_bucket,
        "tail": tail,
        "gates": gates,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Part 8: model card
# ---------------------------------------------------------------------------
def write_model_card(summary: pd.DataFrame, freeze_passed: bool, params: GatedBlowoutParams) -> None:
    v20 = summary[summary["profile"] == "v20_stable"].iloc[0]
    v24 = summary[summary["profile"] == PROFILE_NAME].iloc[0]

    def fmt(metric: str) -> str:
        return f"{v20[metric]:.4f} -> {v24[metric]:.4f}"

    card = f"""# Model card — V2.4 ABCD no-E gated blowout mixture (`{PROFILE_NAME}`)

**Production status:** shadow / research only (NOT promoted to production).
**W/D/L frozen:** {"CONFIRMED (freeze check PASS)" if freeze_passed else "FAILED — DO NOT USE"}.

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
- exact_top1: {fmt("exact_top1")}
- top3: {fmt("top3")}
- top5: {fmt("top5")}
- winner_direction: {fmt("winner_direction")}
- displayed_draw_rate: {fmt("displayed_draw_rate")}
- mean_goals_pred: {fmt("mean_goals_pred")} (actual {v24["mean_goals_actual"]:.4f})
- favorite_scores_4_plus (pred): {fmt("favorite_scores_4_plus_pred")} (actual {v24["favorite_scores_4_plus_actual"]:.4f})
- favorite_scores_5_plus (pred): {fmt("favorite_scores_5_plus_pred")} (actual {v24["favorite_scores_5_plus_actual"]:.4f})
- margin_4_plus (pred): {fmt("margin_4_plus_pred")} (actual {v24["margin_4_plus_actual"]:.4f})
- total_goals_5_plus (pred): {fmt("total_goals_5_plus_pred")} (actual {v24["total_goals_5_plus_actual"]:.4f})
- combined_tail_error: {fmt("combined_tail_error")}

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
(`{PROFILE_NAME}__wdl_freeze_check.csv`) reports
{"PASS with 0 rows changed" if freeze_passed else "FAILURE"}.

## Production status & next step
Shadow only. **Recommended next step: live shadow evaluation** — log V2.4 ABCD
no-E predictions alongside production V2.0 as real 2026 results arrive and score
the tail calibration on live data. Do **not** make further architecture changes
(no Markov, no market odds, no V2.6 representative selector, no agent context)
until live shadow evidence justifies it.

## Parameters
```
{params}
```
"""
    (RESEARCH_DIR / f"{PROFILE_NAME}__model_card.md").write_text(card)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize V2.4 ABCD no-E shadow scoreline profile.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--skip-historical", action="store_true",
                        help="Only regenerate the shadow output + freeze check (fast).")
    args = parser.parse_args()

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    params = v24_abcd_no_e_params()

    # --- shadow output + metadata ---
    shadow_df = generate_shadow_predictions(params)
    SHADOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    shadow_df.to_csv(SHADOW_PATH, index=False)
    write_metadata(shadow_df, source_script="src/diagnostics/finalize_v24_abcd_no_e.py")

    # --- W/D/L freeze check (hard gate) ---
    freeze_report, freeze_passed = wdl_freeze_check(shadow_df)
    freeze_report.to_csv(RESEARCH_DIR / f"{PROFILE_NAME}__wdl_freeze_check.csv", index=False)
    print("W/D/L freeze check:")
    print(freeze_report.to_string(index=False))
    if not freeze_passed:
        raise SystemExit(
            "FAILURE: W/D/L probabilities changed in the shadow output. Aborting; "
            "production W/D/L must remain frozen."
        )

    if args.skip_historical:
        print(f"\nShadow output: {SHADOW_PATH}")
        print("Skipped historical comparison (--skip-historical).")
        return

    # --- historical comparison + research files ---
    print("\nRunning historical comparison (V2.0 stable vs V2.4 ABCD no-E)...")
    results = run_historical_comparison(start_date=args.start_date, input_path=args.input)
    results["summary"].to_csv(RESEARCH_DIR / f"{PROFILE_NAME}__comparison_vs_v20_summary.csv", index=False)
    results["by_bucket"].to_csv(RESEARCH_DIR / f"{PROFILE_NAME}__comparison_vs_v20_by_bucket.csv", index=False)
    results["tail"].to_csv(RESEARCH_DIR / f"{PROFILE_NAME}__tail_calibration.csv", index=False)
    results["gates"].to_csv(RESEARCH_DIR / f"{PROFILE_NAME}__gate_diagnostics.csv", index=False)
    results["examples"].to_csv(RESEARCH_DIR / f"{PROFILE_NAME}__example_matches.csv", index=False)
    write_model_card(results["summary"], freeze_passed, params)

    print("\nV2.0 stable vs V2.4 ABCD no-E (overall):")
    show_cols = ["profile", "sample_size", "exact_top1", "top3", "top5", "winner_direction",
                 "displayed_draw_rate", "mean_goals_pred", "favorite_scores_4_plus_pred",
                 "margin_4_plus_pred", "total_goals_5_plus_pred", "combined_tail_error"]
    print(results["summary"][show_cols].to_string(index=False))
    print(f"\nShadow output:   {SHADOW_PATH}")
    print(f"Metadata:        {SHADOW_METADATA_PATH}")
    print(f"Research dir:    {RESEARCH_DIR}")
    print(f"Production clean file untouched: {CLEAN_PATH}")


if __name__ == "__main__":
    main()
