from __future__ import annotations

"""Live shadow evaluation for V2.4 ABCD no-E scoreline research.

This script is diagnostics-only. It compares the production V2.0 displayed
scoreline layer against the frozen V2.4 ABCD no-E shadow scoreline layer as
real 2026 group-stage results arrive.

Hard boundaries:
  * W/D/L probabilities are read only and are never modified.
  * V2.4 gate logic and parameters are not recomputed or tuned here.
  * Production prediction files are not overwritten.
  * Outputs are written only to output/diagnostics/live_shadow_v24_abcd_no_e/.
"""

import argparse
import ast
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V20_PATH = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "archive"
    / "legacy_prediction_versions"
    / "group_stage_predictions__scoreline_v20_stable.csv"
)
DEFAULT_V24_PATH = (
    PROJECT_ROOT
    / "legacy"
    / "outputs"
    / "predictions"
    / "v24_shadow"
    / "group_stage_predictions__scoreline_v24_abcd_no_e_shadow.csv"
)
DEFAULT_LIVE_RESULTS_PATH = PROJECT_ROOT / "output" / "live" / "fixtures_results.csv"
DEFAULT_MANUAL_RESULTS_PATH = (
    PROJECT_ROOT / "output" / "live" / "worldcup_group_stage_actual_results.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "diagnostics" / "live_shadow_v24_abcd_no_e"


def normalize_team_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    aliases = {
        "bosnia herzegovina": "bosnia and herzegovina",
        "bosnia and herzegovina": "bosnia and herzegovina",
        "usa": "united states",
        "us": "united states",
        "ir iran": "iran",
        "cabo verde": "cabo verde",
        "cape verde": "cabo verde",
        "ivory coast": "cote d ivoire",
        "cote d ivoire": "cote d ivoire",
        "curacao": "curacao",
        "turkiye": "turkey",
    }
    collapsed = " ".join(text.split())
    return aliases.get(collapsed, collapsed)


def parse_scoreline(value: object) -> tuple[int | None, int | None]:
    if pd.isna(value):
        return None, None
    match = re.search(r"(\d+)\s*-\s*(\d+)", str(value))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def result_from_goals(goals_a: int | float | None, goals_b: int | float | None) -> str | None:
    if goals_a is None or goals_b is None or pd.isna(goals_a) or pd.isna(goals_b):
        return None
    if int(goals_a) > int(goals_b):
        return "team_a_win"
    if int(goals_a) < int(goals_b):
        return "team_b_win"
    return "draw"


def parse_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return [part.strip() for part in text.split(",") if part.strip()]


def scoreline_predicted_outcome(predicted_scoreline: str) -> str | None:
    pred_a, pred_b = parse_scoreline(predicted_scoreline)
    return result_from_goals(pred_a, pred_b)


def wdl_predicted_outcome(row: pd.Series) -> str:
    """Return the frozen W/D/L argmax outcome.

    This evaluates the W/D/L model only. It is intentionally separate from the
    outcome implied by the displayed top-1 scoreline, because the scoreline
    display layer can choose a draw such as 1-1 while W/D/L probabilities favor
    a team win.
    """
    probs = {
        "team_a_win": float(row["team_a_win_pct"]),
        "draw": float(row["draw_pct"]),
        "team_b_win": float(row["team_b_win_pct"]),
    }
    return max(probs, key=probs.get)


def infer_actual_result(row: pd.Series) -> str | None:
    result = row.get("actual_result")
    if isinstance(result, str) and result in {"team_a_win", "draw", "team_b_win"}:
        return result
    return result_from_goals(row.get("goals_a"), row.get("goals_b"))


def load_completed_results(live_path: Path, manual_path: Path | None = None) -> pd.DataFrame:
    """Load final live results, falling back to manual actuals if needed."""
    if not live_path.exists():
        raise FileNotFoundError(f"Live results file not found: {live_path}")

    live = pd.read_csv(live_path)
    if "status" in live.columns:
        completed = live[live["status"].astype(str).str.lower().eq("final")].copy()
    else:
        completed = live.copy()

    if completed.empty and manual_path and manual_path.exists():
        manual = pd.read_csv(manual_path)
        completed = manual[manual[["goals_a", "goals_b"]].notna().all(axis=1)].copy()
        completed["date"] = ""

    if completed.empty:
        return completed

    completed["actual_team_a_goals"] = completed["goals_a"].astype(int)
    completed["actual_team_b_goals"] = completed["goals_b"].astype(int)
    completed["actual_scoreline"] = (
        completed["actual_team_a_goals"].astype(str)
        + "-"
        + completed["actual_team_b_goals"].astype(str)
    )
    completed["actual_outcome"] = completed.apply(infer_actual_result, axis=1)
    completed["norm_team_a"] = completed["team_a"].map(normalize_team_name)
    completed["norm_team_b"] = completed["team_b"].map(normalize_team_name)
    completed["match_key"] = completed.apply(
        lambda r: "::".join(sorted([r["norm_team_a"], r["norm_team_b"]])), axis=1
    )
    return completed


def prepare_prediction_frame(v20_path: Path, v24_path: Path) -> pd.DataFrame:
    if not v20_path.exists():
        raise FileNotFoundError(f"Production V2.0 prediction file not found: {v20_path}")
    if not v24_path.exists():
        raise FileNotFoundError(f"Shadow V2.4 prediction file not found: {v24_path}")

    v20 = pd.read_csv(v20_path)
    v24 = pd.read_csv(v24_path)

    required_v20 = {"match_id", "team_a", "team_b", "predicted_scoreline"}
    required_v24 = {
        "match_id",
        "team_a",
        "team_b",
        "predicted_scoreline",
        "top_5_scorelines",
        "favorite_team",
        "favorite_bucket",
        "p_blowout_final",
        "tail_risk_index",
        "p_favorite_scores_4_plus",
        "p_favorite_scores_5_plus",
        "p_margin_4_plus",
        "p_total_goals_5_plus",
    }
    missing_v20 = required_v20 - set(v20.columns)
    missing_v24 = required_v24 - set(v24.columns)
    if missing_v20:
        raise ValueError(f"Missing V2.0 columns: {sorted(missing_v20)}")
    if missing_v24:
        raise ValueError(f"Missing V2.4 columns: {sorted(missing_v24)}")

    merged = v20[
        [
            "match_id",
            "group",
            "team_a",
            "team_b",
            "team_a_win_pct",
            "draw_pct",
            "team_b_win_pct",
            "predicted_scoreline",
        ]
    ].rename(columns={"predicted_scoreline": "v20_predicted_scoreline"})
    if "top_5_scorelines" in v20.columns:
        merged["v20_top5_scorelines"] = v20["top_5_scorelines"]
    else:
        merged["v20_top5_scorelines"] = v20["predicted_scoreline"].map(lambda x: [str(x)])

    v24_cols = [
        "match_id",
        "predicted_scoreline",
        "top_5_scorelines",
        "favorite_team",
        "favorite_bucket",
        "p_blowout_final",
        "tail_risk_index",
        "p_favorite_scores_4_plus",
        "p_favorite_scores_5_plus",
        "p_margin_4_plus",
        "p_total_goals_5_plus",
    ]
    merged = merged.merge(
        v24[v24_cols].rename(
            columns={
                "predicted_scoreline": "v24_predicted_scoreline",
                "top_5_scorelines": "v24_top5_scorelines",
            }
        ),
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    merged["norm_team_a"] = merged["team_a"].map(normalize_team_name)
    merged["norm_team_b"] = merged["team_b"].map(normalize_team_name)
    merged["match_key"] = merged.apply(
        lambda r: "::".join(sorted([r["norm_team_a"], r["norm_team_b"]])), axis=1
    )
    return merged


def align_actual_orientation(pred: pd.Series, actual: pd.Series) -> tuple[int, int, str]:
    """Return actual goals oriented to prediction team_a/team_b."""
    actual_a = normalize_team_name(actual["team_a"])
    actual_b = normalize_team_name(actual["team_b"])
    if pred["norm_team_a"] == actual_a and pred["norm_team_b"] == actual_b:
        goals_a = int(actual["actual_team_a_goals"])
        goals_b = int(actual["actual_team_b_goals"])
    elif pred["norm_team_a"] == actual_b and pred["norm_team_b"] == actual_a:
        goals_a = int(actual["actual_team_b_goals"])
        goals_b = int(actual["actual_team_a_goals"])
    else:
        goals_a = int(actual["actual_team_a_goals"])
        goals_b = int(actual["actual_team_b_goals"])
    return goals_a, goals_b, f"{goals_a}-{goals_b}"


def join_predictions_to_actuals(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    if actuals.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    actual_by_match_id = {}
    if "match_id" in actuals.columns:
        actual_by_match_id = {str(row.match_id): row for row in actuals.itertuples(index=False)}
    actual_by_key = {row.match_key: row for row in actuals.itertuples(index=False)}

    for pred in predictions.itertuples(index=False):
        actual = actual_by_match_id.get(str(pred.match_id))
        if actual is None:
            actual = actual_by_key.get(pred.match_key)
        if actual is None:
            continue

        pred_s = pd.Series(pred._asdict())
        actual_s = pd.Series(actual._asdict())
        goals_a, goals_b, actual_scoreline = align_actual_orientation(pred_s, actual_s)
        actual_outcome = result_from_goals(goals_a, goals_b)
        favorite_side = "team_a" if normalize_team_name(pred.favorite_team) == pred.norm_team_a else "team_b"
        favorite_goals = goals_a if favorite_side == "team_a" else goals_b
        underdog_goals = goals_b if favorite_side == "team_a" else goals_a
        margin = favorite_goals - underdog_goals
        total_goals = goals_a + goals_b

        v20_top5 = parse_list(pred.v20_top5_scorelines)
        v24_top5 = parse_list(pred.v24_top5_scorelines)
        wdl_outcome = wdl_predicted_outcome(pred_s)
        v20_scoreline_outcome = scoreline_predicted_outcome(pred.v20_predicted_scoreline)
        v24_scoreline_outcome = scoreline_predicted_outcome(pred.v24_predicted_scoreline)

        rows.append(
            {
                "match_id": pred.match_id,
                "date": actual_s.get("date", ""),
                "team_a": pred.team_a,
                "team_b": pred.team_b,
                "actual_scoreline": actual_scoreline,
                "actual_team_a_goals": goals_a,
                "actual_team_b_goals": goals_b,
                "actual_outcome": actual_outcome,
                "wdl_predicted_outcome": wdl_outcome,
                "wdl_direction_hit": wdl_outcome == actual_outcome,
                "v20_predicted_scoreline": pred.v20_predicted_scoreline,
                "v20_top5_scorelines": v20_top5,
                "v20_exact_hit": pred.v20_predicted_scoreline == actual_scoreline,
                "v20_top5_hit": actual_scoreline in v20_top5,
                "v20_scoreline_predicted_outcome": v20_scoreline_outcome,
                "v20_scoreline_direction_hit": v20_scoreline_outcome == actual_outcome,
                "v24_predicted_scoreline": pred.v24_predicted_scoreline,
                "v24_top5_scorelines": v24_top5,
                "v24_exact_hit": pred.v24_predicted_scoreline == actual_scoreline,
                "v24_top5_hit": actual_scoreline in v24_top5,
                "v24_scoreline_predicted_outcome": v24_scoreline_outcome,
                "v24_scoreline_direction_hit": v24_scoreline_outcome == actual_outcome,
                "v24_p_blowout_final": float(pred.p_blowout_final),
                "v24_tail_risk_index": float(pred.tail_risk_index),
                "v24_p_favorite_scores_4_plus": float(pred.p_favorite_scores_4_plus),
                "v24_p_favorite_scores_5_plus": float(pred.p_favorite_scores_5_plus),
                "v24_p_margin_4_plus": float(pred.p_margin_4_plus),
                "v24_p_total_goals_5_plus": float(pred.p_total_goals_5_plus),
                "favorite_team": pred.favorite_team,
                "favorite_bucket": pred.favorite_bucket,
                "actual_favorite_scores_4_plus": favorite_goals >= 4,
                "actual_favorite_scores_5_plus": favorite_goals >= 5,
                "actual_margin_4_plus": margin >= 4,
                "actual_total_goals_5_plus": total_goals >= 5,
            }
        )

    return pd.DataFrame(rows)


def safe_mean(values: Iterable[object]) -> float:
    series = pd.Series(values)
    if series.empty:
        return math.nan
    return float(series.mean())


def summarize_overall(match_level: pd.DataFrame) -> pd.DataFrame:
    if match_level.empty:
        return pd.DataFrame(
            [
                {
                    "model": "v20_stable",
                    "matches_evaluated": 0,
                    "wdl_direction_hit_rate": math.nan,
                    "exact_top1": math.nan,
                    "top5_hit_rate": math.nan,
                    "scoreline_direction_hit_rate": math.nan,
                },
                {
                    "model": "v24_abcd_no_e_shadow",
                    "matches_evaluated": 0,
                    "wdl_direction_hit_rate": math.nan,
                    "exact_top1": math.nan,
                    "top5_hit_rate": math.nan,
                    "scoreline_direction_hit_rate": math.nan,
                },
            ]
        )

    return pd.DataFrame(
        [
            {
                "model": "v20_stable",
                "matches_evaluated": len(match_level),
                "wdl_direction_hit_rate": safe_mean(match_level["wdl_direction_hit"]),
                "exact_top1": safe_mean(match_level["v20_exact_hit"]),
                "top5_hit_rate": safe_mean(match_level["v20_top5_hit"]),
                "scoreline_direction_hit_rate": safe_mean(
                    match_level["v20_scoreline_direction_hit"]
                ),
            },
            {
                "model": "v24_abcd_no_e_shadow",
                "matches_evaluated": len(match_level),
                "wdl_direction_hit_rate": safe_mean(match_level["wdl_direction_hit"]),
                "exact_top1": safe_mean(match_level["v24_exact_hit"]),
                "top5_hit_rate": safe_mean(match_level["v24_top5_hit"]),
                "scoreline_direction_hit_rate": safe_mean(
                    match_level["v24_scoreline_direction_hit"]
                ),
            },
        ]
    )


def summarize_by_bucket(match_level: pd.DataFrame) -> pd.DataFrame:
    if match_level.empty:
        return pd.DataFrame()
    rows = []
    for bucket, group in match_level.groupby("favorite_bucket", dropna=False):
        rows.append(
            {
                "favorite_bucket": bucket,
                "matches_evaluated": len(group),
                "wdl_direction_hit_rate": safe_mean(group["wdl_direction_hit"]),
                "v20_exact_top1": safe_mean(group["v20_exact_hit"]),
                "v20_top5_hit_rate": safe_mean(group["v20_top5_hit"]),
                "v20_scoreline_direction_hit_rate": safe_mean(
                    group["v20_scoreline_direction_hit"]
                ),
                "v24_exact_top1": safe_mean(group["v24_exact_hit"]),
                "v24_top5_hit_rate": safe_mean(group["v24_top5_hit"]),
                "v24_scoreline_direction_hit_rate": safe_mean(
                    group["v24_scoreline_direction_hit"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("favorite_bucket")


def summarize_tail(match_level: pd.DataFrame) -> pd.DataFrame:
    if match_level.empty:
        return pd.DataFrame(
            [
                {
                    "scope": "overall",
                    "matches_evaluated": 0,
                    "predicted_favorite_scores_4_plus_mean": math.nan,
                    "actual_favorite_scores_4_plus_rate": math.nan,
                    "predicted_favorite_scores_5_plus_mean": math.nan,
                    "actual_favorite_scores_5_plus_rate": math.nan,
                    "predicted_margin_4_plus_mean": math.nan,
                    "actual_margin_4_plus_rate": math.nan,
                    "predicted_total_goals_5_plus_mean": math.nan,
                    "actual_total_goals_5_plus_rate": math.nan,
                }
            ]
        )

    rows = []
    for scope, group in [("overall", match_level), *match_level.groupby("favorite_bucket")]:
        rows.append(
            {
                "scope": scope,
                "matches_evaluated": len(group),
                "predicted_favorite_scores_4_plus_mean": safe_mean(
                    group["v24_p_favorite_scores_4_plus"]
                ),
                "actual_favorite_scores_4_plus_rate": safe_mean(
                    group["actual_favorite_scores_4_plus"]
                ),
                "predicted_favorite_scores_5_plus_mean": safe_mean(
                    group["v24_p_favorite_scores_5_plus"]
                ),
                "actual_favorite_scores_5_plus_rate": safe_mean(
                    group["actual_favorite_scores_5_plus"]
                ),
                "predicted_margin_4_plus_mean": safe_mean(group["v24_p_margin_4_plus"]),
                "actual_margin_4_plus_rate": safe_mean(group["actual_margin_4_plus"]),
                "predicted_total_goals_5_plus_mean": safe_mean(
                    group["v24_p_total_goals_5_plus"]
                ),
                "actual_total_goals_5_plus_rate": safe_mean(
                    group["actual_total_goals_5_plus"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_delta(match_level: pd.DataFrame) -> pd.DataFrame:
    if match_level.empty:
        return pd.DataFrame(
            [
                {
                    "matches_evaluated": 0,
                    "v24_exact_minus_v20_exact": math.nan,
                    "v24_top5_minus_v20_top5": math.nan,
                    "v24_scoreline_direction_minus_v20_scoreline_direction": math.nan,
                    "matches_where_v24_exact_and_v20_not": 0,
                    "matches_where_v20_exact_and_v24_not": 0,
                    "matches_where_v24_top5_and_v20_not": 0,
                    "matches_where_v20_top5_and_v24_not": 0,
                    "matches_where_v24_scoreline_direction_and_v20_not": 0,
                    "matches_where_v20_scoreline_direction_and_v24_not": 0,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "matches_evaluated": len(match_level),
                "v24_exact_minus_v20_exact": safe_mean(match_level["v24_exact_hit"])
                - safe_mean(match_level["v20_exact_hit"]),
                "v24_top5_minus_v20_top5": safe_mean(match_level["v24_top5_hit"])
                - safe_mean(match_level["v20_top5_hit"]),
                "v24_scoreline_direction_minus_v20_scoreline_direction": safe_mean(
                    match_level["v24_scoreline_direction_hit"]
                )
                - safe_mean(match_level["v20_scoreline_direction_hit"]),
                "matches_where_v24_exact_and_v20_not": int(
                    (match_level["v24_exact_hit"] & ~match_level["v20_exact_hit"]).sum()
                ),
                "matches_where_v20_exact_and_v24_not": int(
                    (match_level["v20_exact_hit"] & ~match_level["v24_exact_hit"]).sum()
                ),
                "matches_where_v24_top5_and_v20_not": int(
                    (match_level["v24_top5_hit"] & ~match_level["v20_top5_hit"]).sum()
                ),
                "matches_where_v20_top5_and_v24_not": int(
                    (match_level["v20_top5_hit"] & ~match_level["v24_top5_hit"]).sum()
                ),
                "matches_where_v24_scoreline_direction_and_v20_not": int(
                    (
                        match_level["v24_scoreline_direction_hit"]
                        & ~match_level["v20_scoreline_direction_hit"]
                    ).sum()
                ),
                "matches_where_v20_scoreline_direction_and_v24_not": int(
                    (
                        match_level["v20_scoreline_direction_hit"]
                        & ~match_level["v24_scoreline_direction_hit"]
                    ).sum()
                ),
            }
        ]
    )


def write_report(
    output_dir: Path,
    match_level: pd.DataFrame,
    overall: pd.DataFrame,
    tail: pd.DataFrame,
    delta: pd.DataFrame,
) -> None:
    n = len(match_level)
    v20 = overall[overall["model"].eq("v20_stable")].iloc[0]
    v24 = overall[overall["model"].eq("v24_abcd_no_e_shadow")].iloc[0]
    delta_row = delta.iloc[0]
    tail_row = tail[tail["scope"].eq("overall")].iloc[0]

    if n < 12:
        decision = "Sample is too small for promotion decisions; keep V2.4 in shadow."
    elif n < 24:
        decision = "Only tentative conclusions are allowed; keep V2.4 in shadow."
    else:
        exact_ok = delta_row["v24_exact_minus_v20_exact"] >= 0
        top5_ok = delta_row["v24_top5_minus_v20_top5"] >= 0
        scoreline_direction_ok = (
            delta_row["v24_scoreline_direction_minus_v20_scoreline_direction"] >= 0
        )
        decision = (
            "V2.4 may remain a shadow candidate for continued monitoring."
            if exact_ok and top5_ok and scoreline_direction_ok
            else "Hold V2.4 in shadow; do not promote on current live evidence."
        )

    lines = [
        "# Live Shadow Evaluation: V2.0 vs V2.4 ABCD no-E",
        "",
        "This is an evaluation-only report. It reads frozen predictions and final live results.",
        "It does not modify W/D/L probabilities, scoreline model parameters, production files, or simulation logic.",
        "",
        "## Sample Size",
        "",
        f"- Live matches evaluated: {n}",
        "",
        "## Metric Definitions",
        "",
        "- W/D/L direction evaluates the frozen W/D/L model: argmax(team_a_win_pct, draw_pct, team_b_win_pct).",
        "- Scoreline direction evaluates the outcome implied by the displayed top-1 scoreline.",
        "- Exact scoreline hit implies scoreline direction hit.",
        "- W/D/L direction and scoreline direction may differ when the top-1 scoreline is draw but W/D/L argmax is a win, or vice versa.",
        "",
        "## Overall Metrics",
        "",
        "| Model | W/D/L direction | Exact top-1 | Top-5 hit | Scoreline direction |",
        "|---|---:|---:|---:|---:|",
        f"| V2.0 stable | {v20['wdl_direction_hit_rate']:.3f} | {v20['exact_top1']:.3f} | {v20['top5_hit_rate']:.3f} | {v20['scoreline_direction_hit_rate']:.3f} |",
        f"| V2.4 ABCD no-E shadow | {v24['wdl_direction_hit_rate']:.3f} | {v24['exact_top1']:.3f} | {v24['top5_hit_rate']:.3f} | {v24['scoreline_direction_hit_rate']:.3f} |",
        "",
        "## Delta vs V2.0",
        "",
        f"- Exact delta: {delta_row['v24_exact_minus_v20_exact']:.3f}",
        f"- Top-5 delta: {delta_row['v24_top5_minus_v20_top5']:.3f}",
        f"- Scoreline-direction delta: {delta_row['v24_scoreline_direction_minus_v20_scoreline_direction']:.3f}",
        "",
        "## V2.4 Tail Calibration",
        "",
        f"- Predicted favorite 4+ mean: {tail_row['predicted_favorite_scores_4_plus_mean']:.3f}",
        f"- Actual favorite 4+ rate: {tail_row['actual_favorite_scores_4_plus_rate']:.3f}",
        f"- Predicted favorite 5+ mean: {tail_row['predicted_favorite_scores_5_plus_mean']:.3f}",
        f"- Actual favorite 5+ rate: {tail_row['actual_favorite_scores_5_plus_rate']:.3f}",
        f"- Predicted margin 4+ mean: {tail_row['predicted_margin_4_plus_mean']:.3f}",
        f"- Actual margin 4+ rate: {tail_row['actual_margin_4_plus_rate']:.3f}",
        f"- Predicted total goals 5+ mean: {tail_row['predicted_total_goals_5_plus_mean']:.3f}",
        f"- Actual total goals 5+ rate: {tail_row['actual_total_goals_5_plus_rate']:.3f}",
        "",
        "## Decision",
        "",
        decision,
        "",
        "Promotion requires a larger live sample and no major tail overshoot.",
        "",
        "## Confirmation",
        "",
        "- W/D/L probabilities were untouched.",
        "- V2.4 ABCD no-E parameters were untouched.",
        "- Production predictions were not overwritten.",
    ]
    (output_dir / "live_shadow_report.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = prepare_prediction_frame(Path(args.v20_predictions), Path(args.v24_predictions))
    actuals = load_completed_results(Path(args.live_results), Path(args.manual_results))
    match_level = join_predictions_to_actuals(predictions, actuals)

    # Store lists as JSON-ish strings for CSV readability.
    for column in ["v20_top5_scorelines", "v24_top5_scorelines"]:
        if column in match_level.columns:
            match_level[column] = match_level[column].map(lambda x: str(x))

    overall = summarize_overall(match_level)
    by_bucket = summarize_by_bucket(match_level)
    tail = summarize_tail(match_level)
    delta = summarize_delta(match_level)

    match_level.to_csv(output_dir / "live_shadow_match_level.csv", index=False)
    overall.to_csv(output_dir / "live_shadow_summary_overall.csv", index=False)
    by_bucket.to_csv(output_dir / "live_shadow_summary_by_favorite_bucket.csv", index=False)
    tail.to_csv(output_dir / "live_shadow_tail_calibration.csv", index=False)
    delta.to_csv(output_dir / "live_shadow_delta_vs_v20.csv", index=False)
    write_report(output_dir, match_level, overall, tail, delta)

    print(f"Live shadow evaluation written to: {output_dir}")
    print(f"Matches evaluated: {len(match_level)}")
    print(overall.to_string(index=False))
    print(delta.to_string(index=False))
    if len(match_level) < 12:
        print("Sample is too small for promotion decisions; keep V2.4 in shadow.")
    elif len(match_level) < 24:
        print("Only tentative conclusions are allowed; keep V2.4 in shadow.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v20-predictions", default=str(DEFAULT_V20_PATH))
    parser.add_argument("--v24-predictions", default=str(DEFAULT_V24_PATH))
    parser.add_argument("--live-results", default=str(DEFAULT_LIVE_RESULTS_PATH))
    parser.add_argument("--manual-results", default=str(DEFAULT_MANUAL_RESULTS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
