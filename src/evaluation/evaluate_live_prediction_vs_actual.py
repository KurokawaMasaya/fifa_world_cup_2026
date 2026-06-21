from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "live"
    / "group_stage_round1_predictions__v24_abcd_no_e.csv"
)
DEFAULT_ADDITIONAL_PREDICTION_PATHS = [
    PROJECT_ROOT / "output" / "predictions" / "live" / "group_stage_round2_predictions_clean.csv",
]
FALLBACK_PREDICTIONS_PATH = (
    PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_clean.csv"
)
DEFAULT_ACTUALS_PATH = PROJECT_ROOT / "output" / "live" / "fixtures_results.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "output" / "diagnostics" / "live_prediction_vs_actual"
)


def normalize_team_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    collapsed = " ".join(text.split())
    aliases = {
        "bosnia herzegovina": "bosnia and herzegovina",
        "bosnia and herzegovina": "bosnia and herzegovina",
        "usa": "united states",
        "us": "united states",
        "united states": "united states",
        "ir iran": "iran",
        "cabo verde": "cabo verde",
        "cape verde": "cabo verde",
        "ivory coast": "cote d ivoire",
        "cote d ivoire": "cote d ivoire",
        "turkiye": "turkey",
        "turkey": "turkey",
        "morroco": "morocco",
        "dr congo": "dr congo",
        "congo dr": "dr congo",
        "congo democratic republic": "dr congo",
        "democratic republic of congo": "dr congo",
    }
    return aliases.get(collapsed, collapsed)


def match_key(team_a: object, team_b: object) -> str:
    return "::".join(sorted([normalize_team_name(team_a), normalize_team_name(team_b)]))


def parse_scoreline_list(value: object) -> list[str]:
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


def parse_probability_list(value: object) -> list[float]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return []


DEFAULT_CLOSE_WIN_RATE_THRESHOLD = 25.0


def outcome_from_goals(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a < goals_b:
        return "team_b_win"
    return "draw"


def scoreline_outcome(scoreline: str | None) -> str | None:
    if not scoreline or "-" not in str(scoreline):
        return None
    try:
        goals_a_text, goals_b_text = str(scoreline).split("-", maxsplit=1)
        return outcome_from_goals(int(goals_a_text), int(goals_b_text))
    except ValueError:
        return None


def predicted_wdl_outcome(row: pd.Series) -> str:
    probs = {
        "team_a_win": float(row["team_a_win_pct"]),
        "draw": float(row["draw_pct"]),
        "team_b_win": float(row["team_b_win_pct"]),
    }
    return max(probs, key=probs.get)


def wdl_evaluation_outcome(
    row: pd.Series,
    top1_scoreline: str | None,
    actual_outcome: str,
    exact_top1_correct: bool,
    close_win_rate_threshold: float = DEFAULT_CLOSE_WIN_RATE_THRESHOLD,
) -> tuple[str, str, bool, bool]:
    """Return the W/D/L outcome used for evaluation.

    Official W/D/L probabilities are not changed. This daily-report rule gives
    credit when close team win rates make the displayed draw scoreline meaningful:

    * If team win rates are within the threshold and the top-1 scoreline is a
      draw that exactly hits, count the W/D/L as correct for a draw.

    The rule is evaluation-only and keeps the argmax outcome visible for audit.
    """
    argmax_outcome = predicted_wdl_outcome(row)
    top1_outcome = scoreline_outcome(top1_scoreline)
    team_a_win_pct = float(row["team_a_win_pct"])
    team_b_win_pct = float(row["team_b_win_pct"])
    team_win_gap = abs(team_a_win_pct - team_b_win_pct)
    close_win_rates = team_win_gap <= close_win_rate_threshold
    higher_win_side = "team_a_win" if team_a_win_pct >= team_b_win_pct else "team_b_win"
    exact_draw_credit = (
        top1_outcome == "draw"
        and exact_top1_correct
        and close_win_rates
        and actual_outcome == "draw"
    )
    daily_rule_correct = exact_draw_credit or argmax_outcome == actual_outcome
    evaluation_outcome = "draw" if exact_draw_credit else argmax_outcome
    return evaluation_outcome, argmax_outcome, exact_draw_credit, daily_rule_correct


def actual_probability(row: pd.Series, actual_outcome: str) -> float:
    column = {
        "team_a_win": "team_a_win_pct",
        "draw": "draw_pct",
        "team_b_win": "team_b_win_pct",
    }[actual_outcome]
    return float(row[column])


def _load_prediction_file(source_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(source_path)
    required = {
        "match_id",
        "team_a",
        "team_b",
        "team_a_win_pct",
        "draw_pct",
        "team_b_win_pct",
        "top_5_scorelines",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {sorted(missing)}")
    predictions = predictions.copy()
    predictions["prediction_source_file"] = str(source_path.relative_to(PROJECT_ROOT))
    return predictions


def load_predictions(path: Path) -> tuple[pd.DataFrame, list[Path]]:
    if path != DEFAULT_PREDICTIONS_PATH:
        source_paths = [path if path.exists() else FALLBACK_PREDICTIONS_PATH]
    else:
        source_paths = [path] if path.exists() else [FALLBACK_PREDICTIONS_PATH]
        source_paths.extend(
            additional_path
            for additional_path in DEFAULT_ADDITIONAL_PREDICTION_PATHS
            if additional_path.exists()
        )

    frames = [_load_prediction_file(source_path) for source_path in source_paths]
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions.drop_duplicates(subset=["match_id"], keep="last")
    predictions["_match_key"] = predictions.apply(
        lambda row: match_key(row["team_a"], row["team_b"]),
        axis=1,
    )
    return predictions, source_paths


def load_completed_actuals(path: Path) -> pd.DataFrame:
    actuals = pd.read_csv(path)
    required = {"team_a", "team_b", "goals_a", "goals_b", "status"}
    missing = required - set(actuals.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    completed = actuals[actuals["status"].astype(str).str.lower().eq("final")].copy()
    completed = completed.dropna(subset=["goals_a", "goals_b"])
    completed["_match_key"] = completed.apply(
        lambda row: match_key(row["team_a"], row["team_b"]),
        axis=1,
    )
    return completed


def evaluate(
    predictions_path: Path,
    actuals_path: Path,
    output_dir: Path,
    close_win_rate_threshold: float = DEFAULT_CLOSE_WIN_RATE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, prediction_source = load_predictions(predictions_path)
    actuals = load_completed_actuals(actuals_path)

    rows: list[dict] = []
    for _, actual in actuals.iterrows():
        matches = predictions[predictions["_match_key"].eq(actual["_match_key"])]
        if matches.empty:
            continue
        pred = matches.iloc[0]

        same_orientation = normalize_team_name(actual["team_a"]) == normalize_team_name(pred["team_a"])
        if same_orientation:
            goals_a = int(actual["goals_a"])
            goals_b = int(actual["goals_b"])
        else:
            goals_a = int(actual["goals_b"])
            goals_b = int(actual["goals_a"])

        actual_scoreline = f"{goals_a}-{goals_b}"
        actual_outcome = outcome_from_goals(goals_a, goals_b)
        top5 = parse_scoreline_list(pred["top_5_scorelines"])
        top5_probs = parse_probability_list(pred.get("top_5_scoreline_probability_pct"))
        top1 = top5[0] if top5 else None
        top1_outcome = scoreline_outcome(top1)
        exact_top1_correct = actual_scoreline == top1
        wdl_pick, wdl_argmax_pick, exact_draw_credit_used, wdl_daily_rule_correct = wdl_evaluation_outcome(
            pred,
            top1,
            actual_outcome,
            exact_top1_correct,
            close_win_rate_threshold=close_win_rate_threshold,
        )

        rows.append(
            {
                "match_id": pred["match_id"],
                "espn_event_id": actual.get("espn_event_id"),
                "date": actual.get("date"),
                "group": pred.get("group"),
                "team_a": pred["team_a"],
                "team_b": pred["team_b"],
                "actual_goals_a": goals_a,
                "actual_goals_b": goals_b,
                "actual_scoreline": actual_scoreline,
                "actual_outcome": actual_outcome,
                "team_a_win_pct": int(pred["team_a_win_pct"]),
                "draw_pct": int(pred["draw_pct"]),
                "team_b_win_pct": int(pred["team_b_win_pct"]),
                "wdl_argmax_outcome": wdl_argmax_pick,
                "top1_scoreline_implied_outcome": top1_outcome,
                "wdl_exact_draw_credit_used": exact_draw_credit_used,
                "wdl_predicted_outcome": wdl_pick,
                "wdl_direction_correct": wdl_daily_rule_correct,
                "wdl_actual_outcome_probability_pct": actual_probability(pred, actual_outcome),
                "top1_scoreline": top1,
                "top3_scorelines": json.dumps(top5[:3]),
                "top5_scorelines": json.dumps(top5[:5]),
                "top1_scoreline_probability_pct": top5_probs[0] if top5_probs else None,
                "top3_scoreline_probability_pct": json.dumps(top5_probs[:3]),
                "top5_scoreline_probability_pct": json.dumps(top5_probs[:5]),
                "exact_top1_correct": exact_top1_correct,
                "top3_correct": actual_scoreline in top5[:3],
                "top5_correct": actual_scoreline in top5[:5],
            }
        )

    match_level = pd.DataFrame(rows)
    if match_level.empty:
        summary = pd.DataFrame(
            [
                {
                    "matches_evaluated": 0,
                    "prediction_source": ";".join(str(path.relative_to(PROJECT_ROOT)) for path in prediction_source),
                    "actual_source": str(actuals_path.relative_to(PROJECT_ROOT)),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "matches_evaluated": len(match_level),
                    "wdl_correct_rate": match_level["wdl_direction_correct"].mean(),
                    "mean_wdl_actual_outcome_probability_pct": match_level[
                        "wdl_actual_outcome_probability_pct"
                    ].mean(),
                    "scoreline_exact_top1_rate": match_level["exact_top1_correct"].mean(),
                    "scoreline_top3_rate": match_level["top3_correct"].mean(),
                    "scoreline_top5_rate": match_level["top5_correct"].mean(),
                    "wdl_exact_draw_credit_count": int(match_level["wdl_exact_draw_credit_used"].sum()),
                    "close_win_rate_threshold_pct": close_win_rate_threshold,
                    "wdl_evaluation_rule": (
                        "argmax plus close-win-rate credit only when top-1 draw scoreline exactly hits"
                    ),
                    "prediction_source": ";".join(str(path.relative_to(PROJECT_ROOT)) for path in prediction_source),
                    "actual_source": str(actuals_path.relative_to(PROJECT_ROOT)),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    match_level.to_csv(output_dir / "live_prediction_vs_actual_match_level.csv", index=False)
    summary.to_csv(output_dir / "live_prediction_vs_actual_summary.csv", index=False)
    return match_level, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate live production predictions against completed actual results."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--actuals", type=Path, default=DEFAULT_ACTUALS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--close-win-rate-threshold",
        type=float,
        default=DEFAULT_CLOSE_WIN_RATE_THRESHOLD,
        help=(
            "If top-1 scoreline is draw and team win percentages are within this "
            "many points, count W/D/L evaluation outcome as draw."
        ),
    )
    args = parser.parse_args()

    match_level, summary = evaluate(
        args.predictions,
        args.actuals,
        args.output_dir,
        close_win_rate_threshold=args.close_win_rate_threshold,
    )
    print(f"Saved match-level evaluation to {args.output_dir / 'live_prediction_vs_actual_match_level.csv'}")
    print(f"Saved summary to {args.output_dir / 'live_prediction_vs_actual_summary.csv'}")
    print(summary.to_string(index=False))
    if not match_level.empty:
        display_cols = [
            "match_id",
            "team_a",
            "team_b",
            "actual_scoreline",
            "wdl_argmax_outcome",
            "top1_scoreline_implied_outcome",
            "wdl_exact_draw_credit_used",
            "wdl_predicted_outcome",
            "wdl_direction_correct",
            "wdl_actual_outcome_probability_pct",
            "top1_scoreline",
            "exact_top1_correct",
            "top3_correct",
            "top5_correct",
        ]
        print(match_level[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
