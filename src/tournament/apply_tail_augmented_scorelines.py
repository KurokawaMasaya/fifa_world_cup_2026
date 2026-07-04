from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT / "output" / "predictions" / "live" / "group_stage_round2_predictions_clean.csv"
)
DEFAULT_DETAIL = (
    PROJECT_ROOT / "output" / "predictions" / "live" / "group_stage_round2_predictions.csv"
)
DEFAULT_OUTPUT = DEFAULT_INPUT
DEFAULT_BACKUP = DEFAULT_INPUT.with_name(DEFAULT_INPUT.stem + "__pre_tail_backup.csv")
DEFAULT_METADATA = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "live"
    / "group_stage_round2_predictions__metadata.json"
)


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


def parse_scoreline(scoreline: str | None) -> tuple[int, int] | None:
    if not scoreline or "-" not in str(scoreline):
        return None
    try:
        goals_a, goals_b = str(scoreline).split("-", maxsplit=1)
        return int(goals_a), int(goals_b)
    except ValueError:
        return None


def scoreline_total(scoreline: str) -> int:
    parsed = parse_scoreline(scoreline)
    return 0 if parsed is None else parsed[0] + parsed[1]


def scoreline_margin(scoreline: str) -> int | None:
    parsed = parse_scoreline(scoreline)
    if parsed is None:
        return None
    return parsed[0] - parsed[1]


def scoreline_outcome(scoreline: str) -> str | None:
    margin = scoreline_margin(scoreline)
    if margin is None:
        return None
    if margin > 0:
        return "team_a_win"
    if margin < 0:
        return "team_b_win"
    return "draw"


def favorite_side(row: pd.Series) -> str:
    return "team_a" if float(row["team_a_win_pct"]) >= float(row["team_b_win_pct"]) else "team_b"


def upset_risk_tier(favorite_win_pct: float) -> str:
    if favorite_win_pct >= 80:
        return "major_upset_risk"
    if favorite_win_pct >= 65:
        return "moderate_upset_risk"
    if favorite_win_pct >= 55:
        return "minor_upset_risk"
    return "balanced_match"


def draw_pressure_weight(favorite_win_pct: float) -> float:
    """Convert draw probability into partial upset pressure.

    An underdog win is a direct upset path. A draw is not the same event, but it
    can be part of favorite-fail pressure, especially when the favorite is very
    strong. This keeps the displayed risk from being a literal draw + underdog
    win sum.
    """
    if favorite_win_pct >= 80:
        return 0.60
    if favorite_win_pct >= 65:
        return 0.45
    if favorite_win_pct >= 55:
        return 0.30
    return 0.20


def upset_pressure_probability(
    favorite_win_pct: pd.Series,
    draw_pct: pd.Series,
    underdog_win_pct: pd.Series,
) -> pd.Series:
    favorite_win_pct = pd.to_numeric(favorite_win_pct, errors="coerce")
    draw_pct = pd.to_numeric(draw_pct, errors="coerce")
    underdog_win_pct = pd.to_numeric(underdog_win_pct, errors="coerce")
    weights = favorite_win_pct.apply(lambda value: draw_pressure_weight(float(value)))
    underdog_hazard = underdog_win_pct / 100.0
    draw_pressure_hazard = (draw_pct / 100.0) * weights
    return (1.0 - (1.0 - underdog_hazard) * (1.0 - draw_pressure_hazard)) * 100.0


def add_risk_column(clean: pd.DataFrame) -> pd.DataFrame:
    clean = clean.copy()
    favorite_win_pct = clean[["team_a_win_pct", "team_b_win_pct"]].max(axis=1)
    underdog_win_pct = clean[["team_a_win_pct", "team_b_win_pct"]].min(axis=1)
    clean["draw_risk"] = pd.to_numeric(clean["draw_pct"], errors="coerce")
    clean["upset_win_risk"] = pd.to_numeric(underdog_win_pct, errors="coerce")
    clean["draw_pressure_weight"] = favorite_win_pct.apply(
        lambda value: draw_pressure_weight(float(value))
    )
    clean["risk"] = upset_pressure_probability(
        favorite_win_pct,
        clean["draw_risk"],
        clean["upset_win_risk"],
    ).round().astype("Int64")
    clean["favorite_fail_to_win_risk"] = clean["draw_risk"] + clean["upset_win_risk"]
    return clean


def add_detail_risk_columns(clean: pd.DataFrame) -> pd.DataFrame:
    clean = clean.copy()
    p_team_a = pd.to_numeric(clean["v2_p_team_a_win"], errors="coerce") * 100
    p_draw = pd.to_numeric(clean["v2_p_draw"], errors="coerce") * 100
    p_team_b = pd.to_numeric(clean["v2_p_team_b_win"], errors="coerce") * 100
    win_probs = pd.concat([p_team_a, p_team_b], axis=1)
    favorite_win_pct = win_probs.max(axis=1)
    underdog_win_pct = win_probs.min(axis=1)
    clean["draw_risk"] = p_draw.round().astype("Int64")
    clean["upset_win_risk"] = underdog_win_pct.round().astype("Int64")
    clean["draw_pressure_weight"] = favorite_win_pct.apply(
        lambda value: draw_pressure_weight(float(value))
    )
    clean["risk"] = upset_pressure_probability(
        favorite_win_pct,
        p_draw,
        underdog_win_pct,
    ).round().astype("Int64")
    clean["favorite_fail_to_win_risk"] = clean["draw_risk"] + clean["upset_win_risk"]
    return clean


def ensure_risk_columns(clean: pd.DataFrame) -> pd.DataFrame:
    if {"team_a_win_pct", "draw_pct", "team_b_win_pct"}.issubset(clean.columns):
        return add_risk_column(clean)
    if {"v2_p_team_a_win", "v2_p_draw", "v2_p_team_b_win"}.issubset(clean.columns):
        return add_detail_risk_columns(clean)
    clean = clean.copy()
    for column in [
        "draw_risk",
        "upset_win_risk",
        "draw_pressure_weight",
        "risk",
        "favorite_fail_to_win_risk",
    ]:
        if column not in clean.columns:
            clean[column] = pd.NA
    return clean


def donor_templates(side: str) -> set[str]:
    if side == "team_a":
        return {"1-0", "2-0", "2-1", "3-0", "3-1"}
    return {"0-1", "0-2", "1-2", "0-3", "1-3"}


def receiver_templates(side: str) -> list[str]:
    if side == "team_a":
        return ["3-1", "4-1", "4-0", "4-2", "5-1", "5-0", "6-1", "6-0"]
    return ["1-3", "1-4", "0-4", "2-4", "1-5", "0-5", "1-6", "0-6"]


def same_favorite_win_region(scoreline: str, side: str) -> bool:
    margin = scoreline_margin(scoreline)
    if margin is None:
        return False
    return margin > 0 if side == "team_a" else margin < 0


def shift_amount(favorite_win_pct: float) -> float:
    if favorite_win_pct >= 85:
        return 8.0
    if favorite_win_pct >= 75:
        return 6.0
    if favorite_win_pct >= 65:
        return 4.0
    if favorite_win_pct >= 55:
        return 2.0
    return 0.0


def receiver_score(scoreline: str, row: pd.Series, side: str, applied_shift: float) -> float:
    total = scoreline_total(scoreline)
    margin = scoreline_margin(scoreline) or 0
    favorite_margin = margin if side == "team_a" else -margin
    favorite_win_pct = float(row["favorite_win_pct"])
    predicted_mean_goals = float(row["predicted_mean_goals"])

    score = applied_shift
    if total == 4:
        score += 0.5
    if total == 5 and favorite_win_pct >= 70:
        score += 0.2
    if favorite_margin == 3:
        score += 0.2
    if predicted_mean_goals >= 2.1:
        score += 0.2
    if total >= 6:
        score -= 0.4
    if favorite_margin >= 5:
        score -= 0.2

    weight = 1.0
    if total == 4:
        weight += 0.25
    elif total == 5:
        weight += 0.10 if favorite_win_pct >= 65 else -0.05
    elif total >= 6:
        weight -= 0.25
    if favorite_margin == 3:
        weight += 0.15
    if favorite_margin >= 4:
        weight -= 0.10
    if predicted_mean_goals >= 2.1:
        weight += 0.10
    return max(0.0, score) * max(0.05, weight)


def augment_top5(row: pd.Series) -> tuple[list[str], bool, list[str]]:
    original = parse_list(row["top_5_scorelines"])[:5]
    if len(original) < 5:
        return original, False, []
    side = favorite_side(row)
    favorite_win_pct = max(float(row["team_a_win_pct"]), float(row["team_b_win_pct"]))
    predicted_mean_goals = float(row["predicted_mean_goals"])
    high_goal_count = sum(scoreline_total(scoreline) >= 4 for scoreline in original)
    eligible = favorite_win_pct >= 55 and predicted_mean_goals >= 1.8 and high_goal_count <= 1
    if not eligible:
        return original, False, []

    probabilities = [float(value) for value in parse_list(row["top_5_scoreline_probability_pct"])[:5]]
    while len(probabilities) < len(original):
        probabilities.append(7.0)
    entries = [
        {"scoreline": scoreline, "score": probability, "rank": index + 1}
        for index, (scoreline, probability) in enumerate(zip(original, probabilities))
    ]

    protected_top3 = set(original[:3])
    replaceable = [
        entry
        for entry in entries[3:]
        if entry["scoreline"] in donor_templates(side)
        and same_favorite_win_region(entry["scoreline"], side)
        and scoreline_outcome(entry["scoreline"]) != "draw"
    ]
    if not replaceable:
        return original, False, []

    applied_shift = min(shift_amount(favorite_win_pct), sum(entry["score"] * 0.35 for entry in replaceable))
    if applied_shift <= 0:
        return original, False, []

    candidates = [
        scoreline
        for scoreline in receiver_templates(side)
        if scoreline not in original and scoreline not in protected_top3
    ]
    receiver_entries = [
        {"scoreline": scoreline, "score": receiver_score(scoreline, row, side, applied_shift)}
        for scoreline in candidates
    ]
    receiver_entries = [entry for entry in receiver_entries if entry["score"] > 0]
    if not receiver_entries:
        return original, False, []

    selected = original.copy()
    injected: list[str] = []
    open_slots = sorted(replaceable, key=lambda item: (item["score"], -item["rank"]))
    for receiver in sorted(receiver_entries, key=lambda item: (-item["score"], item["scoreline"])):
        if not open_slots or len(injected) >= 2:
            break
        weakest = open_slots[0]
        if receiver["score"] < weakest["score"] * 0.35:
            continue
        selected[weakest["rank"] - 1] = receiver["scoreline"]
        injected.append(receiver["scoreline"])
        open_slots.pop(0)

    return selected, bool(injected), injected


def load_predicted_mean_goals(clean: pd.DataFrame, detail_path: Path) -> pd.DataFrame:
    clean = clean.copy()
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        if {"match_id", "lambda_a", "lambda_b"}.issubset(detail.columns):
            detail = detail[["match_id", "lambda_a", "lambda_b"]].copy()
            detail["predicted_mean_goals"] = (
                pd.to_numeric(detail["lambda_a"], errors="coerce")
                + pd.to_numeric(detail["lambda_b"], errors="coerce")
            )
            clean = clean.merge(detail[["match_id", "predicted_mean_goals"]], on="match_id", how="left")
    if "predicted_mean_goals" not in clean.columns:
        clean["predicted_mean_goals"] = 2.0
    clean["predicted_mean_goals"] = pd.to_numeric(clean["predicted_mean_goals"], errors="coerce").fillna(2.0)
    clean["favorite_win_pct"] = clean[["team_a_win_pct", "team_b_win_pct"]].max(axis=1)
    return clean


def source_for_apply(input_path: Path, output_path: Path, backup_path: Path) -> Path:
    if input_path.resolve() == output_path.resolve() and backup_path.exists():
        return backup_path
    return input_path


def apply_tail_augmented_scorelines(
    input_path: Path,
    detail_path: Path,
    output_path: Path,
    backup_path: Path,
) -> pd.DataFrame:
    source_path = source_for_apply(input_path, output_path, backup_path)
    if input_path.resolve() == output_path.resolve() and not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(input_path).to_csv(backup_path, index=False)
    clean = pd.read_csv(source_path)
    original_wdl = clean[["match_id", "team_a_win_pct", "draw_pct", "team_b_win_pct"]].copy()
    working = load_predicted_mean_goals(clean, detail_path)
    changed_rows = []
    for index, row in working.iterrows():
        augmented, changed, injected = augment_top5(row)
        if not changed:
            continue
        clean.at[index, "top_5_scorelines"] = str(augmented)
        changed_rows.append(
            {
                "match_id": int(row["match_id"]),
                "team_a": row["team_a"],
                "team_b": row["team_b"],
                "original_top5": row["top_5_scorelines"],
                "tail_augmented_top5": str(augmented),
                "injected_scorelines": str(injected),
            }
        )

    merged = clean.merge(original_wdl, on="match_id", suffixes=("", "_original"))
    for column in ["team_a_win_pct", "draw_pct", "team_b_win_pct"]:
        if not merged[column].equals(merged[f"{column}_original"]):
            raise ValueError(f"W/D/L column changed unexpectedly: {column}")

    clean = ensure_risk_columns(clean)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    audit_path = output_path.with_name(output_path.stem + "__tail_augmented_audit.csv")
    pd.DataFrame(changed_rows).to_csv(audit_path, index=False)
    update_detail_file(output_path, detail_path)
    return pd.DataFrame(changed_rows)


def update_detail_file(clean_path: Path, detail_path: Path) -> None:
    if not detail_path.exists():
        return
    clean = pd.read_csv(clean_path)
    detail = pd.read_csv(detail_path)
    if "match_id" not in detail.columns:
        return
    columns_to_merge = ["match_id", "top_5_scorelines", "top_5_scoreline_probability_pct"]
    for column in [
        "draw_risk",
        "upset_win_risk",
        "draw_pressure_weight",
        "risk",
        "favorite_fail_to_win_risk",
    ]:
        if column in clean.columns:
            columns_to_merge.append(column)
    clean_top5 = clean[columns_to_merge].copy()
    detail = detail.drop(
        columns=[
            column
            for column in [
                "top_5_scorelines",
                "top_5_scoreline_probability_pct",
                "draw_risk",
                "upset_win_risk",
                "draw_pressure_weight",
                "risk",
                "favorite_fail_to_win_risk",
            ]
            if column in detail.columns
        ],
        errors="ignore",
    )
    detail = detail.merge(clean_top5, on="match_id", how="left")
    detail.to_csv(detail_path, index=False)


def write_metadata(metadata_path: Path, input_path: Path, output_path: Path, changed: pd.DataFrame) -> None:
    previous = {}
    if metadata_path.exists():
        try:
            previous = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            previous = {}
    metadata = {
        **previous,
        "scoreline_top5_mode": "tail_augmented_group_stage_round2",
        "tail_augmented_top5_active": True,
        "tail_augmented_scope": "group_stage_round2_only",
        "wdl_probabilities_modified": False,
        "top1_top3_preserved_by_design": True,
        "source_file": str(input_path.relative_to(PROJECT_ROOT)),
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
        "tail_augmented_rows": int(len(changed)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Live 2026 group-stage results are currently more right-tailed than the larger "
            "historical pre-Cup sample. Use the tail-augmented Top5 display for remaining "
            "Round 2 group-stage predictions only; switch back to original Top5 for knockout "
            "unless later diagnostics justify otherwise."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply conservative tail-augmented Top5 scorelines to a live prediction CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    changed = apply_tail_augmented_scorelines(args.input, args.detail, args.output, args.backup)
    write_metadata(args.metadata, args.input, args.output, changed)
    print(f"Saved tail-augmented predictions to {args.output}")
    print(f"Rows changed: {len(changed)}")
    if not changed.empty:
        print(changed.to_string(index=False))


if __name__ == "__main__":
    main()
