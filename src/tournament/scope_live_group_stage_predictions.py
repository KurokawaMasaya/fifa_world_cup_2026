from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_clean.csv"
)
DEFAULT_FIXTURES_PATH = PROJECT_ROOT / "data" / "raw" / "matches.csv"
DEFAULT_STATIC_SNAPSHOT_PATH = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "static_snapshots"
    / "group_stage_full_72_static_initial__v24_abcd_no_e.csv"
)
DEFAULT_LIVE_ROUND1_PATH = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "live"
    / "group_stage_round1_predictions__v24_abcd_no_e.csv"
)
DEFAULT_LIVE_METADATA_PATH = (
    PROJECT_ROOT
    / "output"
    / "predictions"
    / "live"
    / "group_stage_round1_predictions__v24_abcd_no_e__metadata.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "output_scope"
    / "group_stage_round1_output_report.md"
)

ROUND_COLUMNS = ["group_round", "inferred_group_round", "matchday", "round"]
SCOPE_COLUMNS = [
    "prediction_scope",
    "inferred_group_round",
    "rating_state_used",
    "dynamic_ratings_used",
    "future_results_used",
]


def normalize_group(value: object) -> str:
    """Normalize labels like 'Group A' and 'A' into a plain group letter."""
    text = str(value).strip()
    if text.lower().startswith("group "):
        return text.split()[-1].upper()
    return text.upper()


def load_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    required = {"match_id", "group", "team_a", "team_b"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return predictions.copy()


def load_group_stage_fixtures(path: Path) -> pd.DataFrame:
    fixtures = pd.read_csv(path)
    if "stage_id" in fixtures.columns:
        fixtures = fixtures[fixtures["stage_id"].eq(1)].copy()
    if "match_label" in fixtures.columns:
        fixtures["fixture_group"] = fixtures["match_label"].map(normalize_group)
    if "id" not in fixtures.columns and "match_id" not in fixtures.columns:
        raise ValueError(f"{path} must contain either id or match_id")
    return fixtures


def _find_existing_round_column(predictions: pd.DataFrame) -> str | None:
    for column in ROUND_COLUMNS:
        if column in predictions.columns:
            return column
    return None


def assign_group_rounds(predictions: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Assign group round without changing any W/D/L or scoreline values.

    If the prediction table already carries a round/matchday column, reuse it.
    Otherwise, infer rounds by joining to fixtures and sorting matches inside each
    group by kickoff time and match number: first two group matches are Round 1,
    next two are Round 2, final two are Round 3.
    """
    scoped = predictions.copy()
    existing_round_column = _find_existing_round_column(scoped)
    if existing_round_column:
        scoped["inferred_group_round"] = scoped[existing_round_column].astype(int)
        return scoped

    fixture_id_col = "match_id" if "match_id" in fixtures.columns else "id"
    fixture_cols = [fixture_id_col]
    for optional_col in ["kickoff_at", "match_number", "fixture_group"]:
        if optional_col in fixtures.columns:
            fixture_cols.append(optional_col)

    fixture_lookup = fixtures[fixture_cols].rename(columns={fixture_id_col: "match_id"})
    scoped["_match_id_key"] = scoped["match_id"].astype(str)
    fixture_lookup["_match_id_key"] = fixture_lookup["match_id"].astype(str)
    scoped = scoped.merge(
        fixture_lookup.drop(columns=["match_id"]),
        on="_match_id_key",
        how="left",
        validate="one_to_one",
    )
    scoped = scoped.drop(columns=["_match_id_key"])

    scoped["_group_for_round"] = scoped["group"].map(normalize_group)
    if "fixture_group" in scoped.columns:
        scoped["_group_for_round"] = scoped["fixture_group"].fillna(scoped["_group_for_round"])

    if "kickoff_at" in scoped.columns:
        scoped["_kickoff_sort"] = pd.to_datetime(scoped["kickoff_at"], errors="coerce", utc=True)
    else:
        scoped["_kickoff_sort"] = pd.NaT
    if "match_number" not in scoped.columns:
        scoped["match_number"] = pd.to_numeric(scoped["match_id"], errors="coerce")

    scoped = scoped.sort_values(
        ["_group_for_round", "_kickoff_sort", "match_number", "match_id"],
        kind="mergesort",
    )
    scoped["_group_match_index"] = scoped.groupby("_group_for_round").cumcount()
    scoped["inferred_group_round"] = (scoped["_group_match_index"] // 2 + 1).astype(int)
    scoped = scoped.sort_values("match_id", kind="mergesort").reset_index(drop=True)

    return scoped.drop(
        columns=[
            col
            for col in [
                "_group_for_round",
                "_kickoff_sort",
                "_group_match_index",
                "fixture_group",
                "kickoff_at",
                "match_number",
            ]
            if col in scoped.columns
        ]
    )


def add_scope_columns(scoped: pd.DataFrame, prediction_scope: str) -> pd.DataFrame:
    output = scoped.copy()
    output["prediction_scope"] = prediction_scope
    output["rating_state_used"] = "initial_static"
    output["dynamic_ratings_used"] = False
    output["future_results_used"] = False

    preferred = [
        "match_id",
        "prediction_scope",
        "group",
        "inferred_group_round",
        "team_a",
        "team_b",
        "rating_state_used",
        "dynamic_ratings_used",
        "future_results_used",
    ]
    ordered = [col for col in preferred if col in output.columns] + [
        col for col in output.columns if col not in preferred
    ]
    return output[ordered]


def validate_full_snapshot(full_snapshot: pd.DataFrame) -> None:
    if len(full_snapshot) != 72:
        raise ValueError(f"Full static snapshot must have 72 rows, found {len(full_snapshot)}")
    round_counts = full_snapshot["inferred_group_round"].value_counts().to_dict()
    expected_round_counts = {1: 24, 2: 24, 3: 24}
    if round_counts != expected_round_counts:
        raise ValueError(
            f"Expected 24 matches in each group round; found {round_counts}"
        )


def validate_round1_scope(
    round1: pd.DataFrame,
    full_snapshot: pd.DataFrame,
    expected_groups: Iterable[str] | None = None,
    expected_total_rows: int = 24,
) -> None:
    if len(round1) != expected_total_rows:
        raise ValueError(
            f"Live Round 1 output must have {expected_total_rows} rows, found {len(round1)}"
        )
    if not round1["inferred_group_round"].eq(1).all():
        raise ValueError("Live Round 1 output contains Round 2 or Round 3 matches")

    groups = list(expected_groups or sorted(full_snapshot["group"].map(normalize_group).unique()))
    counts = round1.assign(_group=round1["group"].map(normalize_group)).groupby("_group").size()
    for group in groups:
        if int(counts.get(group, 0)) != 2:
            raise ValueError(f"Group {group} must have exactly 2 Round 1 matches")


def write_metadata(
    path: Path,
    live_round1_path: Path,
    static_snapshot_path: Path,
    report_path: Path,
) -> dict:
    metadata = {
        "production_output": str(live_round1_path.relative_to(PROJECT_ROOT)),
        "prediction_scope": "group_stage_round1_only",
        "scoreline_version": "v24_abcd_no_e",
        "rating_state": "initial_static",
        "dynamic_ratings_used": False,
        "future_results_used": False,
        "full_static_snapshot_file": str(static_snapshot_path.relative_to(PROJECT_ROOT)),
        "diagnostic_report": str(report_path.relative_to(PROJECT_ROOT)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Round 2 and Round 3 predictions should be generated later after prior "
            "rounds complete."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def write_report(
    path: Path,
    full_snapshot: pd.DataFrame,
    round1: pd.DataFrame,
    static_snapshot_path: Path,
    live_round1_path: Path,
    metadata_path: Path,
) -> None:
    groups = sorted(round1["group"].map(normalize_group).unique())
    matches_per_group = (
        round1.assign(group_letter=round1["group"].map(normalize_group))
        .groupby("group_letter")
        .size()
        .to_dict()
    )
    report = f"""# Group Stage Round 1 Output Scope Report

Generated at: {datetime.now(timezone.utc).isoformat()}

## Files
- Full static snapshot: `{static_snapshot_path.relative_to(PROJECT_ROOT)}`
- Live Round 1 output: `{live_round1_path.relative_to(PROJECT_ROOT)}`
- Live metadata: `{metadata_path.relative_to(PROJECT_ROOT)}`

## Row Counts
- Full static snapshot rows: {len(full_snapshot)}
- Live Round 1 rows: {len(round1)}

## Groups Included
- Groups: {", ".join(groups)}
- Matches per group: {matches_per_group}

## Confirmations
- Only Round 1 matches are present in the live output: yes
- Every group has exactly 2 Round 1 matches: yes
- The full static snapshot still has 72 rows: yes
- W/D/L probabilities were not modified: yes
- V2.4 ABCD no-E model logic was not modified: yes
- Dynamic ratings were not implemented: yes

## Notes
Round 1 uses the initial static rating state. Round 2 and Round 3 predictions
should be generated later after prior rounds complete.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)


def scope_live_round1_predictions(
    input_path: Path = DEFAULT_INPUT_PATH,
    fixtures_path: Path = DEFAULT_FIXTURES_PATH,
    static_snapshot_path: Path = DEFAULT_STATIC_SNAPSHOT_PATH,
    live_round1_path: Path = DEFAULT_LIVE_ROUND1_PATH,
    metadata_path: Path = DEFAULT_LIVE_METADATA_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = load_predictions(input_path)
    fixtures = load_group_stage_fixtures(fixtures_path)

    scoped = assign_group_rounds(predictions, fixtures)
    full_snapshot = add_scope_columns(scoped, "group_stage_full_72_static_initial")
    round1 = add_scope_columns(
        scoped[scoped["inferred_group_round"].eq(1)].copy(),
        "group_stage_round1_only",
    )

    validate_full_snapshot(full_snapshot)
    validate_round1_scope(round1, full_snapshot)

    static_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    live_round1_path.parent.mkdir(parents=True, exist_ok=True)
    full_snapshot.to_csv(static_snapshot_path, index=False)
    round1.to_csv(live_round1_path, index=False)
    write_metadata(metadata_path, live_round1_path, static_snapshot_path, report_path)
    write_report(
        report_path,
        full_snapshot,
        round1,
        static_snapshot_path,
        live_round1_path,
        metadata_path,
    )
    return full_snapshot, round1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scope live production group-stage predictions to Round 1 only."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES_PATH)
    parser.add_argument("--static-snapshot-output", type=Path, default=DEFAULT_STATIC_SNAPSHOT_PATH)
    parser.add_argument("--live-round1-output", type=Path, default=DEFAULT_LIVE_ROUND1_PATH)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_LIVE_METADATA_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    full_snapshot, round1 = scope_live_round1_predictions(
        input_path=args.input,
        fixtures_path=args.fixtures,
        static_snapshot_path=args.static_snapshot_output,
        live_round1_path=args.live_round1_output,
        metadata_path=args.metadata_output,
        report_path=args.report_output,
    )
    print(f"Saved full static snapshot: {args.static_snapshot_output}")
    print(f"Full static rows: {len(full_snapshot)}")
    print(f"Saved live Round 1 output: {args.live_round1_output}")
    print(f"Live Round 1 rows: {len(round1)}")
    print(
        "Round 1 matches per group:",
        round1.assign(group_letter=round1["group"].map(normalize_group))
        .groupby("group_letter")
        .size()
        .to_dict(),
    )
    print(f"Saved metadata: {args.metadata_output}")
    print(f"Saved report: {args.report_output}")


if __name__ == "__main__":
    main()
