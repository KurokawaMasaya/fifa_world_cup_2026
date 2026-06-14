from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
TARGET_DIRS = {
    "live": OUTPUT_DIR / "live",
    "predictions": OUTPUT_DIR / "predictions",
    "simulations": OUTPUT_DIR / "simulations",
    "diagnostics": OUTPUT_DIR / "diagnostics",
    "brackets": OUTPUT_DIR / "brackets",
    "reports": OUTPUT_DIR / "reports",
}

NEVER_MOVE_DIRS = {
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "config",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "tests",
}

MODEL_INPUT_FILES = {
    PROJECT_ROOT / "data" / "processed" / "team_strength_default.csv",
    PROJECT_ROOT / "data" / "processed" / "team_ratings_world_cup_elo.csv",
    PROJECT_ROOT / "data" / "raw" / "matches.csv",
    PROJECT_ROOT / "data" / "raw" / "squad_values.csv",
}

LIVE_NAMES = {
    "worldcup_group_stage_locked_predictions.csv",
    "worldcup_group_stage_actual_results.csv",
    "worldcup_group_stage_live_evaluation.csv",
    "worldcup_group_stage_live_summary.csv",
    "worldcup_group_stage_live_calibration.csv",
}


def ensure_output_dirs() -> None:
    for path in TARGET_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def is_protected(path: Path) -> bool:
    if path in MODEL_INPUT_FILES:
        return True
    return any(is_under(path, protected) for protected in NEVER_MOVE_DIRS)


def target_category(path: Path) -> str | None:
    name = path.name
    lower = name.lower()

    if name in LIVE_NAMES:
        return "live"
    if "prediction" in lower or "predictions" in lower:
        return "predictions"
    if "monte" in lower or "simulation" in lower or lower.startswith("tournament_simulation"):
        return "simulations"
    if "bracket" in lower:
        return "brackets"
    if (
        "diagnostic" in lower
        or "evaluation" in lower
        or "calibration" in lower
        or "tuning" in lower
        or "comparison" in lower
        or "performance" in lower
        or "player_impacted" in lower
        or name == "player_values_standardized.csv"
        or lower.startswith("test_")
    ):
        return "diagnostics"
    if lower.endswith((".md", ".html", ".pdf", ".txt")):
        return "reports"
    return None


def legacy_candidates() -> list[Path]:
    candidates: list[Path] = []
    search_roots = [
        OUTPUT_DIR,
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "live",
        PROJECT_ROOT / "live",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path == OUTPUT_DIR / "START_HERE.md":
                continue
            if is_under(path, OUTPUT_DIR) and path.parent != OUTPUT_DIR:
                # Files already inside an output subfolder are considered
                # organized. Do not flatten nested reports/diagnostics back
                # into top-level output category folders.
                continue
            category = target_category(path)
            if (
                category is not None
                and is_under(path, TARGET_DIRS[category])
            ):
                continue
            if path.name in {".DS_Store", ".gitkeep"}:
                continue
            candidates.append(path)
    return sorted(candidates)


def move_or_skip(path: Path, delete_legacy: bool) -> None:
    if is_protected(path):
        print(f"SKIP protected: {path.relative_to(PROJECT_ROOT)}")
        return

    category = target_category(path)
    if category is None:
        print(f"SKIP no target rule: {path.relative_to(PROJECT_ROOT)}")
        return

    target = TARGET_DIRS[category] / path.name
    if path == target:
        print(f"SKIP already in target: {path.relative_to(PROJECT_ROOT)}")
        return

    if target.exists():
        if delete_legacy:
            path.unlink()
            print(
                "DELETE legacy duplicate: "
                f"{path.relative_to(PROJECT_ROOT)} -> {target.relative_to(PROJECT_ROOT)}"
            )
        else:
            print(
                "SKIP target exists: "
                f"{path.relative_to(PROJECT_ROOT)} -> {target.relative_to(PROJECT_ROOT)}"
            )
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    print(f"MOVE {path.relative_to(PROJECT_ROOT)} -> {target.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move generated legacy outputs into output subfolders.")
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="Delete only legacy duplicates when the target file already exists.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    for path in legacy_candidates():
        move_or_skip(path, delete_legacy=args.delete_legacy)


if __name__ == "__main__":
    main()
