from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.evaluate_v2_uncertainty import evaluate_v2_uncertainty  # noqa: E402
from src.evaluation.tune_v2_uncertainty import BEST_CONFIG_PATH, TUNING_RESULTS_PATH  # noqa: E402
from src.models.v2_club_form_adjustment import DEFAULT_CLUB_FORM_FEATURES_PATH  # noqa: E402
from src.models.v2_uncertainty_adjustment import DEFAULT_SQUAD_VALUES_PATH  # noqa: E402
from src.models.v2_superstar_adjustment import DEFAULT_SUPERSTAR_FEATURES_PATH  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "output" / "diagnostics" / "model_performance_summary.csv"


def load_best_config(path: Path = BEST_CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Best V2 uncertainty config not found: {path}. "
            "Run python3 src/evaluation/tune_v2_uncertainty.py first."
        )
    return json.loads(path.read_text())


def summarize_model_performance(
    train_start: str = "2014-01-01",
    test_start: str = "2022-01-01",
    test_end: str | None = None,
    best_config_path: Path = BEST_CONFIG_PATH,
    squad_values_path: Path = DEFAULT_SQUAD_VALUES_PATH,
    superstar_features_path: Path = DEFAULT_SUPERSTAR_FEATURES_PATH,
) -> pd.DataFrame:
    config = load_best_config(best_config_path)
    summary, _ = evaluate_v2_uncertainty(
        train_start=train_start,
        test_start=test_start,
        test_end=test_end,
        squad_values_path=squad_values_path,
        superstar_features_path=superstar_features_path,
        club_form_features_path=DEFAULT_CLUB_FORM_FEATURES_PATH,
        config_path=best_config_path,
    )
    selected = summary.loc[
        summary["model_version"].isin(
            ["v1_default", "v2_uncertainty_superstar_club_form"]
        )
    ].copy()
    selected.loc[
        selected["model_version"].eq("v2_uncertainty_superstar_club_form"),
        "model_status",
    ] = config.get("model_status", "default")
    selected["max_shift"] = pd.NA
    selected["volatility_weight"] = pd.NA
    selected["stability_edge_weight"] = pd.NA
    selected["max_star_shift"] = pd.NA
    selected["star_weight"] = pd.NA
    selected["max_club_form_shift"] = pd.NA
    selected["club_form_weight"] = pd.NA
    v2_mask = selected["model_version"].eq("v2_uncertainty_superstar_club_form")
    selected.loc[v2_mask, "max_shift"] = float(config["max_shift"])
    selected.loc[v2_mask, "volatility_weight"] = float(config["volatility_weight"])
    selected.loc[v2_mask, "stability_edge_weight"] = float(config["stability_edge_weight"])
    selected.loc[v2_mask, "max_star_shift"] = float(config.get("max_star_shift", 0.025))
    selected.loc[v2_mask, "star_weight"] = float(config.get("star_weight", 0.006))
    selected.loc[v2_mask, "max_club_form_shift"] = float(
        config.get("max_club_form_shift", 0.025)
    )
    selected.loc[v2_mask, "club_form_weight"] = float(config.get("club_form_weight", 0.008))
    output = selected.reset_index(drop=True)
    output["improves_log_loss_vs_v1"] = [
        pd.NA,
        output.loc[1, "mean_log_loss"] < output.loc[0, "mean_log_loss"],
    ]
    output["improves_brier_score_vs_v1"] = [
        pd.NA,
        output.loc[1, "mean_brier_score"] < output.loc[0, "mean_brier_score"],
    ]
    output["tuning_results_file"] = str(TUNING_RESULTS_PATH)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CupCast model performance.")
    parser.add_argument("--train-start", default="2014-01-01")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--config", type=Path, default=BEST_CONFIG_PATH)
    parser.add_argument("--squad-values", type=Path, default=DEFAULT_SQUAD_VALUES_PATH)
    parser.add_argument("--superstar-features", type=Path, default=DEFAULT_SUPERSTAR_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    summary = summarize_model_performance(
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
        best_config_path=args.config,
        squad_values_path=args.squad_values,
        superstar_features_path=args.superstar_features,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Saved model performance summary to {args.output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
