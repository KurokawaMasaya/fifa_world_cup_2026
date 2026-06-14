from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_v2_uncertainty_tuned.csv"
FALLBACK_CLEAN_PREDICTIONS_PATH = PROJECT_ROOT / "output" / "predictions" / "group_stage_predictions_clean.csv"
DEFAULT_BACKTEST_PATH = (
    PROJECT_ROOT
    / "output"
    / "diagnostics"
    / "v2_research"
    / "v2_uncertainty_match_level_evaluation.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "diagnostics" / "wdl_audit"
EPSILON = 1e-15
PROBABILITY_BINS = [i / 10 for i in range(11)]
PROBABILITY_BIN_LABELS = [f"{i/10:.2f}-{(i+1)/10:.2f}" for i in range(10)]


def load_group_stage_predictions(path: Path | None = None) -> tuple[pd.DataFrame, Path, str]:
    """Load existing group-stage W/D/L outputs without regenerating predictions."""
    source_path = path or DEFAULT_PREDICTIONS_PATH
    if source_path.exists():
        df = pd.read_csv(source_path)
        if {"v2_p_team_a_win", "v2_p_draw", "v2_p_team_b_win"}.issubset(df.columns):
            out = df.copy()
            out["p_team_a_win"] = pd.to_numeric(out["v2_p_team_a_win"], errors="coerce")
            out["p_draw"] = pd.to_numeric(out["v2_p_draw"], errors="coerce")
            out["p_team_b_win"] = pd.to_numeric(out["v2_p_team_b_win"], errors="coerce")
            return out, source_path, "probability"
    if FALLBACK_CLEAN_PREDICTIONS_PATH.exists():
        df = pd.read_csv(FALLBACK_CLEAN_PREDICTIONS_PATH)
        out = df.copy()
        out["p_team_a_win"] = pd.to_numeric(out["team_a_win_pct"], errors="coerce") / 100.0
        out["p_draw"] = pd.to_numeric(out["draw_pct"], errors="coerce") / 100.0
        out["p_team_b_win"] = pd.to_numeric(out["team_b_win_pct"], errors="coerce") / 100.0
        return out, FALLBACK_CLEAN_PREDICTIONS_PATH, "percent_rounded"
    raise FileNotFoundError(f"Could not find {source_path} or {FALLBACK_CLEAN_PREDICTIONS_PATH}")


def load_backtest(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {
        "team_a",
        "team_b",
        "actual_result",
        "v2_full_p_team_a_win",
        "v2_full_p_draw",
        "v2_full_p_team_b_win",
    }
    if not required.issubset(df.columns):
        return None
    out = df.copy()
    out["p_team_a_win"] = pd.to_numeric(out["v2_full_p_team_a_win"], errors="coerce")
    out["p_draw"] = pd.to_numeric(out["v2_full_p_draw"], errors="coerce")
    out["p_team_b_win"] = pd.to_numeric(out["v2_full_p_team_b_win"], errors="coerce")
    return out


def enrich_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["p_team_a_win", "p_draw", "p_team_b_win"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["probability_sum"] = out[["p_team_a_win", "p_draw", "p_team_b_win"]].sum(axis=1)
    out["favorite_side"] = out.apply(
        lambda row: "team_a" if row["p_team_a_win"] >= row["p_team_b_win"] else "team_b",
        axis=1,
    )
    out["favorite_team"] = out.apply(
        lambda row: row.get("team_a", row.get("home_team", "team_a"))
        if row["favorite_side"] == "team_a"
        else row.get("team_b", row.get("away_team", "team_b")),
        axis=1,
    )
    out["favorite_win_prob"] = out[["p_team_a_win", "p_team_b_win"]].max(axis=1)
    out["underdog_win_prob"] = out[["p_team_a_win", "p_team_b_win"]].min(axis=1)
    out["favorite_gap"] = out["favorite_win_prob"] - out[["p_draw", "underdog_win_prob"]].max(axis=1)
    out["lambda_gap"] = (
        pd.to_numeric(out.get("lambda_a", pd.Series(index=out.index, dtype=float)), errors="coerce")
        - pd.to_numeric(out.get("lambda_b", pd.Series(index=out.index, dtype=float)), errors="coerce")
    ).abs()
    out["abs_strength_diff"] = pd.to_numeric(
        out.get("strength_diff", pd.Series(index=out.index, dtype=float)),
        errors="coerce",
    ).abs()
    return out


def favorite_bucket(favorite_win_prob: float, favorite_gap: float) -> str:
    if pd.isna(favorite_win_prob):
        return "unknown"
    if favorite_win_prob < 0.40 or favorite_gap < 0.05:
        return "balanced"
    if favorite_win_prob < 0.50:
        return "slight_favorite"
    if favorite_win_prob < 0.60:
        return "moderate_favorite"
    if favorite_win_prob < 0.70:
        return "strong_favorite"
    if favorite_win_prob < 0.82:
        return "heavy_favorite"
    return "extreme_favorite"


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["favorite_bucket"] = out.apply(
        lambda row: favorite_bucket(row["favorite_win_prob"], row["favorite_gap"]),
        axis=1,
    )
    out["rating_gap_bucket"] = pd.cut(
        out["abs_strength_diff"],
        bins=[-0.001, 50, 100, 200, 350, 10_000],
        labels=["0-50", "50-100", "100-200", "200-350", "350+"],
    ).astype(str)
    out.loc[out["abs_strength_diff"].isna(), "rating_gap_bucket"] = "not_available"
    out["lambda_gap_bucket"] = pd.cut(
        out["lambda_gap"],
        bins=[-0.001, 0.10, 0.25, 0.50, 0.90, 10_000],
        labels=["0-0.10", "0.10-0.25", "0.25-0.50", "0.50-0.90", "0.90+"],
    ).astype(str)
    out.loc[out["lambda_gap"].isna(), "lambda_gap_bucket"] = "not_available"
    return out


def probability_validity(df: pd.DataFrame, source_file: Path, probability_scale: str) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        values = [row["p_team_a_win"], row["p_draw"], row["p_team_b_win"]]
        issues = []
        if any(pd.isna(value) for value in values):
            issues.append("nan_probability")
        if any((not pd.isna(value)) and value < 0 for value in values):
            issues.append("negative_probability")
        if any((not pd.isna(value)) and value > 1 for value in values):
            issues.append("probability_above_1")
        tolerance = 0.0100001 if probability_scale == "percent_rounded" else 1e-6
        if not pd.isna(row["probability_sum"]) and abs(float(row["probability_sum"]) - 1.0) > tolerance:
            issues.append("sum_not_1")
        rows.append(
            {
                "row_number": idx + 1,
                "match_id": row.get("match_id", ""),
                "team_a": row.get("team_a", row.get("home_team", "")),
                "team_b": row.get("team_b", row.get("away_team", "")),
                "p_team_a_win": row["p_team_a_win"],
                "p_draw": row["p_draw"],
                "p_team_b_win": row["p_team_b_win"],
                "probability_sum": row["probability_sum"],
                "source_file": str(source_file),
                "probability_scale": probability_scale,
                "is_valid": len(issues) == 0,
                "issues": ";".join(issues) if issues else "none",
            }
        )
    return pd.DataFrame(rows)


def distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "p_team_a_win",
        "p_draw",
        "p_team_b_win",
        "favorite_win_prob",
        "underdog_win_prob",
        "favorite_gap",
    ]
    for metric in metrics:
        series = pd.to_numeric(df[metric], errors="coerce")
        quantiles = series.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        rows.append(
            {
                "metric": metric,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p01": quantiles.loc[0.01],
                "p05": quantiles.loc[0.05],
                "p10": quantiles.loc[0.10],
                "p25": quantiles.loc[0.25],
                "median": quantiles.loc[0.50],
                "p75": quantiles.loc[0.75],
                "p90": quantiles.loc[0.90],
                "p95": quantiles.loc[0.95],
                "p99": quantiles.loc[0.99],
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


def by_favorite_bucket(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bucket_order = [
        "balanced",
        "slight_favorite",
        "moderate_favorite",
        "strong_favorite",
        "heavy_favorite",
        "extreme_favorite",
        "unknown",
    ]
    for bucket in bucket_order:
        subset = df.loc[df["favorite_bucket"].eq(bucket)]
        if subset.empty:
            continue
        rows.append(
            {
                "favorite_bucket": bucket,
                "match_count": len(subset),
                "avg_favorite_win_prob": subset["favorite_win_prob"].mean(),
                "avg_draw_pct": subset["p_draw"].mean(),
                "avg_underdog_win_prob": subset["underdog_win_prob"].mean(),
                "avg_favorite_gap": subset["favorite_gap"].mean(),
                "min_favorite_win_prob": subset["favorite_win_prob"].min(),
                "max_favorite_win_prob": subset["favorite_win_prob"].max(),
            }
        )
    return pd.DataFrame(rows)


def draw_behavior(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col, label in [
        ("favorite_bucket", "favorite_bucket"),
        ("rating_gap_bucket", "rating_gap_bucket"),
        ("lambda_gap_bucket", "lambda_gap_bucket"),
    ]:
        for bucket, subset in df.groupby(group_col, sort=False, dropna=False):
            rows.append(
                {
                    "group_type": label,
                    "bucket": bucket,
                    "match_count": len(subset),
                    "avg_draw_pct": subset["p_draw"].mean(),
                    "min_draw_pct": subset["p_draw"].min(),
                    "max_draw_pct": subset["p_draw"].max(),
                    "avg_favorite_win_prob": subset["favorite_win_prob"].mean(),
                    "avg_favorite_gap": subset["favorite_gap"].mean(),
                    "avg_abs_strength_diff": subset["abs_strength_diff"].mean(),
                    "avg_lambda_gap": subset["lambda_gap"].mean(),
                }
            )
    return pd.DataFrame(rows)


def scoreline_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    has_full_grid = False
    has_top_json = "nb_top_scorelines_json" in df.columns
    for _, row in df.iterrows():
        output = {
            "match_id": row.get("match_id", ""),
            "team_a": row.get("team_a", ""),
            "team_b": row.get("team_b", ""),
            "p_team_a_win": row["p_team_a_win"],
            "p_draw": row["p_draw"],
            "p_team_b_win": row["p_team_b_win"],
            "scoreline_implied_team_a_win_pct": pd.NA,
            "scoreline_implied_draw_pct": pd.NA,
            "scoreline_implied_team_b_win_pct": pd.NA,
            "diff_team_a_win_pct": pd.NA,
            "diff_draw_pct": pd.NA,
            "diff_team_b_win_pct": pd.NA,
            "max_abs_diff": pd.NA,
            "mean_abs_diff": pd.NA,
            "coverage_scoreline_probability_mass": pd.NA,
            "status": "full_scoreline_grid_not_available",
            "note": "Existing prediction output contains top scorelines only, not a full scoreline probability grid.",
        }
        if has_full_grid:
            pass
        elif has_top_json and isinstance(row.get("nb_top_scorelines_json"), str):
            try:
                items = json.loads(row["nb_top_scorelines_json"])
                mass = {"team_a_win": 0.0, "draw": 0.0, "team_b_win": 0.0}
                for item in items:
                    mass[str(item["implied_result"])] += float(item["probability"])
                output.update(
                    {
                        "scoreline_implied_team_a_win_pct": mass["team_a_win"],
                        "scoreline_implied_draw_pct": mass["draw"],
                        "scoreline_implied_team_b_win_pct": mass["team_b_win"],
                        "coverage_scoreline_probability_mass": sum(mass.values()),
                        "status": "partial_top_scorelines_only",
                        "note": "Only top displayed scorelines were available; differences are not full-grid consistency errors.",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic should not crash on one row
                output["status"] = "parse_error"
                output["note"] = str(exc)
        rows.append(output)
    return pd.DataFrame(rows)


def actual_probability(row: pd.Series, prefix: str = "") -> float:
    actual = row["actual_result"]
    if actual == "team_a_win":
        return row[f"{prefix}p_team_a_win"]
    if actual == "team_b_win":
        return row[f"{prefix}p_team_b_win"]
    return row[f"{prefix}p_draw"]


def brier_score(row: pd.Series) -> float:
    return sum(
        (row[col] - (1.0 if outcome == row["actual_result"] else 0.0)) ** 2
        for col, outcome in [
            ("p_team_a_win", "team_a_win"),
            ("p_draw", "draw"),
            ("p_team_b_win", "team_b_win"),
        ]
    )


def log_loss(row: pd.Series) -> float:
    return -math.log(max(EPSILON, min(1.0, actual_probability(row))))


def backtest_calibration(backtest: pd.DataFrame | None) -> pd.DataFrame:
    if backtest is None or backtest.empty:
        return pd.DataFrame([{"section": "overall", "status": "historical_backtest_not_available"}])
    df = enrich_probabilities(backtest)
    df = add_buckets(df)
    df["predicted_outcome"] = df[["p_team_a_win", "p_draw", "p_team_b_win"]].idxmax(axis=1).map(
        {
            "p_team_a_win": "team_a_win",
            "p_draw": "draw",
            "p_team_b_win": "team_b_win",
        }
    )
    df["favorite_actual_win"] = df.apply(
        lambda row: row["actual_result"] == ("team_a_win" if row["favorite_side"] == "team_a" else "team_b_win"),
        axis=1,
    )
    df["underdog_actual_win"] = df.apply(
        lambda row: row["actual_result"] == ("team_b_win" if row["favorite_side"] == "team_a" else "team_a_win"),
        axis=1,
    )
    df["draw_actual"] = df["actual_result"].eq("draw")
    df["brier_score"] = df.apply(brier_score, axis=1)
    df["log_loss"] = df.apply(log_loss, axis=1)

    rows = [
        {
            "section": "overall",
            "bin": "all",
            "match_count": len(df),
            "accuracy": df["predicted_outcome"].eq(df["actual_result"]).mean(),
            "brier_score": df["brier_score"].mean(),
            "multiclass_log_loss": df["log_loss"].mean(),
            "avg_actual_outcome_probability": df.apply(actual_probability, axis=1).mean(),
            "avg_predicted_favorite_win_prob": df["favorite_win_prob"].mean(),
            "actual_favorite_win_rate": df["favorite_actual_win"].mean(),
            "avg_predicted_draw_prob": df["p_draw"].mean(),
            "actual_draw_rate": df["draw_actual"].mean(),
            "avg_predicted_underdog_win_prob": df["underdog_win_prob"].mean(),
            "actual_underdog_win_rate": df["underdog_actual_win"].mean(),
            "calibration_error": (
                abs(df["favorite_win_prob"].mean() - df["favorite_actual_win"].mean())
                + abs(df["p_draw"].mean() - df["draw_actual"].mean())
                + abs(df["underdog_win_prob"].mean() - df["underdog_actual_win"].mean())
            )
            / 3.0,
        }
    ]
    for metric, section, actual_col in [
        ("favorite_win_prob", "favorite_win_probability_bin", "favorite_actual_win"),
        ("p_draw", "draw_probability_bin", "draw_actual"),
        ("underdog_win_prob", "underdog_win_probability_bin", "underdog_actual_win"),
    ]:
        binned = pd.cut(df[metric], PROBABILITY_BINS, labels=PROBABILITY_BIN_LABELS, include_lowest=True)
        for bin_label, subset in df.groupby(binned, observed=False):
            if subset.empty:
                continue
            rows.append(
                {
                    "section": section,
                    "bin": str(bin_label),
                    "match_count": len(subset),
                    "accuracy": subset["predicted_outcome"].eq(subset["actual_result"]).mean(),
                    "brier_score": subset["brier_score"].mean(),
                    "multiclass_log_loss": subset["log_loss"].mean(),
                    "avg_actual_outcome_probability": subset.apply(actual_probability, axis=1).mean(),
                    "avg_predicted_favorite_win_prob": subset["favorite_win_prob"].mean(),
                    "actual_favorite_win_rate": subset["favorite_actual_win"].mean(),
                    "avg_predicted_draw_prob": subset["p_draw"].mean(),
                    "actual_draw_rate": subset["draw_actual"].mean(),
                    "avg_predicted_underdog_win_prob": subset["underdog_win_prob"].mean(),
                    "actual_underdog_win_rate": subset["underdog_actual_win"].mean(),
                    "calibration_error": abs(subset[metric].mean() - subset[actual_col].mean()),
                }
            )
    return pd.DataFrame(rows)


def team_bias(backtest: pd.DataFrame | None) -> pd.DataFrame:
    if backtest is None or backtest.empty:
        return pd.DataFrame([{"team": "not_available", "status": "historical_backtest_not_available"}])
    rows = []
    for _, row in backtest.iterrows():
        for side, team_col, win_col, loss_col in [
            ("team_a", "team_a", "p_team_a_win", "p_team_b_win"),
            ("team_b", "team_b", "p_team_b_win", "p_team_a_win"),
        ]:
            actual = row["actual_result"]
            win = actual == f"{side}_win"
            draw = actual == "draw"
            points = 3 if win else 1 if draw else 0
            rows.append(
                {
                    "team": row[team_col],
                    "predicted_win_prob": row[win_col],
                    "predicted_draw_prob": row["p_draw"],
                    "predicted_points": 3 * row[win_col] + row["p_draw"],
                    "actual_win": win,
                    "actual_draw": draw,
                    "actual_points": points,
                }
            )
    long = pd.DataFrame(rows)
    grouped = long.groupby("team", as_index=False).agg(
        matches=("team", "size"),
        avg_predicted_win_prob=("predicted_win_prob", "mean"),
        actual_win_rate=("actual_win", "mean"),
        avg_predicted_draw_prob=("predicted_draw_prob", "mean"),
        actual_draw_rate=("actual_draw", "mean"),
        avg_predicted_points=("predicted_points", "mean"),
        actual_points_per_match=("actual_points", "mean"),
    )
    grouped["calibration_error"] = (
        (grouped["avg_predicted_win_prob"] - grouped["actual_win_rate"]).abs()
        + (grouped["avg_predicted_draw_prob"] - grouped["actual_draw_rate"]).abs()
        + ((grouped["avg_predicted_points"] - grouped["actual_points_per_match"]) / 3.0).abs()
    ) / 3.0
    return grouped.sort_values(["calibration_error", "matches"], ascending=[False, False])


def extreme_cases(df: pd.DataFrame, consistency: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_cases(label: str, subset: pd.DataFrame, sort_col: str, ascending: bool = False, n: int = 10) -> None:
        for _, row in subset.sort_values(sort_col, ascending=ascending).head(n).iterrows():
            rows.append(
                {
                    "case_type": label,
                    "match_id": row.get("match_id", ""),
                    "group": row.get("group", ""),
                    "team_a": row.get("team_a", ""),
                    "team_b": row.get("team_b", ""),
                    "p_team_a_win": row["p_team_a_win"],
                    "p_draw": row["p_draw"],
                    "p_team_b_win": row["p_team_b_win"],
                    "favorite_team": row["favorite_team"],
                    "favorite_win_prob": row["favorite_win_prob"],
                    "underdog_win_prob": row["underdog_win_prob"],
                    "favorite_gap": row["favorite_gap"],
                    "favorite_bucket": row["favorite_bucket"],
                    "strength_diff": row.get("strength_diff", pd.NA),
                    "lambda_a": row.get("lambda_a", pd.NA),
                    "lambda_b": row.get("lambda_b", pd.NA),
                }
            )

    add_cases("favorite_win_prob_very_high", df.loc[df["favorite_win_prob"] >= df["favorite_win_prob"].quantile(0.95)], "favorite_win_prob")
    add_cases("draw_pct_unusually_high", df.loc[df["p_draw"] >= df["p_draw"].quantile(0.95)], "p_draw")
    add_cases("draw_pct_unusually_low", df.loc[df["p_draw"] <= df["p_draw"].quantile(0.05)], "p_draw", ascending=True)
    add_cases("favorite_gap_very_large", df.loc[df["favorite_gap"] >= df["favorite_gap"].quantile(0.95)], "favorite_gap")
    add_cases(
        "balanced_match_suspiciously_low_draw",
        df.loc[df["favorite_bucket"].eq("balanced") & (df["p_draw"] <= df["p_draw"].quantile(0.10))],
        "p_draw",
        ascending=True,
    )
    add_cases(
        "heavy_favorite_suspiciously_high_draw",
        df.loc[df["favorite_bucket"].isin(["heavy_favorite", "extreme_favorite"]) & (df["p_draw"] >= df["p_draw"].quantile(0.90))],
        "p_draw",
    )

    if not consistency.empty and "max_abs_diff" in consistency.columns:
        candidate = consistency.dropna(subset=["max_abs_diff"])
        if not candidate.empty:
            for _, row in candidate.sort_values("max_abs_diff", ascending=False).head(10).iterrows():
                rows.append(
                    {
                        "case_type": "wdl_scoreline_implied_diff_largest",
                        "match_id": row.get("match_id", ""),
                        "group": "",
                        "team_a": row.get("team_a", ""),
                        "team_b": row.get("team_b", ""),
                        "p_team_a_win": row.get("p_team_a_win", pd.NA),
                        "p_draw": row.get("p_draw", pd.NA),
                        "p_team_b_win": row.get("p_team_b_win", pd.NA),
                        "favorite_team": "",
                        "favorite_win_prob": pd.NA,
                        "underdog_win_prob": pd.NA,
                        "favorite_gap": pd.NA,
                        "favorite_bucket": "",
                        "strength_diff": pd.NA,
                        "lambda_a": pd.NA,
                        "lambda_b": pd.NA,
                    }
                )
    return pd.DataFrame(rows).drop_duplicates()


def write_report(
    output_dir: Path,
    validity: pd.DataFrame,
    dist: pd.DataFrame,
    buckets: pd.DataFrame,
    draw: pd.DataFrame,
    consistency: pd.DataFrame,
    calibration: pd.DataFrame,
    team_bias_df: pd.DataFrame,
    source_file: Path,
    backtest_path: Path,
) -> None:
    valid_count = int(validity["is_valid"].sum()) if "is_valid" in validity.columns else 0
    invalid_count = int((~validity["is_valid"]).sum()) if "is_valid" in validity.columns else 0
    mean_draw = float(dist.loc[dist["metric"].eq("p_draw"), "mean"].iloc[0])
    mean_favorite = float(dist.loc[dist["metric"].eq("favorite_win_prob"), "mean"].iloc[0])
    overall = calibration.loc[calibration["section"].eq("overall")].head(1)
    backtest_line = "Historical backtest not available."
    if not overall.empty and "accuracy" in overall.columns:
        row = overall.iloc[0]
        backtest_line = (
            f"Backtest accuracy {row['accuracy']:.3f}, Brier {row['brier_score']:.3f}, "
            f"log loss {row['multiclass_log_loss']:.3f}, actual outcome probability "
            f"{row['avg_actual_outcome_probability']:.3f}."
        )
    consistency_status = consistency["status"].value_counts().to_dict() if "status" in consistency.columns else {}
    high_bias = team_bias_df.head(10) if "calibration_error" in team_bias_df.columns else pd.DataFrame()
    report = [
        "# W/D/L Model Read-Only Audit",
        "",
        "This audit reads existing prediction/evaluation outputs only. It does not modify model code, ratings, probabilities, scoreline logic, or production prediction files.",
        "",
        "## Sources",
        "",
        f"- Group-stage W/D/L source: `{source_file}`",
        f"- Historical calibration source: `{backtest_path}`",
        "",
        "## Probability Validity",
        "",
        f"- Valid rows: {valid_count}",
        f"- Invalid rows: {invalid_count}",
        "",
        "## Overall Distribution",
        "",
        f"- Mean draw probability: {mean_draw:.3f}",
        f"- Mean favorite win probability: {mean_favorite:.3f}",
        "",
        "## Historical Backtest",
        "",
        backtest_line,
        "",
        "## Draw Behavior",
        "",
        "Draw behavior by favorite/rating/lambda bucket is saved in `wdl_draw_behavior.csv`. Review monotonicity there; this report does not apply parameter changes.",
        "",
        "## Scoreline Consistency",
        "",
        f"Scoreline consistency status counts: `{consistency_status}`",
        "",
        "Full scoreline grids are not present in the current prediction CSV. The consistency file therefore marks rows as partial top-scoreline diagnostics rather than full-grid W/D/L consistency errors.",
        "",
        "## Team-Level Bias",
        "",
        "Largest team-level calibration errors are saved in `wdl_team_bias_audit.csv`.",
    ]
    if not high_bias.empty:
        report.append("")
        report.append(high_bias[["team", "matches", "calibration_error"]].to_markdown(index=False) if _has_tabulate() else high_bias[["team", "matches", "calibration_error"]].to_string(index=False))
    report.extend(
        [
            "",
            "## Structural Issues Worth Investigating",
            "",
            "- Draw is never the argmax pick in existing historical summaries, even though mean draw probability is meaningful. This may be acceptable for argmax classification but is important for presentation.",
            "- Full scoreline-grid consistency cannot be audited from current prediction files because full grids are not saved.",
            "- Extreme cases are listed in `wdl_extreme_cases.csv` for manual review.",
            "",
            "## Confirmation",
            "",
            "- No W/D/L model files were modified.",
            "- No prediction probabilities were modified.",
            "- No scoreline logic was modified.",
            "- No production prediction files were overwritten.",
        ]
    )
    (output_dir / "wdl_audit_report.md").write_text("\n".join(report) + "\n")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only W/D/L model audit.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--backtest", type=Path, default=DEFAULT_BACKTEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    predictions, source_file, probability_scale = load_group_stage_predictions(args.predictions)
    predictions = add_buckets(enrich_probabilities(predictions))
    backtest = load_backtest(args.backtest)
    if backtest is not None:
        backtest = add_buckets(enrich_probabilities(backtest))

    validity = probability_validity(predictions, source_file, probability_scale)
    dist = distribution_summary(predictions)
    buckets = by_favorite_bucket(predictions)
    draw = draw_behavior(predictions)
    consistency = scoreline_consistency(predictions)
    calibration = backtest_calibration(backtest)
    team_bias_df = team_bias(backtest)
    extremes = extreme_cases(predictions, consistency)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validity.to_csv(args.output_dir / "wdl_probability_validity.csv", index=False)
    dist.to_csv(args.output_dir / "wdl_distribution_summary.csv", index=False)
    buckets.to_csv(args.output_dir / "wdl_by_favorite_bucket.csv", index=False)
    draw.to_csv(args.output_dir / "wdl_draw_behavior.csv", index=False)
    consistency.to_csv(args.output_dir / "wdl_scoreline_consistency.csv", index=False)
    calibration.to_csv(args.output_dir / "wdl_backtest_calibration.csv", index=False)
    team_bias_df.to_csv(args.output_dir / "wdl_team_bias_audit.csv", index=False)
    extremes.to_csv(args.output_dir / "wdl_extreme_cases.csv", index=False)
    write_report(
        output_dir=args.output_dir,
        validity=validity,
        dist=dist,
        buckets=buckets,
        draw=draw,
        consistency=consistency,
        calibration=calibration,
        team_bias_df=team_bias_df,
        source_file=source_file,
        backtest_path=args.backtest,
    )

    print("W/D/L audit outputs:", args.output_dir)
    print("Invalid probability rows:", int((~validity["is_valid"]).sum()))
    print(dist.to_string(index=False))
    if not calibration.empty:
        print(calibration.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
