from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.favorite_blowout_mixture_scoreline import (
    GatedBlowoutParams,
    _nb_independent_grid,
    _normalize_grid,
    bucket_cap_for_favorite_bucket,
    gated_favorite_blowout_mixture_scoreline_grid,
    normalize_win_probability,
)


FEATURE_COLUMNS = [
    "favorite_win_prob_feature",
    "lambda_imbalance_feature",
    "favorite_scoring_capacity_feature",
    "underdog_suppression_feature",
    "interaction_favorite_capacity_x_suppression",
    "interaction_dominance_x_imbalance",
]


@dataclass(frozen=True)
class LearnedGate:
    """Interpretable learned V2.5 blowout gate.

    The model predicts only the pre-match probability of activating the
    blowout-tail mixture. It does not change W/D/L probabilities, ratings, or
    the normal scoreline distribution.
    """

    pipeline: Pipeline
    feature_columns: list[str]
    global_k: float = 1.0

    def raw_probability(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self.pipeline.predict_proba(features[self.feature_columns])[:, 1],
            index=features.index,
        )

    def logit(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self.pipeline.decision_function(features[self.feature_columns]),
            index=features.index,
        )


def build_learned_gate_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Build pre-match V2.5 gate features from existing prediction diagnostics."""
    features = matches.copy()
    features["favorite_win_prob_feature"] = features["favorite_win_prob"].map(
        normalize_win_probability
    )
    features["lambda_favorite"] = features["base_lambda_fav"]
    features["lambda_underdog"] = features["base_lambda_dog"]
    features["lambda_gap"] = features["lambda_favorite"] - features["lambda_underdog"]
    features["raw_total_lambda"] = features["lambda_a"] + features["lambda_b"]
    features["lambda_imbalance_feature"] = (
        features["lambda_gap"].abs() / features["raw_total_lambda"].clip(lower=1e-9)
    )
    features["favorite_scoring_capacity_feature"] = features["lambda_favorite"]
    features["underdog_suppression_feature"] = (
        1.0 - (features["lambda_underdog"] / 1.5).clip(lower=0.0, upper=1.0)
    )
    features["interaction_favorite_capacity_x_suppression"] = (
        features["favorite_scoring_capacity_feature"]
        * features["underdog_suppression_feature"]
    )
    features["interaction_dominance_x_imbalance"] = (
        features["favorite_win_prob_feature"]
        * features["lambda_imbalance_feature"]
    )
    return features


def build_composite_tail_target(matches: pd.DataFrame) -> pd.Series:
    fav5 = matches["v23_actual_favorite_scores_5_plus"].astype(bool)
    margin4 = matches["v23_actual_margin_4_plus"].astype(bool)
    total5_favorite_win = (matches["actual_total_goals"] >= 5) & (
        matches["actual_result"]
        == matches["favorite_team"].eq(matches["team_a"]).map(
            {True: "team_a_win", False: "team_b_win"}
        )
    )
    fav4_clean_sheet = matches["v23_actual_favorite_scores_4_plus"].astype(bool) & (
        (matches["actual_goals_a"] == 0) | (matches["actual_goals_b"] == 0)
    )
    return (fav5 | margin4 | total5_favorite_win | fav4_clean_sheet).astype(int)


def fit_logistic_gate(
    train_features: pd.DataFrame,
    target: pd.Series,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    global_k: float = 1.0,
) -> LearnedGate:
    feature_columns = list(feature_columns)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    penalty="l2",
                    C=0.8,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(train_features[feature_columns], target)
    return LearnedGate(pipeline=pipeline, feature_columns=feature_columns, global_k=global_k)


def learned_bucket_capped_probability(
    gate: LearnedGate,
    features: pd.DataFrame,
    buckets: pd.Series,
) -> pd.DataFrame:
    raw = gate.raw_probability(features)
    logits = gate.logit(features)
    caps = buckets.map(bucket_cap_for_favorite_bucket).astype(float)
    final = (caps * raw * gate.global_k).clip(lower=0.0)
    final = pd.concat([final, caps], axis=1).min(axis=1)
    return pd.DataFrame(
        {
            "p_blowout_v25_raw": raw,
            "learned_gate_logit": logits,
            "bucket_cap": caps,
            "p_blowout_v25_final": final,
            "global_k": gate.global_k,
        },
        index=features.index,
    )


def learned_gated_scoreline_grid(
    row: pd.Series,
    p_blowout_final: float,
    max_goals: int = 10,
):
    """Use the V2.4 grid builder with a forced learned p_blowout_final."""
    params = GatedBlowoutParams(
        blowout_k_factor=1.0,
        use_favorite_dominance_gate=False,
        use_lambda_imbalance_gate=False,
        use_favorite_scoring_capacity_gate=False,
        use_underdog_suppression_gate=False,
        use_motivation_gate=False,
    )
    _, metadata = gated_favorite_blowout_mixture_scoreline_grid(
        lambda_a=float(row["lambda_a"]),
        lambda_b=float(row["lambda_b"]),
        p_team_a_win=float(row["p_team_a_win"]),
        p_draw=float(row["p_draw"]),
        p_team_b_win=float(row["p_team_b_win"]),
        rating_gap=float(row["strength_diff"]),
        max_goals=max_goals,
        params=params,
    )
    normal_grid = _nb_independent_grid(
        lambda_a=float(row["lambda_a"]),
        lambda_b=float(row["lambda_b"]),
        dispersion_k=params.normal_k,
        max_goals=max_goals,
    )
    favorite_is_a = metadata["favorite_side"] == "team_a"
    blowout_lambda_a = (
        float(metadata["blowout_lambda_fav"])
        if favorite_is_a
        else float(metadata["blowout_lambda_dog"])
    )
    blowout_lambda_b = (
        float(metadata["blowout_lambda_dog"])
        if favorite_is_a
        else float(metadata["blowout_lambda_fav"])
    )
    blowout_grid = _nb_independent_grid(
        lambda_a=blowout_lambda_a,
        lambda_b=blowout_lambda_b,
        dispersion_k=params.blowout_k,
        max_goals=max_goals,
    )
    p_final = float(p_blowout_final)
    mixed = {
        scoreline: (1.0 - p_final) * normal_grid[scoreline]
        + p_final * blowout_grid[scoreline]
        for scoreline in normal_grid
    }
    grid = _normalize_grid(
        mixed,
        lambda_a=float(row["lambda_a"]),
        lambda_b=float(row["lambda_b"]),
    )
    metadata["p_blowout_v25_final"] = float(p_blowout_final)
    return grid, metadata


def gate_auc(target: pd.Series, probabilities: pd.Series) -> float | None:
    if target.nunique() < 2:
        return None
    return float(roc_auc_score(target, probabilities))
