from __future__ import annotations

import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

SQUAD_PLAYERS_PATH = RAW_DIR / "worldcup2026_squad_players.csv"
PLAYERS_PATH = RAW_DIR / "players.csv"
VALUATIONS_PATH = RAW_DIR / "player_valuations.csv"
V1_STRENGTH_PATH = PROCESSED_DIR / "team_ratings_world_cup_elo.csv"

PLAYER_VALUES_OUTPUT_PATH = PROCESSED_DIR / "player_values_standardized.csv"
SQUAD_VALUES_OUTPUT_PATH = RAW_DIR / "squad_values.csv"
QUALITY_REPORT_OUTPUT_PATH = OUTPUT_DIR / "player_value_match_quality_report.csv"
V2_STRENGTH_OUTPUT_PATH = PROCESSED_DIR / "team_strength_v2_player_impacted.csv"
V1_V2_COMPARISON_OUTPUT_PATH = OUTPUT_DIR / "v1_vs_v2_player_impacted_comparison.csv"

MATCH_SCORE_THRESHOLD = 0.55
TEAM_ALIASES = {
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Côte D'Ivoire": "Cote d'Ivoire",
    "Curaçao": "Curacao",
    "Congo DR": "DR Congo",
    "Ivory Coast": "Cote d'Ivoire",
    "Iran": "IR Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "United States": "USA",
    "US": "USA",
}


@dataclass(frozen=True)
class MatchCandidate:
    player_id: int | None
    transfermarkt_name: str | None
    match_score: float
    match_method: str
    duplicate_candidate_count: int


def normalize_country_name(team_name: str) -> str:
    return TEAM_ALIASES.get(str(team_name), str(team_name))


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(value: object) -> set[str]:
    return set(normalize_text(value).split())


def sequence_similarity(left: object, right: object) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def token_overlap_score(left: object, right: object) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def squad_name_variants(row: pd.Series) -> list[str]:
    """Build conservative FIFA-squad name variants for Transfermarkt matching."""
    first_names = str(row["first_names"])
    last_names = str(row["last_names"])
    player_name = str(row["player_name"])
    name_parts = player_name.split()

    variants = [
        f"{first_names} {last_names}",
        f"{row['name_on_shirt']} {first_names.split()[0] if first_names.split() else ''}",
    ]
    if name_parts:
        variants.append(f"{' '.join(name_parts[1:])} {name_parts[0]}")
    if first_names.split() and last_names.split():
        variants.append(f"{first_names.split()[0]} {last_names.split()[0]}")

    deduped = []
    seen = set()
    for variant in variants:
        normalized = normalize_text(variant)
        if normalized and normalized not in seen:
            deduped.append(variant)
            seen.add(normalized)
    return deduped


def score_candidate(row: pd.Series, player: pd.Series) -> float:
    scores = []
    for variant in squad_name_variants(row):
        scores.append(sequence_similarity(variant, player["name"]))
        scores.append(token_overlap_score(variant, player["name"]))
    return max(scores) if scores else 0.0


def build_dob_player_index(players: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        dob: group.copy()
        for dob, group in players.dropna(subset=["date_of_birth_clean"]).groupby(
            "date_of_birth_clean"
        )
    }


def match_squad_player(row: pd.Series, players_by_dob: dict[str, pd.DataFrame]) -> MatchCandidate:
    """Match primarily on date of birth, then choose the best normalized name score.

    Date of birth keeps common-name collisions visible and avoids relying on
    club names, which can differ between FIFA squad pages and Transfermarkt.
    The score and duplicate-candidate count are saved for auditability.
    """
    candidates = players_by_dob.get(row["date_of_birth_clean"], pd.DataFrame())
    if candidates.empty:
        return MatchCandidate(None, None, 0.0, "no_dob_candidates", 0)

    scored = []
    for _, player in candidates.iterrows():
        scored.append((score_candidate(row, player), player))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_player = scored[0]
    duplicate_count = sum(1 for score, _ in scored if abs(score - best_score) < 1e-9)

    if best_score < MATCH_SCORE_THRESHOLD:
        return MatchCandidate(None, None, float(best_score), "below_threshold", duplicate_count)
    return MatchCandidate(
        int(best_player["player_id"]),
        str(best_player["name"]),
        float(best_score),
        "dob_name_similarity",
        duplicate_count,
    )


def load_latest_valuations(path: Path = VALUATIONS_PATH) -> pd.DataFrame:
    valuations = pd.read_csv(path)
    valuations["valuation_date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations = valuations.dropna(subset=["player_id", "valuation_date"])
    valuations = valuations.sort_values(["player_id", "valuation_date"])
    latest = valuations.groupby("player_id", as_index=False).tail(1)
    return latest[
        ["player_id", "valuation_date", "market_value_in_eur", "current_club_name"]
    ].rename(columns={"current_club_name": "valuation_club_name"})


def build_player_values_standardized() -> pd.DataFrame:
    squad = pd.read_csv(SQUAD_PLAYERS_PATH)
    players = pd.read_csv(
        PLAYERS_PATH,
        usecols=[
            "player_id",
            "name",
            "country_of_citizenship",
            "date_of_birth",
            "current_club_name",
        ],
    )
    latest_values = load_latest_valuations()

    squad = squad.copy()
    squad["team_name"] = squad["team"].map(normalize_country_name)
    squad["date_of_birth_clean"] = pd.to_datetime(
        squad["dob"], format="%d/%m/%Y", errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    players = players.copy()
    players["date_of_birth_clean"] = pd.to_datetime(
        players["date_of_birth"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    players_by_dob = build_dob_player_index(players)

    match_rows = []
    for _, row in squad.iterrows():
        match = match_squad_player(row, players_by_dob)
        match_rows.append(
            {
                "player_id": match.player_id,
                "transfermarkt_name": match.transfermarkt_name,
                "match_score": match.match_score,
                "match_method": match.match_method,
                "duplicate_candidate_count": match.duplicate_candidate_count,
            }
        )

    output = pd.concat([squad.reset_index(drop=True), pd.DataFrame(match_rows)], axis=1)
    output = output.merge(latest_values, on="player_id", how="left")
    output["matched"] = output["player_id"].notna()
    output["missing_market_value"] = output["matched"] & output["market_value_in_eur"].isna()
    output["market_value_in_eur"] = pd.to_numeric(
        output["market_value_in_eur"], errors="coerce"
    )

    columns = [
        "team_name",
        "team",
        "team_code",
        "squad_number",
        "position",
        "player_name",
        "first_names",
        "last_names",
        "name_on_shirt",
        "dob",
        "date_of_birth_clean",
        "age_on_2026_06_11",
        "club",
        "caps",
        "goals",
        "player_id",
        "transfermarkt_name",
        "match_score",
        "match_method",
        "duplicate_candidate_count",
        "valuation_date",
        "valuation_club_name",
        "market_value_in_eur",
        "matched",
        "missing_market_value",
    ]
    return output[columns].sort_values(["team_name", "squad_number"]).reset_index(drop=True)


def safe_log(series: pd.Series) -> pd.Series:
    return series.where(series > 0).map(lambda value: math.log(value) if pd.notna(value) else pd.NA)


def z_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / std


def aggregate_squad_values(player_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team_name, group in player_values.groupby("team_name", sort=True):
        values = group["market_value_in_eur"].dropna().sort_values(ascending=False)
        squad_market_value = float(values.sum()) if not values.empty else 0.0
        top_5_value = float(values.head(5).sum()) if not values.empty else 0.0
        rows.append(
            {
                "team_name": team_name,
                "team_code": group["team_code"].iloc[0],
                "squad_players": len(group),
                "matched_players": int(group["matched"].sum()),
                "unmatched_players": int((~group["matched"]).sum()),
                "match_rate": float(group["matched"].mean()),
                "players_with_market_value": int(group["market_value_in_eur"].notna().sum()),
                "players_missing_market_value": int(group["missing_market_value"].sum()),
                "squad_market_value_eur": squad_market_value,
                "top_5_value_eur": top_5_value,
                "depth_concentration": (
                    top_5_value / squad_market_value if squad_market_value > 0 else pd.NA
                ),
            }
        )

    squad_values = pd.DataFrame(rows)
    squad_values["log_squad_value"] = safe_log(squad_values["squad_market_value_eur"])
    squad_values["log_top5_value"] = safe_log(squad_values["top_5_value_eur"])
    squad_values["squad_value_z"] = z_score(squad_values["log_squad_value"])
    squad_values["top5_z"] = z_score(squad_values["log_top5_value"])
    squad_values["player_value_adjustment"] = 30.0 * squad_values["squad_value_z"]
    squad_values["star_player_adjustment"] = 15.0 * squad_values["top5_z"]
    depth_penalty = -40.0 * (squad_values["depth_concentration"].fillna(0) - 0.55).clip(lower=0)
    squad_values["squad_depth_adjustment"] = depth_penalty.clip(lower=-25.0, upper=0.0)
    squad_values["total_player_impact_adjustment_raw"] = (
        squad_values["player_value_adjustment"]
        + squad_values["star_player_adjustment"]
        + squad_values["squad_depth_adjustment"]
    )
    squad_values["total_player_impact_adjustment"] = squad_values[
        "total_player_impact_adjustment_raw"
    ].clip(lower=-75.0, upper=75.0)
    return squad_values.sort_values("team_name").reset_index(drop=True)


def build_quality_report(player_values: pd.DataFrame) -> pd.DataFrame:
    duplicate_matches = int(
        player_values.loc[player_values["matched"], "player_id"].duplicated(keep=False).sum()
    )
    summary_rows = [
        {"metric": "total_squad_players", "value": len(player_values), "team_name": "ALL"},
        {"metric": "matched_players", "value": int(player_values["matched"].sum()), "team_name": "ALL"},
        {
            "metric": "unmatched_players",
            "value": int((~player_values["matched"]).sum()),
            "team_name": "ALL",
        },
        {"metric": "match_rate", "value": float(player_values["matched"].mean()), "team_name": "ALL"},
        {"metric": "duplicate_player_matches", "value": duplicate_matches, "team_name": "ALL"},
        {
            "metric": "players_with_missing_market_values",
            "value": int(player_values["missing_market_value"].sum()),
            "team_name": "ALL",
        },
    ]

    by_team = (
        player_values.groupby("team_name")
        .agg(
            total_squad_players=("player_name", "size"),
            matched_players=("matched", "sum"),
            missing_market_values=("missing_market_value", "sum"),
        )
        .reset_index()
    )
    by_team["match_rate"] = by_team["matched_players"] / by_team["total_squad_players"]

    team_rows = []
    for _, row in by_team.iterrows():
        team_rows.append(
            {
                "metric": "team_match_rate",
                "value": row["match_rate"],
                "team_name": row["team_name"],
                "total_squad_players": row["total_squad_players"],
                "matched_players": row["matched_players"],
                "missing_market_values": row["missing_market_values"],
                "below_0_80_match_rate": bool(row["match_rate"] < 0.80),
            }
        )

    report = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame(team_rows)], ignore_index=True)
    return report


def build_v2_strength(squad_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    v1 = pd.read_csv(V1_STRENGTH_PATH)
    required = {"team_name", "anchored_final_strength"}
    missing = required - set(v1.columns)
    if missing:
        raise ValueError(f"{V1_STRENGTH_PATH} is missing columns: {sorted(missing)}")

    v2 = v1.merge(squad_values, on="team_name", how="left", suffixes=("", "_squad"))
    adjustment_columns = [
        "player_value_adjustment",
        "star_player_adjustment",
        "squad_depth_adjustment",
        "total_player_impact_adjustment",
    ]
    for column in adjustment_columns:
        v2[column] = pd.to_numeric(v2[column], errors="coerce").fillna(0.0)

    v2["final_player_impacted_strength"] = (
        v2["anchored_final_strength"] + v2["total_player_impact_adjustment"]
    )
    v2["v2_model_version"] = "v2"
    v2["v2_layer"] = "player_impacted_transfermarkt_values"

    comparison_columns = [
        "team_name",
        "fifa_code",
        "group_letter",
        "anchored_final_strength",
        "squad_market_value_eur",
        "top_5_value_eur",
        "depth_concentration",
        "match_rate",
        "player_value_adjustment",
        "star_player_adjustment",
        "squad_depth_adjustment",
        "total_player_impact_adjustment",
        "final_player_impacted_strength",
    ]
    comparison = v2[comparison_columns].copy()
    comparison["v1_rank"] = comparison["anchored_final_strength"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["v2_rank"] = comparison["final_player_impacted_strength"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison["rank_change"] = comparison["v1_rank"] - comparison["v2_rank"]
    comparison = comparison.sort_values("final_player_impacted_strength", ascending=False)
    return v2, comparison


def save_outputs() -> dict[str, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    player_values = build_player_values_standardized()
    squad_values = aggregate_squad_values(player_values)
    quality_report = build_quality_report(player_values)
    v2_strength, comparison = build_v2_strength(squad_values)

    player_values.to_csv(PLAYER_VALUES_OUTPUT_PATH, index=False)
    squad_values.to_csv(SQUAD_VALUES_OUTPUT_PATH, index=False)
    quality_report.to_csv(QUALITY_REPORT_OUTPUT_PATH, index=False)
    v2_strength.to_csv(V2_STRENGTH_OUTPUT_PATH, index=False)
    comparison.to_csv(V1_V2_COMPARISON_OUTPUT_PATH, index=False)
    return {
        "player_values": PLAYER_VALUES_OUTPUT_PATH,
        "squad_values": SQUAD_VALUES_OUTPUT_PATH,
        "quality_report": QUALITY_REPORT_OUTPUT_PATH,
        "v2_strength": V2_STRENGTH_OUTPUT_PATH,
        "comparison": V1_V2_COMPARISON_OUTPUT_PATH,
    }


def main() -> None:
    outputs = save_outputs()
    report = pd.read_csv(QUALITY_REPORT_OUTPUT_PATH)
    summary = report.loc[report["team_name"].eq("ALL"), ["metric", "value"]]
    low_match = report.loc[
        report["metric"].eq("team_match_rate") & report["below_0_80_match_rate"].eq(True),
        ["team_name", "value"],
    ].sort_values("value")

    print("Saved V2 player-impacted strength outputs:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")
    print("\nQuality summary:")
    print(summary.to_string(index=False))
    print("\nTeams below 0.80 match rate:")
    print(low_match.to_string(index=False) if not low_match.empty else "  None")


if __name__ == "__main__":
    main()
