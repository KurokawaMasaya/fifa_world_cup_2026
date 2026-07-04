from __future__ import annotations

import csv
import re
import subprocess
import sys
import unicodedata
from io import StringIO
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_R32_SOURCE_PATH = PROJECT_ROOT / "data" / "processed" / "R32.rtf"
NORMALIZED_R32_SOURCE_PATH = PROJECT_ROOT / "data" / "processed" / "r32_bracket_source.csv"


def normalize_key(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())
    aliases = {
        "belgiun": "belgium",
        "belgium": "belgium",
        "cote d ivoire": "cote d ivoire",
        "cote divoire": "cote d ivoire",
        "ivory coast": "cote d ivoire",
        "bosnia herzegovina": "bosnia and herzegovina",
        "bosnia and herzegovina": "bosnia and herzegovina",
        "dr congo": "dr congo",
        "congo dr": "dr congo",
        "democratic republic of congo": "dr congo",
        "usa": "usa",
        "united states": "usa",
        "us": "usa",
    }
    return aliases.get(text, text)


def rtf_to_text(path: Path) -> str:
    if path.suffix.lower() != ".rtf":
        return path.read_text(encoding="utf-8-sig")
    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        raw = path.read_text(errors="ignore")
        raw = re.sub(r"\\'[0-9a-fA-F]{2}", "", raw)
        body = raw.split("\\cf0", maxsplit=1)[-1]
        body = body.replace("\\\n", "\n").replace("\\", "\n")
        body = re.sub(r"[{}]", "", body)
        return body


def load_uploaded_r32_source(path: Path = DEFAULT_R32_SOURCE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing uploaded R32 source file: {path}")
    text = rtf_to_text(path).strip()
    reader = csv.DictReader(StringIO(text))
    rows = []
    for index, row in enumerate(reader, start=73):
        team_a = str(row.get("teams", "")).strip()
        team_b = str(row.get("opponent", "")).strip()
        section = str(row.get("sections", "")).strip()
        if not team_a or not team_b:
            continue
        rows.append(
            {
                "match_number": index,
                "round": "R32",
                "section": section,
                "raw_team_a": team_a,
                "raw_team_b": team_b,
            }
        )
    bracket = pd.DataFrame(rows)
    if len(bracket) != 16:
        raise ValueError(f"Uploaded R32 source should contain 16 fixtures, found {len(bracket)}")
    return bracket


def normalize_r32_bracket(
    ratings_df: pd.DataFrame,
    source_path: Path = DEFAULT_R32_SOURCE_PATH,
    output_path: Path = NORMALIZED_R32_SOURCE_PATH,
) -> pd.DataFrame:
    source = load_uploaded_r32_source(source_path)
    ratings_lookup = {normalize_key(name): name for name in ratings_df["team_name"].dropna().astype(str)}
    rows = []
    unresolved = []
    for _, row in source.iterrows():
        key_a = normalize_key(row["raw_team_a"])
        key_b = normalize_key(row["raw_team_b"])
        team_a = ratings_lookup.get(key_a)
        team_b = ratings_lookup.get(key_b)
        if team_a is None:
            unresolved.append(str(row["raw_team_a"]))
        if team_b is None:
            unresolved.append(str(row["raw_team_b"]))
        rows.append(
            {
                "match_number": int(row["match_number"]),
                "round": "R32",
                "section": row["section"],
                "slot_a": f"uploaded_{row['section']}_{int(row['match_number'])}_A",
                "slot_b": f"uploaded_{row['section']}_{int(row['match_number'])}_B",
                "team_a": team_a,
                "team_b": team_b,
                "raw_team_a": row["raw_team_a"],
                "raw_team_b": row["raw_team_b"],
                "original_match_label": f"{row['raw_team_a']} vs {row['raw_team_b']}",
                "resolved_match_label": f"{team_a} vs {team_b}" if team_a and team_b else "",
                "bracket_source": str(source_path.relative_to(PROJECT_ROOT)),
                "uses_random_pairing": False,
                "official_bracket": True,
                "random_seed_used_for": "['scoreline_sampling', 'tiebreakers']",
            }
        )
    if unresolved:
        print("Unresolved R32 team names:", sorted(set(unresolved)), file=sys.stderr)
        raise ValueError("Uploaded R32 bracket contains unresolved team names")
    bracket = pd.DataFrame(rows)
    if len(set(bracket["team_a"]).union(set(bracket["team_b"]))) != 32:
        raise ValueError("Uploaded R32 bracket must contain exactly 32 unique teams")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bracket.to_csv(output_path, index=False)
    return bracket
