import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.tournament.scope_live_group_stage_predictions import (
    assign_group_rounds,
    validate_round1_scope,
)


class PredictionOutputScopeTest(unittest.TestCase):
    def test_assign_group_rounds_infers_first_two_matches_per_group_as_round1(self) -> None:
        predictions = pd.DataFrame(
            {
                "match_id": [1, 2, 3, 4, 5, 6],
                "group": ["A"] * 6,
                "team_a": ["A1", "A3", "A1", "A2", "A4", "A2"],
                "team_b": ["A2", "A4", "A3", "A4", "A1", "A3"],
            }
        )
        fixtures = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "match_number": [1, 2, 3, 4, 5, 6],
                "stage_id": [1] * 6,
                "kickoff_at": pd.date_range("2026-06-11", periods=6, freq="D").astype(str),
                "match_label": ["Group A"] * 6,
            }
        )

        scoped = assign_group_rounds(predictions, fixtures)

        self.assertEqual(scoped["inferred_group_round"].tolist(), [1, 1, 2, 2, 3, 3])


    def test_validate_round1_scope_rejects_missing_round1_match(self) -> None:
        round1 = pd.DataFrame(
            {
                "match_id": [1, 2],
                "group": ["A", "B"],
                "inferred_group_round": [1, 1],
            }
        )
        full = pd.DataFrame(
            {
                "match_id": [1, 2, 3, 4, 5, 6],
                "group": ["A"] * 6,
                "inferred_group_round": [1, 1, 2, 2, 3, 3],
            }
        )

        with self.assertRaisesRegex(ValueError, "exactly 2 Round 1 matches"):
            validate_round1_scope(round1, full, expected_groups=["A"], expected_total_rows=2)


if __name__ == "__main__":
    unittest.main()
