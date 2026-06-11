import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live.evaluate_live_group_stage import (
    build_live_evaluation,
    build_live_summary,
    load_completed_results,
)
from src.live.lock_group_stage_predictions import lock_predictions


class TestLiveGroupStageValidation(unittest.TestCase):
    def test_lock_predictions_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = []
            for match_id in range(1, 73):
                rows.append(
                    {
                        "match_id": match_id,
                        "group": "A",
                        "team_a": f"Team A {match_id}",
                        "team_b": f"Team B {match_id}",
                        "p_team_a_win": 0.5,
                        "p_draw": 0.25,
                        "p_team_b_win": 0.25,
                    }
                )
            input_path = tmp_path / "predictions.csv"
            output_path = tmp_path / "locked.csv"
            template_path = tmp_path / "actual.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)

            lock_predictions(
                input_path=input_path,
                output_path=output_path,
                actual_template_path=template_path,
                overwrite=False,
            )
            self.assertTrue(output_path.exists())
            self.assertTrue(template_path.exists())
            with self.assertRaises(FileExistsError):
                lock_predictions(
                    input_path=input_path,
                    output_path=output_path,
                    actual_template_path=template_path,
                    overwrite=False,
                )

    def test_live_evaluation_computes_actual_from_goals(self):
        locked = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "group": "A",
                    "team_a": "Mexico",
                    "team_b": "South Africa",
                    "p_team_a_win": 0.7,
                    "p_draw": 0.2,
                    "p_team_b_win": 0.1,
                    "model_version": "v2",
                }
            ]
        )
        actual = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "team_a": "Mexico",
                    "team_b": "South Africa",
                    "goals_a": 2,
                    "goals_b": 1,
                    "actual_result": "",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            actual_path = Path(tmp) / "actual.csv"
            actual.to_csv(actual_path, index=False)
            completed = load_completed_results(actual_path)

        evaluation = build_live_evaluation(locked, completed)
        summary = build_live_summary(evaluation)
        self.assertEqual("team_a_win", evaluation.loc[0, "actual_result"])
        self.assertTrue(bool(evaluation.loc[0, "correct"]))
        self.assertEqual(1, int(summary.loc[0, "n_completed_matches"]))
        self.assertAlmostEqual(0.7, evaluation.loc[0, "actual_outcome_probability"])


if __name__ == "__main__":
    unittest.main()
