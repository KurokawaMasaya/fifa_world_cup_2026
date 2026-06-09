import unittest

import pandas as pd

from src.evaluation.tune_parameters import (
    apply_draw_boost,
    best_parameters_from_results,
    evaluate_final_test,
    parameter_grid,
)


class TuneParametersTests(unittest.TestCase):
    def test_parameter_grid_size(self):
        self.assertEqual(2880, len(parameter_grid()))

    def test_draw_boost_preserves_probability_sum_and_cap(self):
        probabilities = {"home_win": 0.4, "draw": 0.44, "away_win": 0.16}
        boosted = apply_draw_boost(
            probabilities,
            strength_diff=0,
            draw_boost_max=0.045,
            draw_boost_scale=100,
        )

        self.assertAlmostEqual(1.0, sum(boosted.values()))
        self.assertLessEqual(boosted["draw"], 0.45)
        self.assertGreaterEqual(boosted["home_win"], 0)
        self.assertGreaterEqual(boosted["away_win"], 0)

    def test_best_parameters_uses_sorted_valid_rows(self):
        results = pd.DataFrame(
            [
                {
                    "status": "ok",
                    "base_total_goals": 2.55,
                    "share_scale": 250,
                    "mismatch_total_bonus": 0.7,
                    "mismatch_scale": 650,
                    "draw_boost_max": 0.025,
                    "draw_boost_scale": 150,
                    "mean_log_loss": 0.9,
                }
            ]
        )

        best = best_parameters_from_results(results)

        self.assertEqual(2.55, best["base_total_goals"])
        self.assertEqual(250.0, best["share_scale"])

    def test_final_test_evaluation_runs_on_synthetic_data(self):
        matches = pd.DataFrame(
            {
                "date": ["2014-01-01", "2022-01-01", "2024-01-01"],
                "home_team": ["A", "A", "A"],
                "away_team": ["B", "B", "B"],
                "home_score": [2, 1, 1],
                "away_score": [0, 1, 0],
                "tournament": ["Friendly", "Friendly", "Friendly"],
                "neutral": [False, False, False],
            }
        )
        matches["date"] = pd.to_datetime(matches["date"])
        params = {
            "base_total_goals": 2.65,
            "share_scale": 250,
            "mismatch_total_bonus": 0.7,
            "mismatch_scale": 650,
            "draw_boost_max": 0.025,
            "draw_boost_scale": 150,
        }

        final_test = evaluate_final_test(matches, params=params)

        self.assertEqual(1, len(final_test))
        self.assertIn("mean_log_loss", final_test.columns)


if __name__ == "__main__":
    unittest.main()
