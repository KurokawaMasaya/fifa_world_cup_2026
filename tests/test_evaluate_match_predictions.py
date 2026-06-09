import unittest

import pandas as pd

from src.evaluation.evaluate_match_predictions import (
    ELO_ONLY_NAME,
    FINAL_MODEL_NAME,
    HISTORICAL_FREQUENCY_NAME,
    RANDOM_UNIFORM_NAME,
    build_calibration_table,
    evaluate_predictions,
    score_prediction,
)


class EvaluateMatchPredictionsTests(unittest.TestCase):
    def test_score_prediction_metrics(self):
        probabilities = {"home_win": 0.7, "draw": 0.2, "away_win": 0.1}

        brier, log_loss, actual_probability, predicted, correct = score_prediction(
            probabilities,
            "home_win",
        )

        self.assertEqual("home_win", predicted)
        self.assertTrue(correct)
        self.assertAlmostEqual(0.7, actual_probability)
        self.assertGreater(brier, 0)
        self.assertGreater(log_loss, 0)

    def test_evaluate_predictions_outputs_expected_columns(self):
        matches = pd.DataFrame(
            {
                "date": [
                    "2014-01-01",
                    "2015-01-01",
                    "2022-01-01",
                    "2022-01-02",
                ],
                "home_team": ["A", "B", "A", "B"],
                "away_team": ["B", "A", "B", "A"],
                "home_score": [2, 0, 1, 0],
                "away_score": [0, 1, 1, 2],
                "tournament": ["Friendly", "Friendly", "Friendly", "Friendly"],
                "neutral": [False, False, False, True],
            }
        )
        matches["date"] = pd.to_datetime(matches["date"])

        (
            predictions,
            summary,
            model_comparison,
            favorite_calibration,
            draw_calibration,
            strength_diff,
        ) = evaluate_predictions(matches)

        self.assertEqual(8, len(predictions))
        self.assertEqual(1, len(summary))
        self.assertEqual(4, len(model_comparison))
        self.assertEqual(6, len(favorite_calibration))
        self.assertEqual(1, len(draw_calibration))
        self.assertEqual(5, len(strength_diff))
        self.assertIn("p_home_win", predictions.columns)
        self.assertIn("goal_mae", summary.columns)
        self.assertEqual(
            {
                RANDOM_UNIFORM_NAME,
                HISTORICAL_FREQUENCY_NAME,
                ELO_ONLY_NAME,
                FINAL_MODEL_NAME,
            },
            set(model_comparison["model_version"]),
        )
        self.assertTrue(predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1).between(0.999, 1.001).all())

    def test_calibration_table_has_fixed_bins(self):
        predictions = pd.DataFrame(
            {
                "p_home_win": [0.45, 0.55, 0.65],
                "p_away_win": [0.30, 0.20, 0.10],
                "actual_outcome": ["home_win", "draw", "away_win"],
            }
        )

        calibration = build_calibration_table(predictions)

        self.assertEqual(
            [
                "0.40-0.50",
                "0.50-0.60",
                "0.60-0.70",
                "0.70-0.80",
                "0.80-0.90",
                "0.90-1.00",
            ],
            calibration["favorite_win_probability_bin"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
