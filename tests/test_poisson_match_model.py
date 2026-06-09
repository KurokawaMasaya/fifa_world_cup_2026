import unittest

import pandas as pd

from src.models.poisson_match_model import (
    compare_prediction_sensitivity,
    expected_goals_from_strength,
    estimate_base_goals,
    outcome_probabilities,
    poisson_pmf,
    predict_from_expected_goals,
    predict_from_ratings,
    predict_from_strength,
    predict_match_from_expected_goals,
    predict_match_from_ratings,
    predict_match_from_strength,
    predict_match_poisson,
    scoreline_probabilities,
)


class PoissonMatchModelTests(unittest.TestCase):
    def test_poisson_pmf_returns_valid_probabilities(self):
        probabilities = [poisson_pmf(k, 1.35) for k in range(20)]

        self.assertTrue(all(0 <= probability <= 1 for probability in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
        self.assertAlmostEqual(poisson_pmf(0, 1.35), 0.2592402606458915)

    def test_scoreline_probabilities_sum_to_one(self):
        score_probs = scoreline_probabilities(1.4, 1.1, max_goals=8)

        self.assertAlmostEqual(sum(score_probs.values()), 1.0, places=9)

    def test_outcome_probabilities_sum_to_one(self):
        score_probs = scoreline_probabilities(1.4, 1.1, max_goals=8)
        outcome_probs = outcome_probabilities(score_probs)

        self.assertAlmostEqual(sum(outcome_probs.values()), 1.0, places=9)

    def test_estimate_base_goals_from_historical_matches(self):
        matches_df = pd.DataFrame(
            {
                "date": ["2013-01-01", "2014-01-01", "2015-01-01"],
                "home_score": [10, 2, 1],
                "away_score": [10, 0, 3],
            }
        )

        self.assertAlmostEqual(estimate_base_goals(matches_df), 1.5)

    def test_equal_strengths_produce_roughly_symmetric_win_probabilities(self):
        expected_goals = expected_goals_from_strength(1700, 1700)
        prediction = predict_from_strength(
            "Team A",
            "Team B",
            strength_a=1700,
            strength_b=1700,
            base_goals=1.35,
        )

        self.assertAlmostEqual(
            prediction["p_team_a_win"],
            prediction["p_team_b_win"],
            places=9,
        )
        self.assertAlmostEqual(expected_goals["lambda_a"], expected_goals["lambda_b"], places=9)
        self.assertAlmostEqual(expected_goals["goal_share_a"], 0.5, places=9)
        self.assertEqual(prediction["mapping_mode"], "total_share")

    def test_higher_strength_a_increases_team_a_win_probability(self):
        even_prediction = predict_from_strength(
            "Team A",
            "Team B",
            strength_a=1700,
            strength_b=1700,
        )
        stronger_prediction = predict_from_strength(
            "Team A",
            "Team B",
            strength_a=1900,
            strength_b=1600,
        )
        even_expected_goals = expected_goals_from_strength(1700, 1700)
        stronger_expected_goals = expected_goals_from_strength(1900, 1600)

        self.assertGreater(
            stronger_expected_goals["lambda_a"],
            even_expected_goals["lambda_a"],
        )
        self.assertLess(stronger_expected_goals["lambda_b"], 1.325)
        self.assertGreater(
            stronger_prediction["p_team_a_win"],
            even_prediction["p_team_a_win"],
        )

    def test_lambdas_are_clipped_within_bounds(self):
        expected_goals = expected_goals_from_strength(
            strength_a=3000,
            strength_b=1000,
            lambda_min=0.2,
            lambda_max=4.0,
        )

        self.assertLessEqual(expected_goals["lambda_a"], 4.0)
        self.assertGreaterEqual(expected_goals["lambda_b"], 0.2)

    def test_exp_symmetric_mapping_is_available_for_comparison(self):
        expected_goals = expected_goals_from_strength(
            1900,
            1600,
            mapping_mode="exp_symmetric",
            base_goals=1.35,
            scale=800,
        )

        self.assertEqual(expected_goals["mapping_mode"], "exp_symmetric")
        self.assertGreater(expected_goals["lambda_a"], expected_goals["lambda_b"])

    def test_predict_match_from_ratings_looks_up_strengths(self):
        ratings_df = pd.DataFrame(
            {
                "team_name": ["Team A", "Team B"],
                "final_rating": [1900, 1600],
            }
        )

        prediction = predict_from_ratings("Team A", "Team B", ratings_df)

        self.assertGreater(prediction["lambda_a"], prediction["lambda_b"])
        self.assertGreater(prediction["p_team_a_win"], prediction["p_team_b_win"])
        self.assertIn("top_5_scorelines", prediction)
        self.assertEqual(len(prediction["top_5_scorelines"]), 5)

    def test_predict_match_from_ratings_raises_for_missing_team(self):
        ratings_df = pd.DataFrame(
            {
                "team_name": ["Team A"],
                "final_rating": [1900],
            }
        )

        with self.assertRaisesRegex(ValueError, "Team 'Team B' was not found"):
            predict_from_ratings("Team A", "Team B", ratings_df)

    def test_manual_expected_goals_predictor_is_kept_for_debugging(self):
        prediction = predict_from_expected_goals(
            lambda_a=1.5,
            lambda_b=1.0,
            team_a="Team A",
            team_b="Team B",
        )

        self.assertGreater(prediction["p_team_a_win"], prediction["p_team_b_win"])

    def test_legacy_function_names_still_work(self):
        expected_goals_prediction = predict_match_from_expected_goals(
            "Team A",
            "Team B",
            lambda_a=1.5,
            lambda_b=1.0,
        )
        strength_prediction = predict_match_from_strength(
            "Team A",
            "Team B",
            strength_a=1900,
            strength_b=1600,
        )
        ratings_prediction = predict_match_from_ratings(
            "Team A",
            "Team B",
            pd.DataFrame(
                {
                    "team_name": ["Team A", "Team B"],
                    "final_rating": [1900, 1600],
                }
            ),
        )

        self.assertGreater(
            expected_goals_prediction["p_team_a_win"],
            expected_goals_prediction["p_team_b_win"],
        )
        self.assertGreater(strength_prediction["lambda_a"], strength_prediction["lambda_b"])
        self.assertGreater(ratings_prediction["lambda_a"], ratings_prediction["lambda_b"])

    def test_backward_compatible_predict_match_poisson_wrapper(self):
        prediction = predict_match_poisson(
            "Team A",
            "Team B",
            strength_a=1900,
            strength_b=1600,
        )

        self.assertGreater(prediction["lambda_a"], prediction["lambda_b"])

    def test_sensitivity_function_returns_named_matchups(self):
        ratings_df = pd.DataFrame(
            {
                "team_name": ["Spain", "France", "South Africa", "Germany", "Curacao"],
                "final_rating": [1900, 1850, 1550, 1750, 1450],
            }
        )

        sensitivity = compare_prediction_sensitivity(
            ratings_df,
            rating_col="final_rating",
        )

        self.assertEqual(len(sensitivity), 3)
        self.assertTrue((sensitivity["total_expected_goals"] > 0).all())


if __name__ == "__main__":
    unittest.main()
