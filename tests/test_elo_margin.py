import math
import unittest


from src.ratings.build_elo_ratings import (
    BASE_K,
    MARGIN_DAMPING_FLOOR,
    expected_score,
    margin_multiplier,
)


def rating_change(goal_diff, team_rating, opponent_rating, actual_score=1.0):
    expected = expected_score(team_rating, opponent_rating)
    return BASE_K * margin_multiplier(goal_diff, team_rating, opponent_rating) * (
        actual_score - expected
    )


class EloMarginTests(unittest.TestCase):
    def test_evenly_matched_blowout_has_more_impact_than_mismatch_blowout(self):
        even_match_change = rating_change(5, 1500, 1500)
        mismatch_change = rating_change(5, 1900, 1200)

        self.assertGreater(even_match_change, mismatch_change)

    def test_draws_have_margin_multiplier_at_least_one(self):
        self.assertGreaterEqual(margin_multiplier(0, 1500, 1500), 1.0)
        self.assertGreaterEqual(margin_multiplier(0, 1900, 1200), 1.0)

    def test_margin_multiplier_is_capped_for_very_large_wins(self):
        even_large_win = margin_multiplier(20, 1500, 1500)
        even_absurd_win = margin_multiplier(100, 1500, 1500)

        self.assertTrue(math.isclose(even_large_win, 1.7))
        self.assertTrue(math.isclose(even_absurd_win, 1.7))
        self.assertLess(even_absurd_win, 2.0)

    def test_mismatch_damping_has_floor(self):
        extreme_mismatch = margin_multiplier(5, 2400, 900)

        self.assertGreaterEqual(extreme_mismatch, 1 + (1.7 - 1) * MARGIN_DAMPING_FLOOR)


if __name__ == "__main__":
    unittest.main()
