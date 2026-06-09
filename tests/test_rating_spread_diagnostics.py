import unittest

import pandas as pd

from src.ratings.diagnose_rating_spread import (
    CALIBRATED_COL,
    RATING_COL,
    calibrate_ratings,
    ranking_order,
    rating_difference,
)
from src.simulation.group_stage_simulator import load_default_ratings


class RatingSpreadDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame(
            {
                "team_name": ["Spain", "France", "Germany", "Curacao"],
                RATING_COL: [1880.0, 1830.0, 1740.0, 1490.0],
            }
        )
        self.calibrated = calibrate_ratings(self.ratings, spread_factor=1.20)

    def test_germany_curacao_diff_increases(self):
        before = abs(rating_difference(self.ratings, "Germany", "Curacao", RATING_COL))
        after = abs(
            rating_difference(self.calibrated, "Germany", "Curacao", CALIBRATED_COL)
        )

        self.assertGreater(after, before)

    def test_spain_france_diff_remains_moderate(self):
        after = abs(
            rating_difference(self.calibrated, "Spain", "France", CALIBRATED_COL)
        )

        self.assertLess(after, 120)

    def test_rankings_do_not_change(self):
        self.assertEqual(
            ranking_order(self.ratings, RATING_COL),
            ranking_order(self.calibrated, CALIBRATED_COL),
        )

    def test_default_simulation_ratings_use_anchored_strength(self):
        ratings, rating_col = load_default_ratings()

        self.assertEqual("anchored_final_strength", rating_col)
        self.assertIn("anchored_final_strength", ratings.columns)


if __name__ == "__main__":
    unittest.main()
