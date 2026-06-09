import random
import unittest
from pathlib import Path

import pandas as pd

from src.simulation.group_stage_simulator import (
    EXPECTED_DIRECT_QUALIFIERS,
    EXPECTED_GROUP_COUNT,
    EXPECTED_GROUP_MATCH_COUNT,
    EXPECTED_THIRD_PLACE_QUALIFIERS,
    EXPECTED_TOTAL_QUALIFIERS,
    build_sample_output,
    load_base_total_goals,
    load_default_ratings,
    load_group_stage_fixtures,
    run_sanity_checks,
    run_group_stage_monte_carlo,
    run_monte_carlo_sanity_checks,
    sample_scoreline,
    simulate_all_groups,
    simulate_one_group,
)


class TestGroupStageSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = load_group_stage_fixtures()
        cls.teams = pd.read_csv("data/raw/teams.csv")
        cls.ratings, cls.rating_col = load_default_ratings()
        cls.base_total_goals = load_base_total_goals()

    def test_loads_exact_group_stage_fixture_set(self):
        self.assertEqual(EXPECTED_GROUP_COUNT, self.fixtures["group_letter"].nunique())
        self.assertEqual(EXPECTED_GROUP_MATCH_COUNT, len(self.fixtures))
        self.assertEqual(6, self.fixtures.groupby("group_letter").size().min())
        self.assertEqual(6, self.fixtures.groupby("group_letter").size().max())

    def test_sample_scoreline_uses_probability_distribution(self):
        rng = random.Random(1)
        score = sample_scoreline({"0-0": 0.0, "1-0": 1.0}, rng)
        self.assertEqual((1, 0), score)

    def test_simulates_one_group(self):
        standings, matches = simulate_one_group(
            group_letter="A",
            fixtures=self.fixtures,
            teams_df=self.teams,
            ratings_df=self.ratings,
            rating_col=self.rating_col,
            base_total_goals=self.base_total_goals,
            rng=random.Random(2026),
        )

        self.assertEqual(4, len(standings))
        self.assertEqual(6, len(matches))
        self.assertTrue((standings["matches_played"] == 3).all())
        self.assertEqual([1, 2, 3, 4], standings["group_rank"].tolist())

    def test_simulates_all_groups_and_selects_qualifiers(self):
        standings, matches, qualifiers = simulate_all_groups(seed=2026)

        self.assertEqual(48, len(standings))
        self.assertEqual(EXPECTED_GROUP_MATCH_COUNT, len(matches))
        self.assertEqual(EXPECTED_TOTAL_QUALIFIERS, len(qualifiers))
        self.assertEqual(
            EXPECTED_DIRECT_QUALIFIERS,
            int((qualifiers["qualification_status"] == "qualified_top_2").sum()),
        )
        self.assertEqual(
            EXPECTED_THIRD_PLACE_QUALIFIERS,
            int((qualifiers["qualification_status"] == "qualified_best_third").sum()),
        )
        run_sanity_checks(self.fixtures, standings=standings, qualifiers=qualifiers)

    def test_sample_output_has_qualification_labels(self):
        standings, _, qualifiers = simulate_all_groups(seed=2026)
        output = build_sample_output(standings=standings, qualifiers=qualifiers)

        self.assertEqual(48, len(output))
        self.assertEqual(EXPECTED_TOTAL_QUALIFIERS, int(output["qualified"].sum()))
        self.assertIn("qualification_status", output.columns)

    def test_monte_carlo_results_have_rank_probabilities(self):
        results = run_group_stage_monte_carlo(
            n_simulations=200,
            seed=2026,
            output_path=Path("data/processed/test_group_stage_monte_results.csv"),
        )

        self.assertEqual(48, len(results))
        rank_probability_sum = results[
            [
                "group_rank_1_probability",
                "group_rank_2_probability",
                "group_rank_3_probability",
                "group_rank_4_probability",
            ]
        ].sum(axis=1)
        self.assertTrue(((rank_probability_sum - 1.0).abs() < 1e-12).all())
        self.assertAlmostEqual(
            EXPECTED_DIRECT_QUALIFIERS,
            results["qualified_top_2_probability"].sum(),
        )
        self.assertAlmostEqual(
            EXPECTED_THIRD_PLACE_QUALIFIERS,
            results["qualified_best_third_probability"].sum(),
        )
        self.assertAlmostEqual(
            EXPECTED_TOTAL_QUALIFIERS,
            results["qualification_probability"].sum(),
        )
        run_monte_carlo_sanity_checks(results, n_simulations=200)


if __name__ == "__main__":
    unittest.main()
