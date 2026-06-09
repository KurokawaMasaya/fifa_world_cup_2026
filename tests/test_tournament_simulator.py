import unittest
from pathlib import Path

import pandas as pd

from src.simulation.group_stage_simulator import (
    EXPECTED_TOTAL_QUALIFIERS,
    TEAMS_PATH,
    load_base_total_goals,
    load_default_ratings,
    load_group_stage_fixtures,
    precompute_fixture_predictions,
    _simulate_group_records,
)
from src.simulation.tournament_simulator import (
    BRACKET_MODE,
    BRACKET_MAPPING_NOTE,
    BRACKET_SOURCE,
    OFFICIAL_BRACKET,
    RANDOM_SEED_USED_FOR,
    SAMPLE_KNOCKOUT_BRACKET_PATH,
    USES_RANDOM_PAIRING,
    build_fixed_r32_bracket,
    elo_tiebreak_probability,
    load_knockout_schedule,
    resolve_knockout_placeholder,
    run_tournament_monte_carlo,
    run_tournament_sanity_checks,
    select_qualifier_records,
    validate_round_pairings,
)


class TestTournamentSimulator(unittest.TestCase):
    def test_elo_tiebreak_probability_is_rating_sensitive(self):
        self.assertAlmostEqual(0.5, elo_tiebreak_probability(1700, 1700))
        self.assertGreater(elo_tiebreak_probability(1800, 1600), 0.5)
        self.assertLess(elo_tiebreak_probability(1600, 1800), 0.5)

    def test_builds_fixed_r32_bracket_from_schedule_slots(self):
        fixtures = load_group_stage_fixtures()
        teams = pd.read_csv(TEAMS_PATH)
        ratings, rating_col = load_default_ratings()
        fixture_predictions = precompute_fixture_predictions(
            fixtures=fixtures,
            ratings_df=ratings,
            rating_col=rating_col,
            base_total_goals=load_base_total_goals(),
        )
        grouped_fixtures = {
            group_letter: group.sort_values("match_number").to_dict("records")
            for group_letter, group in fixtures.groupby("group_letter")
        }
        grouped_teams = {
            group_letter: group.to_dict("records")
            for group_letter, group in teams.groupby("group_letter")
        }

        import random

        rng = random.Random(2026)
        ranked_groups = [
            _simulate_group_records(
                group_fixtures=grouped_fixtures[group_letter],
                group_teams=grouped_teams[group_letter],
                fixture_predictions=fixture_predictions,
                rng=rng,
            )
            for group_letter in sorted(grouped_fixtures)
        ]
        winners, runners_up, best_third = select_qualifier_records(ranked_groups)
        bracket = build_fixed_r32_bracket(
            winners=winners,
            runners_up=runners_up,
            best_third=best_third,
            knockout_schedule=load_knockout_schedule(),
        )

        teams_in_bracket = [team for match in bracket for team in [match["team_a"], match["team_b"]]]
        self.assertEqual(16, len(bracket))
        self.assertEqual(EXPECTED_TOTAL_QUALIFIERS, len(teams_in_bracket))
        self.assertEqual(EXPECTED_TOTAL_QUALIFIERS, len(set(teams_in_bracket)))
        self.assertEqual(list(range(73, 89)), [match["match_number"] for match in bracket])
        self.assertTrue(all(match["bracket_mode"] == BRACKET_MODE for match in bracket))
        self.assertTrue(all(match["uses_random_pairing"] is False for match in bracket))
        self.assertTrue(all(match["official_bracket"] is False for match in bracket))
        validate_round_pairings(
            "R32",
            [(match["team_a"], match["team_b"]) for match in bracket],
        )

    def test_tournament_monte_carlo_results_have_stage_probabilities(self):
        results = run_tournament_monte_carlo(
            n_simulations=100,
            seed=2026,
            output_path=Path("data/processed/test_tournament_simulation_results.csv"),
        )

        self.assertEqual(48, len(results))
        self.assertAlmostEqual(32, results["r32_probability"].sum())
        self.assertAlmostEqual(16, results["r16_probability"].sum())
        self.assertAlmostEqual(8, results["qf_probability"].sum())
        self.assertAlmostEqual(4, results["sf_probability"].sum())
        self.assertAlmostEqual(2, results["final_probability"].sum())
        self.assertAlmostEqual(1, results["champion_probability"].sum())
        self.assertIn("bracket_mapping_note", results.columns)
        self.assertIn("bracket_mode", results.columns)
        self.assertIn("bracket_source", results.columns)
        self.assertIn("uses_random_pairing", results.columns)
        self.assertIn("official_bracket", results.columns)
        self.assertIn("random_seed_used_for", results.columns)
        self.assertTrue((results["bracket_mapping_note"] == BRACKET_MAPPING_NOTE).all())
        self.assertTrue((results["bracket_mode"] == BRACKET_MODE).all())
        self.assertTrue((results["bracket_source"] == BRACKET_SOURCE).all())
        self.assertTrue((results["uses_random_pairing"] == USES_RANDOM_PAIRING).all())
        self.assertTrue((results["official_bracket"] == OFFICIAL_BRACKET).all())
        self.assertTrue((results["random_seed_used_for"] == str(RANDOM_SEED_USED_FOR)).all())
        sample_bracket = pd.read_csv(SAMPLE_KNOCKOUT_BRACKET_PATH)
        self.assertEqual(32, len(sample_bracket))
        self.assertEqual(16, int((sample_bracket["round"] == "R32").sum()))
        self.assertEqual(1, int((sample_bracket["round"] == "Third Place").sum()))
        run_tournament_sanity_checks(results, n_simulations=100)

    def test_winner_and_runner_up_placeholders_resolve_only_after_match(self):
        self.assertEqual(
            "France",
            resolve_knockout_placeholder("W73", winners={73: "France"}, losers={73: "Brazil"}),
        )
        self.assertEqual(
            "Brazil",
            resolve_knockout_placeholder("RU73", winners={73: "France"}, losers={73: "Brazil"}),
        )
        with self.assertRaisesRegex(ValueError, "before Match 74 was simulated"):
            resolve_knockout_placeholder("W74", winners={73: "France"}, losers={73: "Brazil"})

    def test_tournament_diagnostic_outputs_are_saved(self):
        run_tournament_monte_carlo(
            n_simulations=25,
            seed=2026,
            output_path=Path("data/processed/test_tournament_simulation_results.csv"),
            save_diagnostics=True,
            team_strength_diagnostics_path=Path("data/processed/test_diagnostics_team_strength.csv"),
            head_to_head_diagnostics_path=Path("data/processed/test_diagnostics_head_to_head.csv"),
            path_difficulty_diagnostics_path=Path("data/processed/test_diagnostics_path_difficulty.csv"),
            sample_bracket_path=Path("data/processed/test_sample_knockout_bracket.csv"),
        )

        strength = pd.read_csv("data/processed/test_diagnostics_team_strength.csv")
        head_to_head = pd.read_csv("data/processed/test_diagnostics_head_to_head.csv")
        path = pd.read_csv("data/processed/test_diagnostics_path_difficulty.csv")

        self.assertEqual(12, len(strength))
        self.assertIn("final_strength_used_by_poisson", strength.columns)
        self.assertIn("rank_by_final_strength", strength.columns)
        self.assertEqual(7, len(head_to_head))
        self.assertIn("top_5_scorelines", head_to_head.columns)
        self.assertEqual(48, len(path))
        self.assertIn("avg_r32_opponent_strength", path.columns)
        self.assertIn("avg_final_opponent_strength", path.columns)


if __name__ == "__main__":
    unittest.main()
