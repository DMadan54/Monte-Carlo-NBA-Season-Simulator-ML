import unittest

import pandas as pd

from src.models.mvp_model import build_candidates, evaluate, resolve_votes


class MvpModelTests(unittest.TestCase):
    def test_alias_ambiguity_and_zero_vote_candidates(self):
        players = pd.DataFrame([
            {"SEASON": "2024-25", "PLAYER_ID": "1", "PLAYER_NAME": "Jimmy Butler III", "TEAM_ID": "10", "TEAM_ABBREVIATION": "MIA", "GAME_ID": "a",
             "MIN": 30, "PTS": 20, "REB": 5, "AST": 4, "STL": 1, "BLK": 0, "FGA": 15, "FTA": 5, "PLUS_MINUS": 2},
            {"SEASON": "2024-25", "PLAYER_ID": "2", "PLAYER_NAME": "Other Player", "TEAM_ID": "10", "TEAM_ABBREVIATION": "MIA", "GAME_ID": "a",
             "MIN": 20, "PTS": 5, "REB": 2, "AST": 1, "STL": 0, "BLK": 0, "FGA": 5, "FTA": 0, "PLUS_MINUS": -2}])
        votes = pd.DataFrame([{"season": "2024-25", "player_name": "Jimmy Butler", "team_abbr": "MIA",
                               "vote_share": 0.5, "is_winner": True}])
        aliases = pd.DataFrame([{"vote_name": "Jimmy Butler", "player_id": "1"}])
        mapped, issues = resolve_votes(votes, players, aliases)
        self.assertTrue(issues.empty)
        self.assertEqual(mapped.player_id.iloc[0], "1")
        teams = pd.DataFrame([{"SEASON": "2024-25", "TEAM_ID": "10", "GAME_ID": "a", "WL": "W"}])
        candidates = build_candidates(players, teams, mapped)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates.loc[candidates.player_id == "2", "vote_share"].iloc[0], 0)
        ambiguous = pd.concat([players, players.iloc[[0]].assign(PLAYER_ID="3", PLAYER_NAME="Jimmy Butler"),
                               players.iloc[[0]].assign(PLAYER_ID="4", PLAYER_NAME="Jimmy Butler")])
        _, issues = resolve_votes(votes, ambiguous, pd.DataFrame(columns=["vote_name", "player_id"]))
        self.assertEqual(issues.reason.iloc[0], "ambiguous")

    def test_future_target_mutation_does_not_change_prior_predictions(self):
        from src.models.mvp_model import FEATURES
        rows = []
        for year in range(2016, 2026):
            for player in range(3):
                row = {name: float(player + 1 + year / 100) for name in FEATURES}
                row.update(season=f"{year-1}-{str(year)[-2:]}", season_end_year=year,
                           player_id=str(player), player_name=f"Player {player}",
                           vote_share=0.8 if player == 2 else 0.0, is_winner=player == 2)
                rows.append(row)
        frame = pd.DataFrame(rows)
        original, _, _ = evaluate(frame)
        changed = frame.copy()
        changed.loc[changed.season_end_year == 2025, "vote_share"] = [0.5, 0.5, 0.5]
        later, _, _ = evaluate(changed)
        pd.testing.assert_series_equal(original.loc[original.season_end_year < 2025, "pred_share"].reset_index(drop=True),
                                       later.loc[later.season_end_year < 2025, "pred_share"].reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
