import unittest

import pandas as pd

from src.models.mvp_cutoff_model import FEATURES, build_cutoff_candidates, evaluate


class MvpCutoffTests(unittest.TestCase):
    def test_future_games_cannot_change_cutoff_features_and_trade_team(self):
        player_rows = [
            dict(SEASON="2024-25", PLAYER_ID="1", PLAYER_NAME="Player One", TEAM_ID="A",
                 GAME_ID="a", GAME_DATE="2024-12-01", MIN=30, PTS=20, REB=5, AST=5,
                 STL=1, BLK=0, FGA=15, FTA=5, PLUS_MINUS=2),
            dict(SEASON="2024-25", PLAYER_ID="1", PLAYER_NAME="Player One", TEAM_ID="B",
                 GAME_ID="b", GAME_DATE="2025-02-20", MIN=32, PTS=24, REB=6, AST=6,
                 STL=1, BLK=1, FGA=17, FTA=6, PLUS_MINUS=3),
            dict(SEASON="2024-25", PLAYER_ID="1", PLAYER_NAME="Player One", TEAM_ID="B",
                 GAME_ID="c", GAME_DATE="2025-03-01", MIN=35, PTS=50, REB=10, AST=10,
                 STL=2, BLK=1, FGA=30, FTA=10, PLUS_MINUS=20)]
        team_rows = [dict(SEASON="2024-25", TEAM_ID=team, GAME_ID=game,
                          GAME_DATE=date, WL="W", PLUS_MINUS=5)
                     for team, game, date in [("A", "a", "2024-12-01"),
                                              ("B", "b", "2025-02-20"),
                                              ("B", "c", "2025-03-01")]]
        mapped = pd.DataFrame([dict(season="2024-25", player_id="1", vote_share=.5, is_winner=True)])
        for month, expected_games, expected_team, expected_points in [
                (1, 1, "A", 20), (2, 1, "A", 20), (3, 2, "B", 22)]:
            original = build_cutoff_candidates(pd.DataFrame(player_rows), pd.DataFrame(team_rows), mapped, month=month)
            changed = pd.DataFrame(player_rows)
            changed.loc[2, "PTS"] = 1000
            later = build_cutoff_candidates(changed, pd.DataFrame(team_rows), mapped, month=month)
            self.assertEqual(original.current_team_id.iloc[0], expected_team)
            self.assertEqual(original.games.iloc[0], expected_games)
            self.assertEqual(original.pts_per_game.iloc[0], expected_points)
            self.assertEqual(original.pts_per_game.iloc[0], later.pts_per_game.iloc[0])
            self.assertEqual(str(original.cutoff.iloc[0].date()), f"2025-{month:02d}-01")
            self.assertEqual(original.normal_path_open.iloc[0], 1)

    def test_later_winner_label_does_not_change_prior_forecast(self):
        rows = []
        for year in range(2016, 2026):
            for player in range(3):
                row = {name: float(player + 1 + year / 100) for name in FEATURES}
                row.update(season=f"{year-1}-{str(year)[-2:]}", season_end_year=year,
                           cutoff=pd.Timestamp(year, 3, 1), player_id=str(player),
                           player_name=f"Player {player}", current_team_id="A",
                           vote_share=0.8 if player == 2 else 0.0, is_winner=player == 2,
                           qualifying_games_so_far=40, remaining_team_games_assumed=20)
                rows.append(row)
        frame = pd.DataFrame(rows)
        original, _, _ = evaluate(frame)
        altered = frame.copy()
        altered.loc[altered.season_end_year == 2025, "is_winner"] = altered.loc[
            altered.season_end_year == 2025, "player_id"].eq("1")
        later, _, _ = evaluate(altered)
        before = original[original.season_end_year < 2025].model_probability.reset_index(drop=True)
        after = later[later.season_end_year < 2025].model_probability.reset_index(drop=True)
        pd.testing.assert_series_equal(before, after)


if __name__ == "__main__":
    unittest.main()
