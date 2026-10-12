import unittest

import pandas as pd

from scripts.build_mvp_2026_votes import validated_votes
from src.models.mvp_cutoff_model import FEATURES, evaluate


class NewHoldoutTests(unittest.TestCase):
    def test_official_vote_transcription_reconciles(self):
        votes = validated_votes()
        self.assertEqual(len(votes), 8)
        self.assertEqual(votes.vote_points.sum(), 2600)
        self.assertEqual(votes.loc[votes.is_winner, "player_name"].iloc[0], "Shai Gilgeous-Alexander")
        self.assertEqual(votes.vote_share.max(), .939)

    def test_2026_vote_outcome_is_not_an_input_to_2026_prediction(self):
        rows = []
        for year in range(2016, 2027):
            for player in range(3):
                row = {feature: float(player + 1 + year / 100) for feature in FEATURES}
                row.update(season=f"{year-1}-{str(year)[-2:]}", season_end_year=year,
                           cutoff=pd.Timestamp(year, 3, 1), player_id=str(player),
                           player_name=f"Player {player}", current_team_id="A",
                           vote_share=.8 if player == 2 else 0., is_winner=player == 2,
                           qualifying_games_so_far=40, remaining_team_games_assumed=20)
                rows.append(row)
        frame = pd.DataFrame(rows)
        original, _, _ = evaluate(frame, first_test=2026)
        changed = frame.copy()
        changed.loc[changed.season_end_year == 2026, "is_winner"] = changed.loc[
            changed.season_end_year == 2026, "player_id"].eq("1")
        changed.loc[changed.season_end_year == 2026, "vote_share"] = [0, .8, 0]
        later, _, _ = evaluate(changed, first_test=2026)
        pd.testing.assert_series_equal(original.model_probability, later.model_probability)
        pd.testing.assert_series_equal(original.baseline_probability, later.baseline_probability)


if __name__ == "__main__":
    unittest.main()
