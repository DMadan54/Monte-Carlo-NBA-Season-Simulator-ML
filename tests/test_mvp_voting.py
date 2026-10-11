import unittest

import pandas as pd

from scripts.build_mvp_voting import chronological_folds, normalize


class VotingImportTests(unittest.TestCase):
    def test_duplicate_transcription_is_removed_but_conflict_fails(self):
        years = range(1981, 2026)
        rows = [dict(year=y, Player=f"Player {y}", Tm="ABC", Rank="1",
                     First=10, **{"Pts Won": 100, "Pts Max": 100, "Share": 1.0}) for y in years]
        frame = pd.DataFrame(rows + [rows[0]])
        votes = normalize(frame)
        self.assertEqual(len(votes), 45)
        self.assertEqual(len(chronological_folds(votes)), 8)
        conflict = pd.concat([frame, pd.DataFrame([{**rows[0], "Pts Won": 50, "Share": 0.5}])])
        with self.assertRaisesRegex(ValueError, "Duplicate player-season"):
            normalize(conflict)

    def test_future_voting_cannot_enter_prior_training_fold(self):
        rows = [dict(year=y, Player=f"Player {y}", Tm="ABC", Rank="1",
                     First=10, **{"Pts Won": 100, "Pts Max": 100, "Share": 1.0})
                for y in range(1981, 2026)]
        folds = chronological_folds(normalize(pd.DataFrame(rows)))
        self.assertTrue((folds.train_through_season_end_year < folds.test_season_end_year).all())
        self.assertEqual(folds.test_season_end_year.tolist(), list(range(2018, 2026)))


if __name__ == "__main__":
    unittest.main()
