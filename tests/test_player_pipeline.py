import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.player_profiles import build_candidates, project_minutes, INPUTS
from src.ingest.player_data import load_dated_context


def fixture():
    teams, players = [], []
    for day in range(5):
        for team in ['A', 'B']:
            date = pd.Timestamp('2023-11-01') + pd.Timedelta(days=day * 2)
            teams.append(dict(GAME_ID=str(day), TEAM_ID=team, GAME_DATE=date, SEASON_YEAR=2024, MIN=240))
            for number in range(6):
                pid = team + str(number)
                # Swap two players starting on day 3; cannot be anticipated from logs.
                if day >= 3 and number == 0:
                    pid = ('B' if team == 'A' else 'A') + '0'
                players.append(dict(GAME_ID=str(day), TEAM_ID=team, PLAYER_ID=pid, GAME_DATE=date,
                    MIN=40, PTS=20, REB=5, AST=3, STL=1, BLK=1, TOV=2, FGA=15, FGM=8, FG3A=5, FG3M=2))
    return pd.DataFrame(teams), pd.DataFrame(players)


class PlayerPipelineTests(unittest.TestCase):
    def test_rotation_constraints(self):
        result = project_minutes([200, 10, 3, 0, 0, 0])
        self.assertAlmostEqual(result.sum(), 240)
        self.assertTrue(((result >= 0) & (result <= 48)).all())
        self.assertAlmostEqual(project_minutes(np.zeros(10)).sum(), 240)
        with self.assertRaises(ValueError):
            project_minutes([1, 2, 3, 4])

    def test_future_player_stats_do_not_change_pregame_features(self):
        team, players = fixture()
        cutoff = pd.Timestamp('2023-11-05')
        mutated = players.copy()
        mutated.loc[mutated.GAME_DATE >= cutoff, ['MIN', 'PTS']] = [1, 60]
        before = build_candidates(team, players)
        after = build_candidates(team, mutated)
        columns = ['GAME_ID', 'PLAYER_ID', 'TEAM_ID', *INPUTS]
        assert_frame_equal(before.loc[before.GAME_DATE <= cutoff, columns], after.loc[after.GAME_DATE <= cutoff, columns])

    def test_trade_is_not_known_until_observed(self):
        team, players = fixture()
        candidates = build_candidates(team, players)
        trade_day = candidates[(candidates.GAME_ID == '3') & (candidates.TEAM_ID == 'A')]
        self.assertIn('A0', set(trade_day.PLAYER_ID))
        self.assertNotIn('B0', set(trade_day.PLAYER_ID))
        next_game = candidates[(candidates.GAME_ID == '4') & (candidates.TEAM_ID == 'A')]
        self.assertIn('B0', set(next_game.PLAYER_ID))
        self.assertNotIn('A0', set(next_game.PLAYER_ID))
        self.assertEqual(float(trade_day.loc[trade_day.PLAYER_ID == 'A0', 'TARGET_MIN'].iloc[0]), 0)

    def test_invalid_context_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'coaches.csv'
            with self.assertRaises(FileNotFoundError):
                load_dated_context(path, 'coaches')
            pd.DataFrame([dict(TEAM_ID='A', COACH_ID='x', START_DATE='2023-01-01', END_DATE='2024-01-01',
                               KNOWN_AT='2023-01-01', SOURCE='fixture'),
                          dict(TEAM_ID='A', COACH_ID='y', START_DATE='2023-06-01', END_DATE='2025-01-01',
                               KNOWN_AT='2023-06-01', SOURCE='fixture')]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_dated_context(path, 'coaches')


if __name__ == '__main__':
    unittest.main()
