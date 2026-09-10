import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from scripts.run_development_experiment import annual_dataset
from src.models.coach_model import attach_coaches


class ContextTests(unittest.TestCase):
    def test_annual_features_ignore_future_stats_and_exits_keep_zero_minutes(self):
        players = pd.DataFrame([dict(PLAYER_ID=pid, SEASON_YEAR=year, MIN=1000, PTS=500,
            REB=200, AST=100, GAME_ID=f'{pid}_{year}') for pid in ['A', 'B'] for year in range(2020, 2026)
            if not (pid == 'B' and year == 2025)])
        births = pd.DataFrame({'PLAYER_ID': ['A', 'B'], 'BIRTH_DATE': pd.to_datetime(['1995-01-01', '1997-01-01'])})
        baseline = annual_dataset(players, births)
        changed = players.copy()
        changed.loc[changed.SEASON_YEAR == 2025, 'PTS'] = 99999
        mutated = annual_dataset(changed, births)
        features = ['PLAYER_ID', 'SEASON_YEAR', 'PTS_36', 'PRIOR_PTS36', 'PTS_CHANGE', 'AGE_NEXT_OCT']
        assert_frame_equal(baseline[features], mutated[features])
        exit_row = baseline[(baseline.PLAYER_ID == 'B') & (baseline.TARGET_SEASON == 2025)].iloc[0]
        self.assertEqual(exit_row.TARGET_MIN, 0)
        self.assertTrue(pd.isna(exit_row.TARGET_PTS36))

    def test_coach_assignment_uses_known_at_not_future_announcement(self):
        games = pd.DataFrame({'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-02-01']),
                              'HOME_TEAM': ['A', 'A'], 'AWAY_TEAM': ['B', 'B']})
        assignments = pd.DataFrame({'TEAM_ID': ['A'], 'COACH_ID': ['coach'],
            'START_DATE': pd.to_datetime(['2024-01-01']), 'END_DATE': pd.to_datetime(['2025-01-01']),
            'KNOWN_AT': pd.to_datetime(['2024-01-15'])})
        result = attach_coaches(games, assignments)
        self.assertEqual(result.HOME_COACH.tolist(), ['UNKNOWN', 'coach'])


if __name__ == '__main__':
    unittest.main()
