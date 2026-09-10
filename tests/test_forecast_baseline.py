import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.build_team_features import add_basic_fields, add_rolling_features


def logs():
    rows = []
    for game in range(7):
        for team in [1, 2]:
            rows.append(dict(GAME_ID=str(game), TEAM_ID=team,
                GAME_DATE=pd.Timestamp('2020-09-27') + pd.Timedelta(days=game * 2),
                SEASON='2019-20', WL='W' if team == 1 else 'L',
                PTS=110 if team == 1 else 90, FGM=40, FGA=85, FG3M=10,
                TOV=12, OREB=10, DREB=30 if team == 1 else 40, FTM=20, FTA=25,
                MATCHUP='AAA vs. BBB' if team == 1 else 'BBB @ AAA'))
    return pd.DataFrame(rows)


class BaselineTests(unittest.TestCase):
    def test_bubble_season_and_rebounds(self):
        frame = add_basic_fields(logs())
        self.assertEqual(set(frame.SEASON_YEAR), {2020})
        home = frame[frame.TEAM_ID == 1].iloc[0]
        self.assertAlmostEqual(home.ORB_PCT, 10 / 50)
        self.assertAlmostEqual(home.OPP_ORB_PCT, 10 / 40)

    def test_current_and_future_results_do_not_change_pregame_features(self):
        original = logs()
        changed = original.copy()
        cutoff = original.GAME_DATE.unique()[4]
        changed.loc[changed.GAME_DATE >= cutoff, 'PTS'] += 50
        changed.loc[changed.GAME_DATE >= cutoff, 'WL'] = 'L'
        a = add_rolling_features(add_basic_fields(original))
        b = add_rolling_features(add_basic_fields(changed))
        columns = [c for c in a if c.startswith('ROLL_') or c in
                   ['ELO_RATING', 'OPP_ELO_RATING', 'SEASON_POINT_DIFF', 'EMA_POINT_DIFF']]
        assert_frame_equal(a.loc[a.GAME_DATE <= cutoff, columns], b.loc[b.GAME_DATE <= cutoff, columns])

    def test_requires_authoritative_season(self):
        with self.assertRaises(ValueError):
            add_basic_fields(logs().drop(columns='SEASON'))

    def test_neutral_game_has_one_orientation_and_equal_initial_elo(self):
        raw = logs().iloc[:2].copy()
        raw['MATCHUP'] = ['AAA @ BBB', 'BBB @ AAA']
        frame = add_rolling_features(add_basic_fields(raw))
        self.assertEqual(int(frame.IS_HOME.sum()), 1)
        self.assertTrue(frame.NEUTRAL_GAME.eq(1).all())
        self.assertTrue(frame.ELO_RATING.eq(1500).all())


if __name__ == '__main__':
    unittest.main()
