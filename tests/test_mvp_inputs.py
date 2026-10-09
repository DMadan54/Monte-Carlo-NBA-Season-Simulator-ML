import unittest

import numpy as np
import pandas as pd

from src.models.mvp_inputs import historical_inputs, preseason_inputs, evidence_report


class MVPInputsTests(unittest.TestCase):
    def fixtures(self):
        players = pd.DataFrame([
            dict(season='2021-22', player_id='a', player_name='A', source_max_date='2022-04-10', last_team_id='t',
                 points=100, REB=20, AST=30, minutes=200, games=10, FGA=80, FTA=20),
            dict(season='2021-22', player_id='b', player_name='B', source_max_date='2022-04-10', last_team_id='t',
                 points=0, REB=0, AST=0, minutes=0, games=1, FGA=0, FTA=0)])
        teams = pd.DataFrame([dict(season='2021-22', team_id='t', wins=40, games=72)])
        return players, teams

    def test_feature_formulas_and_undefined_efficiency(self):
        p, t = self.fixtures()
        result = historical_inputs(p, t).set_index('player_id')
        self.assertEqual(result.loc['a', 'assists_per_game'], 3)
        self.assertEqual(result.loc['a', 'rebounds_per_game'], 2)
        self.assertAlmostEqual(result.loc['a', 'true_shooting'], 100/(2*(80+.44*20)))
        self.assertEqual(result.loc['a', 'relative_true_shooting'], 0)
        self.assertEqual(result.loc['a', 'team_win_pct'], 40/72)
        self.assertEqual(result.loc['a', 'games_share'], 10/72)
        self.assertTrue(np.isnan(result.loc['b', 'true_shooting']))
        self.assertEqual(result.loc['b', 'games'], 1)

    def test_missing_teams_and_partial_seasons_fail(self):
        p, t = self.fixtures()
        with self.assertRaises(ValueError):
            historical_inputs(p, t.iloc[:0])
        p.loc[0, 'AST'] = np.nan
        with self.assertRaises(ValueError):
            historical_inputs(p, t)

    def test_preseason_ignores_future_population_and_rejects_future_history(self):
        p, t = self.fixtures()
        history = historical_inputs(p, t)
        future = history.iloc[:1].assign(season='2022-23', player_id='new', ppg=1000)
        field = preseason_inputs(pd.concat([history, future]), '2022-23', '2022-10-01')
        self.assertEqual(set(field.player_id), {'a', 'b'})
        history.loc[0, 'source_max_date'] = '2022-10-01'
        with self.assertRaises(ValueError):
            preseason_inputs(history, '2022-23', '2022-10-01')

    def test_evidence_does_not_fill_absent_votes_or_use_target_metrics(self):
        p, t = self.fixtures()
        prior = historical_inputs(p, t)
        target = prior.assign(season='2022-23', source_max_date='2023-04-10', ppg=[0, 999])
        labels = pd.DataFrame([dict(season='2022-23', award='mvp', player_id='a', winner=True,
                                   vote_points=100, label_available_at='2023-05-01')])
        result = evidence_report(pd.concat([prior, target]), labels)
        ppg = next(row for row in result['preseason_checks'] if row['feature'] == 'ppg')
        self.assertEqual(ppg['winner_rank'], 1)
        self.assertEqual(result['recipient_associations'][0]['observed_recipients'], 1)
        self.assertFalse(result['probability_release_approved'])
        self.assertEqual(len(labels), 1)


if __name__ == '__main__':
    unittest.main()
