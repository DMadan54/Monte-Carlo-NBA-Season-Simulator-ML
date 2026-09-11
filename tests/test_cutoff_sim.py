import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.models.cutoff_sim import simulate_cutoff, apply_trade
from src.models.player_game_model import TEAM_COLUMNS
from src.features.player_profiles import PROFILE


class RecordingClassifier:
    def __init__(self):
        self.calls = []

    def predict_proba(self, values):
        self.calls.append(values.copy())
        p = 1 / (1 + np.exp(-.1 * values.DIFF_ROSTER_PTS_36.to_numpy()))
        return np.column_stack([1 - p, p])


def scenario():
    snapshot, profiles = [], []
    for tid in ['A', 'B']:
        snapshot.append(dict(TEAM_ID=tid, GAME_DATE='2025-01-01', LAST_GAME_DATE='2024-12-29',
                             **{c: 0. for c in TEAM_COLUMNS}))
        for player in range(6):
            row = dict(TEAM_ID=tid, PLAYER_ID=tid + str(player), GAME_ID='snapshot', GAME_DATE='2025-01-01',
                       SOURCE_MAX_DATE='2024-12-29', BASE_MIN=30., ML_MIN=30., DAYS_ABSENT=3.)
            row.update({c: 1. for c in PROFILE})
            row['PTS_36'] = row['ML_PTS36'] = 25. if tid == 'A' else 15.
            row['ML_PTS36_SD'] = 4.
            profiles.append(row)
    schedule = pd.DataFrame([dict(GAME_ID=str(i), HOME_TEAM='A', AWAY_TEAM='B', NEUTRAL_GAME=0,
        GAME_DATE=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i * 2)) for i in range(4)])
    model = RecordingClassifier()
    bundle = dict(model=model, features=['DIFF_ROSTER_PTS_36'], name='hybrid_development_ml', trained_through='2024-06-01')
    return bundle, pd.DataFrame(snapshot), pd.DataFrame(profiles), schedule


class CutoffTests(unittest.TestCase):
    def test_conservation_reproducibility_and_expected_wins(self):
        args = scenario()
        first = simulate_cutoff(*args, '2025-01-01', {'A': 10, 'B': 12}, n_sims=5000, seed=12)
        second = simulate_cutoff(*args, '2025-01-01', {'A': 10, 'B': 12}, n_sims=5000, seed=12)
        assert_frame_equal(first, second)
        self.assertAlmostEqual(first.mean_wins.sum(), 26.)
        self.assertTrue(((first.mean_wins - first.expected_wins).abs() < .06).all())
        self.assertTrue({'p10', 'p25', 'p50', 'p75', 'p90'}.issubset(first.columns))
        self.assertTrue((first.p25 <= first.p50).all())
        self.assertTrue((first.p50 <= first.p75).all())

    def test_uncertainty_is_persistent_within_trial(self):
        args = scenario()
        simulate_cutoff(*args, '2025-01-01', n_sims=10, seed=12, player_points_sd=3)
        calls = args[0]['model'].calls
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(call.DIFF_ROSTER_PTS_36.nunique() == 1 for call in calls))
        self.assertGreater(len(set(float(call.iloc[0, 0]) for call in calls)), 1)

    def test_learned_uncertainty_requires_source_and_broadens_intervals(self):
        args = scenario()
        learned = simulate_cutoff(*args, '2025-01-01', n_sims=750, seed=12,
                                  learned_player_uncertainty=True)
        outcome_only = simulate_cutoff(*args, '2025-01-01', n_sims=750, seed=12)
        self.assertGreater(learned.std_wins.sum(), outcome_only.std_wins.sum())
        bundle, snapshot, profiles, schedule = scenario()
        profiles = profiles.drop(columns='ML_PTS36_SD')
        with self.assertRaises(ValueError):
            simulate_cutoff(bundle, snapshot, profiles, schedule, '2025-01-01',
                            learned_player_uncertainty=True)

    def test_rejects_future_sources_and_model(self):
        bundle, snapshot, profiles, schedule = scenario()
        profiles.loc[0, 'SOURCE_MAX_DATE'] = '2025-01-01'
        with self.assertRaises(ValueError):
            simulate_cutoff(bundle, snapshot, profiles, schedule, '2025-01-01')
        bundle, snapshot, profiles, schedule = scenario()
        bundle['trained_through'] = '2025-01-01'
        with self.assertRaises(ValueError):
            simulate_cutoff(bundle, snapshot, profiles, schedule, '2025-01-01')

    def test_trade_moves_player_once_and_reallocates(self):
        bundle, snapshot, profiles, schedule = scenario()
        moved = apply_trade(profiles, 'A0', 'B')
        self.assertEqual(len(moved), len(profiles))
        self.assertFalse(moved.PLAYER_ID.duplicated().any())
        self.assertEqual((moved.TEAM_ID == 'A').sum(), 5)
        result = simulate_cutoff(bundle, snapshot, moved, schedule, '2025-01-01', n_sims=10)
        self.assertAlmostEqual(result.mean_wins.sum(), 4)


if __name__ == '__main__':
    unittest.main()
