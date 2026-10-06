import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.models.awards import (aggregate_choice_draws, attach_votes, build_awards,
                               eligibility, leading_probabilities, load_awards,
                               save_awards, validate_joint)
from src.models.award_evaluation import preseason_baselines, ranking_metrics
from src.models.cutoff_sim import simulate_cutoff, apply_trade
from test_cutoff_sim import scenario

ROOT = Path(__file__).resolve().parents[1]


class EligibilityTests(unittest.TestCase):
    def test_qualifying_games_and_two_short_game_limit(self):
        self.assertTrue(eligibility('mvp', '2026-27', minutes=[20]*63+[15]*2))
        self.assertFalse(eligibility('mvp', '2026-27', minutes=[20]*62+[19]*3, exception='none'))
        self.assertFalse(eligibility('mip', '2026-27', minutes=[19.99]*65, exception='none'))
        self.assertIsNone(eligibility('dpoy', '2026-27', minutes=[20]*64))

    def test_no_inferred_injury_or_rookie_exceptions(self):
        self.assertIsNone(eligibility('mvp', '2026-27', minutes=[20]*62))
        self.assertTrue(eligibility('mvp', '2026-27', exception='approved'))
        self.assertIsNone(eligibility('roy', '2026-27'))
        self.assertFalse(eligibility('roy', '2026-27', rookie=False))
        self.assertTrue(eligibility('roy', '2026-27', rookie=True))
        self.assertIsNone(eligibility('mvp', '2022-23', minutes=[20]*70))

    def test_sixth_man_strict_bench_majority_no_65_game_gate(self):
        self.assertTrue(eligibility('sixth', '2026-27', minutes=[10]*11, starts=5))
        self.assertFalse(eligibility('sixth', '2026-27', minutes=[10]*10, starts=5))
        self.assertIsNone(eligibility('sixth', '2026-27', minutes=[30]*80))
        with self.assertRaises(ValueError):
            eligibility('sixth', '2026-27', minutes=[10], starts=2)

    def test_invalid_season_and_nonfinite_minutes(self):
        for season in ['2026-28', 'unknown']:
            with self.assertRaises(ValueError):
                eligibility('mvp', season)
        with self.assertRaises(ValueError):
            eligibility('mvp', '2026-27', minutes=[np.nan])


class AwardAccountingTests(unittest.TestCase):
    def test_joint_probability_integration_and_other_preserve_mass(self):
        field = aggregate_choice_draws([[.6, .3, .1], [.2, .5, .3]], ['A', 'B', 'C'])
        np.testing.assert_allclose(field.probability, [.4, .4, .2])
        top = leading_probabilities(field, 1)
        self.assertAlmostEqual(top.probability.sum(), 1)
        self.assertAlmostEqual(top[top.player_id == 'Other'].probability.iloc[0], .6)

    def test_partial_or_invalid_probability_field_rejected(self):
        for values in [[[.2, .3]], [[np.nan, 1]], [[-.2, 1.2]], []]:
            with self.assertRaises(ValueError):
                aggregate_choice_draws(values, ['A', 'B'])
        with self.assertRaises(ValueError):
            aggregate_choice_draws([[.5, .5]], ['A', 'A'])

    def test_unlisted_is_not_verified_zero(self):
        population = pd.DataFrame({'season': ['2024-25']*2, 'award': ['mvp']*2, 'player_id': ['A', 'B']})
        votes = population.iloc[:1].assign(vote_points=0)
        result = attach_votes(population, votes)
        self.assertEqual(result.iloc[0].vote_status, 'observed')
        self.assertEqual(result.iloc[1].vote_status, 'unlisted_unknown')
        self.assertTrue(pd.isna(result.iloc[1].vote_points))
        with self.assertRaises(ValueError):
            attach_votes(population, votes.assign(player_id='C'))


class SharedDrawTests(unittest.TestCase):
    def check_export(self, **options):
        args = scenario()
        joint = {}
        actual = simulate_cutoff(*args, '2025-01-01', n_sims=40, seed=12, joint_draws=joint, **options)
        control = simulate_cutoff(*scenario(), '2025-01-01', n_sims=40, seed=12, **options)
        assert_frame_equal(actual, control)
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'joint.npz'
            np.savez_compressed(file, **joint)
            with np.load(file, allow_pickle=False) as saved:
                validate_joint(saved, actual, args[2], '2025-01-01')
                np.testing.assert_allclose(saved['team_wins'].sum(axis=1), 4)
                if options:
                    self.assertGreater(float(saved['player_pts36'].std(axis=0).max()), 0)
        return actual, joint, args

    def test_export_preserves_all_simulation_paths(self):
        self.check_export()
        self.check_export(learned_player_uncertainty=True)
        self.check_export(player_points_sd=3)

    def test_draws_are_the_same_rates_used_by_classifier(self):
        actual, joint, args = self.check_export(player_points_sd=3)
        for i, call in enumerate(args[0]['model'].calls):
            rates, memberships = joint['player_pts36'][i], joint['player_team_ids']
            diff = rates[memberships == 'A'].mean() - rates[memberships == 'B'].mean()
            np.testing.assert_allclose(call.DIFF_ROSTER_PTS_36, diff)

    def test_rejects_wrong_team_and_cutoff(self):
        actual, joint, args = self.check_export()
        # numpy scalar conversion matches NPZ reader semantics.
        joint['cutoff'] = np.asarray(joint['cutoff'])
        with self.assertRaises(ValueError):
            validate_joint(joint, actual, apply_trade(args[2], 'A0', 'B'), '2025-01-01')
        with self.assertRaises(ValueError):
            validate_joint(joint, actual, args[2], '2025-01-02')
        with self.assertRaises(ValueError):
            validate_joint(joint, actual.assign(mean_wins=0), args[2], '2025-01-01')


class ChronologyTests(unittest.TestCase):
    def history(self):
        return pd.DataFrame([dict(season=s, player_id=p, ppg=v, points=100*v, source_max_date=d)
                             for s, d in [('2021-22', '2022-04-10'), ('2022-23', '2023-04-09'), ('2023-24', '2024-04-14')]
                             for p, v in [('A', 10.), ('B', 20.)]])

    def test_future_stats_do_not_change_preseason_inputs(self):
        history = self.history()
        original = preseason_baselines(history, '2023-24', '2023-10-01')
        history.loc[history.season == '2023-24', ['ppg', 'points']] = 99999
        assert_frame_equal(original, preseason_baselines(history, '2023-24', '2023-10-01'))
        history.loc[history.season == '2022-23', 'source_max_date'] = '2023-10-01'
        with self.assertRaises(ValueError):
            preseason_baselines(history, '2023-24', '2023-10-01')

    def test_missing_winner_is_a_miss_and_ties_are_average_rank(self):
        field = pd.DataFrame({'player_id': ['A', 'B', 'C'], 'score': [1, 1, 2]})
        self.assertEqual(ranking_metrics(field, 'score', 'A')['winner_rank'], 2.5)
        self.assertFalse(ranking_metrics(field, 'score', 'D')['winner_present'])
        self.assertEqual(ranking_metrics(field, 'score', 'D')['reciprocal_rank'], 0)


class ReleaseTests(unittest.TestCase):
    def test_saved_release_is_incomplete_and_version_bound(self):
        source = ROOT / 'data/raw/live_preseason/v1_20260922T113549Z_c79af6'
        release = ROOT / 'results/awards/v3_20261006/awards.json'
        artifact = load_awards(release, source)
        self.assertEqual(artifact['forecast_cutoff'], '2026-09-22T11:35:49.701781+00:00')
        self.assertEqual(len(artifact['awards']), 5)
        for award in artifact['awards'].values():
            self.assertIsNone(award['probabilities'])
            self.assertFalse(award['candidates'])
            self.assertTrue(award['missing_requirements'])

    def test_no_overwrite_and_rejects_source_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'version'
            source.mkdir()
            manifest = source / 'source_metadata.json'
            manifest.write_text(json.dumps({'retrieved_at_utc': '2026-09-22T00:00:00Z', 'files': {}}))
            output = Path(temp) / 'awards.json'
            save_awards(source, output)
            with self.assertRaises(FileExistsError):
                save_awards(source, output)
            manifest.write_text(json.dumps({'retrieved_at_utc': '2026-09-23T00:00:00Z', 'files': {}}))
            with self.assertRaises(ValueError):
                load_awards(output, source)

    def test_historical_artifacts_without_draws_are_not_upgraded(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / 'report.json').write_text(json.dumps({'cutoff': '2025-01-01', 'season': '2024-25'}))
            artifact = build_awards(source)
            self.assertIn('Missing shared artifact: joint_draws.npz', artifact['awards']['mvp']['missing_requirements'])

    def test_real_history_preserves_population_and_reproduces_evaluation(self):
        from src.models.award_evaluation import evaluate_preseason
        path = ROOT / 'data/processed/awards/history_v4_20261006'
        candidates = pd.read_parquet(path / 'candidate_history.parquet')
        self.assertEqual(len(candidates), 8400)
        self.assertEqual(candidates[candidates.season == '2023-24'].player_id.nunique(), 572)
        self.assertTrue(candidates.loc[candidates.vote_status == 'unlisted_unknown', 'vote_points'].isna().all())
        history = pd.read_parquet(path / 'player_history.parquet')
        votes = pd.DataFrame(json.loads((path / 'votes.json').read_text(encoding='utf-8')))
        expected = json.loads((path / 'evaluation.json').read_text())
        self.assertEqual(evaluate_preseason(history, votes), expected['rows'])


if __name__ == '__main__':
    unittest.main()
