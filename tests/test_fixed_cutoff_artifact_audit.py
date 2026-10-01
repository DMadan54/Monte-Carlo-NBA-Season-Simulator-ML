import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import unittest

from scripts.audit_fixed_cutoff_artifact import (canonical_game_id,
    multi_cutoff_manifest, validate_calibration_inputs)


class FixedCutoffArtifactAuditTests(unittest.TestCase):
    def test_canonical_game_id_restores_csv_leading_zeroes(self):
        values = pd.Series([22400556, "0022400557", "22400558.0"])

        self.assertEqual(canonical_game_id(values).tolist(), [
            "0022400556",
            "0022400557",
            "0022400558",
        ])

    def _inputs(self):
        schedule = pd.DataFrame([dict(GAME_ID='0022400001', GAME_DATE='2025-01-02', HOME_TEAM=1,
                                      AWAY_TEAM=2, TARGET=1),
                                 dict(GAME_ID='0022400002', GAME_DATE='2025-01-03', HOME_TEAM=2,
                                      AWAY_TEAM=1, TARGET=0)])
        snapshot = pd.DataFrame([dict(TEAM_ID=1, GAME_DATE='2025-01-01', LAST_GAME_DATE='2024-12-30'),
                                 dict(TEAM_ID=2, GAME_DATE='2025-01-01', LAST_GAME_DATE='2024-12-29')])
        profiles = pd.DataFrame([dict(PLAYER_ID=1, SOURCE_MAX_DATE='2024-12-30')])
        values = pd.DataFrame({'DIFF_FORM': [.2, -.2]})
        bundle = {'features': ['DIFF_FORM']}
        probabilities = [.7, .3]
        return schedule, snapshot, profiles, values, bundle, probabilities

    def test_manifest_rejects_swapped_teams_and_post_cutoff_sources(self):
        schedule, snapshot, profiles, values, bundle, probabilities = self._inputs()
        swapped = schedule.copy()
        swapped[['HOME_TEAM', 'AWAY_TEAM']] = swapped[['AWAY_TEAM', 'HOME_TEAM']]
        first = multi_cutoff_manifest(schedule, snapshot, profiles, values, probabilities, bundle, '2025-01-01')
        second = multi_cutoff_manifest(swapped, snapshot, profiles, values, probabilities, bundle, '2025-01-01')
        self.assertNotEqual(first['ordered_games_hash'], second['ordered_games_hash'])
        profiles.loc[0, 'SOURCE_MAX_DATE'] = '2025-01-01'
        with self.assertRaisesRegex(ValueError, 'source'):
            multi_cutoff_manifest(schedule, snapshot, profiles, values, probabilities, bundle, '2025-01-01')

    def test_manifest_rejects_model_feature_schema_mismatch(self):
        schedule, snapshot, profiles, values, bundle, probabilities = self._inputs()
        with self.assertRaisesRegex(ValueError, 'feature columns'):
            multi_cutoff_manifest(schedule, snapshot, profiles,
                                  values.rename(columns={'DIFF_FORM': 'DIFF_OTHER'}),
                                  probabilities, bundle, '2025-01-01')

    def test_calibration_validation_rejects_target_order_and_probability_changes(self):
        schedule, snapshot, profiles, values, bundle, probabilities = self._inputs()
        manifest = multi_cutoff_manifest(schedule, snapshot, profiles, values, probabilities, bundle, '2025-01-01')
        rows = pd.DataFrame({
            'season': ['2024-25'] * 2, 'cutoff': ['2025-01-01'] * 2,
            'model': ['hybrid_development_ml'] * 2, 'learned_uncertainty': [False] * 2,
            'game_id': schedule.GAME_ID, 'game_date': schedule.GAME_DATE,
            'home_team': schedule.HOME_TEAM, 'away_team': schedule.AWAY_TEAM,
            'target': schedule.TARGET, 'probability': probabilities,
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); audits = root / 'scoring_audits'; audits.mkdir()
            (audits / '2024-25_20250101_hybrid_development_ml.json').write_text(json.dumps(manifest))
            artifact = root / 'game_probabilities.parquet'; rows.to_parquet(artifact, index=False)
            stored = pd.read_parquet(artifact)
            validate_calibration_inputs(root, stored)
            changed_target = stored.iloc[::-1].reset_index(drop=True)
            with self.assertRaisesRegex(ValueError, 'identity/order'):
                validate_calibration_inputs(root, changed_target)
            changed_probability = stored.copy(); changed_probability.loc[0, 'probability'] = .6
            with self.assertRaisesRegex(ValueError, 'probability output'):
                validate_calibration_inputs(root, changed_probability)

    def test_probability_parquet_round_trip_is_bit_exact(self):
        original = np.array([0.1, np.nextafter(0.1, 1.0), 0.9999999999999999], dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'probabilities.parquet'
            pd.DataFrame({'probability': original}).to_parquet(path, index=False)
            restored = pd.read_parquet(path).probability.to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(original.view(np.uint64), restored.view(np.uint64))
