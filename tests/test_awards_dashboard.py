import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard.awards import load_awards
from scripts.build_awards_example import DEFAULT_OUTPUT
from scripts.verify_portfolio_evidence import ROOT


class AwardsDashboardTests(unittest.TestCase):
    def test_bundled_holdout_is_complete(self):
        manifest, predictions, metrics = load_awards(DEFAULT_OUTPUT)
        self.assertEqual(manifest["season"], "2025-26")
        self.assertEqual(len(predictions), 1566)
        self.assertEqual(len(metrics), 54)
        self.assertEqual(predictions.groupby("cutoff_month").is_winner.sum().tolist(), [1, 1, 1])

    def test_rejects_changed_probability(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            folder = Path(temporary)
            for name in ("manifest.json", "season_metrics.csv"):
                (folder / name).write_bytes((DEFAULT_OUTPUT / name).read_bytes())
            predictions = pd.read_csv(DEFAULT_OUTPUT / "predictions_2026.csv")
            predictions.loc[0, "model_probability"] += 0.1
            predictions.to_csv(folder / "predictions_2026.csv", index=False)
            with self.assertRaisesRegex(ValueError, "sum to one"):
                load_awards(folder)

    def test_missing_awards_files_explains_requirements(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "manifest.json"):
                load_awards(temporary)


if __name__ == "__main__":
    unittest.main()
