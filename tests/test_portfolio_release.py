import json
import tempfile
import unittest
from pathlib import Path

from dashboard.artifact import load_artifact
from scripts.verify_portfolio_evidence import EXAMPLE, GAME_RUN, MULTI_RUN, MVP_RUN, ROOT, SCENARIO_RUN, verify


class PortfolioReleaseTests(unittest.TestCase):
    def test_bundled_scenario_smoke(self):
        report, standings = load_artifact(EXAMPLE)
        self.assertEqual(len(standings), 30)
        self.assertEqual(report["n_sims"], 2000)

    @unittest.skipUnless(all(path.is_dir() for path in (SCENARIO_RUN, GAME_RUN, MULTI_RUN, MVP_RUN)),
                         "Local experiment runs are not bundled with Git")
    def test_saved_evidence_audit(self):
        self.assertEqual(verify()["seven_season_28_cutoff_backtest"]["cutoffs"], 28)

    def test_rejects_changed_league_wins(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary)
            report, standings = load_artifact(EXAMPLE)
            (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
            changed = standings.copy()
            changed.loc[0, ["mean_wins", "p95"]] += 1
            changed.to_csv(path / "standings.csv", index=False)
            with self.assertRaisesRegex(ValueError, "conserved"):
                load_artifact(path)

    def test_missing_artifact_explains_required_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "report.json and standings.csv"):
                load_artifact(temporary)


if __name__ == "__main__":
    unittest.main()
