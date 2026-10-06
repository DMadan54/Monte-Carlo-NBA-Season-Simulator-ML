"""Exercise saved-artifact UI states; no network or model training."""
from pathlib import Path
import tempfile
import json
import unittest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'dashboard/app.py'


class AwardsDashboardTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(APP), default_timeout=20).run()

    def test_upcoming_missing_season_awards_and_roster(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any('incomplete' in x.value for x in app.warning))
        app.radio[0].set_value('Awards').run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(sum('Incomplete' in e.label for e in app.expander), 5)
        self.assertEqual(len(app.dataframe[0].value), 12)
        app.radio[0].set_value('Teams / players').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 573)

    def test_missing_and_mismatched_artifacts_are_readable(self):
        app = self.app()
        app.radio[0].set_value('Awards').run()
        app.sidebar.text_input[1].set_value('does-not-exist.json').run()
        self.assertTrue(any('no saved awards artifact' in x.value for x in app.info))
        with tempfile.TemporaryDirectory() as temp:
            bad = Path(temp) / 'bad.json'
            bad.write_text(json.dumps({'schema_version': 1, 'forecast_version': 'wrong'}))
            app.sidebar.text_input[1].set_value(str(bad)).run()
            self.assertTrue(any('version mismatch' in e.value for e in app.error))
            self.assertFalse(app.exception)

    def test_historical_scenario_without_awards_keeps_standings(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'report.json').write_text(json.dumps({'cutoff': '2025-01-01', 'n_sims': 20, 'games_remaining': 4}))
            (path / 'standings.csv').write_text('TEAM_ID,team,mean_wins,p05,p95,actual_wins\n1,A,2,0,4,3\n2,B,2,0,4,1\n')
            app = self.app()
            app.sidebar.selectbox[0].set_value('Custom version').run()
            app.sidebar.text_input[0].set_value(str(path)).run()
            self.assertFalse(app.error)
            self.assertEqual(app.dataframe[0].value.mean_wins.sum(), 4)
            app.radio[0].set_value('Awards').run()
            self.assertTrue(any('no saved awards artifact' in x.value for x in app.info))
            self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
