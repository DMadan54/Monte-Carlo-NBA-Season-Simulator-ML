import unittest
import pandas as pd
from src.models.roster_compatibility import compatibility, scenario_label

class CompatibilityTests(unittest.TestCase):
    def test_transferable_aggregate_features_and_noncausal_label(self):
        fields = dict(PTS_36=[20.], AST_36=[5.], FG3A_36=[7.], REB_36=[6.], TOV_36=[2.], BLK_36=[1.], EFG=[.55])
        result = compatibility(pd.DataFrame(fields), pd.DataFrame(fields))
        self.assertEqual(set(result), {'FIT_CREATE_SPACE','FIT_SHOOTING_EFFICIENCY','FIT_REBOUND_PROTECTION','FIT_TURNOVER_BURDEN'})
        self.assertIn('not a causal trade estimate', scenario_label().lower())
