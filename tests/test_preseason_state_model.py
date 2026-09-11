import unittest
from src.models.preseason_state_model import participation_metrics

class PreseasonStateTests(unittest.TestCase):
    def test_participation_keeps_exits_and_is_bounded(self):
        result = participation_metrics([0, 600], [0, 600])
        self.assertEqual(result['actual_rate'], .5)
        self.assertEqual(result['brier'], 0.)
