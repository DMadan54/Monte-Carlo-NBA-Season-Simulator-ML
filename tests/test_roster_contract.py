import unittest
import pandas as pd
from src.ingest.roster_contract import reconstruct_roster, validate_events


def events(rows):
    return pd.DataFrame(rows, columns=['PLAYER_ID','TEAM_ID','EVENT_TYPE','EFFECTIVE_DATE','KNOWN_AT','SOURCE_ID','RETRIEVED_AT','STATUS','CONFIDENCE'])


class RosterContractTests(unittest.TestCase):
    def test_known_at_and_trade_membership(self):
        data = events([['P','A','SIGNING','2024-01-01','2024-01-01','a','2024-01-01','verified','high'],
                       ['P','A','TRADE_OUT','2024-02-01','2024-02-01','b','2024-02-01','verified','high'],
                       ['P','B','TRADE_IN','2024-02-01','2024-02-01','b','2024-02-01','verified','high']])
        roster, audit = reconstruct_roster(data, '2024-02-02')
        self.assertEqual(roster.iloc[0].TEAM_ID, 'B')
        later = data.copy(); later.loc[2, 'KNOWN_AT'] = '2024-02-03'
        roster, _ = reconstruct_roster(later, '2024-02-02')
        self.assertTrue(roster.empty)

    def test_unknown_and_duplicates_fail_closed(self):
        data = events([['P','A','UNKNOWN','2024-01-01','2024-01-01','a','2024-01-01','unknown','low']])
        with self.assertRaises(ValueError): reconstruct_roster(data, '2024-02-01')
        with self.assertRaises(ValueError): validate_events(pd.concat([data, data]))

    def test_inactive_return_sequence(self):
        data = events([['P','A','SIGNING','2024-01-01','2024-01-01','a','2024-01-01','verified','high'],
                       ['P','A','INACTIVE','2024-01-10','2024-01-10','b','2024-01-10','verified','high'],
                       ['P','A','RETURN','2024-01-20','2024-01-20','c','2024-01-20','verified','high']])
        roster, _ = reconstruct_roster(data, '2024-01-15'); self.assertTrue(roster.empty)
        roster, _ = reconstruct_roster(data, '2024-01-21'); self.assertEqual(roster.iloc[0].TEAM_ID, 'A')
