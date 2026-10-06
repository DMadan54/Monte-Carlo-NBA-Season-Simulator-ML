from types import SimpleNamespace
import unittest
from scripts.collect_award_votes import parse_totals, parse_ballots


class AwardIngestionTests(unittest.TestCase):
    def test_totals_must_reconcile_every_rank_and_weight(self):
        def pdf(text):
            return SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: text)])
        rows = parse_totals(pdf('A, Team 1 0 1 6\nB, Team 0 1 0 3'), 'roy', 1)
        self.assertEqual(rows[0]['vote_points'], 6)
        with self.assertRaises(ValueError):
            parse_totals(pdf('A, Team 1 0 1 5\nB, Team 0 1 0 3'), 'roy', 1)
        with self.assertRaises(ValueError):
            parse_totals(pdf('A, Team 1 0 1 6'), 'roy', 1)

    def test_ballots_must_be_complete_and_unique(self):
        def pdf(rows):
            return SimpleNamespace(pages=[SimpleNamespace(extract_tables=lambda: [rows])])
        row = ['Voter, Outlet', 'A, Team', 'B, Team', 'C, Team']
        row[0] = 'Person, Outlet'
        self.assertEqual(len(parse_ballots(pdf([row]), 'roy', 1)), 3)
        with self.assertRaises(ValueError):
            parse_ballots(pdf([row, row]), 'roy', 2)
        with self.assertRaises(ValueError):
            parse_ballots(pdf([['Person, Outlet', 'A, Team', 'A, Team', 'C, Team']]), 'roy', 1)


if __name__ == '__main__':
    unittest.main()
