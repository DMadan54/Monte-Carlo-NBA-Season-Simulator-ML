"""Focused, offline checks for the tanking-review flag definitions."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.flag_tanked_games import add_elimination_flags, flag_likely_tanking_games


def _season_rows() -> pd.DataFrame:
    """A tiny 2018-19 synthetic season with a strict win-ceiling elimination."""
    rows = []
    # The 15th East team has no remaining wins and can no longer reach the
    # eighth team's current win total at the start of the final date.
    east = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"]
    for position, team in enumerate(east):
        for game_number in range(2):
            rows.append({
                "SEASON": "2018-19", "GAME_DATE": pd.Timestamp("2019-04-09") + pd.Timedelta(days=game_number),
                "GAME_ID": f"00{game_number}{position}", "TEAM_ID": position,
                "TEAM_ABBREVIATION": team, "WL": "W" if position < 8 else "L",
                "PTS": 110 if position < 8 else 90, "OPP_PTS": 90 if position < 8 else 110,
            })
    return pd.DataFrame(rows)


def test_elimination_is_conservative() -> None:
    flagged = add_elimination_flags(_season_rows())
    # The top-eight teams should never be called eliminated by a win ceiling.
    assert not flagged[flagged["TEAM_ABBREVIATION"].isin(["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND"])]["MATHEMATICALLY_ELIMINATED"].any()


def test_final_flag_requires_elimination_and_evidence() -> None:
    flagged = flag_likely_tanking_games(_season_rows(), include_player_absences=False)
    assert not flagged.loc[~flagged["MATHEMATICALLY_ELIMINATED"], "LIKELY_TANKING_REVIEW_FLAG"].any()


if __name__ == "__main__":
    test_elimination_is_conservative()
    test_final_flag_requires_elimination_and_evidence()
    print("Tanking flag tests passed.")
