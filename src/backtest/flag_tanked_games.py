"""Flag possible post-elimination tanking games for backtest review.

This module never removes games.  It creates a separate, auditable table with
one row per team-game and three independent pieces of evidence:

* conservative mathematical elimination (win ceiling below the conference
  play-in cutoff at the start of that game's date),
* a post-elimination deterioration in recent point differential, and
* optionally, absences among players who were normally high-minute players
  before elimination.

"Tanked" is deliberately not an output label: the data cannot distinguish a
development decision from injury, a difficult schedule, or ordinary variance.
Use ``LIKELY_TANKING_REVIEW_FLAG`` to select games for review.

Examples:
    python src/backtest/flag_tanked_games.py
    python src/backtest/flag_tanked_games.py --season 2024-25
    python src/backtest/flag_tanked_games.py --fetch-player-logs --fetch-inactives
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# A strict less-than comparison makes this conservative: teams tied at the
# cutoff are retained because tie-breaker information is not modelled here.
PLAY_IN_CUTOFF_BY_SEASON = {
    "2018-19": 8,  # Last season before the play-in.
    "2019-20": 9,  # Bubble play-in involved only the 8/9 seeds.
}
DEFAULT_PLAY_IN_CUTOFF = 10  # 2020-21 onward.

POINT_DIFF_WINDOW = 5
PRE_ELIMINATION_BASELINE_GAMES = 10
POINT_DIFF_DROP_THRESHOLD = -5.0
HIGH_MINUTES_THRESHOLD = 24.0
MIN_PRIOR_PLAYER_GAMES = 10

EASTERN_TEAMS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA",
    "MIL", "NYK", "ORL", "PHI", "TOR", "WAS",
}


def conference_for_team(abbreviation: str) -> str:
    """Return the NBA conference for the stable 2018-25 team abbreviations."""
    return "East" if abbreviation in EASTERN_TEAMS else "West"


def seed_cutoff_for_season(season: str) -> int:
    """Number of conference places that remain play-in eligible."""
    return PLAY_IN_CUTOFF_BY_SEASON.get(season, DEFAULT_PLAY_IN_CUTOFF)


def load_team_logs() -> pd.DataFrame:
    """Load all cached team logs and normalize the fields used by this module."""
    path = RAW_DATA_DIR / "team_game_logs_combined.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the team-log ingest first.")

    logs = pd.read_parquet(path).copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    if "SEASON" not in logs:
        logs["SEASON"] = logs["SEASON_ID"].astype(str).str[-4:].astype(int).map(
            lambda year: f"{year}-{str(year + 1)[-2:]}"
        )
    logs["CONFERENCE"] = logs["TEAM_ABBREVIATION"].map(conference_for_team)
    logs["WIN"] = (logs["WL"] == "W").astype(int)
    logs = logs.sort_values(["SEASON", "GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)

    pair_sizes = logs.groupby(["SEASON", "GAME_ID"]).size()
    if not (pair_sizes == 2).all():
        raise ValueError("Each regular-season GAME_ID must have exactly two team rows.")
    # With two team rows per game, the other team's score is the game total
    # minus this team's score.  Keep it here rather than modifying raw data.
    logs["OPP_PTS"] = logs.groupby(["SEASON", "GAME_ID"])["PTS"].transform("sum") - logs["PTS"]
    logs["POINT_DIFF"] = logs["PTS"] - logs["OPP_PTS"]
    return logs


def add_elimination_flags(logs: pd.DataFrame) -> pd.DataFrame:
    """Add conservative, date-start mathematical-elimination fields.

    A team is marked only when its maximum possible finish (wins already earned
    plus every game remaining on the published schedule) is below the current
    wins of the final play-in-eligible conference slot.  This intentionally
    does *not* use tie-breakers, so it may mark later than the official NBA
    elimination date but will not overstate the result.
    """
    result = logs.copy()
    if "CONFERENCE" not in result:
        result["CONFERENCE"] = result["TEAM_ABBREVIATION"].map(conference_for_team)
    if "WIN" not in result:
        result["WIN"] = (result["WL"] == "W").astype(int)
    result["PLAY_IN_SEED_CUTOFF"] = result["SEASON"].map(seed_cutoff_for_season)
    result["WINS_BEFORE_GAME"] = 0
    result["MAX_POSSIBLE_WINS"] = 0
    result["CUTOFF_WINS_BEFORE_GAME"] = 0
    result["MATHEMATICALLY_ELIMINATED"] = False
    result["ELIMINATION_DATE"] = pd.NaT

    for season, season_rows in result.groupby("SEASON", sort=True):
        team_totals = season_rows.groupby("TEAM_ABBREVIATION").size().to_dict()
        conferences = season_rows.drop_duplicates("TEAM_ABBREVIATION").set_index(
            "TEAM_ABBREVIATION"
        )["CONFERENCE"].to_dict()
        wins = {team: 0 for team in team_totals}
        played = {team: 0 for team in team_totals}
        eliminated = {team: False for team in team_totals}
        eliminated_on: dict[str, pd.Timestamp | pd.NaT] = {team: pd.NaT for team in team_totals}
        cutoff_seed = seed_cutoff_for_season(season)

        for game_date, day_rows in season_rows.groupby("GAME_DATE", sort=True):
            cutoff_wins: dict[str, int] = {}
            for conference in ("East", "West"):
                conference_wins = sorted(
                    (wins[team] for team, conf in conferences.items() if conf == conference),
                    reverse=True,
                )
                # Defensive fallback for small synthetic test data. NBA seasons
                # always have 15 teams per conference.
                cutoff_wins[conference] = (
                    conference_wins[cutoff_seed - 1]
                    if len(conference_wins) >= cutoff_seed
                    else 0
                )

            # All same-day games see the standings at the start of that date.
            for index, row in day_rows.iterrows():
                team = row["TEAM_ABBREVIATION"]
                max_possible_wins = wins[team] + (team_totals[team] - played[team])
                cutoff = cutoff_wins[row["CONFERENCE"]]
                if not eliminated[team] and max_possible_wins < cutoff:
                    eliminated[team] = True
                    eliminated_on[team] = game_date

                result.loc[index, "WINS_BEFORE_GAME"] = wins[team]
                result.loc[index, "MAX_POSSIBLE_WINS"] = max_possible_wins
                result.loc[index, "CUTOFF_WINS_BEFORE_GAME"] = cutoff
                result.loc[index, "MATHEMATICALLY_ELIMINATED"] = eliminated[team]
                result.loc[index, "ELIMINATION_DATE"] = eliminated_on[team]

            # Update standings only after each game's result is known.
            for _, game_rows in day_rows.groupby("GAME_ID"):
                for _, row in game_rows.iterrows():
                    team = row["TEAM_ABBREVIATION"]
                    wins[team] += int(row["WIN"])
                    played[team] += 1

    return result


def add_point_differential_flags(logs: pd.DataFrame) -> pd.DataFrame:
    """Measure point-differential decline relative to the pre-elimination baseline."""
    result = logs.copy()
    result["PRE_ELIMINATION_POINT_DIFF"] = np.nan
    result["POST_ELIM_ROLLING_POINT_DIFF"] = np.nan
    result["POINT_DIFF_DROP"] = np.nan
    result["POINT_DIFF_DROP_FLAG"] = False

    for _, group in result.groupby(["SEASON", "TEAM_ABBREVIATION"], sort=False):
        group = group.sort_values(["GAME_DATE", "GAME_ID"])
        eliminated_positions = np.flatnonzero(group["MATHEMATICALLY_ELIMINATED"].to_numpy())
        if len(eliminated_positions) == 0:
            continue

        first_eliminated = int(eliminated_positions[0])
        baseline_start = max(0, first_eliminated - PRE_ELIMINATION_BASELINE_GAMES)
        point_diff = group["POINT_DIFF"] if "POINT_DIFF" in group else group["PTS"] - group["OPP_PTS"]
        baseline = point_diff.iloc[baseline_start:first_eliminated].to_numpy()
        if len(baseline) < PRE_ELIMINATION_BASELINE_GAMES:
            continue
        baseline_mean = float(np.mean(baseline))

        post = group.iloc[first_eliminated:].copy()
        post_point_diff = post["POINT_DIFF"] if "POINT_DIFF" in post else post["PTS"] - post["OPP_PTS"]
        post_rolling = post_point_diff.rolling(POINT_DIFF_WINDOW, min_periods=POINT_DIFF_WINDOW).mean()
        drop = post_rolling - baseline_mean
        indices = post.index
        result.loc[indices, "PRE_ELIMINATION_POINT_DIFF"] = baseline_mean
        result.loc[indices, "POST_ELIM_ROLLING_POINT_DIFF"] = post_rolling.to_numpy()
        result.loc[indices, "POINT_DIFF_DROP"] = drop.to_numpy()
        result.loc[indices, "POINT_DIFF_DROP_FLAG"] = (drop <= POINT_DIFF_DROP_THRESHOLD).to_numpy()

    return result


def season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def player_logs_path(season: str) -> Path:
    return RAW_DATA_DIR / f"player_game_logs_{season}.parquet"


def fetch_player_logs(seasons: Iterable[str], force: bool = False) -> None:
    """Cache player game logs, used to define pre-elimination regulars.

    This is intentionally separate from the base detector: it makes one NBA
    request per season and is unnecessary for the point-differential flags.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "ingest"))
    from nba_api.stats.endpoints import leaguegamelog
    from nba_client import REQUEST_TIMEOUT_SEC, fetch_with_retry

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for season in sorted(set(seasons)):
        path = player_logs_path(season)
        if path.exists() and not force:
            print(f"Using cached player logs for {season}")
            continue
        print(f"Pulling player logs for {season}...")

        def pull() -> pd.DataFrame:
            endpoint = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="P",
                timeout=REQUEST_TIMEOUT_SEC,
            )
            return endpoint.get_data_frames()[0]

        player_logs = fetch_with_retry(pull, label=f"Player logs {season}")
        player_logs["SEASON"] = season
        player_logs.to_parquet(path, index=False)
        time.sleep(1.5)


def _minutes_to_float(minutes: pd.Series) -> pd.Series:
    """Convert NBA values such as '31:42' (or numeric values) to minutes."""
    text = minutes.fillna("").astype(str)
    values = pd.to_numeric(text, errors="coerce")
    clock = text.str.extract(r"^(\d+):(\d+(?:\.\d+)?)$")
    clock_values = pd.to_numeric(clock[0], errors="coerce") + pd.to_numeric(
        clock[1], errors="coerce"
    ) / 60
    return values.where(~text.str.contains(":"), clock_values).fillna(0)


def add_high_minutes_absence_flags(logs: pd.DataFrame) -> pd.DataFrame:
    """Flag post-elimination games missing pre-elimination high-minute players.

    A player qualifies only after averaging at least 24 minutes over ten
    pre-elimination appearances.  This is an absence signal, not an assertion
    of rest; NBA's InactivePlayers endpoint does not return absence reasons.
    """
    result = logs.copy()
    result["HIGH_MINUTES_ABSENCES"] = 0
    result["HIGH_MINUTES_ABSENCE_SPIKE"] = False
    result["HIGH_MINUTES_ABSENT_PLAYERS"] = ""
    result["PLAYER_ABSENCE_DATA_AVAILABLE"] = False

    for season, season_rows in result.groupby("SEASON", sort=True):
        path = player_logs_path(season)
        if not path.exists():
            continue
        players = pd.read_parquet(path).copy()
        required = {"TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "GAME_ID", "GAME_DATE", "MIN"}
        if not required.issubset(players.columns):
            print(f"Skipping absence flags for {season}: player-log columns differ from expectation.")
            continue
        players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])
        players["MINUTES_FLOAT"] = _minutes_to_float(players["MIN"])

        for (team_id, team_abbreviation), team_games in season_rows.groupby(["TEAM_ID", "TEAM_ABBREVIATION"]):
            team_games = team_games.sort_values(["GAME_DATE", "GAME_ID"])
            eliminated = np.flatnonzero(team_games["MATHEMATICALLY_ELIMINATED"].to_numpy())
            if len(eliminated) == 0:
                continue
            first_eliminated = int(eliminated[0])
            pre_games = team_games.iloc[:first_eliminated]
            if pre_games.empty:
                continue
            team_players = players[players["TEAM_ID"] == team_id]
            pre_player_games = team_players[team_players["GAME_ID"].isin(pre_games["GAME_ID"])]
            player_summary = pre_player_games.groupby(["PLAYER_ID", "PLAYER_NAME"])["MINUTES_FLOAT"].agg(
                appearances="count", average_minutes="mean"
            )
            regulars = player_summary[
                (player_summary["appearances"] >= MIN_PRIOR_PLAYER_GAMES)
                & (player_summary["average_minutes"] >= HIGH_MINUTES_THRESHOLD)
            ]
            if regulars.empty:
                continue

            regular_ids = set(regulars.index.get_level_values("PLAYER_ID"))
            regular_names = regulars.reset_index().set_index("PLAYER_ID")["PLAYER_NAME"].to_dict()
            # Build this once per team; repeatedly filtering the full player
            # log per game is prohibitively slow across ten seasons.
            appeared_by_game = team_players.groupby("GAME_ID")["PLAYER_ID"].agg(set).to_dict()
            baseline_absences = []
            for game_id in pre_games["GAME_ID"]:
                appeared = appeared_by_game.get(game_id, set())
                baseline_absences.append(len(regular_ids - appeared))
            baseline_mean = float(np.mean(baseline_absences))

            for index, game in team_games.iloc[first_eliminated:].iterrows():
                appeared = appeared_by_game.get(game["GAME_ID"], set())
                absent = regular_ids - appeared
                absent_names = sorted(regular_names[player_id] for player_id in absent)
                absence_count = len(absent_names)
                result.loc[index, "PLAYER_ABSENCE_DATA_AVAILABLE"] = True
                result.loc[index, "HIGH_MINUTES_ABSENCES"] = absence_count
                result.loc[index, "HIGH_MINUTES_ABSENT_PLAYERS"] = "; ".join(absent_names)
                # A two-player rise is intentionally stringent and reduces injury noise.
                result.loc[index, "HIGH_MINUTES_ABSENCE_SPIKE"] = absence_count >= max(2, baseline_mean + 2)

    return result


def flag_likely_tanking_games(logs: pd.DataFrame, include_player_absences: bool = True) -> pd.DataFrame:
    """Return all team-games with diagnostic tanking-review fields attached."""
    flagged = add_elimination_flags(logs)
    flagged = add_point_differential_flags(flagged)
    if include_player_absences:
        flagged = add_high_minutes_absence_flags(flagged)
    else:
        flagged["HIGH_MINUTES_ABSENCE_SPIKE"] = False
        flagged["PLAYER_ABSENCE_DATA_AVAILABLE"] = False
    flagged["LIKELY_TANKING_REVIEW_FLAG"] = (
        flagged["MATHEMATICALLY_ELIMINATED"]
        & (flagged["POINT_DIFF_DROP_FLAG"] | flagged["HIGH_MINUTES_ABSENCE_SPIKE"])
    )
    flagged["REVIEW_REASONS"] = ""
    point_reason = flagged["POINT_DIFF_DROP_FLAG"]
    absence_reason = flagged["HIGH_MINUTES_ABSENCE_SPIKE"]
    flagged.loc[point_reason, "REVIEW_REASONS"] = "post-elimination point-differential drop"
    flagged.loc[absence_reason, "REVIEW_REASONS"] = flagged.loc[absence_reason, "REVIEW_REASONS"].map(
        lambda reason: f"{reason}; high-minute-player absence spike".strip("; ")
    )
    return flagged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", action="append", help="Season to report, e.g. 2024-25. Repeatable.")
    parser.add_argument("--fetch-player-logs", action="store_true", help="Cache one player-game-log pull per requested season.")
    parser.add_argument("--force", action="store_true", help="Refresh cached player logs when fetching.")
    args = parser.parse_args()

    team_logs = load_team_logs()
    seasons = args.season or sorted(team_logs["SEASON"].unique())
    missing = sorted(set(seasons) - set(team_logs["SEASON"]))
    if missing:
        raise ValueError(f"No team logs available for: {', '.join(missing)}")
    team_logs = team_logs[team_logs["SEASON"].isin(seasons)].copy()
    if args.fetch_player_logs:
        fetch_player_logs(seasons, force=args.force)

    output = flag_likely_tanking_games(team_logs)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "backtest_tanking_flags.parquet"
    review_path = PROCESSED_DATA_DIR / "backtest_tanking_review_games.csv"
    output.to_parquet(output_path, index=False)
    review_columns = [
        "SEASON", "GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION", "MATCHUP", "WL",
        "MATHEMATICALLY_ELIMINATED", "ELIMINATION_DATE", "MAX_POSSIBLE_WINS",
        "CUTOFF_WINS_BEFORE_GAME", "PRE_ELIMINATION_POINT_DIFF",
        "POST_ELIM_ROLLING_POINT_DIFF", "POINT_DIFF_DROP", "HIGH_MINUTES_ABSENCES",
        "HIGH_MINUTES_ABSENT_PLAYERS", "REVIEW_REASONS",
    ]
    review = output[output["LIKELY_TANKING_REVIEW_FLAG"]][review_columns].sort_values(
        ["SEASON", "GAME_DATE", "TEAM_ABBREVIATION"]
    )
    review.to_csv(review_path, index=False)
    summary = output.groupby("SEASON").agg(
        eliminated_team_games=("MATHEMATICALLY_ELIMINATED", "sum"),
        review_games=("LIKELY_TANKING_REVIEW_FLAG", "sum"),
    )
    print("\nTanking-review summary (all games retained):")
    print(summary.to_string())
    print(f"\nAll diagnostic flags: {output_path}")
    print(f"Review subset:         {review_path}")


if __name__ == "__main__":
    main()
