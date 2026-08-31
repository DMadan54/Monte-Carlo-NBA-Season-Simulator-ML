"""
Pull team-level game logs for a range of NBA seasons using nba_api,
and cache them locally as Parquet so we never re-hit the API for the
same season twice.

Usage:
    python src/ingest/pull_team_game_logs.py --start_season 2018 --end_season 2024
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def season_str(start_year: int) -> str:
    """Convert e.g. 2018 -> '2018-19' (the format nba_api expects)."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def pull_season(start_year: int, season_type: str = "Regular Season") -> pd.DataFrame:
    """Pull all team game logs for a single season."""
    season = season_str(start_year)
    print(f"Pulling {season} ({season_type})...")

    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",  # team-level, not player-level
    )
    df = log.get_data_frames()[0]
    df["SEASON"] = season
    return df


def main(start_season: int, end_season: int):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_seasons = []

    for year in range(start_season, end_season + 1):
        out_path = RAW_DATA_DIR / f"team_game_logs_{season_str(year)}.parquet"

        if out_path.exists():
            print(f"Skipping {season_str(year)}, already cached at {out_path}")
            df = pd.read_parquet(out_path)
        else:
            df = pull_season(year)
            df.to_parquet(out_path, index=False)
            # Be polite to the unofficial API - don't hammer it
            time.sleep(1.0)

        all_seasons.append(df)

    combined = pd.concat(all_seasons, ignore_index=True)
    combined_path = RAW_DATA_DIR / "team_game_logs_combined.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"\nDone. {len(combined)} team-game rows across "
          f"{end_season - start_season + 1} seasons.")
    print(f"Combined file: {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_season", type=int, default=2018,
                         help="First season's starting year, e.g. 2018 for 2018-19")
    parser.add_argument("--end_season", type=int, default=2024,
                         help="Last season's starting year, e.g. 2024 for 2024-25")
    args = parser.parse_args()

    main(args.start_season, args.end_season)
