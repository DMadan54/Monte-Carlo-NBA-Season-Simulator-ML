"""
Turn raw team game logs into a model-ready feature table.

For each team-game, computes rolling pre-game features (so we never leak
future information into a prediction):
    - rolling win % (last 10 games)
    - rolling point differential (last 10 games)
    - rest days since last game
    - home/away flag

Usage:
    python src/features/build_team_features.py
"""

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

ROLLING_WINDOW = 10


def load_combined_logs() -> pd.DataFrame:
    path = RAW_DATA_DIR / "team_game_logs_combined.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/ingest/pull_team_game_logs.py first."
        )
    return pd.read_parquet(path)


def add_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WIN"] = (df["WL"] == "W").astype(int)

    # Each GAME_ID has exactly two rows (one per team). Join each row to
    # its opponent's row on GAME_ID to get the opponent's score, then
    # compute point differential properly.
    opponent_scores = df[["GAME_ID", "TEAM_ID", "PTS"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS"}
    )
    df = df.merge(opponent_scores, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]]  # drop self-join rows
    df["POINT_DIFF"] = df["PTS"] - df["OPP_PTS"]

    # MATCHUP contains "@" for away games, "vs." for home games
    df["IS_HOME"] = (~df["MATCHUP"].str.contains("@")).astype(int)
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("TEAM_ID")

    # shift(1) ensures we only use PAST games, not the current one (no leakage)
    df["ROLL_WIN_PCT"] = grouped["WIN"].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
    )
    df["ROLL_POINT_DIFF"] = grouped["POINT_DIFF"].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean()
    )
    df["DAYS_SINCE_LAST_GAME"] = grouped["GAME_DATE"].transform(
        lambda x: x.diff().dt.days
    )

    return df


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_combined_logs()
    df = add_basic_fields(df)
    df = add_rolling_features(df)

    # Drop early-season rows with no rolling history yet
    df = df.dropna(subset=["ROLL_WIN_PCT", "ROLL_POINT_DIFF"])

    out_path = PROCESSED_DATA_DIR / "team_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} feature rows to {out_path}")
    print(df[["TEAM_ABBREVIATION", "GAME_DATE", "ROLL_WIN_PCT",
              "ROLL_POINT_DIFF", "IS_HOME", "WIN"]].head())


if __name__ == "__main__":
    main()
