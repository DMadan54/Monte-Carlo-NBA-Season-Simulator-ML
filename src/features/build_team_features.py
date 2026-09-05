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
import numpy as np

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
    # NBA season typically starts in October; assign season year as the year the season ends.
    df["SEASON_YEAR"] = df["GAME_DATE"].apply(lambda d: d.year + 1 if d.month >= 10 else d.year)
    df["WIN"] = (df["WL"] == "W").astype(int)

    # Each GAME_ID has exactly two rows (one per team). Join each row to
    # its opponent's row on GAME_ID to get the opponent's score and stats, then
    # compute point differential and Four Factors.
    opp_stats = df[[
        "GAME_ID", "TEAM_ID", "PTS", "FGM", "FGA", "FG3M", "TOV", "OREB", "FTM", "FTA"
    ]].rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "PTS": "OPP_PTS",
            "FGM": "OPP_FGM",
            "FGA": "OPP_FGA",
            "FG3M": "OPP_FG3M",
            "TOV": "OPP_TOV",
            "OREB": "OPP_ORB",
            "FTM": "OPP_FTM",
            "FTA": "OPP_FTA",
        }
    )
    df = df.merge(opp_stats, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]].copy()  # drop self-join rows
    # Point differential
    df["POINT_DIFF"] = df["PTS"] - df["OPP_PTS"]
    # Four Factors for team offense
    import numpy as np
    df["EFG"] = np.where(df["FGA"] > 0, (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"], np.nan)
    df["TOV_PCT"] = np.where((df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) > 0,
                               df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]), np.nan)
    # Create ORB column from OREB
    df["ORB"] = df["OREB"]
    df["ORB_PCT"] = np.where((df["ORB"] + df["OPP_ORB"]) > 0,
                               df["ORB"] / (df["ORB"] + df["OPP_ORB"]), np.nan)
    df["FTR"] = np.where(df["FGA"] > 0, df["FTM"] / df["FGA"], np.nan)
    # Four Factors for opponent defense (treated as opponent offense)
    df["OPP_EFG"] = np.where(df["OPP_FGA"] > 0,
                               (df["OPP_FGM"] + 0.5 * df["OPP_FG3M"]) / df["OPP_FGA"], np.nan)
    df["OPP_TOV_PCT"] = np.where((df["OPP_FGA"] + 0.44 * df["OPP_FTA"] + df["OPP_TOV"]) > 0,
                                   df["OPP_TOV"] / (df["OPP_FGA"] + 0.44 * df["OPP_FTA"] + df["OPP_TOV"]), np.nan)
    df["OPP_ORB_PCT"] = np.where((df["OPP_ORB"] + df["ORB"]) > 0,
                                   df["OPP_ORB"] / (df["OPP_ORB"] + df["ORB"]), np.nan)
    df["OPP_FTR"] = np.where(df["OPP_FGA"] > 0, df["OPP_FTM"] / df["OPP_FGA"], np.nan)

    # MATCHUP contains "@" for away games, "vs." for home games
    df["IS_HOME"] = (~df["MATCHUP"].str.contains("@")).astype(int)
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(["TEAM_ID", "SEASON_YEAR"])

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
    # Rolling Four Factors (offensive)
    df["ROLL_EFG"] = grouped["EFG"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    df["ROLL_TOV_PCT"] = grouped["TOV_PCT"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    df["ROLL_ORB_PCT"] = grouped["ORB_PCT"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    df["ROLL_FTR"] = grouped["FTR"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    # Rolling Four Factors (opponent)
    df["ROLL_EFG_OPP"] = grouped["OPP_EFG"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    df["ROLL_TOV_PCT_OPP"] = grouped["OPP_TOV_PCT"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    df["ROLL_ORB_PCT_OPP"] = grouped["OPP_ORB_PCT"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    df["ROLL_FTR_OPP"] = grouped["OPP_FTR"].transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    # Derived differential rolling features
    df["ROLL_EFG_DIFF"] = df["ROLL_EFG"] - df["ROLL_EFG_OPP"]
    df["ROLL_TOV_PCT_DIFF"] = df["ROLL_TOV_PCT"] - df["ROLL_TOV_PCT_OPP"]
    df["ROLL_ORB_PCT_DIFF"] = df["ROLL_ORB_PCT"] - df["ROLL_ORB_PCT_OPP"]
    df["ROLL_FTR_DIFF"] = df["ROLL_FTR"] - df["ROLL_FTR_OPP"]

    # Long-horizon team-strength features (persistent signals)
    # 1. Season-to-date cumulative net rating (expanding window)
    df["SEASON_POINT_DIFF"] = grouped["POINT_DIFF"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    )
    # 2. Slower-decaying exponential moving average (half-life = 20 games)
    df["EMA_POINT_DIFF"] = grouped["POINT_DIFF"].transform(
        lambda x: x.shift(1).ewm(halflife=20, min_periods=3).mean()
    )

    # Ensure first two games of each season have NaN rolling features to avoid leakage
    # Determine game order within each team-season group
    order = grouped["GAME_DATE"].rank(method="first")
    mask = order <= 2
    for col in df.columns:
        if col.startswith("ROLL_") or col in ["SEASON_POINT_DIFF", "EMA_POINT_DIFF"]:
            df.loc[mask, col] = np.nan

    # Compute Elo ratings (persistent, not windowed)
    df = compute_elo_ratings(df)
    return df


def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple Elo ratings for each team across the full history.

    The rating is updated after every game using a K‑factor, a margin‑of‑victory multiplier,
    and a home‑court advantage constant. This produces persistent ratings that do not reset
    each season, directly addressing the extremes‑compression issue.
    """
    df = df.copy()
    # Parameters (can be tuned later)
    INITIAL_RATING = 1500.0
    K = 20.0
    HOME_ADV = 100.0  # added to home team's effective rating when computing win prob

    # Helper to compute margin multiplier (greater for larger point differentials)
    def margin_multiplier(point_diff):
        # Use a simple log‑based multiplier; ensure at least 1.0
        return max(1.0, np.log(abs(point_diff) + 1.0))

    # Dictionaries to hold current ratings per team
    ratings = {}
    # Store computed ratings
    elo_ratings = []
    opp_elo_ratings = []

    # Ensure deterministic order: sort by date then by team id
    df = df.sort_values(["GAME_DATE", "TEAM_ID"]).reset_index(drop=True)
    for idx, row in df.iterrows():
        team = row["TEAM_ID"]
        opp = row["OPP_TEAM_ID"]
        home = row["IS_HOME"] == 1
        actual = row["WIN"]  # 1 if team won, else 0
        point_diff = row["POINT_DIFF"]

        # Get current ratings (default initial)
        team_rating = ratings.get(team, INITIAL_RATING)
        opp_rating = ratings.get(opp, INITIAL_RATING)

        # Effective ratings for win‑probability calculation
        team_eff = team_rating + (HOME_ADV if home else 0)
        opp_eff = opp_rating + (HOME_ADV if not home else 0)
        expected = 1.0 / (1.0 + 10 ** ((opp_eff - team_eff) / 400.0))

        mult = margin_multiplier(point_diff)
        delta = K * (actual - expected) * mult

        # Update ratings
        ratings[team] = team_rating + delta
        ratings[opp] = opp_rating - delta  # opponent gets opposite adjustment

        elo_ratings.append(team_rating)  # rating before the game
        opp_elo_ratings.append(opp_rating)

    df["ELO_RATING"] = elo_ratings
    df["OPP_ELO_RATING"] = opp_elo_ratings
    return df


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_combined_logs()
    df = add_basic_fields(df)
    df = add_rolling_features(df)

    # Drop early-season rows with no rolling history yet
    # df = df.dropna(subset=["ROLL_WIN_PCT", "ROLL_POINT_DIFF"])  # Retain early-season rows for testing

    out_path = PROCESSED_DATA_DIR / "team_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} feature rows to {out_path}")
    print(df[["TEAM_ABBREVIATION", "GAME_DATE", "ROLL_WIN_PCT",
              "ROLL_POINT_DIFF", "IS_HOME", "WIN", "ELO_RATING"]].head())


if __name__ == "__main__":
    main()
