"""
Quick test script for the Monte Carlo season simulator.
It loads the trained game model, grabs the most recent rolling features
for two teams (default ATL vs BOS), and prints the win probability for
the home team.
"""

import pandas as pd
from pathlib import Path
import joblib

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_PATH = PROCESSED_DATA_DIR / "models" / "game_model.pkl"
FEATURES = ["ROLL_WIN_PCT", "ROLL_POINT_DIFF", "IS_HOME", "DAYS_SINCE_LAST_GAME"]

def load_latest_features(team_abbr: str) -> pd.Series:
    """Return the most recent rolling feature row for *team_abbr*.
    The row already contains the IS_HOME flag (1 for home, 0 for away)."""
    df = pd.read_parquet(PROCESSED_DATA_DIR / "team_features.parquet")
    team_df = df[df["TEAM_ABBREVIATION"] == team_abbr].sort_values("GAME_DATE")
    if team_df.empty:
        raise ValueError(f"No features found for team {team_abbr}")
    # Return the last row (most recent game) as a Series of the needed features
    return team_df.iloc[-1][FEATURES]

def main(home_team: str = "ATL", away_team: str = "BOS"):
    model = joblib.load(MODEL_PATH)
    # Use the home team's latest rolling features (IS_HOME already set to 1)
    home_feat = load_latest_features(home_team)
    prob_home_win = model.predict_proba([home_feat])[0, 1]
    print(f"Predicted win probability for {home_team} (home) vs {away_team}: {prob_home_win:.3%}")

if __name__ == "__main__":
    main()
