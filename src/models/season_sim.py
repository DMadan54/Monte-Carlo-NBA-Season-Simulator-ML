"""
Monte Carlo season simulator.

Simulates the remainder of a season thousands of times using the trained
game-outcome model as the win-probability engine, then aggregates across
trials into: final standings distribution, playoff seeding probability,
and (later) bracket advancement odds.

Usage:
    python src/models/season_sim.py --n_sims 5000
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = PROCESSED_DATA_DIR / "models"

FEATURES = ["ROLL_WIN_PCT", "ROLL_POINT_DIFF", "IS_HOME", "DAYS_SINCE_LAST_GAME"]


def load_model():
    model_path = MODELS_DIR / "game_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run src/models/game_model.py first."
        )
    return joblib.load(model_path)


def load_remaining_schedule() -> pd.DataFrame:
    """
    Load the full 2024‑25 NBA season schedule and attach each team's pre‑game features
    (the rolling Four‑Factors) as they existed immediately before the game.
    Returns a DataFrame with schedule columns plus all model feature columns.
    """
    from nba_api.stats.endpoints import scheduleleaguev2
    from nba_api.stats.static import teams  # import list of NBA teams
    import pandas as pd
    from datetime import datetime

    # --- Load raw schedule ---------------------------------------------------
    season = "2024-25"
    schedule_df = scheduleleaguev2.ScheduleLeagueV2(season=season).get_data_frames()[0]
    schedule = schedule_df[["gameDate", "homeTeam_teamTricode", "awayTeam_teamTricode"]].copy()
    schedule.rename(columns={
        "gameDate": "GAME_DATE",
        "homeTeam_teamTricode": "HOME_TEAM",
        "awayTeam_teamTricode": "AWAY_TEAM",
    }, inplace=True)
    # Filter to NBA teams – remove G‑League, All‑Star, etc.
    valid_abbr = {team["abbreviation"] for team in teams.get_teams()}
    schedule = schedule[schedule["HOME_TEAM"].isin(valid_abbr) & schedule["AWAY_TEAM"].isin(valid_abbr)]
    
    # Keep GAME_DATE as datetime64 for merge_asof compatibility
    schedule["GAME_DATE"] = pd.to_datetime(schedule["GAME_DATE"])
    schedule.sort_values("GAME_DATE", inplace=True)
    schedule.reset_index(drop=True, inplace=True)

    # --- Load pre‑computed team rolling features ----------------------------
    features_path = PROCESSED_DATA_DIR / "team_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found – run src/features/build_team_features.py first."
        )
    team_feat = pd.read_parquet(features_path)
    # Filter team features to NBA teams only
    team_feat = team_feat[team_feat["TEAM_ABBREVIATION"].isin(valid_abbr)]
    # Ensure date column is datetime64[ns] for merge_asof compatibility
    team_feat["GAME_DATE"] = pd.to_datetime(team_feat["GAME_DATE"])
    team_feat.sort_values("GAME_DATE", inplace=True)

    # --- Attach the most recent pre‑game feature row for each HOME_TEAM --------
    # merge_asof performs a left‑join on the nearest prior GAME_DATE per team
    schedule_with_feat = pd.merge_asof(
        schedule,
        team_feat,
        left_on="GAME_DATE",
        right_on="GAME_DATE",
        left_by="HOME_TEAM",
        right_by="TEAM_ABBREVIATION",
        direction="backward",
        suffixes=("", "_home"),
    )

    # Drop any rows where the required model features are still missing (e.g.,
    # the very first games of a season where no history exists). The model
    # expects the FEATURE columns defined in src/models/game_model.py.
    required_cols = [
        "ROLL_WIN_PCT",
        "ROLL_POINT_DIFF",
        "IS_HOME",
        "DAYS_SINCE_LAST_GAME",
        "ROLL_EFG",
        "ROLL_TOV_PCT",
        "ROLL_ORB_PCT",
        "ROLL_FTR",
        "ROLL_EFG_OPP",
        "ROLL_TOV_PCT_OPP",
        "ROLL_ORB_PCT_OPP",
        "ROLL_FTR_OPP",
        "ROLL_EFG_DIFF",
    ]
    schedule_with_feat = schedule_with_feat.dropna(subset=required_cols)
    return schedule_with_feat


def simulate_one_season(schedule: pd.DataFrame, model, current_wins: dict) -> dict:
    """Run a single Monte Carlo trial over the remaining schedule."""
    wins = current_wins.copy()

    win_probs = model.predict_proba(schedule[FEATURES])[:, 1]

    for i, row in enumerate(schedule.itertuples()):
        home_win = np.random.random() < win_probs[i]
        if home_win:
            wins[row.HOME_TEAM] = wins.get(row.HOME_TEAM, 0) + 1
        else:
            wins[row.AWAY_TEAM] = wins.get(row.AWAY_TEAM, 0) + 1

    return wins


def run_simulation(n_sims: int, current_wins: dict) -> pd.DataFrame:
    model = load_model()
    schedule = load_remaining_schedule()

    results = []
    for sim in range(n_sims):
        final_wins = simulate_one_season(schedule, model, current_wins)
        results.append(final_wins)

    results_df = pd.DataFrame(results)

    summary = pd.DataFrame({
        "team": results_df.columns,
        "mean_wins": results_df.mean().values,
        "std_wins": results_df.std().values,
        "min_wins": results_df.min().values,
        "max_wins": results_df.max().values,
    }).sort_values("mean_wins", ascending=False).reset_index(drop=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sims", type=int, default=5000,
                         help="Number of Monte Carlo trials to run")
    args = parser.parse_args()

    print(
        "NOTE: load_remaining_schedule() is a placeholder — wire it up to "
        "a real schedule pull before running this end to end."
    )
    # current_wins = {"LAL": 20, "BOS": 25, ...}  # pull from standings
    # summary = run_simulation(args.n_sims, current_wins)
    # print(summary)
