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
    Placeholder loader for the remaining schedule.

    TODO: replace with a real pull of remaining games for the season
    (nba_api's `scheduleleaguev2` endpoint or similar), joined with each
    team's current rolling form features so each row has the FEATURES
    columns ready to feed into the model.

    Expected columns: GAME_DATE, HOME_TEAM, AWAY_TEAM, plus the FEATURES
    columns computed from each team's current rolling form.
    """
    raise NotImplementedError(
        "Wire this up to a real schedule pull once you're ready — "
        "for now this is a placeholder so the rest of the pipeline runs."
    )


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
