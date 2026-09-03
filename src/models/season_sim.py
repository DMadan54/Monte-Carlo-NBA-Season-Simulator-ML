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
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MODELS_DIR = PROCESSED_DATA_DIR / "models"

FEATURES = [
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

# Standard and known shortened season game counts
SEASON_EXPECTED_GAMES = {
    "2019-20": 1059,  # COVID-shortened regular season (bubble)
    "2020-21": 1080,  # 72-game season per team (30 * 72 / 2 = 1080)
}


def load_model():
    """Load the trained game outcome model (Four Factors LightGBM)."""
    model_path = MODELS_DIR / "game_model_four_factors.pkl"
    if not model_path.exists():
        # Fallback to legacy game_model.pkl if present
        model_path = MODELS_DIR / "game_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run src/models/game_model.py first."
        )
    return joblib.load(model_path)


def season_to_year(season: str) -> int:
    """Convert '2023-24' -> 2024."""
    start_year = int(season.split("-")[0])
    return start_year + 1


def load_season_schedule(season: str = "2024-25") -> pd.DataFrame:
    """
    Load the regular-season schedule for any given season and attach each team's
    pre-game rolling features strictly computed from games within that season.

    Args:
        season: e.g. '2023-24' or '2014-15'

    Returns:
        DataFrame with schedule columns plus all model feature columns.
    """
    from nba_api.stats.static import teams

    valid_abbr = {team["abbreviation"] for team in teams.get_teams()}
    season_year = season_to_year(season)

    # 1. Load team features for the given season
    features_path = PROCESSED_DATA_DIR / "team_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} not found – run src/features/build_team_features.py first."
        )
    team_feat_all = pd.read_parquet(features_path)
    team_feat_all["GAME_DATE"] = pd.to_datetime(team_feat_all["GAME_DATE"])

    # Filter to the requested season year and valid NBA teams
    # (Strict intra-season isolation: no cross-season leakage)
    season_feats = team_feat_all[
        (team_feat_all["SEASON_YEAR"] == season_year)
        & (team_feat_all["TEAM_ABBREVIATION"].isin(valid_abbr))
    ].copy()

    if season_feats.empty:
        raise ValueError(
            f"No features found for season {season} (SEASON_YEAR={season_year})."
        )

    # Backfill early-season missing rolling features within the same team & season
    # so that all regular season games can be simulated.
    season_feats.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"], inplace=True)
    roll_cols = [c for c in FEATURES if c != "IS_HOME"]
    season_feats[roll_cols] = (
        season_feats.groupby("TEAM_ABBREVIATION")[roll_cols].bfill().ffill()
    )

    # 2. Extract home schedule (every game appears exactly once as home)
    schedule = season_feats[season_feats["IS_HOME"] == 1].copy()
    schedule.rename(columns={"TEAM_ABBREVIATION": "HOME_TEAM"}, inplace=True)

    # Extract away team abbreviation from MATCHUP e.g. 'ATL vs. BOS' -> 'BOS'
    if "AWAY_TEAM" not in schedule.columns:
        schedule["AWAY_TEAM"] = schedule["MATCHUP"].str.extract(r"vs\.\s*(\w+)")
        if schedule["AWAY_TEAM"].isna().any():
            team_id_map = team_feat_all.drop_duplicates("TEAM_ID").set_index("TEAM_ID")["TEAM_ABBREVIATION"].to_dict()
            schedule["AWAY_TEAM"] = schedule["AWAY_TEAM"].fillna(schedule["OPP_TEAM_ID"].map(team_id_map))

    schedule = schedule[
        schedule["HOME_TEAM"].isin(valid_abbr)
        & schedule["AWAY_TEAM"].isin(valid_abbr)
    ]
    schedule.drop_duplicates(subset=["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"], inplace=True)

    # Expected game count validation
    expected_games = SEASON_EXPECTED_GAMES.get(season, 1230)
    actual_games = len(schedule)
    if actual_games != expected_games:
        print(
            f"  [NOTE] Season {season} has {actual_games} games "
            f"(expected {expected_games})."
        )

    schedule.sort_values("GAME_DATE", inplace=True)
    schedule.reset_index(drop=True, inplace=True)
    return schedule


def load_remaining_schedule() -> pd.DataFrame:
    """Wrapper for backward compatibility, loading default current season."""
    return load_season_schedule("2024-25")


def simulate_one_season(
    schedule: pd.DataFrame, model, current_wins: Optional[dict] = None, rng: Optional[np.random.Generator] = None
) -> dict:
    """Run a single Monte Carlo trial over the schedule."""
    wins = current_wins.copy() if current_wins is not None else {}
    if rng is None:
        rng = np.random.default_rng()

    # Model inference
    win_probs = model.predict_proba(schedule[FEATURES])[:, 1]
    random_draws = rng.random(len(schedule))

    for i, row in enumerate(schedule.itertuples()):
        home_win = random_draws[i] < win_probs[i]
        if home_win:
            wins[row.HOME_TEAM] = wins.get(row.HOME_TEAM, 0) + 1
        else:
            wins[row.AWAY_TEAM] = wins.get(row.AWAY_TEAM, 0) + 1

    return wins


def run_simulation_with_schedule(
    schedule: pd.DataFrame, model, n_sims: int = 5000, current_wins: Optional[dict] = None, seed: Optional[int] = 42
) -> pd.DataFrame:
    """Run Monte Carlo simulation on a provided schedule DataFrame."""
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    results = []
    for _ in range(n_sims):
        final_wins = simulate_one_season(schedule, model, current_wins, rng=rng)
        results.append(final_wins)

    results_df = pd.DataFrame(results).fillna(0)

    summary = pd.DataFrame({
        "team": results_df.columns,
        "mean_wins": results_df.mean().values,
        "std_wins": results_df.std().values,
        "min_wins": results_df.min().values,
        "max_wins": results_df.max().values,
    }).sort_values("mean_wins", ascending=False).reset_index(drop=True)

    return summary


def run_simulation(n_sims: int = 5000, current_wins: Optional[dict] = None) -> pd.DataFrame:
    """Run simulation on the default remaining schedule."""
    model = load_model()
    schedule = load_remaining_schedule()
    return run_simulation_with_schedule(schedule, model, n_sims=n_sims, current_wins=current_wins)


def run_multi_season_simulation(
    seasons: List[str], n_sims: int = 1000, seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """Run simulation across multiple seasons independently."""
    model = load_model()
    results = {}
    for season in seasons:
        sched = load_season_schedule(season)
        summary = run_simulation_with_schedule(sched, model, n_sims=n_sims, seed=seed)
        results[season] = summary
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sims", type=int, default=5000,
                        help="Number of Monte Carlo trials to run")
    parser.add_argument("--season", type=str, default="2024-25",
                        help="Season to simulate, e.g. 2023-24")
    args = parser.parse_args()

    model = load_model()
    schedule = load_season_schedule(args.season)
    summary = run_simulation_with_schedule(schedule, model, n_sims=args.n_sims)
    print(f"\n--- Simulation Results for {args.season} ({args.n_sims} sims) ---")
    print(summary.head(10))
