"""
Walk-forward backtest for the NBA Monte Carlo season simulator.

For each target season:
- Train the model using only data from seasons strictly before the target season.
- Evaluate the target season using fixed seed and 1000 simulations.
- Output results to a CSV with actual wins, simulated mean wins, and absolute error.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.game_model import (
    build_model, load_features, add_tanking_review_flag, FEATURES, TARGET
)
from src.models.season_sim import (
    load_season_schedule, run_simulation_with_schedule
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def run_walk_forward_backtest(
    target_seasons: list[str],
    exclude_tanking_flags: bool = False,
    n_sims: int = 1000,
    seed: int = 42
):
    df_features = load_features()
    if exclude_tanking_flags:
        df_features = add_tanking_review_flag(df_features)

    df_features_clean = df_features.dropna(subset=FEATURES + [TARGET])
    
    results = []
    
    for season in target_seasons:
        start_year = int(season.split("-")[0])
        season_year = start_year + 1
        
        # Train strictly on seasons before target
        train_rows = df_features_clean[df_features_clean["SEASON_YEAR"] < season_year].copy()
        
        # Assertion: Zero target-season rows enter training
        assert len(train_rows[train_rows["SEASON_YEAR"] >= season_year]) == 0, \
            "Leakage: Target season rows found in training set!"
            
        if len(train_rows) == 0:
            print(f"Skipping {season} - no prior training data.")
            continue
            
        if exclude_tanking_flags:
            # Exclude only from training rows
            if "LIKELY_TANKING_REVIEW_FLAG" in train_rows.columns:
                train_rows = train_rows.loc[~train_rows["LIKELY_TANKING_REVIEW_FLAG"]]
        
        X_train, y_train = train_rows[FEATURES], train_rows[TARGET]
        
        print(f"Training for target season {season} (using {len(train_rows)} rows prior to {season_year})...")
        model = build_model()
        model.fit(X_train, y_train)
        
        # Actual wins for the season (keeping tanking games in actual standings)
        # We use the full (non-dropna) features to accurately sum all actual wins.
        target_season_actual = df_features[df_features["SEASON_YEAR"] == season_year]
        actual_wins = target_season_actual.groupby("TEAM_ABBREVIATION")["WIN"].sum().to_dict()
        
        # Evaluate target season separately
        schedule = load_season_schedule(season, features=list(model.feature_name_))
        
        # Simulate
        summary = run_simulation_with_schedule(
            schedule=schedule,
            model=model,
            n_sims=n_sims,
            seed=seed
        )
        
        # Compile results
        for _, row in summary.iterrows():
            team = row["team"]
            sim_mean_wins = row["mean_wins"]
            actual = actual_wins.get(team, 0)
            
            results.append({
                "SEASON": season,
                "TEAM": team,
                "ACTUAL_WINS": actual,
                "SIMULATED_MEAN_WINS": sim_mean_wins,
                "ABSOLUTE_ERROR": abs(actual - sim_mean_wins)
            })
            
    # Save to CSV
    results_df = pd.DataFrame(results)
    
    out_name = "walk_forward_backtest_results.csv"
    if exclude_tanking_flags:
        out_name = "walk_forward_backtest_results_no_tanking.csv"
        
    out_path = PROCESSED_DATA_DIR / out_name
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved walk-forward backtest results to {out_path}")
    print(f"Mean Absolute Error across all evaluated seasons: {results_df['ABSOLUTE_ERROR'].mean():.2f}")
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-tanking-flags", action="store_true")
    parser.add_argument("--n_sims", type=int, default=1000)
    args = parser.parse_args()
    
    # We evaluate seasons where we have prior history.
    # The dataset starts in 2015-16. We can evaluate from 2016-17 to 2023-24.
    seasons_to_test = [
        "2016-17", "2017-18", "2018-19", "2019-20", 
        "2020-21", "2021-22", "2022-23", "2023-24"
    ]
    
    run_walk_forward_backtest(
        target_seasons=seasons_to_test,
        exclude_tanking_flags=args.exclude_tanking_flags,
        n_sims=args.n_sims
    )
