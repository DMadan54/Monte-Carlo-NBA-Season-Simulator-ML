"""Phase 3 sensitivity rerun with tanking-review games excluded from fitting.

The model is trained without ``LIKELY_TANKING_REVIEW_FLAG`` team-games, but
the simulation still includes every regular-season game and compares against
the full official actual-wins total. This is a training-data sensitivity
analysis, not a claim that flagged games were intentionally lost.

Usage:
    python src/backtest/run_phase3_excluding_tanked.py --n-sims 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from game_model import FEATURES, train_model
from season_sim import load_season_schedule, run_simulation_with_schedule

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROCESSED_DATA_DIR / "models"
DEFAULT_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]


def actual_wins_by_season(seasons: list[str]) -> pd.DataFrame:
    """Return actual full-season wins; flagged games are intentionally retained."""
    raw = pd.read_parquet(RAW_DATA_DIR / "team_game_logs_combined.parquet")
    if "SEASON" not in raw:
        raise ValueError("Raw team logs require a SEASON column.")
    raw = raw[raw["SEASON"].isin(seasons)].copy()
    raw["Actual Wins"] = (raw["WL"] == "W").astype(int)
    return raw.groupby(["SEASON", "TEAM_ABBREVIATION"], as_index=False)["Actual Wins"].sum()


def run_phase3(seasons: list[str], n_sims: int, exclude_tanking_flags: bool = True) -> pd.DataFrame:
    """Fit a model and simulate the unchanged historical schedules."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    variant = "exclude_flagged_training_games" if exclude_tanking_flags else "all_training_games"
    model_path = MODELS_DIR / f"game_model_phase3_{variant}.pkl"
    model = train_model(exclude_tanking_flags=exclude_tanking_flags, model_output=model_path)
    model_features = list(model.feature_name_)

    actual = actual_wins_by_season(seasons)
    comparisons = []
    for season in seasons:
        print(f"Simulating {season} ({n_sims:,} trials)...", flush=True)
        schedule = load_season_schedule(season, features=model_features)
        simulated = run_simulation_with_schedule(schedule, model, n_sims=n_sims, seed=42)
        simulated = simulated.rename(columns={"team": "TEAM_ABBREVIATION", "mean_wins": "Sim Mean Wins", "std_wins": "Sim Std", "min_wins": "Sim Min", "max_wins": "Sim Max"})
        comparison = actual[actual["SEASON"] == season].merge(
            simulated[["TEAM_ABBREVIATION", "Sim Mean Wins", "Sim Std", "Sim Min", "Sim Max"]],
            on="TEAM_ABBREVIATION",
            how="left",
            validate="one_to_one",
        )
        comparison["Abs Error"] = (comparison["Actual Wins"] - comparison["Sim Mean Wins"]).abs()
        comparison.insert(0, "Model Variant", variant)
        comparison = comparison.rename(columns={"SEASON": "Season", "TEAM_ABBREVIATION": "Team"})
        comparisons.append(comparison)
        print(f"  {season} MAE: {comparison['Abs Error'].mean():.2f}", flush=True)

    output = pd.concat(comparisons, ignore_index=True).sort_values(["Season", "Actual Wins"], ascending=[True, False])
    output_path = PROCESSED_DATA_DIR / f"phase3_{variant}_standings_comparison.csv"
    output.to_csv(output_path, index=False)
    print(f"\nPhase 3 MAE by season ({variant}):")
    print(output.groupby("Season")["Abs Error"].mean().round(2).to_string())
    print(f"Overall MAE: {output['Abs Error'].mean():.2f}")
    print(f"Saved model: {model_path}")
    print(f"Saved comparison: {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--season", action="append", help="Season to backtest; repeatable.")
    parser.add_argument(
        "--include-flagged-training",
        action="store_true",
        help="Run the matched control with all valid training rows included.",
    )
    args = parser.parse_args()
    run_phase3(
        args.season or DEFAULT_SEASONS,
        args.n_sims,
        exclude_tanking_flags=not args.include_flagged_training,
    )


if __name__ == "__main__":
    main()
