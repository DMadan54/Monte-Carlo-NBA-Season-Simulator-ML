"""Read-only checks for the numerical claims in the portfolio release."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from dashboard.artifact import load_artifact

ROOT = Path(__file__).resolve().parents[1]
GAME_RUN = ROOT / "data/processed/player_runs/v1_20260910T200443Z_6a86c4"
SCENARIO_RUN = ROOT / "data/processed/player_runs/v1_20260910T215749Z_35f88a/scenario_20260910T220652Z"
MULTI_RUN = ROOT / "data/processed/multi_cutoff_runs/v1_20260911T110925Z_9a9b27"
MVP_RUN = ROOT / "data/processed/mvp_runs/v4_2026_holdout_20261012T105053Z_78977174"
EXAMPLE = ROOT / "examples/2025-01-01-scenario"


def verify():
    audit = json.loads((GAME_RUN / "data_audit.json").read_text(encoding="utf-8"))
    game_metrics = json.loads((GAME_RUN / "metrics.json").read_text(encoding="utf-8"))
    player_metrics = json.loads((GAME_RUN / "player_metrics.json").read_text(encoding="utf-8"))
    minutes = next((row for row in player_metrics if row["season"] == 2025), None)
    if minutes is None or minutes["candidates"] != 40970:
        raise ValueError("Held-out player minutes coverage is incomplete")
    minutes_improvement = 1 - minutes["ml_minutes_mae"] / minutes["baseline_minutes_mae"]
    if not 0.20 < minutes_improvement < 0.25:
        raise ValueError("Held-out player minutes improvement differs from documented result")
    games = {}
    for name in ("team_linear", "hybrid_development_ml"):
        frame = pd.read_csv(GAME_RUN / f"{name}_test_predictions.csv", dtype={"GAME_ID": str})
        if len(frame) != 1230 or frame.GAME_ID.nunique() != 1230 or not frame.probability.between(0, 1).all():
            raise ValueError(f"Invalid held-out game rows for {name}")
        measured = log_loss(frame.TARGET, frame.probability)
        saved = game_metrics[name]["test"]["log_loss"]
        if not np.isclose(measured, saved, atol=1e-12):
            raise ValueError(f"Held-out log loss disagrees with saved metrics for {name}")
        games[name] = {"games": len(frame), "log_loss": round(measured, 6)}

    report, standings = load_artifact(SCENARIO_RUN)
    load_artifact(EXAMPLE)
    for filename in ("report.json", "standings.csv"):
        original = hashlib.sha256((SCENARIO_RUN / filename).read_bytes()).hexdigest()
        packaged = hashlib.sha256((EXAMPLE / filename).read_bytes()).hexdigest()
        if original != packaged:
            raise ValueError(f"Packaged example differs from saved source: {filename}")
    if report["cutoff"] != "2025-01-01" or report["games_remaining"] != 746:
        raise ValueError("Unexpected historical scenario artifact")
    per_cutoff = pd.read_csv(MULTI_RUN / "per_cutoff_metrics.csv")
    subset = per_cutoff[(per_cutoff.model == "hybrid_development_ml") &
                        (per_cutoff.learned_uncertainty == True)]
    control = per_cutoff[(per_cutoff.model == "team_linear") &
                         (per_cutoff.learned_uncertainty == False)]
    if len(subset) != 28 or len(control) != 28 or subset.season.nunique() != 7:
        raise ValueError("Multi-cutoff coverage is incomplete")
    aggregate = pd.read_csv(MULTI_RUN / "aggregate_metrics.csv")
    saved = aggregate[(aggregate.model == "hybrid_development_ml") &
                      (aggregate.learned_uncertainty == True)].iloc[0]
    baseline = aggregate[(aggregate.model == "team_linear") &
                         (aggregate.learned_uncertainty == False)].iloc[0]
    for column in ("standings_mae", "coverage_90", "rank_correlation"):
        if not np.isclose(subset[column].mean(), saved[column], atol=1e-12):
            raise ValueError(f"Multi-cutoff aggregate mismatch: {column}")
    if not np.isclose(control.standings_mae.mean(), baseline.standings_mae, atol=1e-12):
        raise ValueError("Team baseline aggregate mismatch")

    mvp = pd.read_csv(MVP_RUN / "new_holdout_2026.csv")
    picks = pd.read_csv(MVP_RUN / "top_picks_2026.csv")
    if len(mvp) != 6 or set(mvp.cutoff_month) != {1, 2, 3} or len(picks) != 6:
        raise ValueError("MVP holdout is incomplete")
    if mvp[mvp.method == "model"].winner_correct.any() or not mvp[mvp.method == "baseline"].winner_correct.all():
        raise ValueError("MVP holdout selections differ from documented result")
    return {
        "historical_data": {"seasons": len(audit["seasons"]), "games": audit["games"],
                            "player_appearances": audit["player_rows"]},
        "heldout_2024_25_game_test": games,
        "heldout_2024_25_player_minutes": {"candidates": minutes["candidates"],
                                            "baseline_mae": round(minutes["baseline_minutes_mae"], 3),
                                            "ml_mae": round(minutes["ml_minutes_mae"], 3),
                                            "relative_mae_reduction": round(minutes_improvement, 3)},
        "fixed_cutoff_2025_01_01": {"trials": report["n_sims"], "remaining_games": report["games_remaining"],
                                    "standings_mae": round(report["mean_absolute_error"], 3),
                                    "nominal_90_coverage_teams": int(round(report["interval_coverage"] * len(standings))),
                                    "teams": len(standings)},
        "seven_season_28_cutoff_backtest": {"cutoffs": len(subset), "standings_mae": round(saved.standings_mae, 3),
                                            "team_baseline_mae": round(baseline.standings_mae, 3),
                                            "nominal_90_coverage": round(saved.coverage_90, 3)},
        "new_2025_26_mvp_holdout": {"independent_award_outcomes": 1, "cutoffs": 3,
                                    "learned_exact_winner": int(mvp[mvp.method == "model"].winner_correct.sum()),
                                    "baseline_exact_winner": int(mvp[mvp.method == "baseline"].winner_correct.sum())},
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
