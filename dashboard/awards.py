"""Read-only validation of the bundled historical MVP holdout."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_awards(path):
    directory = Path(path)
    required_files = ("manifest.json", "predictions_2026.csv", "season_metrics.csv")
    if any(not (directory / name).is_file() for name in required_files):
        raise FileNotFoundError("Expected manifest.json, predictions_2026.csv, and season_metrics.csv.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(directory / "predictions_2026.csv", dtype={"player_id": str})
    metrics = pd.read_csv(directory / "season_metrics.csv")
    required_predictions = {"cutoff_month", "cutoff", "player_id", "player_name", "is_winner",
                            "model_probability", "baseline_probability", "model_rank", "baseline_rank"}
    required_metrics = {"test_year", "cutoff_month", "method", "train_through", "winner_correct",
                        "winner_rank", "winner_log_loss", "multiclass_brier"}
    if missing := required_predictions - set(predictions):
        raise ValueError(f"Award predictions lack columns: {sorted(missing)}")
    if missing := required_metrics - set(metrics):
        raise ValueError(f"Award metrics lack columns: {sorted(missing)}")
    if manifest.get("artifact_kind") != "historical_fixed_cutoff_mvp_holdout":
        raise ValueError("Unsupported award artifact kind")
    if set(predictions.cutoff_month) != {1, 2, 3} or set(metrics.cutoff_month) != {1, 2, 3}:
        raise ValueError("Award cutoff coverage is incomplete")
    if predictions.duplicated(["cutoff_month", "player_id"]).any():
        raise ValueError("Duplicate candidate within an award cutoff")
    if predictions[["cutoff_month", "player_id", "player_name"]].isna().any().any():
        raise ValueError("Award candidate identity is missing")
    if set(metrics.method) != {"model", "baseline"} or metrics.duplicated(
            ["test_year", "cutoff_month", "method"]).any():
        raise ValueError("Award season metrics are incomplete or duplicated")
    if metrics.test_year.min() != 2018 or metrics.test_year.max() != 2026 or len(metrics) != 54:
        raise ValueError("Expected nine chronological award test seasons")
    if not (metrics.loc[metrics.test_year.eq(2026), "train_through"] == 2025).all():
        raise ValueError("2026 award holdout uses test-year training")
    for month, frame in predictions.groupby("cutoff_month"):
        if len(frame) < 30 or int(frame.is_winner.sum()) != 1:
            raise ValueError(f"Winner or candidate coverage missing at month {month}")
        for method in ("model", "baseline"):
            probabilities = frame[f"{method}_probability"]
            if not np.isfinite(probabilities).all() or not probabilities.between(0, 1).all():
                raise ValueError(f"Invalid {method} probabilities at month {month}")
            if not np.isclose(probabilities.sum(), 1, atol=1e-8):
                raise ValueError(f"{method} probabilities do not sum to one at month {month}")
            winner = frame.loc[frame.is_winner].iloc[0]
            metric = metrics[(metrics.test_year == 2026) & (metrics.cutoff_month == month) &
                             (metrics.method == method)].iloc[0]
            if not np.isclose(-np.log(winner[f"{method}_probability"]), metric.winner_log_loss, atol=1e-9):
                raise ValueError(f"{method} winner loss disagrees with predictions at month {month}")
            if int(winner[f"{method}_rank"]) != int(metric.winner_rank):
                raise ValueError(f"{method} winner rank disagrees with predictions at month {month}")
    return manifest, predictions, metrics
