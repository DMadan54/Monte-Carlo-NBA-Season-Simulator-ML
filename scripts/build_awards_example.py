"""Package a compact, read-only MVP holdout example for the portfolio app."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/mvp_runs/v4_2026_holdout_20261012T105053Z_78977174"
DEFAULT_OUTPUT = ROOT / "examples/mvp-2026-holdout"
PREDICTION_COLUMNS = [
    "cutoff_month", "cutoff", "player_id", "player_name", "is_winner",
    "model_probability", "baseline_probability", "model_rank", "baseline_rank",
    "games", "pts_per_game", "reb_per_game", "ast_per_game", "team_win_pct",
]
METRIC_COLUMNS = [
    "test_year", "cutoff_month", "method", "train_through", "winner_correct",
    "winner_rank", "winner_top3", "winner_log_loss", "multiclass_brier",
    "recipient_spearman", "top_probability",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source=SOURCE, output=DEFAULT_OUTPUT):
    source, output = Path(source), Path(output)
    predictions = pd.read_parquet(source / "predictions.parquet")
    predictions = predictions[predictions.season_end_year.eq(2026)].copy()
    metrics = pd.read_csv(source / "season_metrics.csv")
    if len(predictions) != 1566 or set(predictions.cutoff_month) != {1, 2, 3}:
        raise ValueError("Unexpected 2026 candidate coverage")
    if metrics.test_year.min() != 2018 or metrics.test_year.max() != 2026:
        raise ValueError("Expected all chronological MVP folds through 2026")
    if not (metrics.loc[metrics.test_year.eq(2026), "train_through"] == 2025).all():
        raise ValueError("2026 holdout was not trained only through 2025")
    output.mkdir(parents=True, exist_ok=False)
    predictions[PREDICTION_COLUMNS].sort_values(["cutoff_month", "model_rank"]).to_csv(
        output / "predictions_2026.csv", index=False, float_format="%.17g")
    metrics[METRIC_COLUMNS].sort_values(["test_year", "cutoff_month", "method"]).to_csv(
        output / "season_metrics.csv", index=False, float_format="%.17g")
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "artifact_kind": "historical_fixed_cutoff_mvp_holdout",
        "season": "2025-26",
        "source_run": str(source.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": {name: sha256(source / name) for name in
                          ("predictions.parquet", "season_metrics.csv", "manifest.json")},
        "official_vote_source": source_manifest["official_vote_source"]["source_release_url"],
        "training_rule": "Each test season trains only on earlier seasons; 2026 fold trained through 2025",
        "mode": "retrospective fixed-date award forecast, not preseason or live",
        "limitations": source_manifest["limitations"] + [
            "The learned model missed the 2025-26 winner at all three cutoffs",
            "Three cutoff dates in 2025-26 are one independent award outcome",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.source, args.output))
