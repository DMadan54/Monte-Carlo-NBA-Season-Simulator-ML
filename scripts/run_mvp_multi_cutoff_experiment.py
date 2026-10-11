"""Run January/February/March MVP forecasts with forward-only season folds."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.mvp_cutoff_model import FEATURES, build_cutoff_candidates, evaluate
from src.models.mvp_model import resolve_votes


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
VOTE_DIR = RAW / "mvp_voting/v1_325de65f7849"
MONTHS = (1, 2, 3)
SUMMARY_COLUMNS = ["winner_correct", "winner_rank", "winner_top3",
                   "recipient_recall_at_10", "winner_log_loss",
                   "multiclass_brier", "recipient_spearman", "top_probability"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def paired_bootstrap(metrics, repetitions=5000, seed=0):
    """Resample seasons, keeping each model/baseline pair together."""
    rng = np.random.default_rng(seed)
    rows = []
    for month, group in metrics.groupby("cutoff_month"):
        pairs = group.pivot(index="test_year", columns="method", values="winner_log_loss")
        delta = (pairs["model"] - pairs["baseline"]).to_numpy()
        samples = rng.choice(delta, size=(repetitions, len(delta)), replace=True).mean(axis=1)
        rows.append({"cutoff_month": month, "metric": "winner_log_loss_model_minus_baseline",
                     "paired_seasons": len(delta), "mean_difference": float(delta.mean()),
                     "bootstrap_95_low": float(np.quantile(samples, .025)),
                     "bootstrap_95_high": float(np.quantile(samples, .975))})
    return pd.DataFrame(rows)


def progression_bootstrap(metrics, repetitions=5000, seed=0):
    """Compare dates using the same eight held-out seasons."""
    rng = np.random.default_rng(seed)
    rows = []
    for method, group in metrics.groupby("method"):
        table = group.pivot(index="test_year", columns="cutoff_month", values="winner_log_loss")
        for first, second in ((1, 2), (2, 3), (1, 3)):
            delta = (table[second] - table[first]).to_numpy()
            samples = rng.choice(delta, size=(repetitions, len(delta)), replace=True).mean(axis=1)
            rows.append({"method": method, "first_month": first, "second_month": second,
                         "metric": "winner_log_loss_second_minus_first", "paired_seasons": len(delta),
                         "mean_difference": float(delta.mean()),
                         "bootstrap_95_low": float(np.quantile(samples, .025)),
                         "bootstrap_95_high": float(np.quantile(samples, .975))})
    return pd.DataFrame(rows)


def main():
    player_paths = sorted(RAW.glob("player_game_logs_*.parquet"))
    team_paths = sorted(p for p in RAW.glob("team_game_logs_20??-??.parquet")
                        if p.stem[-7:] in {x.stem[-7:] for x in player_paths})
    if len(player_paths) != 10 or len(team_paths) != 10:
        raise ValueError("Expected matching 2015-16 through 2024-25 player/team log caches")
    players = pd.concat([pd.read_parquet(p) for p in player_paths], ignore_index=True)
    teams = pd.concat([pd.read_parquet(p) for p in team_paths], ignore_index=True)
    vote_path = VOTE_DIR / "mvp_votes.csv"
    votes = pd.read_csv(vote_path)
    votes = votes[votes.season_end_year.between(2016, 2025)]
    alias_path = RAW / "mvp_voting/name_aliases.csv"
    mapped, issues = resolve_votes(votes, players, pd.read_csv(alias_path, dtype={"player_id": str}))
    if not issues.empty or len(mapped) != len(votes):
        raise ValueError("Voting identity coverage incomplete")

    candidates_all, predictions_all, metrics_all, weights_all, coverage_all = [], [], [], [], []
    for month in MONTHS:
        candidates = build_cutoff_candidates(players, teams, mapped, month=month)
        if candidates.duplicated(["season", "player_id"]).any():
            raise ValueError(f"Duplicate cutoff candidate in month {month}")
        observed = candidates.groupby("season").agg(
            candidates=("player_id", "size"),
            vote_recipients_at_cutoff=("vote_share", lambda x: int((x > 0).sum())),
            winners_at_cutoff=("is_winner", "sum"),
            median_team_games=("team_games", "median"),
            normal_path_closed=("normal_path_open", lambda x: int((x == 0).sum()))).reset_index()
        expected = votes.groupby("season").size().rename("final_vote_recipients").reset_index()
        observed = observed.merge(expected, on="season", validate="one_to_one")
        observed["recipient_coverage"] = observed.vote_recipients_at_cutoff / observed.final_vote_recipients
        observed["cutoff_month"] = month
        if not observed.winners_at_cutoff.eq(1).all():
            raise ValueError(f"Eventual winner missing at month {month}; report coverage before evaluation")
        predictions, metrics, weights = evaluate(candidates)
        for frame in (candidates, predictions, metrics, weights):
            frame["cutoff_month"] = month
        candidates_all.append(candidates)
        predictions_all.append(predictions)
        metrics_all.append(metrics)
        weights_all.append(weights)
        coverage_all.append(observed)

    candidates = pd.concat(candidates_all, ignore_index=True)
    predictions = pd.concat(predictions_all, ignore_index=True)
    metrics = pd.concat(metrics_all, ignore_index=True)
    weights = pd.concat(weights_all, ignore_index=True)
    coverage = pd.concat(coverage_all, ignore_index=True)
    summary = metrics.groupby(["cutoff_month", "method"])[SUMMARY_COLUMNS].mean().reset_index()
    summary["heldout_seasons"] = 8
    delayed_start_sensitivity = metrics[metrics.test_year != 2021].groupby(
        ["cutoff_month", "method"])[SUMMARY_COLUMNS].mean().reset_index()
    delayed_start_sensitivity["heldout_seasons"] = 7
    paired = paired_bootstrap(metrics)
    progress_intervals = progression_bootstrap(metrics)
    progression = metrics[["test_year", "cutoff_month", "method", "cutoff", "winner_rank",
                           "winner_correct", "winner_top3", "winner_log_loss", "multiclass_brier",
                           "top_probability"]].copy()
    progression = progression.merge(coverage[["cutoff_month", "season", "median_team_games"]].assign(
        test_year=lambda x: x.season.str[:4].astype(int) + 1).drop(columns="season"),
        on=["cutoff_month", "test_year"], validate="many_to_one")
    picks = []
    for (month, year), season in predictions.groupby(["cutoff_month", "season_end_year"]):
        winner = season.loc[season.is_winner].iloc[0]
        for method in ("model", "baseline"):
            top = season.loc[season[f"{method}_probability"].idxmax()]
            picks.append({"cutoff_month": month, "season_end_year": year, "method": method,
                          "predicted_winner": top.player_name,
                          "top_probability": top[f"{method}_probability"],
                          "actual_winner": winner.player_name,
                          "actual_winner_rank": int(winner[f"{method}_rank"])})
    now = datetime.now(timezone.utc)
    run = ROOT / "data/processed/mvp_runs" / f"v3_multi_cutoff_{now.strftime('%Y%m%dT%H%M%SZ')}_{digest(vote_path)[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    candidates.to_parquet(run / "cutoff_candidates.parquet", index=False)
    predictions.to_parquet(run / "predictions.parquet", index=False)
    metrics.to_csv(run / "season_metrics.csv", index=False)
    weights.to_csv(run / "standardized_weights.csv", index=False)
    coverage.to_csv(run / "coverage.csv", index=False)
    summary.to_csv(run / "month_summary.csv", index=False)
    delayed_start_sensitivity.to_csv(run / "exclude_2020_21_sensitivity.csv", index=False)
    paired.to_csv(run / "paired_bootstrap.csv", index=False)
    progress_intervals.to_csv(run / "progression_bootstrap.csv", index=False)
    progression.to_csv(run / "season_progression.csv", index=False)
    pd.DataFrame(picks).to_csv(run / "top_picks.csv", index=False)
    inputs = [*player_paths, *team_paths, vote_path, alias_path]
    manifest = {"created_at_utc": now.isoformat(), "mode": "fixed-cutoff retrospective MVP forecast",
        "cutoff_months": list(MONTHS), "cutoff_day": 1, "cutoff_rows_inclusive": False,
        "training_rule": "each test season uses only earlier seasons; separate model and baseline calibration per cutoff month",
        "target": "eventual MVP winner", "features": FEATURES, "seed": 0,
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "coverage": coverage.to_dict("records"),
        "month_summary": summary.to_dict("records"),
        "exclude_delayed_start_2020_21_sensitivity": delayed_start_sensitivity.to_dict("records"),
        "paired_bootstrap": paired.to_dict("records"),
        "progression_bootstrap": progress_intervals.to_dict("records"),
        "limitations": ["Eight independent held-out seasons per cutoff; dates within a season share one outcome",
            "January 2020-21 follows only about five games per team because that season began in December",
            "Historical injuries, transactions before first appearance, and award exceptions are unavailable",
            "Final vote labels are retrospective; no archived contemporaneous award market is available",
            "Top-pick probabilities are not proven calibrated"]}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(run), "summary": manifest["month_summary"],
                      "paired_bootstrap": manifest["paired_bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
