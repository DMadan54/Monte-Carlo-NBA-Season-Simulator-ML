"""Evaluate March-1 MVP forecasts in forward-only season folds."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.mvp_model import resolve_votes
from src.models.mvp_cutoff_model import FEATURES, build_cutoff_candidates, evaluate


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
VOTE_DIR = RAW / "mvp_voting/v1_325de65f7849"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    player_paths = sorted(RAW.glob("player_game_logs_*.parquet"))
    team_paths = sorted(p for p in RAW.glob("team_game_logs_20??-??.parquet")
                        if p.stem[-7:] in {x.stem[-7:] for x in player_paths})
    players = pd.concat([pd.read_parquet(p) for p in player_paths], ignore_index=True)
    teams = pd.concat([pd.read_parquet(p) for p in team_paths], ignore_index=True)
    votes = pd.read_csv(VOTE_DIR / "mvp_votes.csv")
    votes = votes[votes.season_end_year.between(2016, 2025)]
    alias_path = RAW / "mvp_voting/name_aliases.csv"
    mapped, issues = resolve_votes(votes, players, pd.read_csv(alias_path, dtype={"player_id": str}))
    if not issues.empty or len(mapped) != len(votes):
        raise ValueError("Full-season voting identities unresolved")
    candidates = build_cutoff_candidates(players, teams, mapped)
    observed = candidates[candidates.vote_share > 0].groupby("season").size()
    expected = votes.groupby("season").size()
    coverage = pd.DataFrame({"vote_recipients_at_cutoff": observed, "final_vote_recipients": expected}).fillna(0).astype(int)
    coverage["recipient_coverage"] = coverage.vote_recipients_at_cutoff / coverage.final_vote_recipients
    if not candidates.groupby("season").is_winner.sum().eq(1).all():
        raise ValueError("An eventual MVP winner is absent at a March 1 cutoff")
    predictions, metrics, weights = evaluate(candidates)
    now = datetime.now(timezone.utc)
    run = ROOT / "data/processed/mvp_runs" / f"v2_cutoff_{now.strftime('%Y%m%dT%H%M%SZ')}_{digest(VOTE_DIR / 'mvp_votes.csv')[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    candidates.to_parquet(run / "cutoff_candidates.parquet", index=False)
    mapped.to_csv(run / "identity_mapping.csv", index=False)
    coverage.to_csv(run / "coverage.csv")
    predictions.to_parquet(run / "predictions.parquet", index=False)
    metrics.to_csv(run / "season_metrics.csv", index=False)
    weights.to_csv(run / "standardized_weights.csv", index=False)
    picks = []
    for year, season in predictions.groupby("season_end_year"):
        winner = season[season.is_winner].iloc[0]
        for method in ["model", "baseline"]:
            top = season.loc[season[f"{method}_probability"].idxmax()]
            picks.append({"season_end_year": year, "method": method,
                          "predicted_winner": top.player_name,
                          "predicted_winner_probability": top[f"{method}_probability"],
                          "actual_winner": winner.player_name,
                          "actual_winner_rank": winner[f"{method}_rank"]})
    pd.DataFrame(picks).to_csv(run / "top_picks.csv", index=False)
    top_calibration = metrics.groupby("method").agg(
        heldout_seasons=("test_year", "size"), mean_top_probability=("top_probability", "mean"),
        top_pick_hit_rate=("winner_correct", "mean")).reset_index()
    top_calibration.to_csv(run / "top_pick_calibration.csv", index=False)
    eligibility = candidates[candidates.season_end_year >= 2024].groupby("season").agg(
        candidates=("player_id", "size"), normal_path_closed=("normal_path_open", lambda x: int((x == 0).sum())),
        recipients_with_closed_path=("vote_share", lambda x: 0)).reset_index()
    eligibility["recipients_with_closed_path"] = [int(((g.vote_share > 0) & (g.normal_path_open == 0)).sum())
        for _, g in candidates[candidates.season_end_year >= 2024].groupby("season")]
    eligibility.to_csv(run / "eligibility_audit.csv", index=False)
    inputs = [*player_paths, *team_paths, VOTE_DIR / "mvp_votes.csv", alias_path]
    manifest = {"created_at_utc": now.isoformat(), "mode": "March-1 fixed-cutoff retrospective forecast",
        "cutoff_rule": "Only game rows dated before March 1 of each season end year",
        "features": FEATURES, "target": "eventual MVP winner",
        "candidate_universe": "players observed in regular-season logs before the cutoff",
        "team_context": "most recently observed team at cutoff; team record through cutoff only",
        "eligibility": "2023-24 onward: normal 65 qualifying-game path with 20-minute minimum and up to two 15-19-minute games; 82-game season length. Exceptional eligibility unknown; path flag never hard-excludes a candidate.",
        "model": "median imputation, standardization, balanced logistic regression; C selected on earlier-season winner log loss; C=0.3 until inner folds exist",
        "baseline": "points per game * current-team win fraction * games/team games; softmax temperature selected by earlier training seasons",
        "seed": 0, "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "coverage": coverage.reset_index().to_dict("records"),
        "top_pick_calibration": top_calibration.to_dict("records"),
        "mean_metrics": metrics.groupby("method")[["winner_correct", "winner_rank", "winner_top3", "recipient_recall_at_10", "winner_log_loss", "multiclass_brier", "recipient_spearman"]].mean().to_dict("index"),
        "limitations": ["March 1 is a retrospective fixed date, not an archived award forecast date",
            "Future award votes are labels only; final eligibility and future rosters are not inputs",
            "Historical schedule revisions and injury/transaction news are unavailable",
            "Normal eligibility path does not include injury/grievance exceptions",
            "Only eight held-out MVP winners; probability calibration is poorly identified"]}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(run), "coverage": manifest["coverage"],
                      "mean_metrics": manifest["mean_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
