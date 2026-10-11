"""Run the versioned retrospective MVP experiment from cached regular-season logs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.mvp_model import FEATURES, build_candidates, evaluate, resolve_votes


ROOT = Path(__file__).resolve().parents[1]
VOTING = ROOT / "data/raw/mvp_voting/v1_325de65f7849"
RAW = ROOT / "data/raw"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    player_paths = sorted(RAW.glob("player_game_logs_*.parquet"))
    team_paths = sorted(p for p in RAW.glob("team_game_logs_20??-??.parquet")
                        if p.stem[-7:] in {x.stem[-7:] for x in player_paths})
    if len(player_paths) != 10:
        raise ValueError("Expected ten cached player seasons, 2015-16 through 2024-25")
    players = pd.concat([pd.read_parquet(p) for p in player_paths], ignore_index=True)
    teams = pd.concat([pd.read_parquet(p) for p in team_paths], ignore_index=True)
    votes = pd.read_csv(VOTING / "mvp_votes.csv")
    votes = votes[votes.season_end_year.between(2016, 2025)].copy()
    aliases = pd.read_csv(RAW / "mvp_voting/name_aliases.csv", dtype={"player_id": str})
    mapped, issues = resolve_votes(votes, players, aliases)
    if not issues.empty or len(mapped) != len(votes):
        raise ValueError(f"Unresolved voting identities: {issues.to_dict('records')}")
    candidates = build_candidates(players, teams, mapped)
    if len(candidates) != candidates[["season", "player_id"]].drop_duplicates().shape[0]:
        raise ValueError("Duplicate player-season candidates")
    if not candidates.groupby("season").is_winner.sum().eq(1).all():
        raise ValueError("Every season requires exactly one mapped winner")
    predictions, metrics, weights = evaluate(candidates)
    run_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = ROOT / "data/processed/mvp_runs" / f"v1_{run_time}_{sha(VOTING / 'mvp_votes.csv')[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    candidates.to_parquet(run / "candidates.parquet", index=False)
    mapped.to_csv(run / "identity_mapping.csv", index=False)
    issues.to_csv(run / "identity_issues.csv", index=False)
    predictions.to_parquet(run / "predictions.parquet", index=False)
    metrics.to_csv(run / "season_metrics.csv", index=False)
    weights.to_csv(run / "standardized_weights.csv", index=False)
    calibration = predictions.assign(bin=pd.cut(predictions.pred_share, bins=[-0.001, .01, .05, .1, .25, .5, 1], include_lowest=True)).groupby("bin", observed=False).agg(
        count=("vote_share", "size"), mean_predicted=("pred_share", "mean"), mean_observed=("vote_share", "mean")).reset_index()
    calibration.to_csv(run / "share_calibration.csv", index=False)
    coverage = candidates.groupby("season").agg(candidates=("player_id", "size"),
        vote_recipients=("vote_share", lambda x: int((x > 0).sum())), winners=("is_winner", "sum")).reset_index()
    coverage.to_csv(run / "coverage.csv", index=False)
    inputs = [*player_paths, *team_paths, VOTING / "mvp_votes.csv", RAW / "mvp_voting/name_aliases.csv"]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": 0,
        "mode": "retrospective season-end MVP vote-share evaluation", "features": FEATURES,
        "target": "vote points / maximum vote points, zero for nonrecipients",
        "candidate_rule": "all players with at least one cached regular-season game",
        "primary_team_rule": "team with most player minutes that season",
        "baseline": "points per game * primary-team win fraction * player games / season maximum games",
        "model": "median imputation, standardization, ridge; alpha selected by earlier-season folds (10 until inner folds exist)",
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        "coverage": coverage.to_dict("records"), "unmatched_vote_rows": len(issues),
        "mean_metrics": metrics.groupby("method")[["winner_correct", "spearman", "recipient_spearman", "share_mae_all", "share_mae_recipients"]].mean().to_dict("index"),
        "limitations": ["Secondary voting transcription", "Historical MVP eligibility rules not applied",
            "Primary-team record simplifies traded players", "Season-end features do not validate preseason or in-season forecasts",
            "Vote share is not a winner probability; calibration is share-bin comparison"]}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(run), "means": manifest["mean_metrics"], "unmatched": len(issues)}, indent=2))


if __name__ == "__main__":
    main()
