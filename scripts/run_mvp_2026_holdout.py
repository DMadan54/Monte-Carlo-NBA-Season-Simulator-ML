"""Evaluate the frozen MVP cutoff method on the newly available 2025-26 season."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.build_mvp_2026_votes import OUTPUT as OFFICIAL_VOTE_DIR, validated_votes
from src.models.mvp_cutoff_model import FEATURES, build_cutoff_candidates, evaluate
from src.models.mvp_model import resolve_votes


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
LIVE = RAW / "live_preseason/v1_20260922T113549Z_c79af6"
LEGACY_VOTES = RAW / "mvp_voting/v1_325de65f7849/mvp_votes.csv"
OFFICIAL_VOTES = OFFICIAL_VOTE_DIR / "mvp_votes_2025-26.csv"
ALIAS = RAW / "mvp_voting/name_aliases.csv"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_2026_logs(players, teams, metadata):
    for name in ("player_game_logs_2025-26.parquet", "team_game_logs_2025-26.parquet"):
        if digest(LIVE / name) != metadata["files"][name]:
            raise ValueError(f"2025-26 cache hash disagrees with source manifest: {name}")
    if players.SEASON.unique().tolist() != ["2025-26"] or teams.SEASON.unique().tolist() != ["2025-26"]:
        raise ValueError("2025-26 logs contain another season")
    if teams.duplicated(["GAME_ID", "TEAM_ID"]).any() or players.duplicated(["GAME_ID", "PLAYER_ID"]).any():
        raise ValueError("Duplicate 2025-26 game rows")
    if len(teams) != 2460 or teams.GAME_ID.nunique() != 1230:
        raise ValueError("2025-26 team game coverage incomplete")
    if not teams.groupby("GAME_ID").TEAM_ID.nunique().eq(2).all():
        raise ValueError("2025-26 game does not have two teams")
    if not teams.groupby("TEAM_ID").GAME_ID.nunique().eq(82).all():
        raise ValueError("2025-26 team schedule does not contain 82 games each")
    points = players.groupby(["GAME_ID", "TEAM_ID"]).PTS.sum()
    team_points = teams.set_index(["GAME_ID", "TEAM_ID"]).PTS
    if len(points) != len(team_points) or not points.sort_index().equals(team_points.sort_index()):
        raise ValueError("2025-26 player and team points fail reconciliation")
    joined = players[["GAME_ID", "TEAM_ID", "GAME_DATE"]].merge(
        teams[["GAME_ID", "TEAM_ID", "GAME_DATE"]], on=["GAME_ID", "TEAM_ID"],
        suffixes=("_PLAYER", "_TEAM"), validate="many_to_one")
    if len(joined) != len(players) or not pd.to_datetime(joined.GAME_DATE_PLAYER).eq(
            pd.to_datetime(joined.GAME_DATE_TEAM)).all():
        raise ValueError("2025-26 game dates fail reconciliation")
    return {"player_rows": len(players), "team_rows": len(teams), "games": 1230,
            "teams": teams.TEAM_ID.nunique(), "players": players.PLAYER_ID.nunique(),
            "first_game_date": str(pd.to_datetime(teams.GAME_DATE).min().date()),
            "last_game_date": str(pd.to_datetime(teams.GAME_DATE).max().date())}


def main():
    prior_player_paths = sorted(RAW.glob("player_game_logs_*.parquet"))
    prior_team_paths = sorted(p for p in RAW.glob("team_game_logs_20??-??.parquet")
                              if p.stem[-7:] in {x.stem[-7:] for x in prior_player_paths})
    if len(prior_player_paths) != 10 or len(prior_team_paths) != 10:
        raise ValueError("Historical training cache is incomplete")
    new_player_path = LIVE / "player_game_logs_2025-26.parquet"
    new_team_path = LIVE / "team_game_logs_2025-26.parquet"
    new_players = pd.read_parquet(new_player_path)
    new_teams = pd.read_parquet(new_team_path)
    source_manifest = LIVE / "source_metadata.json"
    log_audit = audit_2026_logs(new_players, new_teams, json.loads(source_manifest.read_text(encoding="utf-8")))
    players = pd.concat([*(pd.read_parquet(p) for p in prior_player_paths), new_players], ignore_index=True)
    teams = pd.concat([*(pd.read_parquet(p) for p in prior_team_paths), new_teams], ignore_index=True)
    votes_2026 = pd.read_csv(OFFICIAL_VOTES)
    expected = validated_votes()[votes_2026.columns]
    pd.testing.assert_frame_equal(votes_2026, expected, check_dtype=False)
    votes = pd.concat([pd.read_csv(LEGACY_VOTES), votes_2026], ignore_index=True)
    votes = votes[votes.season_end_year.between(2016, 2026)]
    mapped, issues = resolve_votes(votes, players, pd.read_csv(ALIAS, dtype={"player_id": str}))
    if not issues.empty or len(mapped) != len(votes):
        raise ValueError(f"Vote identity coverage incomplete: {issues.to_dict('records')}")
    all_candidates, all_predictions, all_metrics, all_weights, coverage = [], [], [], [], []
    for month in (1, 2, 3):
        candidates = build_cutoff_candidates(players, teams, mapped, month=month)
        observed = candidates[candidates.season_end_year == 2026]
        if observed.is_winner.sum() != 1 or int((observed.vote_share > 0).sum()) != len(votes_2026):
            raise ValueError(f"2025-26 winner or vote recipient missing at month {month}")
        predictions, metrics, weights = evaluate(candidates, first_test=2018)
        for frame in (candidates, predictions, metrics, weights):
            frame["cutoff_month"] = month
        all_candidates.append(candidates)
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        all_weights.append(weights)
        coverage.append({"cutoff_month": month, "candidates_2026": len(observed),
                         "vote_recipients_2026": int((observed.vote_share > 0).sum()),
                         "median_team_games_2026": float(observed.team_games.median())})
    candidates = pd.concat(all_candidates, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    weights = pd.concat(all_weights, ignore_index=True)
    holdout = metrics[metrics.test_year == 2026]
    now = datetime.now(timezone.utc)
    run = ROOT / "data/processed/mvp_runs" / f"v4_2026_holdout_{now.strftime('%Y%m%dT%H%M%SZ')}_{digest(OFFICIAL_VOTES)[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    candidates.to_parquet(run / "cutoff_candidates.parquet", index=False)
    predictions.to_parquet(run / "predictions.parquet", index=False)
    metrics.to_csv(run / "season_metrics.csv", index=False)
    weights.to_csv(run / "standardized_weights.csv", index=False)
    pd.DataFrame(coverage).to_csv(run / "coverage_2026.csv", index=False)
    holdout.to_csv(run / "new_holdout_2026.csv", index=False)
    picks = []
    for month, season in predictions[predictions.season_end_year == 2026].groupby("cutoff_month"):
        actual = season.loc[season.is_winner].iloc[0]
        for method in ("model", "baseline"):
            top = season.loc[season[f"{method}_probability"].idxmax()]
            picks.append({"cutoff_month": month, "method": method,
                          "predicted_winner": top.player_name,
                          "top_probability": float(top[f"{method}_probability"]),
                          "actual_winner": actual.player_name,
                          "actual_winner_rank": int(actual[f"{method}_rank"]),
                          "actual_winner_probability": float(actual[f"{method}_probability"])})
    pd.DataFrame(picks).to_csv(run / "top_picks_2026.csv", index=False)
    inputs = [*prior_player_paths, *prior_team_paths, new_player_path, new_team_path,
              source_manifest, LEGACY_VOTES, OFFICIAL_VOTES, ALIAS]
    manifest = {"created_at_utc": now.isoformat(), "mode": "new 2025-26 chronological MVP holdout",
        "heldout_rule": "2025-26 labels never used for fitting, hyperparameter selection, or probability scaling",
        "official_vote_source": json.loads((OFFICIAL_VOTE_DIR / "manifest.json").read_text(encoding="utf-8")),
        "log_audit": log_audit, "features": FEATURES,
        "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "coverage_2026": coverage,
        "holdout_metrics": json.loads(holdout.to_json(orient="records")),
        "limitations": ["Only one newly added independent award outcome", "Votes and logs are archived; live injury and eligibility exception data remain absent",
                        "Calendar-date cutoffs are a different fraction of season across unusual years"]}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(run), "log_audit": log_audit,
                      "picks": picks, "holdout_metrics": manifest["holdout_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
