"""Build reproducible MVP vote labels from a pinned public voting table.

The source is a secondary transcription of Basketball-Reference award tables.
Only voting columns are retained; source player statistics are deliberately excluded.
"""
import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


COMMIT = "135b21a8d757b516822df57782d41b564862cd46"
URL = f"https://raw.githubusercontent.com/DavidYoo912/nba_mvp/{COMMIT}/data/master_table.csv"
VOTE_COLUMNS = ["year", "Player", "Tm", "Rank", "First", "Pts Won", "Pts Max", "Share"]


def normalize(source):
    missing = set(VOTE_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f"Missing vote columns: {sorted(missing)}")
    votes = source[VOTE_COLUMNS].copy().rename(columns={
        "year": "season_end_year", "Player": "player_name", "Tm": "team_abbr",
        "Rank": "vote_rank", "First": "first_place_votes", "Pts Won": "vote_points",
        "Pts Max": "max_vote_points", "Share": "vote_share_source"})
    votes["season_end_year"] = pd.to_numeric(votes.season_end_year, errors="raise").astype(int)
    votes = votes[votes.season_end_year.between(1981, 2025)].copy()
    for col in ["first_place_votes", "vote_points", "max_vote_points", "vote_share_source"]:
        votes[col] = pd.to_numeric(votes[col], errors="coerce")
    if votes[["season_end_year", "player_name", "team_abbr", "vote_points", "max_vote_points"]].isna().any().any():
        raise ValueError("Missing required MVP vote values")
    votes = votes.drop_duplicates()
    if votes.duplicated(["season_end_year", "player_name"]).any():
        raise ValueError("Duplicate player-season MVP votes")
    if (votes.vote_points < 0).any() or (votes.max_vote_points <= 0).any():
        raise ValueError("Invalid vote totals")
    if (votes.vote_points > votes.max_vote_points).any():
        raise ValueError("Vote points exceed maximum")
    if not votes.groupby("season_end_year").max_vote_points.nunique().eq(1).all():
        raise ValueError("Inconsistent maximum vote points within season")
    years = sorted(votes.season_end_year.unique())
    if years != list(range(1981, 2026)):
        raise ValueError("Historical MVP voting seasons are incomplete")
    winners = votes.loc[votes.groupby("season_end_year").vote_points.idxmax()]
    if len(winners) != len(years) or (winners.vote_points <= 0).any():
        raise ValueError("Missing season winner")
    votes["vote_share"] = votes.vote_points / votes.max_vote_points
    if ((votes.vote_share - votes.vote_share_source).abs() > 0.0011).any():
        raise ValueError("Source share disagrees with vote totals")
    votes["season"] = votes.season_end_year.sub(1).astype(str) + "-" + votes.season_end_year.mod(100).astype(str).str.zfill(2)
    votes["is_winner"] = votes.index.isin(winners.index)
    return votes.drop(columns="vote_share_source").sort_values(["season_end_year", "vote_points"], ascending=[True, False]).reset_index(drop=True)


def chronological_folds(votes, first_test=2018):
    """One held-out season at a time; no labels from the test year enter training."""
    years = sorted(votes.season_end_year.unique())
    return pd.DataFrame([
        {"test_season_end_year": year, "train_through_season_end_year": year - 1,
         "train_seasons": sum(y < year and y >= 2016 for y in years),
         "test_vote_recipients": int((votes.season_end_year == year).sum())}
        for year in years if first_test <= year <= 2025
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=Path, help="Pinned raw CSV already downloaded")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/mvp_voting"))
    args = parser.parse_args()
    if args.source_file:
        raw = args.source_file.read_bytes()
        retrieved_at = "unknown (local source file)"
    else:
        response = requests.get(URL, timeout=30)
        response.raise_for_status()
        raw = response.content
        retrieved_at = datetime.now(timezone.utc).isoformat()
    votes = normalize(pd.read_csv(io.BytesIO(raw)))
    folds = chronological_folds(votes)
    digest = hashlib.sha256(raw).hexdigest()
    out = args.output_dir / f"v1_{digest[:12]}"
    if out.exists():
        raise FileExistsError(f"Versioned voting import already exists: {out}")
    out.mkdir(parents=True)
    (out / "source_master_table.csv").write_bytes(raw)
    votes.to_csv(out / "mvp_votes.csv", index=False)
    folds.to_csv(out / "chronological_folds.csv", index=False)
    manifest = {"source_url": URL, "source_commit": COMMIT, "retrieved_at_utc": retrieved_at,
                "source_sha256": digest, "source_type": "secondary transcription of Basketball-Reference voting tables",
                "first_season_end_year": 1981, "last_season_end_year": 2025,
                "vote_recipients": len(votes), "evaluation_folds": len(folds),
                "scope": "MVP vote recipients only; nonrecipients require a separate player-season universe",
                "feature_policy": "No source statistics imported; construct features separately at each prediction cutoff"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), **manifest}, indent=2))


if __name__ == "__main__":
    main()
