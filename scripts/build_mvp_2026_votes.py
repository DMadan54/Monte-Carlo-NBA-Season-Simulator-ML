"""Validate a manual transcription of the NBA's official 2025-26 MVP vote table."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://ak-static.cms.nba.com/wp-content/uploads/sites/46/2026/05/Kia-NBA-Most-Valuable-Player-Voting-Results-scaled.jpg"
RELEASE_URL = "https://pr.nba.com/voting-results-2025-26-nba-regular-season-awards/"
SOURCE_SHA256 = "cb4ede55835ff5b95e64526de5fc03499b4abce84ed623b1a7202d31a33f334e"
SOURCE_IMAGE = ROOT / "data/raw/mvp_voting/official_2026_mvp_voting.jpg"
OUTPUT = ROOT / "data/raw/mvp_voting/v2_2026_official_cb4ede55835f"

# Transcribed from the official NBA image. Columns are first through fifth place votes.
ROWS = [
    ("Shai Gilgeous-Alexander", "OKC", 83, 13, 3, 1, 0, 939),
    ("Nikola Jokić", "DEN", 10, 48, 37, 4, 1, 634),
    ("Victor Wembanyama", "SAS", 5, 36, 47, 10, 2, 569),
    ("Luka Dončić", "LAL", 0, 1, 8, 60, 23, 250),
    ("Cade Cunningham", "DET", 2, 1, 3, 11, 42, 117),
    ("Jaylen Brown", "BOS", 0, 1, 2, 14, 30, 89),
    ("Kawhi Leonard", "LAC", 0, 0, 0, 0, 1, 1),
    ("Donovan Mitchell", "CLE", 0, 0, 0, 0, 1, 1),
]
WEIGHTS = (10, 7, 5, 3, 1)


def validated_votes():
    columns = ["player_name", "team_abbr", "first_votes", "second_votes",
               "third_votes", "fourth_votes", "fifth_votes", "vote_points"]
    votes = pd.DataFrame(ROWS, columns=columns)
    ballot = votes[["first_votes", "second_votes", "third_votes", "fourth_votes", "fifth_votes"]]
    if not ballot.sum().eq(100).all():
        raise ValueError("Official voting transcription lacks 100 ballots per rank")
    calculated = ballot.to_numpy() @ WEIGHTS
    if not (calculated == votes.vote_points.to_numpy()).all() or votes.vote_points.sum() != 2600:
        raise ValueError("Official MVP vote points do not reconcile")
    if votes.duplicated("player_name").any():
        raise ValueError("Duplicate vote recipient")
    votes.insert(0, "season", "2025-26")
    votes.insert(1, "season_end_year", 2026)
    votes["max_vote_points"] = 1000
    votes["vote_share"] = votes.vote_points / votes.max_vote_points
    votes["is_winner"] = votes.vote_points.eq(votes.vote_points.max())
    if votes.is_winner.sum() != 1 or votes.loc[votes.is_winner, "player_name"].iloc[0] != "Shai Gilgeous-Alexander":
        raise ValueError("Winner disagrees with NBA release")
    return votes


def main():
    if SOURCE_IMAGE.exists():
        source = SOURCE_IMAGE.read_bytes()
    else:
        response = requests.get(SOURCE_URL, timeout=30)
        response.raise_for_status()
        source = response.content
        SOURCE_IMAGE.write_bytes(source)
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official source image hash changed; re-audit transcription")
    votes = validated_votes()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    votes.to_csv(OUTPUT / "mvp_votes_2025-26.csv", index=False)
    manifest = {"season": "2025-26", "source_release_url": RELEASE_URL,
                "source_image_url": SOURCE_URL, "source_image_sha256": SOURCE_SHA256,
                "transcription_method": "Manually transcribed official NBA vote table; reconciled each rank to 100 ballots and 2600 total points",
                "transcribed_at_utc": datetime.now(timezone.utc).isoformat(),
                "vote_recipients": len(votes), "winner": votes.loc[votes.is_winner, "player_name"].iloc[0],
                "source_image_storage": "Local ignored cache; not committed"}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT), **manifest}, indent=2))


if __name__ == "__main__":
    main()
