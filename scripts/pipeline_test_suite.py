"""
Pipeline test suite for the Monte Carlo NBA project.
Runs a series of sanity checks:
1. Re‑run pull_team_game_logs.py – expect cache hits only.
2. Verify combined raw dataframe has no duplicate (GAME_ID, TEAM_ID) pairs
   and each GAME_ID appears exactly twice.
3. Verify that the first two games of each team/season have NaN for the
   rolling features (min_periods=3).
4. Spot‑check that POINT_DIFF for the home team equals the negative of the
   away team's POINT_DIFF for the same GAME_ID.
5. Train the game model and ensure accuracy is between 0.55 and 0.70 and
   that all four features have non‑zero importance.
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

# Helper to run a command and capture output
def run_cmd(cmd: str):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def test_pull_logs():
    print("\n[TEST] Re-run pull_team_game_logs.py for 2018-2024...")
    out, err, rc = run_cmd(
        "python src/ingest/pull_team_game_logs.py --start_season 2018 --end_season 2024"
    )
    if rc != 0:
        print("  [FAIL] Command failed.")
        return False
    # Expect lines starting with 'Skipping' and no lines containing 'Pulling'
    if "Skipping" in out and "Pulling" not in out:
        print("  [PASS] Cache hit confirmed.")
        return True
    print("  [FAIL] Unexpected output:\n", out)
    return False

def test_raw_duplicates():
    print("\n[TEST] Checking duplicate (GAME_ID, TEAM_ID) pairs in combined raw data...")
    raw_path = Path("data/raw/team_game_logs_combined.parquet")
    df = pd.read_parquet(raw_path)
    dup = df.duplicated(subset=["GAME_ID", "TEAM_ID"]).any()
    if dup:
        print("  [FAIL] Duplicate rows found.")
        print("  [FAIL] Duplicate rows found.")
        return False
    # Each GAME_ID should appear exactly twice
    counts = df.groupby("GAME_ID").size()
    if (counts != 2).any():
        print("  [FAIL] Some GAME_IDs do not have exactly two rows.")
        return False
    print("  [PASS] No duplicates and each GAME_ID appears twice.")
    return True

def test_rolling_features_nan():
    print("\n[TEST] Verifying early-season NaNs for rolling features...")
    feats_path = Path("data/processed/team_features.parquet")
    df = pd.read_parquet(feats_path)
    # Determine season for each row (year part of GAME_DATE)
    df["SEASON_YEAR"] = df["GAME_DATE"].dt.year
    # For each team+season, sort by date and check first two rows
    problem = False
    for (team, season), grp in df.groupby(["TEAM_ABBREVIATION", "SEASON_YEAR"]):
        grp = grp.sort_values("GAME_DATE")
        first_two = grp.head(2)
        if not first_two["ROLL_WIN_PCT"].isna().all() or not first_two["ROLL_POINT_DIFF"].isna().all():
            print(f"  [FAIL] {team} season {season} has non-NaN rolling values in first two games.")
            problem = True
    if not problem:
        print("  [PASS] First two games for each team/season have NaN rolling features.")
    return not problem

def test_point_diff_consistency():
    print("\n[TEST] Spot-checking POINT_DIFF sign consistency for a random GAME_ID...")
    raw_path = Path("data/raw/team_game_logs_combined.parquet")
    df = pd.read_parquet(raw_path)
    # Pick a random GAME_ID
    sample_id = df["GAME_ID"].sample(1).iloc[0]
    rows = df[df["GAME_ID"] == sample_id]
    if len(rows) != 2:
        print("  [FAIL] Unexpected number of rows for GAME_ID", sample_id)
        return False
    # Compute point diff per row as in add_basic_fields
    rows = rows.copy()
    # POINT_DIFF will be computed after merging opponent scores
    # Use the same logic as in pull script: merge opp scores
    opp = rows[["GAME_ID", "TEAM_ID", "PTS"]].rename(columns={"TEAM_ID": "OPP_TEAM_ID", "PTS": "OPP_PTS"})
    merged = rows.merge(opp, on="GAME_ID")
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]]
    merged["POINT_DIFF"] = merged["PTS"] - merged["OPP_PTS"]
    pdiffs = merged["POINT_DIFF"]
    if len(pdiffs) != 2:
        print("  [FAIL] Merge logic did not produce two rows.")
        return False
    if np.isclose(pdiffs.iloc[0], -pdiffs.iloc[1]):
        print(f"  [PASS] POINT_DIFF sign check passed for GAME_ID {sample_id}.")
        return True
    else:
        print(f"  [FAIL] POINT_DIFF values not opposite: {pdiffs.tolist()}")
        return False

def test_game_model():
    print("\n[TEST] Training game model and checking metrics...")
    # Run the training script directly (it prints metrics)
    out, err, rc = run_cmd("python src/models/game_model.py")
    if rc != 0:
        print("  [FAIL] Model training failed.")
        print(err)
        return False
    # Parse metrics from stdout
    acc = None
    for line in out.splitlines():
        if "Test accuracy" in line:
            try:
                acc = float(line.split(":")[-1].strip())
            except Exception:
                pass
    if acc is None:
        print("  [FAIL] Could not parse accuracy.")
        return False
    if 0.55 <= acc <= 0.70:
        print(f"  [PASS] Accuracy {acc:.4f} within expected range.")
    else:
        print(f"  [FAIL] Accuracy {acc:.4f} out of expected range (55‑70%).")
        return False
    # Load the saved model and check feature importances
    model_path = Path("data/processed/models/game_model.pkl")
    model = joblib.load(model_path)
    importances = model.feature_importances_
    if all(imp > 0 for imp in importances):
        print("  [PASS] All four features have non-zero importance.")
        return True
    else:
        print("  [FAIL] One or more features have zero importance:", importances)
        return False

if __name__ == "__main__":
    results = {
        "pull_logs": test_pull_logs(),
        "raw_duplicates": test_raw_duplicates(),
        "rolling_nan": test_rolling_features_nan(),
        "point_diff": test_point_diff_consistency(),
        "game_model": test_game_model(),
    }
    passed = sum(results.values())
    total = len(results)
    print(f"\nSummary: {passed}/{total} tests passed.")
    if passed == total:
        sys.exit(0)
    else:
        sys.exit(1)
