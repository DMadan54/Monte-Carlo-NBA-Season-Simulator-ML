"""Validation for the read-only, historical scenario dashboard."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_artifact(path):
    directory = Path(path)
    report_path, standings_path = directory / "report.json", directory / "standings.csv"
    if not report_path.is_file() or not standings_path.is_file():
        raise FileNotFoundError("Expected report.json and standings.csv in selected scenario directory.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    standings = pd.read_csv(standings_path)
    required_report = {"cutoff", "seed", "n_sims", "games_remaining", "mean_absolute_error", "interval_coverage"}
    required_standings = {"TEAM_ID", "team", "mean_wins", "p05", "p95", "actual_wins"}
    if missing := required_report - report.keys():
        raise ValueError(f"Scenario report is missing: {sorted(missing)}")
    if missing := required_standings - set(standings):
        raise ValueError(f"Standings schema is missing: {sorted(missing)}")
    if len(standings) != 30 or standings.TEAM_ID.nunique() != 30 or standings.team.nunique() != 30:
        raise ValueError("Standings must contain 30 distinct teams.")
    values = standings[["mean_wins", "p05", "p95", "actual_wins"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Standings contain missing or nonfinite win values.")
    if not ((values.p05 <= values.mean_wins) & (values.mean_wins <= values.p95)).all():
        raise ValueError("Win intervals do not contain the simulated means.")
    if int(report["n_sims"]) <= 0 or int(report["games_remaining"]) < 0:
        raise ValueError("Scenario trial or game count is invalid.")
    total_actual = values.actual_wins.sum()
    if not np.isclose(values.mean_wins.sum(), total_actual, atol=0.05):
        raise ValueError("Projected league wins are not conserved.")
    computed_mae = (values.mean_wins - values.actual_wins).abs().mean()
    computed_coverage = ((values.p05 <= values.actual_wins) & (values.actual_wins <= values.p95)).mean()
    if not np.isclose(computed_mae, float(report["mean_absolute_error"]), atol=1e-6):
        raise ValueError("Reported standings MAE does not match the team rows.")
    if not np.isclose(computed_coverage, float(report["interval_coverage"]), atol=1e-9):
        raise ValueError("Reported interval coverage does not match the team rows.")
    return report, standings
