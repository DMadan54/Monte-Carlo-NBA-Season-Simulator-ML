"""Render the matched Phase 3 control/filter report as plain text."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"
REPORT = PROJECT_ROOT / "claude.txt"


def ranking_group(group: pd.DataFrame) -> pd.Series:
    """Label five highest and five lowest actual-win teams in each season."""
    ordered = group.sort_values(["Actual Wins", "Team"], ascending=[False, True]).copy()
    labels = pd.Series("Middle 20", index=ordered.index)
    labels.iloc[:5] = "Top 5"
    labels.iloc[-5:] = "Bottom 5"
    return labels.reindex(group.index)


def display_table(frame: pd.DataFrame, float_columns: list[str]) -> str:
    """Format report tables in the existing fixed-width text-report style."""
    formatted = frame.copy()
    for column in float_columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")
    return formatted.to_string(index=False)


def main() -> None:
    control = pd.read_csv(PROCESSED / "phase3_all_training_games_standings_comparison.csv")
    filtered = pd.read_csv(PROCESSED / "phase3_exclude_flagged_training_games_standings_comparison.csv")
    raw = pd.read_parquet(RAW / "team_game_logs_combined.parquet")

    keys = ["Season", "Team", "Actual Wins"]
    comparison = control.merge(filtered, on=keys, suffixes=(" Control", " Filtered"), validate="one_to_one")
    comparison["Sim Delta"] = comparison["Sim Mean Wins Filtered"] - comparison["Sim Mean Wins Control"]
    comparison["Error Delta"] = comparison["Abs Error Filtered"] - comparison["Abs Error Control"]
    comparison["Ranking Group"] = comparison.groupby("Season", group_keys=False).apply(
        ranking_group, include_groups=False
    )

    games = raw.groupby("SEASON")["GAME_ID"].nunique().rename("Games")
    summary_rows = []
    for season, group in comparison.groupby("Season", sort=True):
        top = group[group["Ranking Group"] == "Top 5"]
        bottom = group[group["Ranking Group"] == "Bottom 5"]
        summary_rows.append({
            "Season": season,
            "Games": games[season],
            "Control MAE": group["Abs Error Control"].mean(),
            "Filtered MAE": group["Abs Error Filtered"].mean(),
            "MAE Delta": group["Error Delta"].mean(),
            "Top-5 MAE Before": top["Abs Error Control"].mean(),
            "Top-5 MAE After": top["Abs Error Filtered"].mean(),
            "Bottom-5 MAE Before": bottom["Abs Error Control"].mean(),
            "Bottom-5 MAE After": bottom["Abs Error Filtered"].mean(),
        })
    summary = pd.DataFrame(summary_rows)

    top5_before = comparison[comparison["Ranking Group"] == "Top 5"]["Abs Error Control"].mean()
    top5_after = comparison[comparison["Ranking Group"] == "Top 5"]["Abs Error Filtered"].mean()
    bottom5_before = comparison[comparison["Ranking Group"] == "Bottom 5"]["Abs Error Control"].mean()
    bottom5_after = comparison[comparison["Ranking Group"] == "Bottom 5"]["Abs Error Filtered"].mean()

    lines = [
        "================================================================================",
        "NBA MONTE CARLO SEASON SIMULATOR: PHASE 3 TANKING-FLAG TRAINING COMPARISON",
        "================================================================================",
        "",
        "--- 1. EXECUTIVE SUMMARY ---",
        "* Comparison design: same rebuilt features, six historical schedules, 1,000 fixed-seed simulations per season.",
        "* Control model: all valid training rows included.",
        "* Filtered model: 388 flagged team-game rows removed from the training split only.",
        "* Actual standings comparison: all regular-season games remain included (7,059 total actual wins).",
        f"* Overall MAE: {comparison['Abs Error Control'].mean():.2f} wins/team (control) vs {comparison['Abs Error Filtered'].mean():.2f} (filtered).",
        f"* Bottom-5 MAE: {bottom5_before:.2f} -> {bottom5_after:.2f} ({bottom5_after - bottom5_before:+.2f}).",
        f"* Top-5 MAE: {top5_before:.2f} -> {top5_after:.2f} ({top5_after - top5_before:+.2f}).",
        "",
        "--- 2. PHASE 3 MAE SUMMARY TABLE (1,000 SIMULATIONS EACH) ---",
        display_table(
            summary,
            [
                "Control MAE", "Filtered MAE", "MAE Delta", "Top-5 MAE Before",
                "Top-5 MAE After", "Bottom-5 MAE Before", "Bottom-5 MAE After",
            ],
        ),
        "",
        "--- 3. COMPLETE TEAM-BY-TEAM CONTROL VS FILTERED STANDINGS ---",
    ]

    display_columns = [
        "Team", "Actual Wins", "Sim Mean Wins Control", "Sim Mean Wins Filtered",
        "Sim Delta", "Abs Error Control", "Abs Error Filtered", "Error Delta", "Ranking Group",
    ]
    renamed = {
        "Sim Mean Wins Control": "Control Sim",
        "Sim Mean Wins Filtered": "Filtered Sim",
        "Abs Error Control": "Control Error",
        "Abs Error Filtered": "Filtered Error",
    }
    for season, group in comparison.groupby("Season", sort=True):
        group = group.sort_values(["Actual Wins", "Team"], ascending=[False, True])
        season_mae_control = group["Abs Error Control"].mean()
        season_mae_filtered = group["Abs Error Filtered"].mean()
        lines.extend([
            "",
            f"==================== SEASON {season} (Control MAE: {season_mae_control:.2f}; Filtered MAE: {season_mae_filtered:.2f}) ====================",
            display_table(
                group[display_columns].rename(columns=renamed),
                ["Control Sim", "Filtered Sim", "Sim Delta", "Control Error", "Filtered Error", "Error Delta"],
            ),
        ])

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
