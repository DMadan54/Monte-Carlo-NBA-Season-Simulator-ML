"""Read-only portfolio viewer for historical NBA standings and MVP experiments."""
import json
from pathlib import Path

import streamlit as st

from dashboard.artifact import load_artifact
from dashboard.awards import load_awards

ROOT = Path(__file__).resolve().parents[1]
STANDINGS_EXAMPLE = ROOT / "examples/2025-01-01-scenario"
AWARDS_EXAMPLE = ROOT / "examples/mvp-2026-holdout"
MONTHS = {"January 1": 1, "February 1": 2, "March 1": 3}


def show_standings(path):
    st.header("Regular-season standings")
    st.caption("2025-01-01 historical fixed-cutoff retrospective scenario. The final played schedule was supplied after the season.")
    try:
        report, standings = load_artifact(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        st.error(str(error))
        return
    a, b, c, d = st.columns(4)
    a.metric("Cutoff", report["cutoff"])
    b.metric("Trials", f"{report['n_sims']:,}")
    c.metric("Remaining games", f"{report['games_remaining']:,}")
    d.metric("Standings MAE", f"{report['mean_absolute_error']:.2f} wins")
    covered = ((standings.actual_wins >= standings.p05) & (standings.actual_wins <= standings.p95)).sum()
    st.caption(f"Seed {report['seed']} · Nominal 90% interval coverage {report['interval_coverage']:.1%} ({covered} of {len(standings)} teams)")
    table = standings[["team", "mean_wins", "p05", "p95", "actual_wins"]].rename(columns={
        "team": "Team", "mean_wins": "Mean wins", "p05": "P05", "p95": "P95", "actual_wins": "Actual wins"})
    st.subheader("Projected standings")
    st.dataframe(table, hide_index=True, width="stretch")
    st.subheader("Projected versus actual wins")
    st.bar_chart(standings.set_index("team")[["mean_wins", "actual_wins"]])
    with st.expander("Scenario assumptions and limits"):
        for assumption in report.get("assumptions", []):
            st.write(f"- {assumption}")
        st.write("This is a historical scenario, not a live, preseason, trade, or injury forecast.")


def show_awards(path):
    st.header("MVP awards experiment")
    st.caption("2025–26 historical holdout. Predictions use performances observed strictly before each date; the eventual vote is the evaluation label.")
    try:
        manifest, predictions, metrics = load_awards(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        st.error(str(error))
        return
    st.warning("Experimental result: the learned model missed the 2025–26 winner at all three cutoffs. The simple baseline picked the winner each time. These dates are one award outcome.")
    cutoff_label = st.selectbox("Prediction date", list(MONTHS), index=2)
    month = MONTHS[cutoff_label]
    current = predictions[predictions.cutoff_month.eq(month)].copy()
    winner = current.loc[current.is_winner].iloc[0]
    model_pick = current.loc[current.model_probability.idxmax()]
    baseline_pick = current.loc[current.baseline_probability.idxmax()]
    a, b, c, d = st.columns(4)
    a.metric("Candidates", f"{len(current):,}")
    b.metric("Actual winner", winner.player_name)
    c.metric("Learned top pick", model_pick.player_name)
    d.metric("Baseline top pick", baseline_pick.player_name)
    model_metric = metrics[(metrics.test_year.eq(2026)) & (metrics.cutoff_month.eq(month)) &
                           (metrics.method.eq("model"))].iloc[0]
    baseline_metric = metrics[(metrics.test_year.eq(2026)) & (metrics.cutoff_month.eq(month)) &
                              (metrics.method.eq("baseline"))].iloc[0]
    st.caption(f"Actual winner rank: learned #{int(model_metric.winner_rank)}, baseline #{int(baseline_metric.winner_rank)}. "
               f"Winner log loss: learned {model_metric.winner_log_loss:.3f}, baseline {baseline_metric.winner_log_loss:.3f} (lower is better).")
    st.subheader("Candidate ranking")
    order = st.radio("Sort candidates by", ["Learned model", "Simple baseline"], horizontal=True)
    probability = "model_probability" if order == "Learned model" else "baseline_probability"
    ranked = current.sort_values(probability, ascending=False)
    search = st.text_input("Find a player", placeholder="Enter a name")
    if search:
        ranked = ranked[ranked.player_name.str.contains(search, case=False, regex=False)]
    else:
        ranked = ranked.head(15)
    display = ranked[["player_name", "model_rank", "model_probability", "baseline_rank",
                      "baseline_probability", "pts_per_game", "team_win_pct", "is_winner"]].rename(columns={
        "player_name": "Player", "model_rank": "Learned rank", "model_probability": "Learned probability",
        "baseline_rank": "Baseline rank", "baseline_probability": "Baseline probability",
        "pts_per_game": "Points per game at cutoff", "team_win_pct": "Team win fraction at cutoff",
        "is_winner": "Actual winner"})
    st.dataframe(display, hide_index=True, width="stretch")
    if not search:
        st.bar_chart(ranked.set_index("player_name")[["model_probability", "baseline_probability"]])
    st.caption("The ranking shows probabilities produced at the chosen cutoff. 'Actual winner' is a later outcome shown only for evaluation; it was not a model feature.")
    st.subheader("Chronological test seasons")
    historical = metrics[metrics.cutoff_month.eq(month)].copy()
    historical = historical[["test_year", "method", "winner_correct", "winner_rank", "winner_log_loss"]]
    historical.columns = ["Season ending", "Method", "Winner picked", "Winner rank", "Winner log loss"]
    st.dataframe(historical.sort_values(["Season ending", "Method"], ascending=[False, True]),
                 hide_index=True, width="stretch")
    st.caption("Each season was tested using only earlier seasons for training. January 2020–21 followed an unusually short opening schedule; calendar dates are not equal season fractions.")
    with st.expander("Sources, methods, and limits"):
        st.markdown(f"[Official NBA 2025–26 MVP voting results]({manifest['official_vote_source']})")
        st.write(manifest["training_rule"])
        st.write(manifest["mode"])
        for limitation in manifest["limitations"]:
            st.write(f"- {limitation}")
        st.write("The learned model remains experimental and is not used for a live award forecast.")


def show_about():
    st.header("Project evidence")
    st.write("This portfolio combines independently trained player minutes/scoring forecasts, a game probability model, and seeded Monte Carlo regular-season scenarios. Historical failures and simple baselines are retained alongside the results.")
    st.write("The strongest held-out player result was a 22% reduction in minutes MAE: 7.489 to 5.827 minutes across 40,970 player-game candidates. On 1,230 held-out games, player-aware log loss was 0.601090 versus 0.602622 for the team-only baseline; that small difference was inconclusive.")
    st.write("Across 28 historical standings cutoffs, player-aware MAE was 3.897 wins versus 3.910 for the team-only control. The model has not established a meaningful standings advantage.")
    st.markdown("[Methods and limitations](https://github.com/DMadan54/Monte-Carlo-NBA-Season-Simulator-ML/blob/main/docs/MODEL_CARD.md) · [CV evidence](https://github.com/DMadan54/Monte-Carlo-NBA-Season-Simulator-ML/blob/main/docs/PORTFOLIO_RELEASE.md)")
    st.info("This dashboard reads saved artifacts only. Live roster, injury, coach, and full preseason schedule inputs are not validated here.")


def main():
    st.set_page_config(page_title="NBA forecasting portfolio", layout="wide")
    st.title("NBA forecasting portfolio")
    st.caption("Historical, read-only model results and Monte Carlo scenarios")
    standings_path = st.sidebar.text_input("Standings artifact directory", str(STANDINGS_EXAMPLE))
    awards_path = st.sidebar.text_input("MVP artifact directory", str(AWARDS_EXAMPLE))
    st.sidebar.caption("Both bundled examples work without the large local training caches. Selecting a directory only reads its saved files.")
    standings_tab, awards_tab, about_tab = st.tabs(["Standings", "MVP awards", "Project evidence"])
    with standings_tab:
        show_standings(standings_path)
    with awards_tab:
        show_awards(awards_path)
    with about_tab:
        show_about()


if __name__ == "__main__":
    main()
