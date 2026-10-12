"""Read-only viewer for versioned fixed-cutoff simulation artifacts."""
from pathlib import Path
import streamlit as st
from dashboard.artifact import load_artifact

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'examples' / '2025-01-01-scenario'

def main():
    st.set_page_config(page_title='NBA fixed-cutoff scenario', layout='wide')
    st.title('NBA fixed-cutoff simulation')
    st.caption('Historical retrospective scenario — not a live forecast or as-published schedule reconstruction.')
    path = st.sidebar.text_input('Scenario artifact directory', str(DEFAULT))
    try:
        report, standings = load_artifact(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        st.error(str(error)); st.stop()
    st.sidebar.success('Read-only artifact loaded; the dashboard never trains or changes a model.')
    a, b, c, d = st.columns(4)
    a.metric('Cutoff', report['cutoff']); b.metric('Trials', f"{report['n_sims']:,}")
    c.metric('Remaining games', f"{report['games_remaining']:,}"); d.metric('Standings MAE', f"{report['mean_absolute_error']:.2f}")
    st.caption(f"Seed: {report['seed']} · 90% interval coverage: {report['interval_coverage']:.1%} · Historical fixed-cutoff retrospective scenario")
    st.subheader('Projected standings')
    table = standings[['team', 'mean_wins', 'p05', 'p95', 'actual_wins']].rename(columns={'team': 'Team', 'mean_wins': 'Mean wins', 'p05': 'P05', 'p95': 'P95', 'actual_wins': 'Actual wins'})
    st.dataframe(table, hide_index=True, width='stretch')
    st.subheader('Projected versus actual wins')
    st.bar_chart(standings.set_index('team')[['mean_wins', 'actual_wins']])
    st.subheader('Assumptions and limitations')
    for item in report.get('assumptions', []): st.write(f'- {item}')
    st.warning('Portfolio evidence only: this is not a current NBA betting, trade, or injury forecast.')

if __name__ == '__main__':
    main()
