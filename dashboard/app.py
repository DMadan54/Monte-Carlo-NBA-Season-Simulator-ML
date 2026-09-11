"""Read-only viewer for versioned fixed-cutoff simulation artifacts."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'data' / 'processed' / 'player_runs' / 'v1_20260910T215749Z_35f88a' / 'scenario_20260910T220652Z'

@st.cache_data
def load_artifact(path_text):
    path = Path(path_text)
    report_path, standings_path = path / 'report.json', path / 'standings.csv'
    if not report_path.exists() or not standings_path.exists():
        raise FileNotFoundError('Expected report.json and standings.csv in selected scenario directory.')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    standings = pd.read_csv(standings_path)
    required = {'team', 'mean_wins', 'p05', 'p95', 'actual_wins'}
    if missing := required - set(standings):
        raise ValueError(f'Standings schema is missing: {sorted(missing)}')
    return report, standings

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
    st.subheader('Projected standings')
    table = standings[['team', 'mean_wins', 'p05', 'p95', 'actual_wins']].rename(columns={'team': 'Team', 'mean_wins': 'Mean wins', 'p05': 'P05', 'p95': 'P95', 'actual_wins': 'Actual wins'})
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.subheader('Projected versus actual wins')
    st.bar_chart(standings.set_index('team')[['mean_wins', 'actual_wins']])
    st.subheader('Assumptions and limitations')
    for item in report.get('assumptions', []): st.write(f'- {item}')
    st.warning('Portfolio evidence only: this is not a current NBA betting, trade, or injury forecast.')

if __name__ == '__main__':
    main()
