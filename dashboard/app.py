"""Read-only season and awards viewer. Every view stays on one artifact version."""
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.awards import load_awards, digest

DEFAULT = ROOT / 'data/processed/player_runs/v1_20260910T215749Z_35f88a/scenario_20260910T220652Z'
UPCOMING = ROOT / 'data/raw/live_preseason/v1_20260922T113549Z_c79af6'
AWARDS_RELEASE = ROOT / 'results/awards/v3_20261006/awards.json'
MVP_EVIDENCE = ROOT / 'data/processed/awards/mvp_inputs_v1_20261009/evidence.json'


def show_mvp_inputs():
    st.subheader('MVP: historical inputs and evidence')
    st.caption('Separate historical research artifact · Experimental inputs · Not predictions for the selected forecast')
    if not MVP_EVIDENCE.exists():
        st.info('MVP feature evidence unavailable. Run scripts.build_mvp_inputs to generate it.')
        return
    report = json.loads(MVP_EVIDENCE.read_text(encoding='utf-8'))
    if digest(MVP_EVIDENCE.parent / 'inputs.parquet') != report['inputs_sha256']:
        raise ValueError('MVP input evidence fingerprint mismatch')
    st.caption(f"Evidence version: {MVP_EVIDENCE.parent.name} · Generated: {report['created_at_utc']}")
    for group, features in report['feature_groups'].items():
        st.write(f"**{group}:** " + '; '.join(report['definitions'][feature] for feature in features))
    st.write('**Past winners and their actual season metrics**')
    st.dataframe(pd.DataFrame(report['winners']).drop(columns='player_id'), hide_index=True, use_container_width=True)
    st.write('**Would last season’s metrics have ranked the next MVP highly?**')
    checks = pd.DataFrame(report['preseason_checks'])
    st.dataframe(checks[['season', 'feature', 'winner_rank', 'top5', 'candidates', 'missing_input_exclusions']], hide_index=True, use_container_width=True)
    with st.expander('Associations with recorded MVP voting'):
        st.write('Spearman rank correlations among recorded vote recipients, separately by season. Positive means higher metrics tended to accompany more votes in this selected group. This does not measure causal influence or establish a model weight.')
        st.dataframe(pd.DataFrame(report['recipient_associations']), hide_index=True, use_container_width=True)
    for limitation in report['limitations']:
        st.write(f'- {limitation}')
    with st.expander('MVP input provenance'):
        st.write(f"{report['player_seasons']} player-seasons; history version {report['history_version']}. Original game-log retrieval timestamps are unknown; file hashes are preserved.")
        for source in report['sources']:
            st.markdown(f"[{source['season']} official MVP voting]({source['url']}) — retrieved {source['retrieved_at_utc']}")
    st.download_button('Download MVP evidence', MVP_EVIDENCE.read_bytes(), file_name='mvp_evidence.json', mime='application/json')


def load_artifact(path_text):
    path = Path(path_text)
    if (path / 'report.json').exists():
        report = json.loads((path / 'report.json').read_text(encoding='utf-8'))
        standings = pd.read_csv(path / 'standings.csv')
        required = {'TEAM_ID', 'team', 'mean_wins', 'p05', 'p95'}
        if missing := required - set(standings):
            raise ValueError(f'Standings schema is missing: {sorted(missing)}')
        return report, standings
    if (path / 'source_metadata.json').exists():
        return json.loads((path / 'source_metadata.json').read_text(encoding='utf-8')), None
    raise FileNotFoundError('Select a scenario with report.json and standings.csv, or a preserved preseason input run.')


def show_awards(path, award_path):
    st.header('Awards')
    if not award_path.exists():
        st.info('Awards incomplete for this version: no saved awards artifact. Historical results remain available.')
        return
    artifact = load_awards(award_path, path)
    st.caption(f"Forecast version: {artifact['forecast_version']} · Cutoff: {artifact['forecast_cutoff']}")
    st.caption(f"Awards artifact generated: {artifact['generated_at_utc']}")
    st.warning('No award probabilities or candidate rankings are supported for this forecast version.')
    population = artifact['population']
    st.write(f"Modelled award candidates: {population['modelled']}. Excluded population: {population['excluded']}.")
    if population.get('roster_count') is not None:
        st.write(f"The source roster contains {population['roster_count']} players; this is not a verified eligible field.")
    for award in artifact['awards'].values():
        with st.expander(f"{award['name']} · Incomplete", expanded=True):
            st.write('**Missing requirements**')
            for reason in award['missing_requirements']:
                st.write(f'- {reason}')
            st.write('**Eligibility contract**')
            st.write(award['eligibility'])
            st.markdown(f"[Rule source]({award['rule_source']})")
    st.info('All-NBA deferred until the five primary awards work reliably.')
    evaluation = artifact.get('development_evaluation')
    if evaluation:
        st.subheader('Historical preseason baseline checks')
        st.caption('Experimental rankings · Development evidence only · Not a validated simulation-to-award pipeline')
        rows = pd.DataFrame(evaluation['rows'])
        st.dataframe(rows[['season', 'award', 'baseline', 'candidates', 'winner_rank', 'top5']], hide_index=True, use_container_width=True)
        for limitation in evaluation['limitations']:
            st.write(f'- {limitation}')
        st.write('Winner log loss, Brier score and calibration: not applicable to rankings; no probabilities were issued.')
        with st.expander('Historical data provenance'):
            provenance = artifact['history_provenance']
            st.write(f"{provenance['vote_recipient_rows']} vote-recipient records; {provenance['candidate_rows']} player-season-award rows.")
            for entry in provenance['sources']:
                st.markdown(f"[{entry['season']} {entry['award'].upper()}]({entry['url']}) — retrieved {entry['retrieved_at_utc']}")
    show_mvp_inputs()
    st.download_button('Download saved awards artifact', award_path.read_bytes(), file_name='awards.json', mime='application/json')


def show_season(report, standings):
    st.header('Season results')
    if standings is None:
        completeness = report.get('schedule_completeness', {})
        st.warning('2026–27 season projection incomplete — no simulated standings have been generated.')
        st.write(completeness.get('reason', 'Shared season outputs are missing.'))
        st.write(f"Named games: {completeness.get('named_games', 'Unknown')} / {completeness.get('required_regular_season_games', 'Unknown')} required.")
        st.caption('The requested September 21 reference date is not a verified roster snapshot date. The preserved September 22 retrieval timestamp governs this input version.')
    else:
        st.caption('Historical retrospective scenario — not a live forecast or as-published schedule reconstruction.')
        a, b = st.columns(2)
        a.metric('Trials', f"{report['n_sims']:,}")
        b.metric('Remaining games', f"{report['games_remaining']:,}")
        columns = [c for c in ['team', 'mean_wins', 'p05', 'p95', 'actual_wins'] if c in standings]
        st.dataframe(standings[columns], hide_index=True, use_container_width=True)
        chart = [c for c in ['mean_wins', 'actual_wins'] if c in standings]
        st.bar_chart(standings.set_index('team')[chart])
        if report.get('mean_absolute_error') is not None:
            st.write(f"Historical standings MAE: {report['mean_absolute_error']:.2f} wins")
    st.subheader('Assumptions and limitations')
    for item in report.get('assumptions', []):
        st.write(f'- {item}')


def show_players(path, standings):
    st.header('Teams and players')
    snapshot = path / 'player_snapshot.parquet'
    if snapshot.exists():
        players = pd.read_parquet(snapshot)
        if 'PLAYER_NAME' not in players:
            players['PLAYER_NAME'] = players.PLAYER_ID.astype(str)
        players['team'] = players.TEAM_ID.astype(str)
        if standings is not None:
            players['team'] = players.TEAM_ID.astype(str).map(standings.assign(TEAM_ID=standings.TEAM_ID.astype(str)).set_index('TEAM_ID').team)
        st.caption('Frozen player inputs from this scenario. Scoring rates are not complete player-season predictions.')
    else:
        files = sorted(path.glob('roster_snapshot_*.parquet'))
        if len(files) != 1:
            st.info('No player snapshot saved for this version.')
            return
        players = pd.read_parquet(files[0]).rename(columns={'PLAYER': 'PLAYER_NAME', 'SNAPSHOT_TEAM_ABBREVIATION': 'team'})
        st.caption('Source roster only. Projected statistics, starts, games played and award eligibility are unavailable.')
    team = st.selectbox('Team', ['All teams', *sorted(players.team.dropna().unique())])
    selected = players if team == 'All teams' else players[players.team == team]
    if standings is not None and team != 'All teams':
        st.dataframe(standings[standings.team == team], hide_index=True, use_container_width=True)
    columns = [c for c in ['PLAYER_ID', 'PLAYER_NAME', 'team', 'ML_MIN', 'ML_PTS36', 'PTS_36', 'POSITION', 'EXP'] if c in selected]
    st.dataframe(selected[columns], hide_index=True, use_container_width=True)
    ids = selected.PLAYER_ID.astype(str).tolist()
    names = dict(zip(ids, selected.PLAYER_NAME))
    player = st.selectbox('Player detail', ['None', *ids], format_func=lambda value: 'Select a player' if value == 'None' else f'{names[value]} ({value})')
    if player != 'None':
        st.subheader(names[player])
        st.dataframe(selected[selected.PLAYER_ID.astype(str) == player][columns], hide_index=True, use_container_width=True)
        st.info('Award eligibility and probabilities are incomplete for this player in this forecast version.')
    st.caption('All rows use the forecast version and cutoff shown above. No causal award explanations are available.')


def main():
    st.set_page_config(page_title='NBA season & awards', layout='wide')
    st.title('NBA season & awards')
    selection = st.sidebar.selectbox('Forecast', ['2026–27 preseason', 'Historical demo', 'Custom version'])
    default = UPCOMING if selection == '2026–27 preseason' else DEFAULT
    path = Path(st.sidebar.text_input('Artifact directory', str(default), key=selection))
    award_default = AWARDS_RELEASE if path.resolve() == UPCOMING.resolve() else path / 'awards.json'
    award_path = Path(st.sidebar.text_input('Awards artifact', str(award_default), key=f'awards-{selection}'))
    st.sidebar.caption('Read-only saved artifacts. This website never trains models or changes results.')
    try:
        report, standings = load_artifact(path)
        cutoff = report.get('cutoff', report.get('retrieved_at_utc', 'Unknown'))
        st.caption(f'Forecast version: {path.name} · Cutoff / source timestamp: {cutoff}')
        section = st.radio('Section', ['Season', 'Awards', 'Teams / players'], horizontal=True)
        if section == 'Season':
            show_season(report, standings)
        elif section == 'Awards':
            show_awards(path, award_path)
        else:
            show_players(path, standings)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        st.error(f'Artifact unavailable or inconsistent: {error}')


if __name__ == '__main__':
    main()
