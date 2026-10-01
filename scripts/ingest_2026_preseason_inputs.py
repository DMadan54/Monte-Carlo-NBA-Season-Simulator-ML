"""Create an immutable current-data input contract for the 2026-27 preseason scenario.

This is deliberately an ingestion step only.  It does not infer injuries,
transactions not present in the roster endpoint, or current-season results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import uuid
import sys

import pandas as pd
import requests
from nba_api.stats.endpoints import commonteamroster, leaguegamelog
from nba_api.stats.static import teams

# Permit both ``python -m scripts...`` and direct script execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingest.nba_client import REQUEST_TIMEOUT_SEC, fetch_with_retry


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_URL = 'https://cdn.nba.com/manage/2026/08/2026-27-NBA-Regular-Season-Schedule-By-Date.pdf'
SEASON = '2026-27'
BASELINE_SEASON = '2025-26'
AS_OF = datetime.now(timezone.utc).date().isoformat()
TEAM_NAME_TO_ABBREV = {
    'Atlanta': 'ATL', 'Boston': 'BOS', 'Brooklyn': 'BKN', 'Charlotte': 'CHA',
    'Chicago': 'CHI', 'Cleveland': 'CLE', 'Dallas': 'DAL', 'Denver': 'DEN',
    'Detroit': 'DET', 'Golden State': 'GSW', 'Houston': 'HOU', 'Indiana': 'IND',
    'LA Clippers': 'LAC', 'LA Lakers': 'LAL', 'Memphis': 'MEM', 'Miami': 'MIA',
    'Milwaukee': 'MIL', 'Minnesota': 'MIN', 'New Orleans': 'NOP', 'New York': 'NYK',
    'Oklahoma City': 'OKC', 'Orlando': 'ORL', 'Philadelphia': 'PHI', 'Phoenix': 'PHX',
    'Portland': 'POR', 'Sacramento': 'SAC', 'San Antonio': 'SAS', 'Toronto': 'TOR',
    'Utah': 'UTA', 'Washington': 'WAS',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pull_logs(kind: str) -> pd.DataFrame:
    if kind not in {'T', 'P'}:
        raise ValueError('kind must be T or P')

    def request():
        endpoint = leaguegamelog.LeagueGameLog(
            season=BASELINE_SEASON, season_type_all_star='Regular Season',
            player_or_team_abbreviation=kind, timeout=REQUEST_TIMEOUT_SEC,
        )
        return endpoint.get_data_frames()[0]

    result = fetch_with_retry(request, label=f'{BASELINE_SEASON} {"team" if kind == "T" else "player"} logs')
    result['SEASON'] = BASELINE_SEASON
    result['GAME_DATE'] = pd.to_datetime(result['GAME_DATE'], errors='raise')
    if result.empty or result['GAME_DATE'].max() < pd.Timestamp('2026-04-01'):
        raise ValueError(f'{kind} log pull does not contain a completed {BASELINE_SEASON} regular season')
    return result


def pull_rosters() -> pd.DataFrame:
    rows = []
    for team in teams.get_teams():
        team_id = int(team['id'])

        def request(team_id=team_id):
            return commonteamroster.CommonTeamRoster(team_id=team_id, season=SEASON,
                                                      timeout=REQUEST_TIMEOUT_SEC).get_data_frames()[0]
        frame = fetch_with_retry(request, label=f'{SEASON} roster {team["abbreviation"]}')
        if frame.empty:
            raise ValueError(f'Empty roster returned for {team["abbreviation"]}')
        frame['SNAPSHOT_TEAM_ID'] = team_id
        frame['SNAPSHOT_TEAM_ABBREVIATION'] = team['abbreviation']
        rows.append(frame)
        time.sleep(0.25)
    result = pd.concat(rows, ignore_index=True)
    if result.groupby('SNAPSHOT_TEAM_ID').size().min() < 5:
        raise ValueError('Roster endpoint returned fewer than five players for a team')
    return result


def parse_schedule(pdf_path: Path) -> pd.DataFrame:
    try:
        from pypdf import PdfReader
    except ImportError as err:
        raise RuntimeError('Schedule parsing requires pypdf; install it in the project virtual environment.') from err
    text = '\n'.join(page.extract_text() or '' for page in PdfReader(str(pdf_path)).pages)
    pattern = re.compile(
        r'(?m)^(\d+)\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.\s+(\d{1,2}/\d{1,2}/\d{2})\s+'
        r'(.+?)\s+(at|vs)\s+(.+?)\s+\d{1,2}:\d{2}\s+(?:AM|PM)\b'
    )
    records = []
    ids = {abbr: int(team['id']) for team in teams.get_teams() for abbr in [team['abbreviation']]}
    for source_number, date, visitor, venue, home in pattern.findall(text):
        visitor, home = visitor.strip(), home.strip()
        if visitor not in TEAM_NAME_TO_ABBREV or home not in TEAM_NAME_TO_ABBREV:
            raise ValueError(f'Unmapped schedule team: {visitor!r} at {home!r}')
        records.append(dict(
            GAME_ID=f'NBA2026-{int(source_number):04d}', SOURCE_GAME_NUMBER=int(source_number),
            GAME_DATE=pd.to_datetime(date, format='%m/%d/%y'),
            AWAY_TEAM=ids[TEAM_NAME_TO_ABBREV[visitor]], HOME_TEAM=ids[TEAM_NAME_TO_ABBREV[home]],
            NEUTRAL_GAME=int(venue == 'vs'),
        ))
    schedule = pd.DataFrame(records).drop_duplicates('SOURCE_GAME_NUMBER').sort_values(
        ['GAME_DATE', 'SOURCE_GAME_NUMBER']).reset_index(drop=True)
    # The August 13 source assigns 1,200 games.  The remaining 30 are NBA Cup
    # knockout/TBD slots and must not be invented or imputed for a full-season run.
    if len(schedule) != 1200 or schedule.SOURCE_GAME_NUMBER.nunique() != 1200:
        raise ValueError(f'Expected 1,200 named games in the published source, parsed {len(schedule)}')
    appearances = pd.concat([schedule.HOME_TEAM, schedule.AWAY_TEAM]).value_counts()
    if len(appearances) != 30 or not appearances.eq(80).all():
        raise ValueError('Published named games do not give exactly 80 games to each team')
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data' / 'raw' / 'live_preseason')
    parser.add_argument('--schedule-pdf', type=Path, help='Optional already-downloaded official schedule PDF')
    args = parser.parse_args()
    run = args.output_root / f'v1_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}'
    run.mkdir(parents=True, exist_ok=False)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    try:
        team_logs = pull_logs('T')
        player_logs = pull_logs('P')
        rosters = pull_rosters()
        pdf_path = run / '2026-27_schedule_by_date.pdf'
        if args.schedule_pdf:
            shutil.copyfile(args.schedule_pdf, pdf_path)
        else:
            response = requests.get(SCHEDULE_URL, timeout=REQUEST_TIMEOUT_SEC)
            response.raise_for_status()
            pdf_path.write_bytes(response.content)
        schedule = parse_schedule(pdf_path)
        team_logs.to_parquet(run / 'team_game_logs_2025-26.parquet', index=False)
        player_logs.to_parquet(run / 'player_game_logs_2025-26.parquet', index=False)
        rosters.to_parquet(run / f'roster_snapshot_retrieved_{AS_OF}.parquet', index=False)
        schedule.to_parquet(run / 'schedule_2026-27.parquet', index=False)
        metadata = {
            'scenario': '2026-27 preseason; not a mid-season forecast',
            'roster_snapshot_effective_date': AS_OF,
            'retrieved_at_utc': retrieved_at, 'baseline_season': BASELINE_SEASON,
            'schedule_url': SCHEDULE_URL, 'roster_source': 'stats.nba.com CommonTeamRoster season=2026-27',
            'team_log_source': 'stats.nba.com LeagueGameLog team regular season 2025-26',
            'player_log_source': 'stats.nba.com LeagueGameLog player regular season 2025-26',
            'assumptions': [
                'Current roster endpoint is a retrieval-time snapshot, not a historical transaction ledger.',
                'No injuries or availability limitations are supplied or assumed.',
                'Returning players will use final-2025-26 minutes share; players new to a team receive a league-average bench-minutes weight.',
            ],
            'files': {p.name: sha256(p) for p in run.iterdir() if p.is_file()},
            'row_counts': {'team_logs': len(team_logs), 'player_logs': len(player_logs),
                           'roster_rows': len(rosters), 'schedule_games': len(schedule)},
            'schedule_completeness': {
                'named_games': len(schedule), 'required_regular_season_games': 1230,
                'missing_games': 1230 - len(schedule),
                'simulation_eligible': False,
                'reason': 'Official August 13 schedule leaves 30 NBA Cup/TBD regular-season slots unassigned.'
            },
        }
        (run / 'source_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'INGESTION_RUN={run}', flush=True)
        print(json.dumps(metadata['row_counts']), flush=True)
        raise RuntimeError('Full 2026-27 simulation blocked: official schedule has 30 unassigned regular-season games.')
    except Exception as err:
        (run / 'FAILED.txt').write_text(f'{type(err).__name__}: {err}\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
