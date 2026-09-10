"""Validated cached NBA inputs. Participation is post-game evidence, not a roster feed."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

COUNTS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGA', 'FGM', 'FG3A', 'FG3M']


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize(frame):
    frame = frame.copy()
    for name in ['GAME_ID', 'TEAM_ID']:
        frame[name] = frame[name].astype(str)
    frame['GAME_DATE'] = pd.to_datetime(frame['GAME_DATE']).dt.normalize()
    year = frame.SEASON.astype(str).str.extract(r'^(\d{4})-\d{2}$', expand=False)
    if year.isna().any():
        raise ValueError('Invalid SEASON; calendar inference is forbidden')
    frame['SEASON_YEAR'] = year.astype(int) + 1
    return frame


def load_cached(raw_dir):
    raw_dir = Path(raw_dir)
    team_path = raw_dir / 'team_game_logs_combined.parquet'
    paths = sorted(raw_dir.glob('player_game_logs_*.parquet'))
    if not paths:
        raise FileNotFoundError('No cached player logs; fetch and validate before training')
    team = normalize(pd.read_parquet(team_path))
    players = normalize(pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True))
    players['PLAYER_ID'] = players.PLAYER_ID.astype(str)
    required = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'GAME_DATE', 'MIN', *COUNTS]
    if players[required].isna().any().any():
        raise ValueError('Missing required player values')
    players['MIN'] = pd.to_numeric(players.MIN, errors='raise')
    if players.duplicated(['GAME_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate player-game rows')
    if team.duplicated(['GAME_ID', 'TEAM_ID']).any() or not team.groupby('GAME_ID').size().eq(2).all():
        raise ValueError('Team games must have exactly two unique teams')
    if not np.isfinite(players[['MIN', *COUNTS]].to_numpy()).all() or (players[['MIN', *COUNTS]] < 0).any().any():
        raise ValueError('Invalid negative/nonfinite player counts')
    if (players.FGM > players.FGA).any() or (players.FG3M > players.FG3A).any():
        raise ValueError('Made shots exceed attempts')
    totals = players.groupby(['GAME_ID', 'TEAM_ID'])[['PTS', 'MIN']].sum()
    joined = team.set_index(['GAME_ID', 'TEAM_ID'])[['PTS', 'MIN', 'GAME_DATE', 'SEASON']].join(totals, rsuffix='_PLAYER')
    if joined.MIN_PLAYER.isna().any() or not joined.PTS.eq(joined.PTS_PLAYER).all():
        raise ValueError('Player coverage/score totals do not reconcile with team logs')
    if len(totals) != len(joined):
        raise ValueError('Player logs contain games outside the team dataset')
    # Cached LeagueGameLog minutes are rounded to whole minutes per player.
    if (joined.MIN - joined.MIN_PLAYER).abs().max() > 8:
        raise ValueError('Rounded player minutes fail team total reconciliation')
    if not ((team.MIN >= 240) & ((team.MIN - 240) % 25 == 0)).all():
        raise ValueError('Team minutes must be regulation plus complete overtime periods')
    linked = players.merge(team[['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'SEASON']], on=['GAME_ID', 'TEAM_ID'], suffixes=('', '_TEAM'), validate='many_to_one')
    if not linked.GAME_DATE.eq(linked.GAME_DATE_TEAM).all() or not linked.SEASON.eq(linked.SEASON_TEAM).all():
        raise ValueError('Player and team dates/seasons disagree')
    audit = dict(player_rows=len(players), team_rows=len(team), games=team.GAME_ID.nunique(),
        seasons=sorted(team.SEASON.unique().tolist()), max_minute_rounding_gap=float((joined.MIN - joined.MIN_PLAYER).abs().max()),
        source='Existing cached NBA LeagueGameLog files', retrieval_time='Unknown: original cache has no retrieval manifest',
        semantics='Game rows become usable only after their game date; same-day outcomes excluded',
        files={str(p): file_hash(p) for p in [team_path, *paths]})
    return team, players, audit


def load_dated_context(path, kind):
    """Fail-fast contracts for optional sourced metadata; never manufacture assignments."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{kind} data unavailable: {path}')
    frame = pd.read_csv(path, dtype=str)
    required = {'birth_dates': ['PLAYER_ID', 'BIRTH_DATE', 'SOURCE'],
                'coaches': ['TEAM_ID', 'COACH_ID', 'START_DATE', 'END_DATE', 'KNOWN_AT', 'SOURCE'],
                'rosters': ['TEAM_ID', 'PLAYER_ID', 'START_DATE', 'END_DATE', 'KNOWN_AT', 'SOURCE']}[kind]
    missing = set(required) - set(frame)
    if missing or frame[required].isna().any().any():
        raise ValueError(f'Incomplete {kind} contract: {sorted(missing)}')
    for column in [c for c in required if c.endswith('DATE') or c == 'KNOWN_AT']:
        frame[column] = pd.to_datetime(frame[column], errors='raise')
    if kind == 'birth_dates':
        if frame.PLAYER_ID.duplicated().any():
            raise ValueError('Duplicate birth dates')
    else:
        if (frame.START_DATE >= frame.END_DATE).any():
            raise ValueError('Date intervals must be nonempty and half-open')
        group = 'TEAM_ID' if kind == 'coaches' else 'PLAYER_ID'
        for _, rows in frame.sort_values('START_DATE').groupby(group):
            if (rows.START_DATE.iloc[1:].reset_index(drop=True) < rows.END_DATE.iloc[:-1].reset_index(drop=True)).any():
                raise ValueError('Overlapping dated assignments')
    return frame
