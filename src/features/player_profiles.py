"""Causal player snapshots and explicitly estimated historical rosters.

No current game's participants are consulted to construct its candidate roster.
The candidate list is a last-observed approximation, not an injury/transaction feed.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from src.ingest.player_data import COUNTS

RATE_COUNTS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGA', 'FG3A']
PROFILE = [f'{c}_36' for c in RATE_COUNTS] + ['EFG']
INPUTS = PROFILE + ['LAST_MIN', 'MEAN_MIN', 'RECENT_MIN', 'GAMES_SEEN', 'DAYS_ABSENT', 'TEAM_GAMES_ABSENT', 'PTS_TREND']


@dataclass
class PlayerState:
    counts: np.ndarray = field(default_factory=lambda: np.zeros(len(COUNTS)))
    exposure: float = 0.
    games: int = 0
    mean_min: float = 18.
    recent_min: float = 18.
    last_min: float = 18.
    recent_pts: float = 18.
    date: object = None
    team: str = ''
    team_game: int = 0


def profile(state, league_counts, league_minutes):
    # 120 pseudo-minutes at a prior estimated only from games already observed.
    neutral = np.array([18, 7, 4, 1, .7, 2, 15, 7, 5, 1.7]) / 36
    prior = league_counts / league_minutes if league_minutes else neutral
    counts = state.counts + 120 * prior
    rates = counts / (state.exposure + 120) * 36
    result = {f'{name}_36': float(rates[COUNTS.index(name)]) for name in RATE_COUNTS}
    result['EFG'] = float((counts[COUNTS.index('FGM')] + .5 * counts[COUNTS.index('FG3M')]) / max(counts[COUNTS.index('FGA')], 1))
    return result


def project_minutes(values, total=240., cap=48.):
    """Project nonnegative weights to feasible regulation minutes by water filling."""
    weights = np.asarray(values, dtype=float)
    if len(weights) < int(np.ceil(total / cap)) or not np.isfinite(weights).all():
        raise ValueError('At least five finite player weights required for regulation')
    weights = np.maximum(weights, 0.)
    result = np.zeros(len(weights))
    active = np.ones(len(weights), dtype=bool)
    remaining = total
    while active.any():
        w = weights[active]
        allocation = remaining * (w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w))
        over = allocation > cap + 1e-10
        indices = np.flatnonzero(active)
        if not over.any():
            result[indices] = allocation
            break
        fixed = indices[over]
        result[fixed] = cap
        active[fixed] = False
        remaining -= cap * len(fixed)
    return result


def build_candidates(team, players):
    """Return pregame candidates with future labels attached only after snapshots.

    Rosters retain players seen in the last 15 team games, until observed on a
    different team. Offseason trades are unknown until first observed. Date-batch
    processing excludes every outcome on the prediction date.
    """
    states, rosters, team_games = {}, {}, {}
    league_counts, league_minutes = np.zeros(len(COUNTS)), 0.
    player_days = {date: rows for date, rows in players.groupby('GAME_DATE')}
    records = []
    for date, games in team.sort_values(['GAME_DATE', 'GAME_ID']).groupby('GAME_DATE', sort=True):
        day_rows = player_days.get(date, pd.DataFrame())
        labels = {(row.GAME_ID, row.PLAYER_ID): row for row in day_rows.itertuples()}
        for game in games.itertuples():
            candidate_ids = sorted(pid for pid in rosters.get(game.TEAM_ID, set())
                if states[pid].team == game.TEAM_ID and team_games.get(game.TEAM_ID, 0) - states[pid].team_game <= 15)
            for pid in candidate_ids:
                state = states[pid]
                values = profile(state, league_counts, league_minutes)
                values.update(GAME_ID=game.GAME_ID, TEAM_ID=game.TEAM_ID, PLAYER_ID=pid,
                    GAME_DATE=date, SOURCE_MAX_DATE=state.date, SEASON_YEAR=game.SEASON_YEAR, LAST_MIN=state.last_min,
                    MEAN_MIN=state.mean_min, RECENT_MIN=state.recent_min, GAMES_SEEN=state.games,
                    DAYS_ABSENT=(date - state.date).days,
                    TEAM_GAMES_ABSENT=team_games.get(game.TEAM_ID, 0) - state.team_game,
                    PTS_TREND=state.recent_pts - values['PTS_36'])
                label = labels.get((game.GAME_ID, pid))
                # A traded player may appear for the opposition: zero minutes on old team.
                played = label is not None and label.TEAM_ID == game.TEAM_ID
                minutes = float(label.MIN) if played else 0.
                values['TARGET_MIN'] = minutes * 240 / float(game.MIN)
                values['TARGET_PTS36'] = float(label.PTS) * 36 / minutes if minutes > 0 else np.nan
                records.append(values)
        for game in games.itertuples():
            team_games[game.TEAM_ID] = team_games.get(game.TEAM_ID, 0) + 1
        for row in day_rows.itertuples():
            if row.MIN <= 0:
                continue
            state = states.setdefault(row.PLAYER_ID, PlayerState())
            elapsed = (date - state.date).days if state.date is not None else 0
            decay = 2 ** (-elapsed / 60.)
            counts = np.array([getattr(row, c) for c in COUNTS], dtype=float)
            state.counts = state.counts * decay + counts
            state.exposure = state.exposure * decay + row.MIN
            state.mean_min = .9 * state.mean_min + .1 * row.MIN
            state.recent_min = .65 * state.recent_min + .35 * row.MIN
            state.last_min = float(row.MIN)
            state.recent_pts = .7 * state.recent_pts + .3 * row.PTS * 36 / row.MIN
            state.games += 1
            state.date, state.team, state.team_game = date, row.TEAM_ID, team_games[row.TEAM_ID]
            rosters.setdefault(row.TEAM_ID, set()).add(row.PLAYER_ID)
            league_counts += counts
            league_minutes += row.MIN
    return pd.DataFrame(records)


def aggregate_profiles(candidates, minute_column, points_column=None):
    records = []
    for (gid, tid), rows in candidates.groupby(['GAME_ID', 'TEAM_ID'], sort=False):
        if len(rows) < 5:
            continue
        minutes = project_minutes(rows[minute_column].to_numpy())
        weights = minutes / 240
        values = rows[PROFILE].to_numpy().copy()
        if points_column:
            values[:, PROFILE.index('PTS_36')] = rows[points_column].to_numpy()
        record = {'GAME_ID': gid, 'TEAM_ID': tid}
        record.update({f'ROSTER_{c}': float(weights @ values[:, i]) for i, c in enumerate(PROFILE)})
        record.update(ROSTER_DEPTH=float((minutes > 12).sum()),
                      ROSTER_MIN_CONCENTRATION=float(np.sum(weights ** 2)),
                      ROSTER_RECENCY=float(weights @ np.minimum(rows.DAYS_ABSENT.to_numpy(), 365)),
                      ROSTER_TOP_PTS=float(np.max(values[:, 0])),
                      ROSTER_PTS_SPREAD=float(np.sqrt(weights @ (values[:, 0] - weights @ values[:, 0]) ** 2)),
                      ROSTER_SUPPORTED=1.)
        records.append(record)
    return pd.DataFrame(records)
