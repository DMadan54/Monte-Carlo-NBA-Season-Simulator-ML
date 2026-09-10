"""Fixed-cutoff scenario simulation: frozen profiles, optional persistent uncertainty.

Schedule dates/venues and roster assumptions are supplied by the caller. This
does not infer unknown future trades or use future actual performance.
"""
import numpy as np
import pandas as pd

from src.features.player_profiles import aggregate_profiles
from src.models.player_game_model import TEAM_COLUMNS, fit_features


def schedule_features(snapshot, schedule, rosters, include_fit=False):
    frame = snapshot.drop(columns=[c for c in snapshot if c.startswith('ROSTER_')])
    if include_fit:
        rosters = fit_features(rosters)
    columns = [c for c in rosters if c.startswith('ROSTER_')]
    frame = frame.merge(rosters[['TEAM_ID', *columns]], on='TEAM_ID', validate='one_to_one')
    frame = frame.set_index('TEAM_ID')
    h = frame.loc[schedule.HOME_TEAM].reset_index(drop=True)
    a = frame.loc[schedule.AWAY_TEAM].reset_index(drop=True)
    values = {'DIFF_NEUTRAL_GAME': schedule.NEUTRAL_GAME.to_numpy()}
    for name in TEAM_COLUMNS + columns:
        values[f'DIFF_{name}'] = h[name].to_numpy() - a[name].to_numpy()
        values[f'AVG_{name}'] = (h[name].to_numpy() + a[name].to_numpy()) / 2
    # Rest is known from supplied schedule and last completed game, not frozen.
    last = snapshot.set_index('TEAM_ID').LAST_GAME_DATE.to_dict()
    rest_h, rest_a = [], []
    for game in schedule.itertuples():
        rest_h.append((game.GAME_DATE - pd.Timestamp(last[game.HOME_TEAM])).days)
        rest_a.append((game.GAME_DATE - pd.Timestamp(last[game.AWAY_TEAM])).days)
        last[game.HOME_TEAM] = last[game.AWAY_TEAM] = game.GAME_DATE
    values['DIFF_DAYS_SINCE_LAST_GAME'] = np.array(rest_h) - rest_a
    values['AVG_DAYS_SINCE_LAST_GAME'] = (np.array(rest_h) + rest_a) / 2
    return pd.DataFrame(values)


def simulate_cutoff(bundle, snapshot, profiles, schedule, cutoff, current_wins=None,
                    n_sims=2000, seed=42, player_points_sd=0., learned_player_uncertainty=False):
    """Return standings with optional learned, persistent player uncertainty.

    Each player's scoring perturbation is drawn once per trial and retained for
    every scheduled game in that trial. Default zero yields outcome-only intervals.
    """
    cutoff = pd.Timestamp(cutoff).normalize()
    if pd.Timestamp(bundle['trained_through']) >= cutoff:
        raise ValueError('Model training reaches or exceeds forecast cutoff')
    if (pd.to_datetime(profiles.GAME_DATE) > cutoff).any():
        raise ValueError('Player snapshot is after cutoff')
    if not pd.to_datetime(snapshot.GAME_DATE).eq(cutoff).all():
        raise ValueError('Team snapshot must be prepared at this cutoff')
    if 'SOURCE_MAX_DATE' not in profiles or (pd.to_datetime(profiles.SOURCE_MAX_DATE) >= cutoff).any():
        raise ValueError('Player history must be strictly before cutoff')
    if not pd.to_datetime(profiles.GAME_DATE).eq(cutoff).all():
        raise ValueError('All profiles must be explicitly prepared at this cutoff')
    if (pd.to_datetime(snapshot.LAST_GAME_DATE) >= cutoff).any():
        raise ValueError('Team history reaches cutoff')
    if n_sims < 2 or seed is None or player_points_sd < 0 or not np.isfinite(player_points_sd):
        raise ValueError('Require >=2 trials, a seed, and finite nonnegative uncertainty')
    if learned_player_uncertainty and 'ML_PTS36_SD' not in profiles:
        raise ValueError('Learned player uncertainty requested but ML_PTS36_SD is unavailable')
    if profiles.duplicated('PLAYER_ID').any():
        raise ValueError('A player cannot belong to multiple scenario rosters')
    if snapshot.TEAM_ID.duplicated().any() or set(profiles.TEAM_ID) != set(snapshot.TEAM_ID):
        raise ValueError('Require one snapshot and a roster for every team')
    schedule = schedule.copy().sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    schedule.GAME_DATE = pd.to_datetime(schedule.GAME_DATE)
    if schedule.GAME_ID.duplicated().any() or (schedule.HOME_TEAM == schedule.AWAY_TEAM).any():
        raise ValueError('Invalid schedule game identities')
    if (schedule.GAME_DATE < cutoff).any():
        raise ValueError('Remaining schedule contains completed games')
    if not set(schedule.HOME_TEAM).union(schedule.AWAY_TEAM).issubset(set(snapshot.TEAM_ID)):
        raise ValueError('Schedule contains unknown teams')
    known = current_wins or {}
    if not set(known).issubset(set(snapshot.TEAM_ID)) or any(v < 0 or int(v) != v for v in known.values()):
        raise ValueError('Invalid current wins')
    seeds = np.random.SeedSequence(seed).spawn(2)
    talent_rng, game_rng = [np.random.default_rng(s) for s in seeds]
    ids = sorted(snapshot.TEAM_ID)
    index = {tid: i for i, tid in enumerate(ids)}
    home = schedule.HOME_TEAM.map(index).to_numpy()
    away = schedule.AWAY_TEAM.map(index).to_numpy()
    base = np.array([known.get(tid, 0) for tid in ids])
    win_matrix = np.tile(base, (n_sims, 1))
    expected = np.zeros((n_sims, len(ids))) + base
    include_fit = any('FIT_' in f for f in bundle['features'])
    minutes_column = 'BASE_MIN' if bundle['name'] == 'hybrid_simple' else 'ML_MIN'
    scoring_column = 'ML_PTS36' if ('development' in bundle['name'] or 'fit' in bundle['name'] or
                                      'boosted' in bundle['name'] or 'age' in bundle['name']) else 'PTS_36'
    base_profiles = profiles.copy()
    base_profiles['SCENARIO_POINTS'] = profiles[scoring_column]
    learned_sd = profiles.ML_PTS36_SD.to_numpy() if learned_player_uncertainty else np.zeros(len(profiles))
    def probabilities(current):
        rosters = aggregate_profiles(current, minutes_column, 'SCENARIO_POINTS')
        values = schedule_features(snapshot, schedule, rosters, include_fit)
        missing = set(bundle['features']) - set(values)
        if missing:
            raise ValueError(f'Scenario missing model features: {sorted(missing)}')
        return bundle['model'].predict_proba(values[bundle['features']])[:, 1]
    fixed = (
        probabilities(base_profiles)
        if player_points_sd == 0 and not learned_player_uncertainty
        else None
    )
    for trial in range(n_sims):
        if fixed is None:
            base_profiles['SCENARIO_POINTS'] = np.clip(profiles[scoring_column] +
                talent_rng.normal(0, np.sqrt(learned_sd ** 2 + player_points_sd ** 2), len(profiles)), 0, 60)
            p = probabilities(base_profiles)
        else:
            p = fixed
        outcomes = game_rng.random(len(schedule)) < p
        np.add.at(win_matrix[trial], home, outcomes.astype(int))
        np.add.at(win_matrix[trial], away, (~outcomes).astype(int))
        np.add.at(expected[trial], home, p)
        np.add.at(expected[trial], away, 1 - p)
    if not np.all(win_matrix.sum(axis=1) == sum(known.values()) + len(schedule)):
        raise AssertionError('League wins not conserved')
    result = pd.DataFrame(dict(TEAM_ID=ids, mean_wins=win_matrix.mean(axis=0), expected_wins=expected.mean(axis=0),
        std_wins=win_matrix.std(axis=0, ddof=1), p05=np.quantile(win_matrix, .05, axis=0),
        p95=np.quantile(win_matrix, .95, axis=0)))
    return result.sort_values('mean_wins', ascending=False).reset_index(drop=True)


def apply_trade(profiles, player_id, destination_team):
    """Change scenario membership; aggregation reallocates both teams' minutes."""
    result = profiles.copy()
    if destination_team not in set(result.TEAM_ID):
        raise ValueError('Unknown destination team')
    mask = result.PLAYER_ID == player_id
    if mask.sum() != 1:
        raise ValueError('Trade requires exactly one known player')
    result.loc[mask, 'TEAM_ID'] = destination_team
    result['GAME_ID'] = 'scenario'
    if result.groupby('TEAM_ID').size().min() < 5:
        raise ValueError('Trade leaves a team without a feasible rotation')
    return result
