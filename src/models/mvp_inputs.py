"""MVP feature contract and descriptive evidence, never a probability model."""
import numpy as np
import pandas as pd

from src.models.award_evaluation import ranking_metrics


FEATURE_GROUPS = {
    'Production': ['ppg', 'rebounds_per_game', 'assists_per_game'],
    'Efficiency': ['true_shooting', 'relative_true_shooting'],
    'Availability and role': ['games', 'minutes_per_game', 'minutes', 'games_share'],
    'Team success': ['team_win_pct'],
}
FEATURES = [feature for group in FEATURE_GROUPS.values() for feature in group]
DEFINITIONS = {
    'ppg': 'Points / appearances.',
    'rebounds_per_game': 'Rebounds / appearances.',
    'assists_per_game': 'Assists / appearances.',
    'true_shooting': 'PTS / (2 × (FGA + 0.44 × FTA)); undefined with no shooting attempts.',
    'relative_true_shooting': 'Player true shooting minus league true shooting, in proportion units.',
    'games': 'Observed appearances, including appearances rounded to zero minutes.',
    'minutes_per_game': 'Total minutes / appearances.',
    'minutes': 'Total minutes; combines role and availability.',
    'games_share': 'Appearances / final team games. Descriptive proxy, not award eligibility.',
    'team_win_pct': 'Final team wins / games. Traded players use their final team, not a stint-weighted team.',
}


def historical_inputs(players, teams):
    """Accept complete player-season sufficient statistics and team outcomes.

    Missing shooting volume stays missing. A partial prior-history row is rejected,
    not turned into a zero-production season. All inputs here are retrospective.
    """
    required = ['season', 'player_id', 'player_name', 'source_max_date', 'last_team_id',
                'points', 'REB', 'AST', 'minutes', 'games', 'FGA', 'FTA']
    if missing := set(required) - set(players):
        raise ValueError(f'Missing MVP inputs: {sorted(missing)}')
    if players[required].isna().any().any():
        raise ValueError('Incomplete MVP player-season inputs')
    if players.duplicated(['season', 'player_id']).any() or teams.duplicated(['season', 'team_id']).any():
        raise ValueError('Duplicate player or team season')
    p = players.copy()
    for column in ['points', 'REB', 'AST', 'minutes', 'games', 'FGA', 'FTA']:
        if not np.isfinite(p[column]).all() or (p[column] < 0).any():
            raise ValueError(f'Invalid MVP statistic: {column}')
    if (p.games <= 0).any():
        raise ValueError('Appearances must be positive')
    t = teams[['season', 'team_id', 'games', 'wins']].rename(columns={'games': 'team_games', 'wins': 'team_wins'})
    p = p.merge(t, left_on=['season', 'last_team_id'], right_on=['season', 'team_id'], how='left', validate='many_to_one')
    if p[['team_games', 'team_wins']].isna().any().any() or (p.team_games <= 0).any() or not p.team_wins.between(0, p.team_games).all():
        raise ValueError('Missing or invalid team outcomes')
    p['ppg'] = p.points / p.games
    p['rebounds_per_game'], p['assists_per_game'] = p.REB / p.games, p.AST / p.games
    p['minutes_per_game'] = p.minutes / p.games
    p['shooting_possessions'] = p.FGA + .44 * p.FTA
    p['true_shooting'] = p.points / (2 * p.shooting_possessions.replace(0, np.nan))
    league = p.groupby('season')[['points', 'shooting_possessions']].sum()
    league_ts = league.points / (2 * league.shooting_possessions.replace(0, np.nan))
    p['relative_true_shooting'] = p.true_shooting - p.season.map(league_ts)
    p['games_share'], p['team_win_pct'] = p.games / p.team_games, p.team_wins / p.team_games
    return p[['season', 'player_id', 'player_name', 'source_max_date', 'last_team_id', *FEATURES]]


def preseason_inputs(history, season, cutoff):
    """Carry forward prior-season inputs; never join target-season participants."""
    year = int(season[:4])
    prior = f'{year-1}-{str(year)[-2:]}'
    field = history[history.season == prior].copy()
    if field.empty or field.source_max_date.isna().any() or (pd.to_datetime(field.source_max_date) >= pd.Timestamp(cutoff)).any():
        raise ValueError('Missing or post-cutoff MVP history')
    if field.player_id.duplicated().any():
        raise ValueError('Duplicate preseason player')
    field['forecast_season'], field['forecast_cutoff'] = season, cutoff
    return field


def evidence_report(history, votes):
    """Audit winners and single-feature preseason baselines on the full prior field.

    Vote correlations cover observed recipients only. Unlisted votes remain unknown.
    No inferred zero votes, fitted weights, p-values or probability outputs.
    """
    votes = votes[votes.award == 'mvp'].copy()
    if votes.duplicated(['season', 'player_id']).any():
        raise ValueError('Duplicate MVP voting label')
    winners, associations, checks = [], [], []
    for season, labels in votes.groupby('season', sort=True):
        cutoff = f'{season[:4]}-10-01'
        if (pd.to_datetime(labels.label_available_at) <= pd.Timestamp(cutoff)).any():
            raise ValueError('MVP outcome must be after forecast cutoff')
        target = history[history.season == season]
        winner_labels = labels[labels.winner]
        if len(winner_labels) != 1:
            raise ValueError('Exactly one MVP winner required')
        winner_id = winner_labels.iloc[0].player_id
        winner = target[target.player_id == winner_id]
        if len(winner) != 1:
            raise ValueError('Winner missing from historical population')
        winner_row = winner.iloc[0]
        winners.append({key: winner_row[key] for key in ['season', 'player_id', 'player_name', *FEATURES]})
        observed = target.merge(labels[['player_id', 'vote_points']], on='player_id', validate='one_to_one')
        if len(observed) != len(labels) or observed.vote_points.isna().any():
            raise ValueError('Missing observed vote recipient or points')
        past = preseason_inputs(history, season, cutoff)
        for feature in FEATURES:
            pairs = observed[[feature, 'vote_points']].dropna()
            # Rank correlation is descriptive among recorded recipients only.
            corr = pairs.rank().corr().iloc[0, 1] if len(pairs) > 2 else np.nan
            associations.append({'season': season, 'feature': feature, 'observed_recipients': len(pairs),
                                 'recipient_spearman': None if pd.isna(corr) else float(corr)})
            valid = past.dropna(subset=[feature])
            checks.append({'season': season, 'cutoff': cutoff, 'feature_max_date': str(valid.source_max_date.max()),
                           'feature': feature, 'candidates': len(valid), 'missing_input_exclusions': len(past)-len(valid),
                           'target_players_without_inputs': len(set(target.player_id)-set(valid.player_id)),
                           **ranking_metrics(valid, feature, winner_id)})
    return {'status': 'experimental_inputs', 'probability_release_approved': False,
            'feature_groups': FEATURE_GROUPS, 'definitions': DEFINITIONS, 'winners': winners,
            'recipient_associations': associations, 'preseason_checks': checks,
            'limitations': [
                'Three MVP outcomes; every examined season is development evidence, not a held-out final test.',
                'Associations use actual end-of-season statistics and observed vote recipients only; selection bias and correlated inputs prevent causal conclusions.',
                'Preseason checks separately carry forward prior-season metrics. They retain exits, exclude entrants and do not incorporate roster moves or improvement.',
                'Single-feature rankings are baselines, not fitted feature weights or calibrated probabilities.',
                'Team context uses the final historical team; traded-player stint weighting remains missing.',
                'Games and rounded minutes do not certify qualifying games or eligibility exceptions.',
                'The shared simulation still needs joint rebounds, assists, shooting attempts/makes, games and minutes before these inputs can drive a 2026–27 award model.',
                'More historical voting seasons and full preseason pipeline evaluation are required to learn and validate MVP weights.',
            ]}
