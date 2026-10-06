"""Chronological preseason baselines; rankings never imply probabilities."""
import numpy as np
import pandas as pd


def preseason_baselines(history, season, cutoff):
    """Carry forward last season, retaining exits and never selecting on target data.

    This is a returning-player universe, not a reconstructed preseason roster.
    Rookies and other entrants remain a disclosed coverage gap.
    """
    year = int(season[:4])
    past = history[history.season == f'{year-1}-{str(year)[-2:]}'].copy()
    if past.empty or (pd.to_datetime(past.source_max_date) >= pd.Timestamp(cutoff)).any():
        raise ValueError('Missing or future preseason feature history')
    if past.player_id.duplicated().any():
        raise ValueError('Duplicate prior player season')
    previous = history[history.season == f'{year-2}-{str(year-1)[-2:]}']
    if (pd.to_datetime(previous.source_max_date) >= pd.Timestamp(cutoff)).any():
        raise ValueError('Future improvement baseline')
    past = past.merge(previous[['player_id', 'ppg']], on='player_id', how='left', suffixes=('', '_previous'), validate='one_to_one')
    past['points_growth'] = past.ppg - past.ppg_previous
    # Missing previous year is explicit, not imputed as zero production.
    past['forecast_season'], past['forecast_cutoff'] = season, cutoff
    return past


def ranking_metrics(field, score_column, winner_id):
    values = field[score_column]
    if values.isna().any() or not np.isfinite(values.to_numpy()).all() or field.player_id.duplicated().any():
        raise ValueError('Rankings require complete finite scores and unique IDs')
    winner = field[field.player_id == winner_id]
    if len(winner) != 1:
        return {'winner_present': False, 'winner_rank': None, 'reciprocal_rank': 0., 'top5': False}
    score = winner.iloc[0][score_column]
    rank = 1 + float((values > score).sum()) + float((values == score).sum() - 1) / 2
    return {'winner_present': True, 'winner_rank': rank, 'reciprocal_rank': 1 / rank, 'top5': rank <= 5}


def evaluate_preseason(history, votes):
    rows = []
    for season in sorted(votes.season.unique()):
        cutoff = f'{season[:4]}-10-01'
        field = preseason_baselines(history, season, cutoff)
        for award, baselines in {'mvp': ['ppg', 'points'], 'mip': ['points_growth', 'ppg']}.items():
            labels = votes[(votes.season == season) & (votes.award == award)]
            winners = labels[labels.winner]
            if len(winners) != 1 or (pd.to_datetime(labels.label_available_at) <= pd.Timestamp(cutoff)).any():
                raise ValueError('Target labels require exactly one post-cutoff winner')
            for baseline in baselines:
                # Compare MIP baselines on the same two-year-history population.
                valid = field.dropna(subset=['points_growth'] if award == 'mip' else [baseline])
                row = dict(season=season, award=award, baseline=baseline, cutoff=cutoff,
                           evidence='development', feature_max_date=valid.source_max_date.max(),
                           candidates=len(valid), excluded_prior_players=len(field)-len(valid),
                           winner_id=str(winners.iloc[0].player_id),
                           **ranking_metrics(valid, baseline, winners.iloc[0].player_id))
                # Post-season population is used only for this coverage audit,
                # after ranking; it never selects or filters the forecast field.
                target_ids = set(history.loc[history.season == season, 'player_id'])
                row['target_players_without_preseason_inputs'] = len(target_ids - set(valid.player_id))
                rows.append(row)
    return rows
