"""Independently trained LightGBM player forecasts with annual out-of-fold output."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.features.player_profiles import INPUTS


def regressor(seed):
    return LGBMRegressor(n_estimators=100, learning_rate=.04, num_leaves=9,
        min_child_samples=150, reg_lambda=10., random_state=seed, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True)


def fit_forecasts(train, seed=42):
    minutes = regressor(seed).fit(train[INPUTS], train.TARGET_MIN)
    active = train.TARGET_MIN >= 5
    scoring = regressor(seed).fit(train.loc[active, INPUTS], train.loc[active, 'TARGET_PTS36'],
        sample_weight=train.loc[active, 'TARGET_MIN'])
    return {'minutes': minutes, 'scoring': scoring, 'features': INPUTS,
            'trained_through': str(train.GAME_DATE.max().date()),
            'targets': ['regulation-equivalent candidate minutes including absence', 'points per 36 conditional on >=5 minutes']}


def forecast_profiles(rows, bundle):
    result = rows.copy()
    result['ML_MIN'] = np.clip(bundle['minutes'].predict(rows[INPUTS]), 0, 48)
    result['ML_PTS36'] = np.clip(bundle['scoring'].predict(rows[INPUTS]), 0, 60)
    return result


def chronological_forecasts(candidates, seed=42):
    result = candidates.copy()
    # Deterministic past-only baselines, also used during the initial training seasons.
    result['BASE_MIN'] = result.MEAN_MIN * np.exp(-.12 * result.TEAM_GAMES_ABSENT)
    result['ML_MIN'] = result.BASE_MIN
    result['ML_PTS36'] = result.PTS_36
    diagnostics, latest = [], None
    seasons = sorted(result.SEASON_YEAR.unique())
    for season in seasons[2:]:
        train = result[result.SEASON_YEAR < season]
        mask = result.SEASON_YEAR == season
        test = result.loc[mask]
        bundle = fit_forecasts(train, seed)
        predictions = forecast_profiles(test, bundle)
        result.loc[mask, ['ML_MIN', 'ML_PTS36']] = predictions[['ML_MIN', 'ML_PTS36']].to_numpy()
        active = test.TARGET_MIN >= 5
        def weighted_rmse(values):
            return float(np.sqrt(np.average((values[active] - test.loc[active, 'TARGET_PTS36']) ** 2,
                                           weights=test.loc[active, 'TARGET_MIN'])))
        diagnostics.append(dict(season=int(season), candidates=len(test), active_scoring_targets=int(active.sum()),
            baseline_minutes_mae=float(np.abs(test.BASE_MIN - test.TARGET_MIN).mean()),
            ml_minutes_mae=float(np.abs(predictions.ML_MIN - test.TARGET_MIN).mean()),
            baseline_scoring_rmse=weighted_rmse(test.PTS_36),
            ml_scoring_rmse=weighted_rmse(predictions.ML_PTS36),
            trained_through=bundle['trained_through']))
        latest = bundle
        print(f'Player out-of-fold forecasts: {season}, {len(test):,} candidates', flush=True)
    return result, diagnostics, latest
