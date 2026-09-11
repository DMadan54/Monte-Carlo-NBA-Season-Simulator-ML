"""Independently trained LightGBM player forecasts with annual out-of-fold output."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.features.player_profiles import INPUTS


def regressor(seed):
    return LGBMRegressor(n_estimators=100, learning_rate=.04, num_leaves=9,
        min_child_samples=150, reg_lambda=10., random_state=seed, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True)


def fit_forecasts(train, seed=42, inputs=None):
    inputs = INPUTS if inputs is None else inputs
    minutes = regressor(seed).fit(train[inputs], train.TARGET_MIN)
    active = train.TARGET_MIN >= 5
    scoring = regressor(seed).fit(train.loc[active, inputs], train.loc[active, 'TARGET_PTS36'],
        sample_weight=train.loc[active, 'TARGET_MIN'])
    return {'minutes': minutes, 'scoring': scoring, 'features': inputs,
            'trained_through': str(train.GAME_DATE.max().date()),
            'targets': ['regulation-equivalent candidate minutes including absence', 'points per 36 conditional on >=5 minutes']}


def fit_scoring_uncertainty(oof_rows, seed=42, inputs=None):
    """Fit player-specific score volatility from strictly earlier OOF residuals.

    The model targets expected absolute error and converts it to a normal-scale
    standard deviation. Training on in-sample residuals would make intervals
    falsely narrow, so callers must provide only past out-of-fold predictions.
    """
    active = oof_rows.TARGET_MIN >= 5
    rows = oof_rows.loc[active].copy()
    if rows.empty:
        raise ValueError('Need active out-of-fold player rows for uncertainty')
    rows['ABS_SCORING_ERROR'] = (rows.TARGET_PTS36 - rows.ML_PTS36).abs()
    inputs = INPUTS if inputs is None else inputs
    model = regressor(seed).fit(rows[inputs], rows.ABS_SCORING_ERROR,
        sample_weight=rows.TARGET_MIN)
    return model


def forecast_profiles(rows, bundle):
    result = rows.copy()
    inputs = bundle.get('features', INPUTS)
    result['ML_MIN'] = np.clip(bundle['minutes'].predict(rows[inputs]), 0, 48)
    result['ML_PTS36'] = np.clip(bundle['scoring'].predict(rows[inputs]), 0, 60)
    uncertainty = bundle.get('scoring_uncertainty')
    if uncertainty is not None:
        # For Normal(0, sigma), E|X| = sigma * sqrt(2 / pi).
        expected_abs = np.clip(uncertainty.predict(rows[inputs]), .1, 25)
        result['ML_PTS36_SD'] = expected_abs * np.sqrt(np.pi / 2)
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
        # Only earlier seasons contribute labels to this season's uncertainty.
        uncertainty_rows = result[(result.SEASON_YEAR < season) & result.ML_PTS36.notna()]
        if not uncertainty_rows.empty:
            bundle['scoring_uncertainty'] = fit_scoring_uncertainty(uncertainty_rows, seed)
        latest = bundle
        print(f'Player out-of-fold forecasts: {season}, {len(test):,} candidates', flush=True)
    return result, diagnostics, latest
