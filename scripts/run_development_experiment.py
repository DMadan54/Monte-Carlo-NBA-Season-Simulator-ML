"""Our age-aware next-season player model, with explicit survivor conditioning."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ingest.player_data import load_cached, load_dated_context, file_hash
from src.models.player_forecast import regressor
from src.models.preseason_state_model import participation_metrics, error_slices


def annual_dataset(players, births):
    totals = players.groupby(['PLAYER_ID', 'SEASON_YEAR']).agg(
        MIN=('MIN', 'sum'), PTS=('PTS', 'sum'), REB=('REB', 'sum'), AST=('AST', 'sum'),
        GAMES=('GAME_ID', 'nunique')).reset_index()
    totals = totals.merge(births[['PLAYER_ID', 'BIRTH_DATE']], on='PLAYER_ID', how='left', validate='many_to_one')
    for stat in ['PTS', 'REB', 'AST']:
        totals[f'{stat}_36'] = totals[stat] * 36 / totals.MIN.clip(lower=1)
    totals['AGE_NEXT_OCT'] = ((pd.to_datetime(totals.SEASON_YEAR.astype(str) + '-10-01') - totals.BIRTH_DATE).dt.days / 365.25)
    totals = totals.sort_values(['PLAYER_ID', 'SEASON_YEAR'])
    totals['PRIOR_PTS36'] = totals.groupby('PLAYER_ID').PTS_36.shift(1)
    totals['PRIOR_YEAR'] = totals.groupby('PLAYER_ID').SEASON_YEAR.shift(1)
    totals.loc[totals.PRIOR_YEAR != totals.SEASON_YEAR - 1, 'PRIOR_PTS36'] = np.nan
    totals['PRIOR_PTS36'] = totals.PRIOR_PTS36.fillna(totals.PTS_36)
    totals['PTS_CHANGE'] = totals.PTS_36 - totals.PRIOR_PTS36
    totals['TARGET_SEASON'] = totals.SEASON_YEAR + 1
    labels = totals[['PLAYER_ID', 'SEASON_YEAR', 'MIN', 'PTS_36', 'REB_36', 'AST_36']].rename(
        columns={'SEASON_YEAR': 'TARGET_SEASON', 'MIN': 'TARGET_MIN', 'PTS_36': 'TARGET_PTS36',
                 'REB_36': 'TARGET_REB36', 'AST_36': 'TARGET_AST36'})
    result = totals.merge(labels, on=['PLAYER_ID', 'TARGET_SEASON'], how='left', validate='one_to_one')
    # An absent next-year row means zero NBA minutes, not zero skill.
    result.TARGET_MIN = result.TARGET_MIN.fillna(0)
    return result[result.TARGET_SEASON <= players.SEASON_YEAR.max()].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--birth-dates', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    _, players, _ = load_cached('data/raw')
    births = load_dated_context(args.birth_dates, 'birth_dates')
    dataset = annual_dataset(players, births)
    features = ['AGE_NEXT_OCT', 'MIN', 'GAMES', 'PTS_36', 'REB_36', 'AST_36', 'PRIOR_PTS36', 'PTS_CHANGE']
    output = args.run_dir / ('development_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(exist_ok=False)
    results = {'birth_date_source': str(args.birth_dates), 'source_hash': file_hash(args.birth_dates),
               'missing_age_rows': int(dataset.AGE_NEXT_OCT.isna().sum()),
               'limitations': ['Rates evaluated conditional on at least 300 minutes in the next season; survivor selection remains',
                              'Minutes targets include exits as zero; unusual season length affects totals',
                              'This model forecasts NBA production, not context-free causal ability; no future teammates are supplied']}
    models = {}
    for year, split in [(2024, 'validation'), (2025, 'test')]:
        train = dataset[dataset.TARGET_SEASON < year]
        test = dataset[dataset.TARGET_SEASON == year]
        scores = {}
        for target, prior in [('TARGET_PTS36', 'PTS_36'), ('TARGET_REB36', 'REB_36'), ('TARGET_AST36', 'AST_36')]:
            active_train = train.TARGET_MIN >= 300
            active_test = test.TARGET_MIN >= 300
            model = regressor(42).fit(train.loc[active_train, features], train.loc[active_train, target],
                sample_weight=train.loc[active_train, 'TARGET_MIN'])
            prediction = model.predict(test.loc[active_test, features])
            base = test.loc[active_test, prior].to_numpy()
            actual = test.loc[active_test, target].to_numpy()
            weights = test.loc[active_test, 'TARGET_MIN'].to_numpy()
            scores[target] = {'players': int(active_test.sum()),
                'baseline_rmse': float(np.sqrt(np.average((base - actual) ** 2, weights=weights))),
                'ml_rmse': float(np.sqrt(np.average((prediction - actual) ** 2, weights=weights)))}
            models[target] = model
            predictions = test.loc[active_test, ['PLAYER_ID', 'TARGET_SEASON', 'AGE_NEXT_OCT', prior, target]].copy()
            predictions['prediction'] = prediction
            predictions.to_csv(output / f'{split}_{target}.csv', index=False)
        minute_model = regressor(42).fit(train[features], train.TARGET_MIN)
        minute_prediction = np.maximum(minute_model.predict(test[features]), 0)
        scores['season_minutes'] = {'players': len(test), 'baseline_mae': float(np.abs(test.MIN - test.TARGET_MIN).mean()),
                                   'ml_mae': float(np.abs(minute_prediction - test.TARGET_MIN).mean()),
                                   'participation': participation_metrics(test.TARGET_MIN, minute_prediction),
                                   'age_slices': error_slices(test, minute_prediction, 'TARGET_MIN')}
        models['minutes'] = minute_model
        results[split] = scores
    dataset.to_parquet(output / 'annual_training_data.parquet', index=False)
    joblib.dump({'models': models, 'features': features, 'max_training_target_season': 2024, 'seed': 42}, output / 'annual_models.joblib')
    (output / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    print(f'DEVELOPMENT_DIR={output}')


if __name__ == '__main__':
    main()
