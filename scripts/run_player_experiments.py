"""Train our player-aware models reproducibly without touching legacy artifacts.

Run: .venv/Scripts/python.exe -m scripts.run_player_experiments
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import uuid

import joblib
import numpy as np
import pandas as pd

from src.ingest.player_data import load_cached
from src.features.build_team_features import add_basic_fields, add_rolling_features
from src.features.player_profiles import build_candidates, aggregate_profiles
from src.models.player_forecast import chronological_forecasts
from src.models.player_game_model import game_frame, fit_features, make_classifier, metrics, standings_replay

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def evaluate(name, games, columns, run, seed, kind='linear'):
    validation, test = 2024, 2025
    result, fitted = {}, None
    for season, label in [(validation, 'validation'), (test, 'test')]:
        train = games[games.SEASON_YEAR < season]
        held = games[games.SEASON_YEAR == season]
        if len(train) == 0 or len(held) == 0:
            raise ValueError('Required chronological split unavailable')
        model = make_classifier(kind, seed).fit(train[columns], train.TARGET)
        probabilities = model.predict_proba(held[columns])[:, 1]
        result[label] = metrics(held.TARGET, probabilities)
        result[label]['rolling_standings'] = standings_replay(held, probabilities)
        result[label]['trained_through'] = str(train.GAME_DATE.max().date())
        output = held[['GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'TARGET']].copy()
        output['probability'] = probabilities
        output.to_csv(run / f'{name}_{label}_predictions.csv', index=False)
        # Calendar-month and extreme-outcome slices are diagnostics, not model inputs.
        early = held.GAME_DATE.dt.month.isin([10, 11])
        result[label]['early_season'] = metrics(held.loc[early, 'TARGET'], probabilities[early])
        fitted = dict(model=model, features=columns, trained_through=str(train.GAME_DATE.max().date()),
            mode='rolling_next_game', name=name, seed=seed, cutoff_rule='All input outcomes strictly before forecast date')
    joblib.dump(fitted, run / f'{name}.joblib')
    print(f'{name}: validation LL={result["validation"]["log_loss"]:.4f}; test LL={result["test"]["log_loss"]:.4f}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = ROOT / 'data' / 'processed' / 'player_runs' / f'v1_{stamp}_{uuid.uuid4().hex[:6]}'
    run.mkdir(parents=True, exist_ok=False)
    print(f'RUN_DIR={run}', flush=True)
    team, players, audit = load_cached(ROOT / 'data' / 'raw')
    write_json(run / 'data_audit.json', audit)
    frozen = subprocess.run([__import__('sys').executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (run / 'requirements.lock.txt').write_text(frozen.stdout, encoding='utf-8')
    features = add_rolling_features(add_basic_fields(team))
    baseline = game_frame(features)
    # Same downstream game set across experiments; first two seasons warm up player training.
    baseline = baseline[baseline.SEASON_YEAR >= 2018].reset_index(drop=True)
    team_columns = [c for c in baseline if c.startswith(('DIFF_', 'AVG_'))]
    results = {}
    results['team_linear'] = evaluate('team_linear', baseline, team_columns, run, args.seed)
    results['team_ml'] = evaluate('team_ml', baseline, team_columns, run, args.seed, 'boosted')
    write_json(run / 'metrics.json', results)
    candidates = build_candidates(team, players)
    print(f'Built {len(candidates):,} causal candidate rows', flush=True)
    forecasts, player_metrics, latest = chronological_forecasts(candidates, args.seed)
    forecasts.to_parquet(run / 'player_forecasts.parquet', index=False)
    write_json(run / 'player_metrics.json', player_metrics)
    joblib.dump(latest, run / 'player_models.joblib')
    # Coverage measures how much actual playing time belonged to pregame candidates.
    coverage = candidates.groupby(['GAME_ID', 'TEAM_ID']).TARGET_MIN.sum().reindex(
        pd.MultiIndex.from_frame(team[['GAME_ID', 'TEAM_ID']]), fill_value=0) / 240
    write_json(run / 'roster_coverage.json', dict(mean_captured_minutes=float(coverage.mean()),
        below_80_percent_team_games=int((coverage < .8).sum()),
        note='Last-observed roster estimate; first appearances and newly traded arrivals are unavailable until observed'))
    variants = {'simple': aggregate_profiles(forecasts, 'BASE_MIN'),
                'minutes_ml': aggregate_profiles(forecasts, 'ML_MIN'),
                'development_ml': aggregate_profiles(forecasts, 'ML_MIN', 'ML_PTS36')}
    variants['fit_ml'] = fit_features(variants['development_ml'])
    for variant, roster in variants.items():
        frame = game_frame(features, roster)
        frame = frame[frame.SEASON_YEAR >= 2018].reset_index(drop=True)
        if not frame.GAME_ID.equals(baseline.GAME_ID):
            raise ValueError('Ablation game sets differ')
        all_columns = [c for c in frame if c.startswith(('DIFF_', 'AVG_'))]
        if variant == 'simple':
            roster_columns = [c for c in all_columns if 'ROSTER_' in c]
            results['roster_linear'] = evaluate('roster_linear', frame, roster_columns, run, args.seed)
        results[f'hybrid_{variant}'] = evaluate(f'hybrid_{variant}', frame, all_columns, run, args.seed)
        if variant == 'fit_ml':
            results['hybrid_boosted'] = evaluate('hybrid_boosted', frame, all_columns, run, args.seed, 'boosted')
        write_json(run / 'metrics.json', results)
    # Select exclusively on validation loss; test is reported once for the frozen candidates.
    selected = min(results, key=lambda name: results[name]['validation']['log_loss'])
    player_selected = min((n for n in results if n.startswith('hybrid')), key=lambda n: results[n]['validation']['log_loss'])
    write_json(run / 'selection.json', dict(validation_selected=selected, experimental_player_selected=player_selected,
        selection_rule='Minimum 2023-24 log loss; predefined models; no tuning on 2024-25',
        seed=args.seed, evaluation_mode='rolling_next_game',
        missing=['Dated transaction/injury history', 'Verified birth dates', 'Dated head-coach assignments', 'Possession/lineup data'],
        model_promotion='No legacy model replaced; player candidate remains experimental'))
    print(f'Selected on validation: {selected}; experimental player candidate: {player_selected}', flush=True)
    print(f'Completed: {run}', flush=True)


if __name__ == '__main__':
    main()
