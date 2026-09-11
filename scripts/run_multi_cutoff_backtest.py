"""Chronological multi-cutoff calibration backtest for fixed-cutoff scenarios.

The final historical remaining schedule is deliberately a retrospective scenario
input.  It is never treated as a schedule archive known at the cutoff.
"""
import argparse
import gc
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss, log_loss

from scripts.run_cutoff_scenario import prepare_snapshot
from src.features.build_team_features import add_basic_fields, add_rolling_features
from src.features.player_profiles import INPUTS, build_candidates, aggregate_profiles
from src.ingest.player_data import file_hash, load_cached
from src.models.cutoff_sim import schedule_features, simulate_cutoff
from src.models.player_forecast import fit_forecasts, fit_scoring_uncertainty, forecast_profiles
from src.models.player_game_model import game_frame, make_classifier

ROOT = Path(__file__).resolve().parents[1]
CUTOFF_MONTHS = (12, 1, 2, 3)
SEASONS = tuple(f'{year}-{str(year + 1)[-2:]}' for year in range(2018, 2025))
NOMINALS = (.50, .80, .90)


def _hash_frame(frame):
    values = pd.util.hash_pandas_object(frame.sort_index(axis=1), index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def season_cutoffs(season):
    start = int(season[:4])
    return (pd.Timestamp(start, 12, 1), pd.Timestamp(start + 1, 1, 1),
            pd.Timestamp(start + 1, 2, 1), pd.Timestamp(start + 1, 3, 1))


def calibration_rows(actual, probabilities, bins=10):
    actual, probabilities = np.asarray(actual), np.asarray(probabilities)
    result = []
    for index in range(bins):
        mask = np.minimum((probabilities * bins).astype(int), bins - 1) == index
        if mask.any():
            result.append(dict(bin=index, games=int(mask.sum()), predicted=float(probabilities[mask].mean()),
                               observed=float(actual[mask].mean())))
    return result


def build_schedule(season_rows, cutoff):
    remaining = season_rows[(season_rows.GAME_DATE >= cutoff) & (season_rows.IS_HOME == 1)].copy()
    return remaining[['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'OPP_TEAM_ID', 'NEUTRAL_GAME', 'WIN']].rename(
        columns={'TEAM_ID': 'HOME_TEAM', 'OPP_TEAM_ID': 'AWAY_TEAM', 'WIN': 'TARGET'})


def fit_bundle(name, train_games, columns, cutoff, seed):
    if train_games.empty:
        raise ValueError(f'No training games before {cutoff.date()}')
    model = make_classifier('linear', seed).fit(train_games[columns], train_games.TARGET)
    return dict(model=model, features=columns, trained_through=str(train_games.GAME_DATE.max().date()),
                name=name, seed=seed, mode='fixed_cutoff',
                cutoff_rule='All model labels and features strictly before forecast cutoff')


def predicted_schedule(bundle, snapshot, profiles, schedule):
    # This mirrors the simulator feature path, so Brier/log loss evaluate the
    # exact probabilities used for the retrospective scenario.
    roster = aggregate_profiles(profiles, 'ML_MIN', 'SCENARIO_POINTS')
    values = schedule_features(snapshot, schedule, roster, include_fit=False)
    return bundle['model'].predict_proba(values[bundle['features']])[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-sims', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data' / 'processed' / 'multi_cutoff_runs')
    parser.add_argument('--oof-forecasts', type=Path,
                        default=ROOT / 'data' / 'processed' / 'player_runs' / 'v1_20260910T200443Z_6a86c4' / 'player_forecasts.parquet',
                        help='Chronological OOF upstream predictions; used only after source-date validation.')
    args = parser.parse_args()
    if args.n_sims < 2:
        raise ValueError('--n-sims must be at least two')
    run = args.output_root / f'v1_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}'
    run.mkdir(parents=True, exist_ok=False)
    raw_team, players, audit = load_cached(ROOT / 'data' / 'raw')
    team = add_basic_fields(raw_team)
    features = add_rolling_features(team)
    oof_schema = pq.ParquetFile(args.oof_forecasts).schema.names
    player_inputs = [column for column in INPUTS if column in oof_schema]
    if not player_inputs:
        raise ValueError('OOF player forecast artifact does not contain a usable feature schema')
    oof_columns = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'GAME_DATE', 'SEASON_YEAR', 'ML_MIN', 'ML_PTS36', 'TARGET_MIN', 'TARGET_PTS36', *player_inputs]
    oof_all = pd.read_parquet(args.oof_forecasts, columns=oof_columns)
    oof_all.GAME_DATE = pd.to_datetime(oof_all.GAME_DATE)
    source = dict(team_rows=len(team), player_rows=len(players), team_hash=_hash_frame(team), player_hash=_hash_frame(players),
                  oof_forecasts=str(args.oof_forecasts), oof_hash=file_hash(args.oof_forecasts), cache_audit=audit)
    (run / 'source_metadata.json').write_text(json.dumps(source, indent=2, default=str), encoding='utf-8')
    frozen = subprocess.run([__import__('sys').executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (run / 'requirements.lock.txt').write_text(frozen.stdout, encoding='utf-8')
    all_rows, game_rows, calibration = [], [], []
    for season in SEASONS:
        season_rows = team[team.SEASON == season].copy()
        final_wins = season_rows.groupby('TEAM_ID').WIN.sum()
        for cutoff in season_cutoffs(season):
            if cutoff <= team.GAME_DATE.min() or not (season_rows.GAME_DATE >= cutoff).any():
                continue
            print(f'{season} {cutoff.date()}', flush=True)
            snapshot, cutoff_profiles = prepare_snapshot(raw_team, players, cutoff, season)
            # Constructing only the historical prefix keeps peak memory bounded;
            # candidate construction itself is past-only, so this equals filtering
            # a full candidate table at the same cutoff.
            history_candidates = build_candidates(raw_team[raw_team.GAME_DATE < cutoff], players[players.GAME_DATE < cutoff])
            if any(column not in history_candidates for column in player_inputs):
                raise ValueError('Cutoff player candidates do not match OOF feature schema')
            player_bundle = fit_forecasts(history_candidates, args.seed, player_inputs)
            # This artifact contains predictions made by the prior-season OOF
            # model.  It is audited above and then filtered strictly by cutoff.
            oof = oof_all[(oof_all.GAME_DATE < cutoff) & (oof_all.SEASON_YEAR >= 2018)].copy()
            source_dates = history_candidates[['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'SOURCE_MAX_DATE']]
            oof = oof.merge(source_dates, on=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'], how='left', validate='one_to_one')
            oof.SOURCE_MAX_DATE = pd.to_datetime(oof.SOURCE_MAX_DATE)
            if oof.SOURCE_MAX_DATE.isna().any() or not (oof.SOURCE_MAX_DATE < oof.GAME_DATE).all():
                raise ValueError('OOF forecast artifact cannot be validated against a strictly earlier source date')
            first_oof_season = 2018
            player_bundle['scoring_uncertainty'] = fit_scoring_uncertainty(
                oof[(oof.GAME_DATE < cutoff) & (oof.SEASON_YEAR >= first_oof_season)], args.seed, player_inputs)
            if pd.Timestamp(player_bundle['trained_through']) >= cutoff:
                raise AssertionError('Player bundle leaked through cutoff')
            profiles = forecast_profiles(cutoff_profiles, player_bundle)
            profiles['SCENARIO_POINTS'] = profiles.ML_PTS36
            schedule = build_schedule(season_rows, cutoff)
            current = season_rows[season_rows.GAME_DATE < cutoff].groupby('TEAM_ID').WIN.sum().to_dict()
            team_game = game_frame(features)
            roster_oof = aggregate_profiles(oof, 'ML_MIN', 'ML_PTS36')
            del oof, history_candidates, cutoff_profiles
            gc.collect()
            hybrid_game = game_frame(features, roster_oof)
            for model_name, games in [('team_linear', team_game), ('hybrid_development_ml', hybrid_game)]:
                games = games[games.GAME_DATE < cutoff].copy()
                columns = [c for c in games if c.startswith(('DIFF_', 'AVG_'))]
                bundle = fit_bundle(model_name, games, columns, cutoff, args.seed)
                for learned in (False, True):
                    # Team-only has no roster features; its learned-uncertainty
                    # counterpart is intentionally a matched no-effect control.
                    result = simulate_cutoff(bundle, snapshot, profiles, schedule.drop(columns='TARGET'), cutoff, current,
                                             args.n_sims, args.seed, 0., learned)
                    probability = predicted_schedule(bundle, snapshot, profiles, schedule)
                    result['actual_wins'] = result.TEAM_ID.map(final_wins)
                    result['season'], result['cutoff'], result['model'], result['learned_uncertainty'] = season, str(cutoff.date()), model_name, learned
                    result.to_csv(run / f'standings_{season}_{cutoff:%Y%m%d}_{model_name}_{"learned" if learned else "outcome"}.csv', index=False)
                    metrics = dict(season=season, cutoff=str(cutoff.date()), model=model_name, learned_uncertainty=learned,
                                   games_remaining=len(schedule), brier=float(brier_score_loss(schedule.TARGET, probability)),
                                   log_loss=float(log_loss(schedule.TARGET, probability, labels=[0, 1])),
                                   standings_mae=float((result.mean_wins - result.actual_wins).abs().mean()),
                                   standings_rmse=float(np.sqrt(((result.mean_wins - result.actual_wins) ** 2).mean())),
                                   standings_bias=float((result.mean_wins - result.actual_wins).mean()),
                                   rank_correlation=float(result.mean_wins.rank().corr(result.actual_wins.rank())),
                                   coverage_50=float(((result.actual_wins >= result.p25) & (result.actual_wins <= result.p75)).mean()),
                                   width_50=float((result.p75 - result.p25).mean()),
                                   coverage_80=float(((result.actual_wins >= result.p10) & (result.actual_wins <= result.p90)).mean()),
                                   width_80=float((result.p90 - result.p10).mean()),
                                   coverage_90=float(((result.actual_wins >= result.p05) & (result.actual_wins <= result.p95)).mean()),
                                   width_90=float((result.p95 - result.p05).mean()),
                                   top_five_mae=float(result.nlargest(5, 'actual_wins').eval('mean_wins - actual_wins').abs().mean()),
                                   bottom_five_mae=float(result.nsmallest(5, 'actual_wins').eval('mean_wins - actual_wins').abs().mean()),
                                   trained_through=bundle['trained_through'], player_trained_through=player_bundle['trained_through'])
                    all_rows.append(metrics)
                    game_rows.extend(dict(season=season, cutoff=str(cutoff.date()), model=model_name, learned_uncertainty=learned,
                                          target=int(y), probability=float(p)) for y, p in zip(schedule.TARGET, probability))
                    calibration.extend(dict(season=season, cutoff=str(cutoff.date()), model=model_name, learned_uncertainty=learned, **row)
                                       for row in calibration_rows(schedule.TARGET, probability))
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(run / 'per_cutoff_metrics.csv', index=False)
    pd.DataFrame(game_rows).to_csv(run / 'game_probabilities.csv', index=False)
    pd.DataFrame(calibration).to_csv(run / 'calibration_bins.csv', index=False)
    aggregate = metrics.groupby(['model', 'learned_uncertainty']).mean(numeric_only=True).reset_index()
    aggregate.to_csv(run / 'aggregate_metrics.csv', index=False)
    protocol = dict(seed=args.seed, n_sims=args.n_sims, seasons=list(SEASONS), cutoff_rule='Strictly before each cutoff',
                    schedule='Final historical remaining schedule: retrospective scenario input, not known-at-cutoff archive',
                    promotion_gate='Learned uncertainty must move mean 90% coverage closer to 90% without raising mean standings MAE by 0.25 wins or more.',
                    interval_note='Quantiles are empirical seeded Monte Carlo intervals.')
    (run / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(f'RUN_DIR={run}', flush=True)


if __name__ == '__main__':
    main()
