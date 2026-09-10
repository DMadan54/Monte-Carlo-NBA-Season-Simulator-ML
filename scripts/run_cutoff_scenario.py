"""Run a historical fixed-cutoff scenario with explicit, limited roster assumptions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import pandas as pd

from src.ingest.player_data import load_cached
from src.features.build_team_features import add_basic_fields, add_rolling_features
from src.features.player_profiles import build_candidates
from src.models.player_forecast import forecast_profiles
from src.models.cutoff_sim import simulate_cutoff, apply_trade


def prepare_snapshot(team, players, cutoff, season):
    cutoff = pd.Timestamp(cutoff)
    history = team[team.GAME_DATE < cutoff].copy()
    player_history = players[players.GAME_DATE < cutoff].copy()
    latest = history.sort_values('GAME_DATE').groupby('TEAM_ID').tail(1).sort_values('TEAM_ID').copy()
    dates = latest.set_index('TEAM_ID').GAME_DATE.to_dict()
    fake = latest.copy().reset_index(drop=True)
    fake.GAME_DATE = cutoff
    fake.SEASON = season
    fake.SEASON_YEAR = int(season[:4]) + 1
    fake.SEASON_ID = '2' + season[:4]
    fake.WL = 'L'
    for column in ['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']:
        fake[column] = 0
    fake.MIN = 240
    for i in range(0, len(fake), 2):
        fake.loc[i:i+1, 'GAME_ID'] = f'snapshot_{i}'
        fake.loc[i, 'MATCHUP'] = 'A vs. B'
        fake.loc[i+1, 'MATCHUP'] = 'B @ A'
    combined = pd.concat([history, fake], ignore_index=True)
    features = add_rolling_features(add_basic_fields(combined))
    snapshot = features[features.GAME_DATE == cutoff].copy()
    snapshot['LAST_GAME_DATE'] = snapshot.TEAM_ID.map(dates)
    profiles = build_candidates(combined, player_history)
    profiles = profiles[profiles.GAME_DATE == cutoff].copy()
    return snapshot, profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cutoff', default='2025-01-01')
    parser.add_argument('--season', default='2024-25')
    parser.add_argument('--n-sims', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--player-points-sd', type=float, default=0.,
                        help='Additional per-player PTS/36 scenario sensitivity standard deviation.')
    parser.add_argument('--learned-player-uncertainty', action='store_true',
                        help='Use player-specific uncertainty trained on prior out-of-fold scoring errors.')
    parser.add_argument('--trade', nargs=2, action='append', metavar=('PLAYER_ID', 'DESTINATION_TEAM_ID'), default=[])
    args = parser.parse_args()
    selection = json.loads((args.run_dir / 'selection.json').read_text())
    name = selection['experimental_player_selected']
    bundle = joblib.load(args.run_dir / f'{name}.joblib')
    player_bundle = joblib.load(args.run_dir / 'player_models.joblib')
    cutoff = pd.Timestamp(args.cutoff)
    if pd.Timestamp(player_bundle['trained_through']) >= cutoff:
        raise ValueError('Player model was trained after the scenario cutoff')
    team, players, _ = load_cached('data/raw')
    snapshot, profiles = prepare_snapshot(team, players, cutoff, args.season)
    profiles = forecast_profiles(profiles, player_bundle)
    profiles['BASE_MIN'] = profiles.MEAN_MIN * __import__('numpy').exp(-.12 * profiles.TEAM_GAMES_ABSENT)
    # Final historical game dates are scenario inputs, not claimed as-published schedules.
    season = add_basic_fields(team[team.SEASON == args.season])
    remaining = season[(season.GAME_DATE >= cutoff) & (season.IS_HOME == 1)].copy()
    schedule = remaining[['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'OPP_TEAM_ID', 'NEUTRAL_GAME']].rename(
        columns={'TEAM_ID': 'HOME_TEAM', 'OPP_TEAM_ID': 'AWAY_TEAM'})
    current = season[season.GAME_DATE < cutoff].groupby('TEAM_ID').WIN.sum().to_dict()
    result = simulate_cutoff(bundle, snapshot, profiles, schedule, cutoff, current, args.n_sims, args.seed,
                             args.player_points_sd, args.learned_player_uncertainty)
    if args.trade:
        original_result = result.copy()
        for player_id, destination in args.trade:
            profiles = apply_trade(profiles, player_id, destination)
        result = simulate_cutoff(bundle, snapshot, profiles, schedule, cutoff, current, args.n_sims, args.seed,
                                 args.player_points_sd, args.learned_player_uncertainty)
        result['baseline_expected_wins'] = result.TEAM_ID.map(original_result.set_index('TEAM_ID').expected_wins)
        result['trade_expected_wins_delta'] = result.expected_wins - result.baseline_expected_wins
    names = team.drop_duplicates('TEAM_ID').set_index('TEAM_ID').TEAM_ABBREVIATION
    result['team'] = result.TEAM_ID.map(names)
    actual = season.groupby('TEAM_ID').WIN.sum()
    result['actual_wins'] = result.TEAM_ID.map(actual)
    output = args.run_dir / ('scenario_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(exist_ok=False)
    result.to_csv(output / 'standings.csv', index=False)
    snapshot.to_parquet(output / 'team_snapshot.parquet', index=False)
    profiles.to_parquet(output / 'player_snapshot.parquet', index=False)
    schedule.to_csv(output / 'schedule.csv', index=False)
    report = dict(cutoff=args.cutoff, model=name, seed=args.seed, n_sims=args.n_sims,
        player_points_sd=args.player_points_sd, learned_player_uncertainty=args.learned_player_uncertainty,
        trades=args.trade, games_remaining=len(schedule),
        mean_absolute_error=float((result.mean_wins - result.actual_wins).abs().mean()),
        interval_coverage=float(((result.actual_wins >= result.p05) & (result.actual_wins <= result.p95)).mean()),
        assumptions=['Frozen past-only profiles', 'Last-observed roster, no future transactions or injuries',
                     'Final historical schedule supplied as scenario; not an as-published schedule archive',
                     'Learned uncertainty uses only prior out-of-fold scoring residuals when enabled',
                     'Additional nonzero player SD is a sensitivity assumption rather than calibrated uncertainty'],
        promotion='Experimental demonstration; one cutoff is not comprehensive preseason validation')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print(f'SCENARIO_DIR={output}', flush=True)


if __name__ == '__main__':
    main()
