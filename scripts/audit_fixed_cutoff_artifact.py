"""Integrity checks for fixed-cutoff scenario and multi-cutoff artifacts."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.ingest.player_data import load_cached
from src.features.build_team_features import add_basic_fields


def canonical_game_id(values: pd.Series) -> pd.Series:
    """Normalise NBA game ids after CSV readers have removed leading zeroes."""
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


IDENTITY_COLUMNS = ['game_id', 'game_date', 'home_team', 'away_team', 'target']


def deterministic_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash ordered, CSV-stable audit records without relying on row indexes."""
    values = frame.loc[:, columns].copy()
    for column in values:
        if pd.api.types.is_datetime64_any_dtype(values[column]):
            values[column] = values[column].dt.strftime('%Y-%m-%d')
        elif pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda value: format(value, '.17g'))
        else:
            values[column] = values[column].astype(str)
    payload = values.to_csv(index=False, lineterminator='\n').encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def probability_output_hash(records: pd.DataFrame) -> str:
    """Hash ordered identity plus the exact IEEE-754 probability bytes."""
    identity = deterministic_hash(records, IDENTITY_COLUMNS).encode('ascii')
    probabilities = np.ascontiguousarray(records.probability.to_numpy(dtype=np.float64))
    return hashlib.sha256(identity + probabilities.tobytes()).hexdigest()


def multi_cutoff_manifest(schedule: pd.DataFrame, snapshot: pd.DataFrame,
                          profiles: pd.DataFrame, values: pd.DataFrame,
                          probabilities, bundle: dict, cutoff) -> dict:
    """Create a self-contained manifest for one raw scoring pass.

    The ordered schedule and the exact model matrix are hashed before any
    calibration can consume the raw probabilities.
    """
    cutoff = pd.Timestamp(cutoff)
    required_schedule = ['GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'TARGET']
    missing = set(required_schedule).difference(schedule.columns)
    if missing:
        raise ValueError(f'Missing schedule audit columns: {sorted(missing)}')
    if pd.to_datetime(snapshot.GAME_DATE).ne(cutoff).any():
        raise ValueError('Snapshot date does not equal cutoff')
    if 'LAST_GAME_DATE' not in snapshot or (pd.to_datetime(snapshot.LAST_GAME_DATE) >= cutoff).any():
        raise ValueError('Snapshot includes a game at or after cutoff')
    if 'SOURCE_MAX_DATE' not in profiles or (pd.to_datetime(profiles.SOURCE_MAX_DATE) >= cutoff).any():
        raise ValueError('Player profile includes a source at or after cutoff')
    feature_columns = list(bundle['features'])
    if list(values.columns) != feature_columns:
        raise ValueError('Scored feature columns do not exactly match model feature columns')
    records = pd.DataFrame({
        'game_id': canonical_game_id(schedule.GAME_ID),
        'game_date': pd.to_datetime(schedule.GAME_DATE),
        'home_team': schedule.HOME_TEAM,
        'away_team': schedule.AWAY_TEAM,
        'target': schedule.TARGET.astype(int),
        'probability': probabilities,
    })
    if records.game_id.duplicated().any() or (records.home_team == records.away_team).any():
        raise ValueError('Invalid ordered game identities')
    return {
        'audit_version': 1,
        'cutoff': str(cutoff.date()),
        'games': int(len(records)),
        'ordered_games_hash': deterministic_hash(records, IDENTITY_COLUMNS),
        'probability_output_hash': probability_output_hash(records),
        'feature_matrix_hash': deterministic_hash(values, feature_columns),
        'feature_schema': feature_columns,
        'model_feature_columns': feature_columns,
        'snapshot_rows': int(len(snapshot)),
        'snapshot_last_game_max': str(pd.to_datetime(snapshot.LAST_GAME_DATE).max().date()),
        'profile_source_max': str(pd.to_datetime(profiles.SOURCE_MAX_DATE).max().date()),
        'checks': {
            'snapshot_strictly_before_cutoff': True,
            'profile_sources_strictly_before_cutoff': True,
            'feature_schema_matches_model': True,
            'unique_ordered_game_ids': True,
        },
    }


def validate_calibration_inputs(run_dir: Path, game_rows: pd.DataFrame) -> None:
    """Require calibration rows to match audited raw scoring outputs exactly."""
    required = {'season', 'cutoff', 'model', 'learned_uncertainty', *IDENTITY_COLUMNS, 'probability'}
    missing = required.difference(game_rows.columns)
    if missing:
        raise ValueError(f'Calibration input lacks audit columns: {sorted(missing)}')
    raw = game_rows[(game_rows.model == 'hybrid_development_ml') &
                    (game_rows.learned_uncertainty == False)].copy()
    audit_dir = run_dir / 'scoring_audits'
    if not audit_dir.is_dir():
        raise ValueError('Calibration input has no scoring audit manifests')
    expected = set()
    for (season, cutoff), rows in raw.groupby(['season', 'cutoff'], sort=False):
        expected.add((season, str(cutoff)))
        path = audit_dir / f'{season}_{pd.Timestamp(cutoff):%Y%m%d}_hybrid_development_ml.json'
        if not path.exists():
            raise ValueError(f'Missing scoring audit manifest for {season} {cutoff}')
        manifest = json.loads(path.read_text(encoding='utf-8'))
        records = rows.loc[:, [*IDENTITY_COLUMNS, 'probability']].copy()
        records['game_id'] = canonical_game_id(records.game_id)
        if deterministic_hash(records, IDENTITY_COLUMNS) != manifest['ordered_games_hash']:
            raise ValueError(f'Calibration game identity/order mismatch for {season} {cutoff}')
        if probability_output_hash(records) != manifest['probability_output_hash']:
            raise ValueError(f'Calibration probability output mismatch for {season} {cutoff}')
        if manifest['feature_schema'] != manifest['model_feature_columns']:
            raise ValueError(f'Model-feature schema mismatch for {season} {cutoff}')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--raw-dir',type=Path,default=Path('data/raw')); args=ap.parse_args()
    report=json.loads((args.artifact/'report.json').read_text()); cutoff=pd.Timestamp(report['cutoff'])
    snapshot=pd.read_parquet(args.artifact/'team_snapshot.parquet'); profiles=pd.read_parquet(args.artifact/'player_snapshot.parquet'); schedule=pd.read_csv(args.artifact/'schedule.csv'); schedule.GAME_DATE=pd.to_datetime(schedule.GAME_DATE)
    team,_,_=load_cached(args.raw_dir); team=add_basic_fields(team)
    expected_last=team[team.GAME_DATE<cutoff].groupby('TEAM_ID').GAME_DATE.max()
    schedule_truth = team[(team.GAME_DATE >= cutoff) & (team.IS_HOME == 1)].copy()
    schedule_ids = canonical_game_id(schedule.GAME_ID)
    truth_ids = canonical_game_id(schedule_truth.GAME_ID)
    checks={
      'snapshot_rows_30': len(snapshot)==30 and snapshot.TEAM_ID.nunique()==30,
      'snapshot_dates_equal_cutoff': bool(pd.to_datetime(snapshot.GAME_DATE).eq(cutoff).all()),
      'snapshot_last_game_matches_raw': bool(
          pd.to_datetime(snapshot.set_index('TEAM_ID').LAST_GAME_DATE).eq(
              expected_last.reindex(snapshot.set_index('TEAM_ID').index)
          ).all()
      ),
      'profile_sources_strictly_before_cutoff': bool((pd.to_datetime(profiles.SOURCE_MAX_DATE)<cutoff).all()),
      'schedule_unique_games': bool(not schedule.GAME_ID.duplicated().any()),
      'schedule_after_cutoff': bool((schedule.GAME_DATE>=cutoff).all()),
      'schedule_has_raw_home_targets': bool(schedule_ids.isin(truth_ids).all()),
    }
    output={
        'cutoff':str(cutoff.date()),
        'schedule_games': int(len(schedule)),
        'raw_home_games_after_cutoff': int(len(schedule_truth)),
        'checks':checks,
        'passed':all(checks.values()),
        'note':'This validates artifact identity/date/orientation inputs; it does not prove model feature-value parity.',
    }
    print(json.dumps(output,indent=2))
    (args.artifact/'integrity_audit.json').write_text(json.dumps(output,indent=2))
    if not output['passed']: raise SystemExit(1)
if __name__=='__main__': main()
