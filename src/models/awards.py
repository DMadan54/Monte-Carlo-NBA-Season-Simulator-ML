"""Conservative award contracts. No production probability model is approved."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

AWARDS = {'mvp': 'Most Valuable Player', 'dpoy': 'Defensive Player of the Year',
          'roy': 'Rookie of the Year', 'sixth': 'Sixth Man of the Year',
          'mip': 'Most Improved Player'}
CBA = 'https://www.nbpa.com/cba/'
SIXTH = 'https://pr.nba.com/keldon-johnson-2025-26-kia-nba-sixth-man-of-the-year/'
RULES = {
    'mvp': '2023 CBA Article XXIX §6: 65 qualifying games; 20 minutes per game, with up to two games of 15–19.999 minutes. Injury exception: 62 qualifying games and 85% participation before an approved season-ending injury. Successful grievances/challenges also qualify.',
    'dpoy': 'Same 2023 CBA qualifying-games and exception contract as MVP.',
    'mip': 'Same 2023 CBA qualifying-games and exception contract as MVP; improvement requires a prior-season baseline.',
    'sixth': 'Bench appearances must strictly exceed starts. No 65-game requirement under Article XXIX §6. Verify season-specific league award instructions before release.',
    'roy': 'Requires verified NBA rookie designation; absence from one season of logs is not proof. No 65-game requirement under Article XXIX §6. Current-season designation feed is missing.',
}
MISSING = {
    'mvp': ['Projected efficiency and all-around production', 'Game-level minutes/availability and eligibility exceptions'],
    'dpoy': ['Validated defensive impact measurements (not just steals/blocks)', 'Opponent, lineup and team defensive context', 'Game-level minutes/availability and eligibility exceptions'],
    'roy': ['Verified rookie candidate population', 'Evaluated priors for players without NBA history'],
    'sixth': ['Joint projected games played and starts/bench roles', 'Season-specific Sixth Man eligibility confirmation'],
    'mip': ['Prior-season production joined by stable player ID', 'Evaluated year-to-year role, efficiency and production changes', 'Game-level minutes/availability and eligibility exceptions'],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligibility(award, season, *, minutes=None, starts=None, rookie=None,
                exception=None):
    """Three-state eligibility; exceptions require externally verified decisions.

    minutes is one value per played game, not average minutes. None means unknown.
    exception: approved, denied, none, or None (unresolved). Do not infer injuries.
    """
    if award not in AWARDS:
        raise ValueError('Unknown award')
    if exception not in {None, 'approved', 'denied', 'none'}:
        raise ValueError('Invalid exception status')
    if award == 'roy':
        if rookie is not None and not isinstance(rookie, (bool, np.bool_)):
            raise ValueError('Rookie designation must be boolean or unknown')
        return None if rookie is None else bool(rookie)
    if minutes is None:
        return None
    values = np.asarray(minutes, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any() or len(values) > 82:
        raise ValueError('Require valid played-game minutes')
    if award == 'sixth':
        if starts is None:
            return None
        if not isinstance(starts, (int, np.integer)) or not 0 <= starts <= len(values):
            raise ValueError('Invalid starts')
        return len(values) - starts > starts
    if int(season[:4]) < 2023:
        return None  # Earlier rules must be independently verified, never backfilled.
    qualifying = int((values >= 20).sum()) + min(2, int(((values >= 15) & (values < 20)).sum()))
    if qualifying >= 65 or exception == 'approved':
        return True
    return False if exception in {'none', 'denied'} else None


def attach_votes(population, votes):
    """Unlisted candidates retain unknown vote counts, never implicit zeroes."""
    keys = ['season', 'award', 'player_id']
    if population.duplicated(keys).any() or votes.duplicated(keys).any():
        raise ValueError('Duplicate candidate or voting record')
    if not set(map(tuple, votes[keys].values)).issubset(set(map(tuple, population[keys].values))):
        raise ValueError('Voting records outside candidate population')
    result = population.merge(votes, on=keys, how='left', validate='one_to_one')
    result['vote_status'] = np.where(result.vote_points.notna(), 'observed', 'unlisted_unknown')
    return result


def aggregate_choice_draws(probabilities, player_ids):
    """Integrate conditional voting probabilities over shared season trials.

    Caller must supply a separately validated choice model; this function does not
    transform scores into probabilities or establish calibration.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or not p.shape[0] or p.shape[1] != len(player_ids) or len(set(player_ids)) != len(player_ids):
        raise ValueError('Require full field with unique players and nonempty trials')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError('Each trial must allocate exactly one winner across the full field')
    return pd.DataFrame({'player_id': player_ids, 'probability': p.mean(axis=0)})


def leading_probabilities(field, limit=10):
    if limit < 1 or field.player_id.duplicated().any() or 'Other' in set(field.player_id):
        raise ValueError('Invalid display field')
    aggregate_choice_draws([field.probability.to_numpy()], field.player_id.tolist())
    ordered = field.sort_values('probability', ascending=False)
    head = ordered.head(limit).copy()
    if len(ordered) > limit:
        head = pd.concat([head, pd.DataFrame([{'player_id': 'Other', 'probability': ordered.iloc[limit:].probability.sum()}])], ignore_index=True)
    return head


def validate_joint(draws, standings, players, cutoff):
    """Check exact player membership, trial alignment and saved team means."""
    if str(draws['cutoff'].item()) != pd.Timestamp(cutoff).normalize().isoformat():
        raise ValueError('Joint draw cutoff mismatch')
    tids, pids = draws['team_ids'].astype(str), draws['player_ids'].astype(str)
    wins, pts = draws['team_wins'], draws['player_pts36']
    if len(set(tids)) != len(tids) or len(set(pids)) != len(pids):
        raise ValueError('Duplicate joint identities')
    roster = players.assign(PLAYER_ID=players.PLAYER_ID.astype(str), TEAM_ID=players.TEAM_ID.astype(str))
    if roster.PLAYER_ID.duplicated().any() or set(roster.PLAYER_ID) != set(pids):
        raise ValueError('Joint player population mismatch')
    if not np.array_equal(roster.set_index('PLAYER_ID').loc[pids].TEAM_ID.to_numpy(), draws['player_team_ids'].astype(str)):
        raise ValueError('Joint player/team membership mismatch')
    if wins.ndim != 2 or wins.shape[0] < 2 or wins.shape[1] != len(tids) or pts.shape != (len(wins), len(pids)):
        raise ValueError('Joint trial shape mismatch')
    if not np.isfinite(wins).all() or not np.isfinite(pts).all() or (wins < 0).any() or (pts < 0).any():
        raise ValueError('Invalid joint draw values')
    saved = standings.assign(TEAM_ID=standings.TEAM_ID.astype(str)).set_index('TEAM_ID')
    if saved.index.duplicated().any() or set(saved.index) != set(tids) or not np.allclose(saved.loc[tids].mean_wins, wins.mean(axis=0)):
        raise ValueError('Joint outcomes disagree with standings')
    if not np.all(wins.sum(axis=1) == wins.sum(axis=1)[0]):
        raise ValueError('Joint wins do not conserve league total')


def build_awards(source):
    """Build a fail-closed release from one immutable season/input directory."""
    source = Path(source)
    is_scenario = (source / 'report.json').exists()
    manifest = source / ('report.json' if is_scenario else 'source_metadata.json')
    report = json.loads(manifest.read_text(encoding='utf-8'))
    cutoff = report.get('cutoff') if is_scenario else report['retrieved_at_utc']
    files = {manifest.name: digest(manifest)}
    shared = ['Chronological full-pipeline preseason award evaluation',
              'Complete eligible populations linked to sourced historical voting labels',
              'Separately evaluated voting-preference uncertainty']
    if not is_scenario:
        shared += ['No shared season simulation outputs exist']
        schedule = report.get('schedule_completeness', {})
        if not schedule.get('simulation_eligible', False):
            shared += [schedule.get('reason', 'Complete regular-season schedule missing')]
        # Verify the preserved upstream input identities before reusing their status.
        for name, expected in report.get('files', {}).items():
            if digest(source / name) != expected:
                raise ValueError(f'Upstream input hash mismatch: {name}')
            files[name] = expected
    else:
        for name in ['standings.csv', 'player_snapshot.parquet', 'joint_draws.npz']:
            if (source / name).exists():
                files[name] = digest(source / name)
            else:
                shared.append(f'Missing shared artifact: {name}')
        if all(name in files for name in ['standings.csv', 'player_snapshot.parquet', 'joint_draws.npz']):
            with np.load(source / 'joint_draws.npz', allow_pickle=False) as draws:
                validate_joint(draws, pd.read_csv(source / 'standings.csv'), pd.read_parquet(source / 'player_snapshot.parquet'), cutoff)
            shared += ['Shared draws contain scoring rates only, not full player seasons']
    return {'schema_version': 1, 'forecast_version': source.name,
            'season': report.get('season', '2026-27' if not is_scenario else 'historical'),
            'forecast_cutoff': cutoff, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_files': files, 'status': 'incomplete',
            'population': {'modelled': 0, 'excluded': 'All players: no award model approved',
                           'roster_count': report.get('row_counts', {}).get('roster_rows')},
            'evaluation': {'status': 'not_validated', 'preseason_test_seasons': 0,
                           'winner_log_loss': None, 'winner_brier': None, 'calibration': None},
            'awards': {key: {'name': name, 'status': 'incomplete', 'candidates': [],
                             'probabilities': None, 'missing_requirements': shared + MISSING[key],
                             'eligibility': RULES[key], 'rule_source': SIXTH if key == 'sixth' else CBA}
                       for key, name in AWARDS.items()},
            'all_nba': {'status': 'deferred', 'reason': 'Five primary awards must work reliably first'}}


def save_awards(source, output):
    artifact = build_awards(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
    return artifact


def load_awards(path, source):
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    if artifact.get('schema_version') != 1 or artifact['forecast_version'] != Path(source).name:
        raise ValueError('Awards forecast version mismatch')
    for name, expected in artifact['source_files'].items():
        if Path(name).name != name or digest(Path(source) / name) != expected:
            raise ValueError('Awards source fingerprint mismatch')
    if set(artifact['awards']) != set(AWARDS):
        raise ValueError('Missing primary awards')
    for award in artifact['awards'].values():
        if award['status'] != 'incomplete' or award['probabilities'] is not None or award['candidates']:
            raise ValueError('This release has no approved award models')
    return artifact
