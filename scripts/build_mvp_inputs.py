"""Build reproducible MVP input evidence from the sourced award history."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from src.models.awards import digest
from src.models.mvp_inputs import historical_inputs, evidence_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.history / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if digest(args.history / name) != expected:
            raise ValueError(f'History fingerprint mismatch: {name}')
    players = pd.read_parquet(args.history / 'player_history.parquet')
    teams = pd.read_parquet(args.history / 'team_history.parquet')
    # The 2020-21 cache contains only PPG for an earlier MIP baseline; disclose it.
    complete_seasons = sorted(set(teams.season))
    inputs = historical_inputs(players[players.season.isin(complete_seasons)], teams)
    votes = pd.read_json(args.history / 'votes.json', dtype={'player_id': str})
    report = evidence_report(inputs, votes)
    report.update({'created_at_utc': datetime.now(timezone.utc).isoformat(),
                   'history_version': args.history.name, 'history_manifest_sha256': digest(args.history / 'manifest.json'),
                   'source_seasons': complete_seasons, 'player_seasons': len(inputs),
                   'excluded_incomplete_source_seasons': sorted(set(players.season)-set(complete_seasons)),
                   'sources': [s for s in manifest['voting_sources'] if s['award'] == 'mvp'],
                   'log_provenance': manifest['logs']})
    args.output.mkdir(parents=True, exist_ok=False)
    inputs.to_parquet(args.output / 'inputs.parquet', index=False)
    report['inputs_sha256'] = digest(args.output / 'inputs.parquet')
    (args.output / 'evidence.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"Built {len(inputs)} player-seasons and {len(report['preseason_checks'])} MVP feature checks in {args.output}")


if __name__ == '__main__':
    main()
