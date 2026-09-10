"""Bounded metadata feasibility probe; no bulk retrieval or invented coach dates."""
import json
from pathlib import Path
from datetime import datetime, timezone
from nba_api.stats.endpoints import commonteamroster


if __name__ == '__main__':
    record = {'requested_at': datetime.now(timezone.utc).isoformat(),
              'endpoint': 'CommonTeamRoster', 'team_id': '1610612738', 'season': '2023-24',
              'coach_limitation': 'Season roster coaching output lacks effective assignment dates and known-at timestamps'}
    try:
        frames = commonteamroster.CommonTeamRoster(team_id='1610612738', season='2023-24', timeout=10).get_data_frames()
        record['status'] = 'available'
        record['columns'] = [list(f.columns) for f in frames]
        record['rows'] = [len(f) for f in frames]
    except Exception as error:
        record['status'] = 'unavailable'
        record['error'] = str(error)
    path = Path('reports/player_metadata_probe.json')
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))
