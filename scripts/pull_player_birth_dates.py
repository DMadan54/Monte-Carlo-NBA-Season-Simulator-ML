"""Cache static NBA birth-date evidence; roster snapshots are NOT dated transactions."""
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import pandas as pd
from nba_api.stats.endpoints import commonteamroster, commonplayerinfo
from src.ingest.player_data import load_cached


def main():
    root = Path('data/raw/player_metadata')
    root.mkdir(parents=True, exist_ok=True)
    team, players, _ = load_cached('data/raw')
    known, failures = {}, []
    for path in root.glob('roster_*.json'):
        for row in json.loads(path.read_text(encoding='utf-8')).get('players', []):
            if row.get('BIRTH_DATE'):
                known[str(row['PLAYER_ID'])] = (row['BIRTH_DATE'], str(path))
    for (season, tid), rows in players.groupby(['SEASON', 'TEAM_ID'], sort=True):
        if set(rows.PLAYER_ID).issubset(known):
            continue
        path = root / f'roster_{season}_{tid}.json'
        if path.exists():
            continue
        try:
            frames = commonteamroster.CommonTeamRoster(team_id=tid, season=season, timeout=10).get_data_frames()
            record = {'source': 'NBA CommonTeamRoster', 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                      'season': season, 'team_id': tid, 'players': frames[0].to_dict(orient='records'),
                      'coaches_undated': frames[1].to_dict(orient='records')}
            path.write_text(json.dumps(record, default=str), encoding='utf-8')
            for row in record['players']:
                if row.get('BIRTH_DATE'):
                    known[str(row['PLAYER_ID'])] = (row['BIRTH_DATE'], str(path))
            print(f'{season} {tid}: {len(known)} birth dates', flush=True)
        except Exception as error:
            failures.append({'season': season, 'team_id': tid, 'error': str(error)})
            print(f'Metadata request failed: {season} {tid}: {error}', flush=True)
            if len(failures) >= 3:
                break
        time.sleep(.2)
    missing = sorted(set(players.PLAYER_ID) - set(known))
    if len(failures) < 3:
        for pid in missing:
            path = root / f'player_{pid}.json'
            try:
                if path.exists():
                    record = json.loads(path.read_text(encoding='utf-8'))
                else:
                    row = commonplayerinfo.CommonPlayerInfo(player_id=pid, timeout=10).get_data_frames()[0].iloc[0].to_dict()
                    record = {'source': 'NBA CommonPlayerInfo', 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'player': row}
                    path.write_text(json.dumps(record, default=str), encoding='utf-8')
                known[pid] = (record['player']['BIRTHDATE'], str(path))
            except Exception as error:
                failures.append({'player_id': pid, 'error': str(error)})
                if len(failures) >= 3:
                    break
            time.sleep(.2)
    rows = [dict(PLAYER_ID=pid, BIRTH_DATE=pd.Timestamp(dob).date().isoformat(), SOURCE=source)
            for pid, (dob, source) in known.items() if pd.notna(dob)]
    output = root / f'birth_dates_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.csv'
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'BIRTH_DATES={output}; missing={len(set(players.PLAYER_ID) - set(known))}', flush=True)
    (root / 'retrieval_status.json').write_text(json.dumps({'failures': failures, 'missing_player_ids': sorted(set(players.PLAYER_ID) - set(known)), 'output': str(output)}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
