"""Join sourced ballots to all observed players and run chronological baselines."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from src.models.awards import AWARDS, attach_votes, digest
from src.models.award_evaluation import evaluate_preseason


def name_key(value):
    text = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode().lower()
    text = re.sub(r'[^a-z ]', '', text)
    return ' '.join(text.split())


# Explicit source spelling differences, checked against NBA player IDs on join.
ALIASES = {'bobby portis jr': 'bobby portis', 'herb jones': 'herbert jones',
           'nicolas claxton': 'nic claxton', 'kj martin': 'kenyon martin jr',
           'jimmy butler': 'jimmy butler iii'}


def aggregate_season(players, teams, season):
    if players.duplicated(['PLAYER_ID', 'GAME_ID']).any() or teams.duplicated(['TEAM_ID', 'GAME_ID']).any():
        raise ValueError('Duplicate game records')
    if set(players.SEASON) != {season} or set(teams.SEASON) != {season}:
        raise ValueError('Season mismatch')
    # An appearance rounded to zero minutes is still a population record.
    # Never silently drop a player's entire season at a rounding boundary.
    players = players.copy()
    observed = players.groupby(['TEAM_ID', 'GAME_ID']).PTS.sum()
    official = teams.set_index(['TEAM_ID', 'GAME_ID']).PTS
    if set(observed.index) != set(official.index) or not observed.sort_index().equals(official.sort_index()):
        raise ValueError('Historical player scores do not reconcile with team results')
    players['GAME_DATE'] = pd.to_datetime(players.GAME_DATE)
    group = players.groupby('PLAYER_ID')
    totals = group[['PTS', 'MIN', 'REB', 'AST', 'FGM', 'FGA', 'FG3M', 'FTM', 'FTA', 'STL', 'BLK', 'TOV']].sum()
    totals['games'] = group.size()
    totals['source_max_date'] = group.GAME_DATE.max().dt.strftime('%Y-%m-%d')
    latest = players.sort_values(['GAME_DATE', 'GAME_ID']).groupby('PLAYER_ID').tail(1).set_index('PLAYER_ID')
    totals['player_name'] = latest.PLAYER_NAME
    totals['last_team_id'] = latest.TEAM_ID.astype(str)
    totals['last_team'] = latest.TEAM_ABBREVIATION
    totals['ppg'], totals['mpg'] = totals.PTS / totals.games, totals.MIN / totals.games
    totals['ts_pct'] = totals.PTS / (2 * (totals.FGA + .44 * totals.FTA)).replace(0, np.nan)
    totals['season'] = season
    totals['player_id'] = totals.index.astype(str)
    totals = totals.rename(columns={'PTS': 'points', 'MIN': 'minutes'})
    # Rounded whole-minute logs cannot certify the precise 20-minute boundary.
    totals['eligibility_status'] = 'unknown: rounded minutes, no starts/rookie/exception ledger'
    team = teams.groupby('TEAM_ID').agg(games=('GAME_ID', 'size'), wins=('WL', lambda x: int((x == 'W').sum())))
    if not team.games.eq(82).all() or team.wins.sum() != 1230 or len(team) != 30:
        raise ValueError('Require complete 82-game historical team season')
    totals['last_team_wins'] = latest.TEAM_ID.map(team.wins)
    team['season'], team['team_id'] = season, team.index.astype(str)
    return totals.reset_index(drop=True), team.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-logs', type=Path, required=True)
    parser.add_argument('--votes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = json.loads((args.votes / 'sources.json').read_text(encoding='utf-8'))
    for source in sources:
        if digest(args.votes / source['file']) != source['sha256']:
            raise ValueError('Voting evidence fingerprint mismatch')
    votes = pd.DataFrame(json.loads((args.votes / 'votes.json').read_text(encoding='utf-8')))
    history, team_history, logs = [], [], []
    for year in range(2020, 2025):
        # 2020-21 was shortened. Only its player history is needed for 2022-23 growth.
        season = f'{year}-{str(year+1)[-2:]}'
        pp, tp = args.raw_logs / f'player_game_logs_{season}.parquet', args.raw_logs / f'team_game_logs_{season}.parquet'
        players, teams = pd.read_parquet(pp), pd.read_parquet(tp)
        if year == 2020:
            h = players.groupby('PLAYER_ID').agg(ppg=('PTS', 'mean'), source_max_date=('GAME_DATE', 'max')).reset_index()
            h['player_id'], h['season'] = h.PLAYER_ID.astype(str), season
        else:
            h, t = aggregate_season(players, teams, season)
            team_history.append(t)
        history.append(h)
        for path in (pp, tp):
            logs.append({'file': path.name, 'sha256': digest(path), 'retrieved_at_utc': None,
                         'source': 'NBA LeagueGameLog local cache', 'retrieval_note': 'Original cache retrieval timestamp is unknown'})
    history = pd.concat(history, ignore_index=True)
    joined_votes = []
    for season, group in votes.groupby('season'):
        population = history[history.season == season].copy()
        population['name_key'] = population.player_name.map(name_key)
        if population.name_key.duplicated().any():
            raise ValueError('Ambiguous name join requires an explicit player-ID mapping')
        ids = population.set_index('name_key').player_id.to_dict()
        for row in group.to_dict('records'):
            key = name_key(row['player_name'])
            key = key if key in ids else ALIASES.get(key, key)
            if key not in ids:
                raise ValueError(f"Unmatched vote recipient: {season} {row['player_name']}")
            row['player_id'] = ids[key]
            joined_votes.append(row)
    votes = pd.DataFrame(joined_votes)
    population = history[history.season.isin(votes.season)].copy()
    population = pd.concat([population.assign(award=award) for award in AWARDS], ignore_index=True)
    candidates = attach_votes(population, votes.drop(columns='player_name'))
    candidates['population_semantics'] = 'All observed regular-season players; eligibility unresolved. Not a preseason roster.'
    candidates.to_parquet(args.output / 'candidate_history.parquet', index=False)
    history.to_parquet(args.output / 'player_history.parquet', index=False)
    pd.concat(team_history, ignore_index=True).to_parquet(args.output / 'team_history.parquet', index=False)
    votes.to_json(args.output / 'votes.json', orient='records', indent=2, force_ascii=False)
    evaluation = {'status': 'development_only', 'probability_release_approved': False,
                  'award_status': {'mvp': 'Preseason production baselines only; eligibility and full pipeline incomplete',
                                   'mip': 'Preseason production/growth baselines only; eligibility and full pipeline incomplete',
                                   'dpoy': 'Not evaluated: defensive impact and eligibility inputs missing',
                                   'roy': 'Not evaluated: rookie field and preseason priors missing',
                                   'sixth': 'Not evaluated: projected starts and bench roles missing'},
                  'preseason_pipeline_validated': False, 'rows': evaluate_preseason(history, votes),
                  'winner_log_loss': None, 'winner_brier': None, 'calibration': None,
                  'limitations': ['Only three outcomes per award; all examined seasons are development evidence.',
                                  'Carry-forward production is a preseason baseline, not the shared simulation pipeline.',
                                  'Prior-season participants retain exits but exclude incoming players; no archived preseason rosters.',
                                  'No learned choice model fitted: insufficient seasons and unresolved eligibility/population coverage.',
                                  'No DPOY, ROY or Sixth Man ranking: defensive impact, rookie priors and starts are missing.',
                                  'MIP growth baseline excludes players missing a second prior season; both MIP baselines use that same field.',
                                  'All award eligibility is unresolved; candidates are not filtered using future games played.',
                                  'Ranking baselines do not define probabilities; log loss, Brier and calibration are not applicable.']}
    (args.output / 'evaluation.json').write_text(json.dumps(evaluation, indent=2, allow_nan=False), encoding='utf-8')
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'logs': logs,
                'voting_sources': sources, 'votes_input_sha256': digest(args.votes / 'votes.json'),
                'candidate_rows': len(candidates), 'vote_recipient_rows': len(votes),
                'files': {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'candidates': len(candidates), 'vote_recipients': len(votes), 'evaluation': evaluation}, indent=2))


if __name__ == '__main__':
    main()
