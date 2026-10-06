"""Download official NBA voting evidence into a new immutable directory.

Requires pdfplumber (optional ingestion dependency). URLs were resolved from the
NBA Communications season voting indexes; never scrape or infer player IDs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

SLUGS = {'mvp': 'Most-Valuable-Player', 'dpoy': 'Defensive-Player-of-the-Year',
         'roy': 'Rookie-of-the-Year', 'sixth': 'Sixth-Man-of-the-Year',
         'mip': 'Most-Improved-Player'}


def sources():
    for year in (2022, 2023, 2024):
        season = f'{year}-{str(year + 1)[-2:]}'
        for award, slug in SLUGS.items():
            month = '04' if year == 2022 and award != 'mvp' else '05'
            if year == 2023 and award == 'mvp':
                slug = 'MVP'
            kind = 'Voter-Selections' if year == 2024 else 'Voting-Results'
            yield {'season': season, 'award': award, 'kind': kind,
                   'voters': 99 if year == 2023 else 100,
                   'index_url': f'https://pr.nba.com/voting-results-{season}-nba-regular-season-awards/',
                   'url': f'https://ak-static.cms.nba.com/wp-content/uploads/sites/46/{year + 1}/{month}/{season}-Kia-NBA-{slug}-{kind}.pdf',
                   # Conservative label availability: date of complete published index.
                   'label_available_at': {2022: '2023-05-11', 2023: '2024-05-23', 2024: '2025-05-24'}[year]}


def parse_totals(pdf, award, voters):
    weights = [10, 7, 5, 3, 1] if award == 'mvp' else [5, 3, 1]
    records = []
    pattern = r'^(.+?,\s*.+?)\s+' + r'\s+'.join([r'(\d+)'] * (len(weights) + 1)) + r'$'
    for page in pdf.pages:
        for line in (page.extract_text() or '').splitlines():
            match = re.match(pattern, line)
            if match:
                values = list(map(int, match.groups()[1:]))
                if sum(w * v for w, v in zip(weights, values)) != values[-1]:
                    raise ValueError('Ballot weighted totals do not reconcile')
                records.append({'player_name': match[1].split(',')[0].strip(),
                                'place_votes': values[:-1], 'vote_points': values[-1]})
    if not records or len({r['player_name'] for r in records}) != len(records):
        raise ValueError('Missing or duplicate voting rows')
    if any(sum(r['place_votes'][i] for r in records) != voters for i in range(len(weights))):
        raise ValueError('Each ballot position must reconcile to the documented electorate')
    return records


def parse_ballots(pdf, award, voters):
    weights = [10, 7, 5, 3, 1] if award == 'mvp' else [5, 3, 1]
    counts, identities = defaultdict(lambda: [0] * len(weights)), set()
    for page in pdf.pages:
        for table in page.extract_tables():
            for row in table:
                if len(row) != len(weights) + 1 or not row[0] or row[0].startswith('Voter,'):
                    continue
                if any(not cell or ',' not in cell for cell in row):
                    raise ValueError('Incomplete ballot table row')
                if row[0] in identities:
                    raise ValueError('Duplicate voter')
                identities.add(row[0])
                names = [' '.join(cell.split(',')[0].split()) for cell in row[1:]]
                if len(set(names)) != len(names):
                    raise ValueError('Duplicate player on one ballot')
                for i, name in enumerate(names):
                    counts[name][i] += 1
    if len(identities) != voters:
        raise ValueError(f'Expected {voters} complete ballots, got {len(identities)}')
    return [{'player_name': name, 'place_votes': votes,
             'vote_points': sum(w * v for w, v in zip(weights, votes))}
            for name, votes in counts.items()]


def main():
    import pdfplumber
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence, records = [], []
    try:
        for source in sources():
            path = args.output / f"{source['season']}_{source['award']}.pdf"
            with urllib.request.urlopen(source['url'], timeout=45) as response:
                data = response.read()
            path.write_bytes(data)
            source.update(file=path.name, retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                          sha256=hashlib.sha256(data).hexdigest())
            evidence.append(source)
            with pdfplumber.open(path) as pdf:
                rows = (parse_ballots if source['kind'] == 'Voter-Selections' else parse_totals)(pdf, source['award'], source['voters'])
            maximum = max(r['vote_points'] for r in rows)
            if sum(r['vote_points'] == maximum for r in rows) != 1:
                raise ValueError('Tied winner requires explicit handling')
            for row in rows:
                records.append({**row, 'season': source['season'], 'award': source['award'],
                                'winner': row['vote_points'] == maximum, 'source_file': path.name,
                                'label_available_at': source['label_available_at']})
            print(f"{source['season']} {source['award']}: {len(rows)} vote recipients", flush=True)
        (args.output / 'votes.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    finally:
        (args.output / 'sources.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
