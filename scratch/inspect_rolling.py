import pandas as pd
from pathlib import Path

feat_path = Path('data/processed/team_features.parquet')
df = pd.read_parquet(feat_path)
# ensure datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
# examine a few teams
for (team, season), grp in df.groupby(['TEAM_ABBREVIATION', 'SEASON_YEAR']):
    grp = grp.sort_values('GAME_DATE')
    first2 = grp.head(2)
    if not first2['ROLL_WIN_PCT'].isna().all() or not first2['ROLL_POINT_DIFF'].isna().all():
        print('Problem', team, season)
        print(first2[['GAME_DATE','ROLL_WIN_PCT','ROLL_POINT_DIFF']])
        break
print('Done')
