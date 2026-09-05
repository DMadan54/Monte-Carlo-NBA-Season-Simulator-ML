import sys, pandas as pd
sys.path.append(r'c:/Users/dhruv/OneDrive/Desktop/Monte Carlo NBA')
from nba_api.stats.endpoints import scheduleleaguev2
# Load raw schedule (no filtering)
schedule_df = scheduleleaguev2.ScheduleLeagueV2(season='2024-25').get_data_frames()[0]
schedule = schedule_df[['gameDate','homeTeam_teamTricode','awayTeam_teamTricode']].copy()
unique_home = schedule['homeTeam_teamTricode'].unique()
unique_away = schedule['awayTeam_teamTricode'].unique()
all_abbr = set(unique_home).union(set(unique_away))
print('Raw schedule unique abbreviations count:', len(all_abbr))
print('Raw schedule abbreviations:', sorted(all_abbr))
# Load team_features parquet
from src.models.season_sim import PROCESSED_DATA_DIR
feat_path = PROCESSED_DATA_DIR / 'team_features.parquet'
team_feat = pd.read_parquet(feat_path)
unique_feat = team_feat['TEAM_ABBREVIATION'].unique()
print('Feature DataFrame unique abbreviations count:', len(unique_feat))
print('Feature abbreviations:', sorted(unique_feat))
