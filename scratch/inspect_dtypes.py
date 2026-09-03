import sys, pandas as pd
sys.path.append(r'c:/Users/dhruv/OneDrive/Desktop/Monte Carlo NBA')
from src.models.season_sim import PROCESSED_DATA_DIR
from nba_api.stats.endpoints import scheduleleaguev2

# Load schedule
schedule_df = scheduleleaguev2.ScheduleLeagueV2(season='2024-25').get_data_frames()[0]
schedule = schedule_df[['gameDate','homeTeam_teamTricode','awayTeam_teamTricode']].copy()
schedule.rename(columns={'gameDate':'GAME_DATE','homeTeam_teamTricode':'HOME_TEAM','awayTeam_teamTricode':'AWAY_TEAM'}, inplace=True)
schedule['GAME_DATE'] = pd.to_datetime(schedule['GAME_DATE'])
print('schedule GAME_DATE dtype:', schedule['GAME_DATE'].dtype)

# Load team features
features_path = PROCESSED_DATA_DIR / 'team_features.parquet'
team_feat = pd.read_parquet(features_path)
team_feat['GAME_DATE'] = pd.to_datetime(team_feat['GAME_DATE'])
print('team_feat GAME_DATE dtype:', team_feat['GAME_DATE'].dtype)
