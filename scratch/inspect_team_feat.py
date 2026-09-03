import sys, pandas as pd
sys.path.append(r'c:/Users/dhruv/OneDrive/Desktop/Monte Carlo NBA')
from src.models.season_sim import PROCESSED_DATA_DIR

features_path = PROCESSED_DATA_DIR / 'team_features.parquet'
team_feat = pd.read_parquet(features_path)
print('Columns:', team_feat.columns.tolist())
print('Dtypes:\n', team_feat.dtypes)
print('Sample rows:')
print(team_feat.head())
