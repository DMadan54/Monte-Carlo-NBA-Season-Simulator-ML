"""Optional roster-conditioned coaching experiment; requires dated source data.

This is a predictive association, not a causal estimate of coaching ability.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def attach_coaches(games, assignments):
    result = games.copy()
    for side in ['HOME', 'AWAY']:
        values, tenure = [], []
        by_team = {key: frame for key, frame in assignments.groupby('TEAM_ID')}
        for row in games.itertuples():
            date = pd.Timestamp(row.GAME_DATE)
            possible = by_team.get(getattr(row, f'{side}_TEAM'))
            eligible = possible.loc[(possible.START_DATE <= date) & (possible.END_DATE > date) &
                                    (possible.KNOWN_AT < date)] if possible is not None else pd.DataFrame()
            if len(eligible) > 1:
                raise ValueError('Ambiguous coach assignment')
            values.append(eligible.COACH_ID.iloc[0] if len(eligible) else 'UNKNOWN')
            tenure.append((date - eligible.START_DATE.iloc[0]).days if len(eligible) else 0)
        result[f'{side}_COACH'] = values
        result[f'{side}_COACH_TENURE'] = tenure
    return result


def fit_coach_model(train, numeric_columns, seed=42):
    numeric = [*numeric_columns, 'HOME_COACH_TENURE', 'AWAY_COACH_TENURE']
    categorical = ['HOME_COACH', 'AWAY_COACH']
    if (train[categorical] == 'UNKNOWN').any().any():
        raise ValueError('Source coaching coverage is incomplete; do not silently estimate missing assignments')
    preprocessing = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
        # Strong ridge shrinkage and unknown=zero pool unseen coaches to baseline.
        ('coach', OneHotEncoder(handle_unknown='ignore'), categorical)])
    model = make_pipeline(preprocessing, LogisticRegression(C=.05, max_iter=1500, random_state=seed))
    return model.fit(train[numeric + categorical], train.TARGET)
