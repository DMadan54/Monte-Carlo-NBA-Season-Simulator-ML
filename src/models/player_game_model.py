"""Game feature assembly and portfolio model training helpers."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

from src.models.game_model import FEATURES

TEAM_COLUMNS = [c for c in FEATURES if c not in ['IS_HOME', 'NEUTRAL_GAME', 'OPP_ELO_RATING']]


def game_frame(team_features, rosters=None):
    frame = team_features.copy()
    if rosters is not None:
        frame = frame.merge(rosters, on=['GAME_ID', 'TEAM_ID'], how='left', validate='one_to_one')
    roster_columns = [c for c in frame if c.startswith('ROSTER_')]
    columns = TEAM_COLUMNS + roster_columns
    home = frame[frame.IS_HOME == 1]
    away = frame[frame.IS_HOME == 0]
    merged = home.merge(away, on='GAME_ID', suffixes=('_H', '_A'), validate='one_to_one')
    result = pd.DataFrame(dict(GAME_ID=merged.GAME_ID, GAME_DATE=merged.GAME_DATE_H,
        SEASON_YEAR=merged.SEASON_YEAR_H, HOME_TEAM=merged.TEAM_ID_H, AWAY_TEAM=merged.TEAM_ID_A,
        TARGET=merged.WIN_H, MARGIN=merged.POINT_DIFF_H))
    # Older cached feature tables predate explicit neutral-site handling.
    features = {'DIFF_NEUTRAL_GAME': merged.get('NEUTRAL_GAME_H', pd.Series(0, index=merged.index))}
    for column in columns:
        features[f'DIFF_{column}'] = merged[f'{column}_H'] - merged[f'{column}_A']
        # Include both levels for nonlinear/context effects; no player or team ID features.
        features[f'AVG_{column}'] = (merged[f'{column}_H'] + merged[f'{column}_A']) / 2
    return pd.concat([result, pd.DataFrame(features)], axis=1)


def fit_features(rosters):
    """Transferable profile interactions: hypotheses, not identified causal chemistry."""
    result = rosters.copy()
    result['ROSTER_FIT_CREATE_SPACE'] = result.ROSTER_AST_36 * result.ROSTER_FG3A_36
    result['ROSTER_FIT_VOLUME_EFF'] = result.ROSTER_FGA_36 * result.ROSTER_EFG
    result['ROSTER_FIT_REBOUND_BLOCK'] = result.ROSTER_REB_36 * result.ROSTER_BLK_36
    return result


def make_classifier(kind, seed=42):
    if kind == 'linear':
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
            StandardScaler(), LogisticRegression(C=.1, max_iter=1500, random_state=seed))
    return LGBMClassifier(n_estimators=180, learning_rate=.025, num_leaves=7,
        min_child_samples=120, reg_lambda=15, random_state=seed, n_jobs=2,
        deterministic=True, force_col_wise=True, verbosity=-1)


def metrics(y, probabilities):
    y, probabilities = np.asarray(y), np.clip(np.asarray(probabilities), 1e-8, 1 - 1e-8)
    bins = np.minimum((probabilities * 10).astype(int), 9)
    calibration = []
    for index in range(10):
        mask = bins == index
        if mask.any():
            calibration.append(dict(bin=index, games=int(mask.sum()), predicted=float(probabilities[mask].mean()),
                                    observed=float(y[mask].mean())))
    ece = sum(row['games'] * abs(row['predicted'] - row['observed']) for row in calibration) / len(y)
    return dict(games=len(y), log_loss=float(log_loss(y, probabilities, labels=[0, 1])),
        brier=float(brier_score_loss(y, probabilities)), accuracy=float(accuracy_score(y, probabilities >= .5)),
        calibration_ece=float(ece), calibration_bins=calibration)


def standings_replay(games, probabilities):
    # Expected wins from rolling pre-game probabilities; NOT a preseason forecast.
    h = pd.DataFrame({'TEAM_ID': games.HOME_TEAM, 'expected': probabilities, 'actual': games.TARGET})
    a = pd.DataFrame({'TEAM_ID': games.AWAY_TEAM, 'expected': 1 - probabilities, 'actual': 1 - games.TARGET})
    totals = pd.concat([h, a]).groupby('TEAM_ID')[['expected', 'actual']].sum()
    totals['error'] = totals.expected - totals.actual
    totals = totals.sort_values('actual', ascending=False)
    return dict(mae=float(totals.error.abs().mean()), top_5_bias=float(totals.head(5).error.mean()),
        bottom_5_bias=float(totals.tail(5).error.mean()),
        label='Rolling next-game expected-win aggregation; not a fixed-cutoff season forecast')
