"""Evaluation helpers for date-safe preseason player state transitions."""
import numpy as np
import pandas as pd


def participation_metrics(actual_minutes, predicted_minutes, threshold=300):
    """Participation is evaluated on all players, including zero-minute exits."""
    actual = (np.asarray(actual_minutes) >= threshold).astype(float)
    probability = np.clip(np.asarray(predicted_minutes) / threshold, 0, 1)
    return {'brier': float(np.mean((probability - actual) ** 2)),
            'actual_rate': float(actual.mean()), 'mean_predicted_probability': float(probability.mean())}


def error_slices(rows, prediction, target, minute_column='TARGET_MIN'):
    """Report transparent slice sizes/errors; no slice is a tuning target."""
    frame = rows.copy(); frame['PREDICTION'] = prediction
    frame['ABS_ERROR'] = (frame.PREDICTION - frame[target]).abs()
    definitions = {
        'age_under_25': frame.AGE_NEXT_OCT < 25,
        'age_25_to_29': frame.AGE_NEXT_OCT.between(25, 29.999),
        'age_30_plus': frame.AGE_NEXT_OCT >= 30,
        'low_minutes': frame[minute_column] < 600,
        'regular_minutes': frame[minute_column] >= 600,
    }
    return {name: {'players': int(mask.sum()), 'mae': float(frame.loc[mask, 'ABS_ERROR'].mean()) if mask.any() else None}
            for name, mask in definitions.items()}
