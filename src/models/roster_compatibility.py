"""Interpretable, non-causal player-to-roster compatibility summaries."""
import numpy as np
import pandas as pd

REQUIRED = ('PTS_36', 'AST_36', 'FG3A_36', 'REB_36', 'TOV_36', 'BLK_36', 'EFG')


def compatibility(incoming, destination):
    """Return transferable interactions; callers must exclude incoming player.

    These are estimated model contributions under stated roster assumptions,
    never causal estimates of a trade's effect.
    """
    if set(REQUIRED) - set(incoming) or set(REQUIRED) - set(destination):
        raise ValueError('Compatibility requires complete aggregate profiles')
    if not np.isfinite(incoming[list(REQUIRED)].to_numpy(dtype=float)).all() or not np.isfinite(destination[list(REQUIRED)].to_numpy(dtype=float)).all():
        raise ValueError('Compatibility profiles must be finite')
    return pd.DataFrame({
        'FIT_CREATE_SPACE': incoming.AST_36.to_numpy() * destination.FG3A_36.to_numpy(),
        'FIT_SHOOTING_EFFICIENCY': incoming.FG3A_36.to_numpy() * destination.EFG.to_numpy(),
        'FIT_REBOUND_PROTECTION': incoming.REB_36.to_numpy() * destination.BLK_36.to_numpy(),
        'FIT_TURNOVER_BURDEN': incoming.TOV_36.to_numpy() * destination.TOV_36.to_numpy(),
    })


def scenario_label():
    return 'Estimated model contribution under stated roster assumptions; not a causal trade estimate.'
