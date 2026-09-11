"""Dated roster, transaction, and availability data contract.

Events are usable at a forecast cutoff only when their known-at timestamp is
strictly earlier than the cutoff.  This intentionally rejects inferred roster
history (for example, a player's first later box-score appearance).
"""
import pandas as pd

REQUIRED = ('PLAYER_ID', 'TEAM_ID', 'EVENT_TYPE', 'EFFECTIVE_DATE', 'KNOWN_AT',
            'SOURCE_ID', 'RETRIEVED_AT', 'STATUS', 'CONFIDENCE')
EVENT_TYPES = {'TRADE_IN', 'TRADE_OUT', 'SIGNING', 'WAIVER', 'INACTIVE', 'RETURN', 'UNKNOWN'}


def validate_events(events):
    result = events.copy()
    missing = set(REQUIRED) - set(result)
    if missing or result[list(REQUIRED)].isna().any().any():
        raise ValueError(f'Incomplete roster-event contract: {sorted(missing)}')
    if not set(result.EVENT_TYPE).issubset(EVENT_TYPES):
        raise ValueError('Unsupported roster event type')
    for column in ('EFFECTIVE_DATE', 'KNOWN_AT', 'RETRIEVED_AT'):
        result[column] = pd.to_datetime(result[column], errors='raise').dt.normalize()
    if result.duplicated(['PLAYER_ID', 'TEAM_ID', 'EVENT_TYPE', 'EFFECTIVE_DATE', 'SOURCE_ID']).any():
        raise ValueError('Duplicate roster event')
    return result.sort_values(['KNOWN_AT', 'EFFECTIVE_DATE', 'PLAYER_ID']).reset_index(drop=True)


def reconstruct_roster(events, cutoff):
    """Return eligible memberships and a per-player audit trail at cutoff.

    Unknown events, same/after-cutoff announcements, and conflicting active
    memberships fail closed rather than silently borrowing future knowledge.
    """
    cutoff = pd.Timestamp(cutoff).normalize()
    events = validate_events(events)
    visible = events[events.KNOWN_AT < cutoff]
    rows = []
    for player_id, history in visible.groupby('PLAYER_ID', sort=False):
        history = history[history.EFFECTIVE_DATE < cutoff].sort_values(['EFFECTIVE_DATE', 'KNOWN_AT'])
        if history.empty:
            continue
        if (history.EVENT_TYPE == 'UNKNOWN').any():
            raise ValueError(f'Unknowable roster history for player {player_id}')
        active = set()
        inactive = False
        for event in history.itertuples():
            if event.EVENT_TYPE in ('TRADE_IN', 'SIGNING', 'RETURN'):
                active.add(event.TEAM_ID)
                inactive = False
            elif event.EVENT_TYPE in ('TRADE_OUT', 'WAIVER'):
                active.discard(event.TEAM_ID)
            elif event.EVENT_TYPE == 'INACTIVE':
                inactive = True
        if len(active) > 1:
            raise ValueError(f'Ambiguous active teams for player {player_id}')
        rows.append(dict(PLAYER_ID=player_id, TEAM_ID=next(iter(active), None), INCLUDED=bool(active) and not inactive,
                         STATUS='inactive' if inactive else ('active' if active else 'not_rostered'),
                         LAST_EVENT=history.EVENT_TYPE.iloc[-1], SOURCE_ID=history.SOURCE_ID.iloc[-1],
                         KNOWN_AT=history.KNOWN_AT.iloc[-1], CUTOFF=cutoff))
    audit = pd.DataFrame(rows)
    return audit[audit.INCLUDED].reset_index(drop=True), audit
