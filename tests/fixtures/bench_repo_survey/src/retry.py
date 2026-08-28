"""Retry helpers with a fixed backoff schedule."""

BACKOFF = [1, 2, 4, 8]


def delay_for(attempt):
    if attempt >= len(BACKOFF):
        return BACKOFF[-1]
    return BACKOFF[attempt]
