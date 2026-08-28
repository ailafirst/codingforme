"""Elapsed-time bookkeeping."""


def elapsed(start, now):
    return max(0, now - start)
