# SPDX: internal
"""A list-backed FIFO queue."""


def push(items, value):
    items.append(value)
    return items


def pop(items):
    return items.pop(0) if items else None
