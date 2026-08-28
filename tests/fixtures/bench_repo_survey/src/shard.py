# SPDX: internal
"""Pick a shard for a key."""


def shard_for(key, count):
    return sum(ord(char) for char in str(key)) % max(1, count)
