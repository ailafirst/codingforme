# SPDX: internal
"""Tiny in-memory cache used by the sample package."""


def get(store, key):
    return store.get(key)


def put(store, key, value):
    store[key] = value
    return store
