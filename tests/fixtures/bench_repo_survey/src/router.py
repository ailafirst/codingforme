# SPDX: internal
"""Route a request name onto a handler key."""


def route(table, name):
    return table.get(name, "default")


def connect(table, name, handler):
    table[name] = handler
    return table
