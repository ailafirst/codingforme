"""Connection pool bookkeeping."""


def acquire(pool):
    return pool.pop() if pool else None


def release(pool, item):
    pool.append(item)
    return pool
