"""Bounded chronological work with an explicitly entered native BLAS policy."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from temporal_fast.backend import single_thread_blas


def ordered_months(function, items, *, workers, guard):
    if workers not in (1, 2, 4, 8, 12):
        raise ValueError('Unsupported bounded worker count')

    def one(item):
        with single_thread_blas():
            guard()
            result = function(item)
            guard()
            return result

    if workers == 1:
        for item in items:
            yield one(item)
        return
    source = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(workers):
            try:
                pending.append(pool.submit(one, next(source)))
            except StopIteration:
                break
        while pending:
            result = pending.popleft().result()
            try:
                pending.append(pool.submit(one, next(source)))
            except StopIteration:
                pass
            yield result
