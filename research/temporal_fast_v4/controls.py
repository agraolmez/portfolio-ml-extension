"""Keep the original current-only control transformation inside bounded workers."""
from dataclasses import replace
from temporal_study.controls import current_only_batch_factory as previous_factory
from .deferred import DeferredMonth


def current_only_batch_factory(basis_factory):
    def batches():
        for value in basis_factory():
            if type(value) is DeferredMonth:yield replace(value,current_only=True)
            else:yield next(previous_factory(lambda:iter([value]))())
    return batches
