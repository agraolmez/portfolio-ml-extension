"""Bounded layer parallelism; inherited economic arithmetic is unchanged.

The accepted temporal engine fixes admission at four workers in its constructor.
This separate adapter widens that execution bound to twelve before any evaluation,
then retains the engine's configuration, provenance and mutation checks. The
caller must enforce the study memory/time limits. Accepted modules are unedited.
"""
import numpy as np
from annual_ind.numerics_expanded import ordered_months
from temporal_fast_v4.deferred import OrderedObjective as AcceptedObjective


class ParallelTemporalObjective(AcceptedObjective):
    def __init__(self, *args, workers, **kwargs):
        if type(workers) is not int or workers not in (1, 2, 4, 8, 12):
            raise ValueError('Explicit one/two/four/eight/twelve worker bound required')
        super().__init__(*args, workers=min(workers, 4), **kwargs)
        self.workers = workers
        self._execution_identity = (workers, self.trig_backend)


def conditional_sums(items, moments, *, columns, workers, guard):
    """Release bulky designs in workers; reduce small moments in input order."""
    score = np.zeros(columns)
    curvature = np.zeros((columns, columns))
    count = 0

    def one(item):
        row = moments(item)
        return row['score'], row['risk'] + row['cost']

    for ss, cc in ordered_months(one, items, workers=workers, guard=guard):
        if (ss.shape != score.shape or cc.shape != curvature.shape
                or not np.isfinite(ss).all() or not np.isfinite(cc).all()):
            raise ValueError('Finite aligned conditional moments required')
        score += ss
        curvature += cc
        count += 1
    if not count:
        raise ValueError('Nonempty conditional history required')
    return score/count, curvature/count
