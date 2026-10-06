"""Group repeated inventory checks around one complete monthly preparation.

Actual file reads retain the accepted path/hash checks. The full consumed-file
inventory is checked before and after preparation, before a batch can be yielded.
Nested lookback requests defer only their redundant full-inventory scans. No
financial arrays, chronological reduction or optimizer settings are changed.
"""
from contextlib import contextmanager
import threading
from annual_ind.cached_inputs import ReusedInputCache


class PreparationFiles:
    def __init__(self, files):
        self._files=files
        self._owner=None
        self._scope_lock=threading.Lock()
        self.deferred_checks=0
        self.completed_batches=0

    def __getattr__(self,name):
        return getattr(self._files,name)

    def _check_owner(self):
        if self._owner is not None and self._owner!=threading.get_ident():
            raise RuntimeError('Monthly preparation cannot share a source-check scope across threads')

    def path(self,relative):
        self._check_owner()
        return self._files.path(relative)

    def unchanged(self):
        self._check_owner()
        if self._owner is None:
            return self._files.unchanged()
        self.deferred_checks+=1

    @contextmanager
    def preparation(self):
        if not self._scope_lock.acquire(blocking=False):
            raise RuntimeError('Nested or concurrent monthly preparation is not permitted')
        self._owner=threading.get_ident()
        try:
            self._files.unchanged()
            try:
                yield
            finally:
                # Also executed when preparation fails; no successful return
                # can escape the final inventory check.
                self._files.unchanged()
        finally:
            self._owner=None
            self._scope_lock.release()
        self.completed_batches+=1


class PreparationCheckedInputCache(ReusedInputCache):
    def get_batch(self,history,day,cutoff,operators):
        if not isinstance(history.files,PreparationFiles):
            raise TypeError('Explicit before/after preparation source verifier required')
        with history.files.preparation():
            result=super().get_batch(history,day,cutoff,operators)
        return result
