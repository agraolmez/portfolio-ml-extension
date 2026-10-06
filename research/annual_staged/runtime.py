"""Persistent status wrapper around the audited resource watchdog."""
import time
from annual_ind.runtime_expanded import Runtime as Original, prior_seconds
from .common import atomic,now

class Runtime(Original):
    def __init__(self,folder,config,phase):
        super().__init__(folder,config,phase);self._display={}
    def event(self,event,**values):
        super().event(event,**values)
        if event=='phase_progress':
            self._display.update({k:values[k] for k in ('label','completed','total','year') if k in values})
        atomic(self.folder/'live.json',dict(at_utc=now(),status='RUNNING',event=event,phase=self.phase,
            **self._display,charged_seconds=self.previous+time.monotonic()-self.start))
    def close(self,status,error=None):
        super().close(status,error)
        charged,_=prior_seconds(self.folder)
        atomic(self.folder/'resource_ledger.json',dict(charged_seconds=charged,cumulative_engineering_cap_seconds=self.budget['total_active_seconds']))
        atomic(self.folder/'live.json',dict(at_utc=now(),status=status,error=None if error is None else str(error),**self._display,charged_seconds=charged))
