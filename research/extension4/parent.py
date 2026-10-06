"""Accepted selected Extension-3 artifacts; no parent reconstruction or edits."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from feature_geometry.history import sha
from feature_geometry.policy import GeometryModel
from feature_geometry.training import FitSettings
from temporal_extension.model import TemporalModel

PARENT = 'output/extension3_full_study_20260919'


@dataclass(frozen=True)
class SelectedParent:
    year: int
    seed: int
    model: TemporalModel
    settings: FitSettings
    receipt: dict
    model_path: str
    selection_available_at: str

    def assert_legal_selection(self, decision_date):
        if pd.Timestamp(decision_date) < pd.Timestamp(self.selection_available_at):
            raise ValueError('Selected Extension-3 penalty was not known by this decision')


class ParentCatalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.consumed = {}
        self._stats = {}
        self._models = {}
        path = self.root/PARENT/'acceptance_manifest.json'
        self.acceptance_sha256 = sha(path)
        accepted = json.loads(path.read_text())
        if (accepted.get('study_complete') is not True or accepted.get('accepted') is not True
                or accepted.get('status') != 'ACCEPTED_COMPLETED_HISTORICAL_TEMPORAL_STUDY'
                or not all(g['passed'] is True for g in accepted['gates'])):
            raise ValueError('Accepted completed Extension-3 study required')
        self.inventory = dict(accepted['artifact_sha256'])
        self.inventory[PARENT+'/acceptance_manifest.json'] = self.acceptance_sha256
        self.path(PARENT+'/acceptance_manifest.json')
        self.selection = self.read(PARENT+'/selection_canonical_v1/temporal.json')
        self.lock = self.read(PARENT+'/study_lock.json')
        if self.selection['status'] != 'SELECTION_FROZEN':
            raise ValueError('Frozen selected temporal family required')

    def path(self, relative):
        path = (self.root/relative).resolve()
        if not path.is_relative_to(self.root) or relative not in self.inventory:
            raise ValueError('File is outside the accepted parent inventory: '+relative)
        stat = path.stat()
        stamp = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if relative not in self.consumed:
            actual = sha(path)
            if actual != self.inventory[relative]:
                raise ValueError('Accepted parent file changed: '+relative)
            self.consumed[relative] = actual
            self._stats[relative] = stamp
        elif self._stats[relative] != stamp:
            raise ValueError('Parent source changed during this execution: '+relative)
        return path

    def read(self, relative):
        return json.loads(self.path(relative).read_text())

    def load(self, year, seed):
        if type(year) is not int or year not in range(1981,2026) or type(seed) is not int or seed not in (1,2,3):
            raise ValueError('Exact archived year and seed required')
        if (year,seed) in self._models:
            for rel in tuple(self.consumed): self.path(rel)
            return self._models[year,seed]
        choice = [r for r in self.selection['selected_by_seed'] if r['seed']==seed]
        if len(choice)!=1 or choice[0]['candidate']!='temporal__rho_0p1':
            raise ValueError('Unexpected selected Extension-3 configuration')
        folder=f'{PARENT}/fits/{year}/seed{seed}/temporal/rho_0p1'
        receipt=self.read(folder+'/fit.json')
        commit=self.read(folder+'/fit_commit.json')
        if commit['status']!='COMPLETE_ARTIFACT': raise ValueError('Incomplete parent fit')
        for rel,expected in commit['artifact_sha256'].items():
            self.path(rel)
            if self.consumed[rel]!=expected: raise ValueError('Parent fit commit differs')
        model_path=folder+'/model.npz'
        model=TemporalModel.load(self.path(model_path),expected_parent_fingerprint=receipt['parent_fingerprint'])
        settings=FitSettings(**receipt['parent_settings'])
        expected=dict(year=year,seed=seed,kind='temporal',rho=.1,candidate='rho_0p1')
        if (receipt['fit_spec']!=expected or receipt['converged'] is not True
                or receipt['eligible_for_portfolio_evaluation'] is not True
                or receipt['neutral_gate']['passed'] is not True
                or receipt['solve_residual']>1e-9
                or model.cutoff!=f'{year-1}-12-31' or settings.cutoff!=model.cutoff
                or model.feature_map.parent.columns!=513):
            raise ValueError('Selected temporal model/receipt/cutoff mismatch')
        np.testing.assert_array_equal(model.gains,receipt['gains'])
        np.testing.assert_array_equal(model.beta,receipt['beta'])
        geo_folder=f'output/feature_geometry_study_20260914/fits/{year}/seed{seed}/full'
        geo=GeometryModel.load(self.path(geo_folder+'/model.npz'))
        meta=self.read(geo_folder+'/fit.json')
        np.testing.assert_array_equal(model.feature_map.theta,geo.theta)
        np.testing.assert_array_equal(model.feature_map.parent.frequencies,geo.feature_map.frequencies)
        if settings.ridge != meta['coefficient_ridge']:
            raise ValueError('Base penalty differs from accepted normal-equation penalty')
        result=SelectedParent(year,seed,model,settings,receipt,model_path,self.selection['selection_available_at'])
        self._models[year,seed]=result
        return result
