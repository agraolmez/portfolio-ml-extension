"""Bounded existing-data replay and independent selection verification; no fit."""
import fcntl
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.full import preflight
from feature_geometry.history import sha
from temporal_extension.history import load_parent
from temporal_study.paths import AcceptedPathSource
from temporal_fast.backend import single_thread_blas
from temporal_fast_v4.history import StudyHistory
from temporal_fast_v4.study import fit_specs, _policies, _json, _verify, write_new
from temporal_continuity.paths import run_paths
from temporal_continuity.study import _select


def main():
    folder = ROOT/'output/extension3_continuity_repair_20260922'
    study = ROOT/'output/extension3_full_study_20260919'
    if (folder/'historical_checks.json').exists():
        raise FileExistsError('Preserve the completed repair verification')
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    with (ROOT/'output/.extension3_heavy_job.lock').open('a+') as heavy, single_thread_blas():
        fcntl.flock(heavy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        config = _json(ROOT/'config/extension3_study_v3_proposed.json')
        lock = _json(study/'study_lock.json'); lock_hash = sha(study/'study_lock.json')
        _verify(ROOT, lock['code_sha256']); _verify(ROOT, lock['authority_sha256'])
        receipt, _ = preflight()
        if receipt != _json(study/'financial_preflight.json'):
            raise ValueError('Financial dependencies changed')
        write_new(folder/'financial_preflight.json', receipt)
        print('Financial preflight passed; checking actual saved portfolios.', flush=True)
        history = StudyHistory(ROOT, receipt, weights=config['reference_weights'])
        source = AcceptedPathSource(history); initial = source.common_initial()
        parents = {(1981,s):load_parent(ROOT,1981,s,expected_cutoff='1980-12-31') for s in config['frequency_seeds']}
        specs = [s for s in fit_specs(config) if s['year']==1981]
        policies = _policies(ROOT, study, config, parents, specs, lock_hash, selected=False)
        dates = pd.date_range('1980-12-31','1981-01-31',freq='ME')
        result = run_paths(source, dates, policies, initial, destination=folder/'bounded_replay',
            save_positions=True, accounting_atol=1e-11,accounting_rtol=1e-10,
            neutral_atol=1e-11,neutral_rtol=1e-10,compare_archived=True,
            progress=lambda item:print(json.dumps(item),flush=True))
        for date in dates:
            name=f'{date.date().isoformat()}.npz'
            with np.load(study/'candidate_paths'/name) as old, np.load(folder/'bounded_replay'/name) as new:
                if old.files != new.files: raise ValueError('Position schema changed')
                for key in old.files:
                    np.testing.assert_array_equal(old[key], new[key],err_msg=f'Actual position change: {key}')
        expected=pd.read_parquet(study/'candidate_paths_canonical_v1/monthly.parquet')
        expected=expected[expected.decision_eom.isin(dates)].reset_index(drop=True)
        actual=result['monthly'].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected.drop(columns='modeled_annual_volatility'),
                                      actual.drop(columns='modeled_annual_volatility'),check_exact=True)
        np.testing.assert_allclose(expected.modeled_annual_volatility,actual.modeled_annual_volatility,
                                   atol=1e-11,rtol=1e-10)
        print('Actual positions and economic results match exactly; verifying selection.',flush=True)
        derivative=study/'candidate_paths_canonical_v1'
        _verify(ROOT,_json(derivative/'stage_commit.json')['artifact_sha256'])
        first_paths=dict(monthly=pd.read_parquet(derivative/'monthly.parquet'),manifest=_json(derivative/'paths_manifest.json'))
        selections=_select(ROOT,study,config,first_paths,initial,lock_hash)
        original=pd.read_parquet(study/'candidate_paths/monthly.parquet')
        score_checks=[]
        for family, selection in selections.items():
            for selected in selection['selected_by_seed']:
                scores=[]
                for rho in config['penalties_in_tie_order']:
                    from temporal_fast_v4.study import candidate_name
                    name=family+'__'+candidate_name(rho)
                    frame=original[(original.seed==selected['seed'])&(original.method==name)
                        &original.return_eom.between(config['selection']['first_return'],config['selection']['last_return'])]
                    if len(frame)!=120:raise ValueError('Selection sample is incomplete')
                    values=frame.net_excess.to_numpy()
                    value=12*(values.mean()-.5*config['selection']['risk_aversion']*values.var(ddof=0))
                    scores.append((name,float(value)))
                winner=max(scores,key=lambda item:item[1])
                if winner[0]!=selected['candidate']:raise ValueError('Independent unchanged-data selection differs')
                np.testing.assert_allclose(winner[1],selected['annual_net_utility'],atol=1e-12,rtol=1e-11)
                score_checks.append(dict(family=family,seed=selected['seed'],candidate=winner[0],passed=True))
        all_specs=fit_specs(config,selections)
        if len(all_specs)!=765 or sum(s['year']>2010 for s in all_specs)!=135:
            raise ValueError('Continuation changes the approved comparison count')
        history.files.unchanged()
        write_new(folder/'historical_checks.json',dict(status='PASS',financial_preflight_unchanged=True,
            replay_months=2,replay_paths=27,all_saved_position_arrays_equal=True,
            all_economic_rows_exactly_equal=True,neutral_position_error=result['manifest']['neutral_maximum_position_error'],
            archive_position_error=result['manifest']['archived_replay_maximum_position_error'],
            selections_match_independent_original_data=score_checks,
            initial_fits_reused=630,remaining_annual_fits=135,total_annual_fits=765,
            original_study_lock_sha256=lock_hash,selection_commit_sha256=sha(study/'selection_canonical_v1/selection_commit.json'),
            elapsed_seconds=time.monotonic()-started,historical_fit_run=False,
            consumed_inputs_sha256=history.files.consumed))
        print('PASS: real-data equivalence, frozen selection and 135 remaining fits.',flush=True)


if __name__=='__main__':main()
