"""Export complete moments and economic inputs for actual pinned R validation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from baseline.full import OUT, PREP, DATES, MOMENT_DATES
from baseline.io import DATA, FEATURES, ROOT, save_json, write_matrix, sha


def main():
    ref = OUT / 'reference'
    if (ref / 'export_manifest.json').exists():
        raise ValueError('Completed reference export must be preserved')
    for gi in [0, 1]:
        folder = ref / f'fit/g{gi}'; folder.mkdir(parents=True, exist_ok=True)
        signal_folder = ref / f'signals/g{gi}'; signal_folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'denom.bin').open('wb') as a, (folder / 'rtilde.bin').open('wb') as r:
            for d in MOMENT_DATES:
                with np.load(PREP / f'moments/g{gi}/{d:%Y-%m-%d}.npz') as z:
                    np.asarray(z['denom'], dtype='<f8').tofile(a)
                    np.asarray(z['r_tilde'], dtype='<f8').tofile(r)
                    if d in DATES:
                        write_matrix(signal_folder / f'{d:%Y-%m-%d}.bin', z['signal_t'])
        save_json(folder / 'dates.json', [str(d.date()) for d in MOMENT_DATES])
        print(f'Exported all moments and decision signals: bandwidth {gi}', flush=True)
    dates = pd.date_range(DATES[0] - pd.offsets.MonthEnd(12), DATES[-1], freq='ME')
    cols = ['id','eom','valid','me','ret_ld1','tr_ld0','tr_ld1','mu_ld0','mu_ld1',
            'lambda_diagonal','wealth_usd',*FEATURES]
    data = pd.read_parquet(DATA / 'monthly_inputs.parquet', columns=list(dict.fromkeys(cols)),
        filters=[('eom','>=',dates[0]),('eom','<=',dates[-1])]).sort_values(['eom','id'])
    data['valid'] = data.valid.astype(int)
    data.to_csv(ref / 'data.csv', index=False, float_format='%.17g')
    rates = pd.read_parquet(DATA / 'monthly_rates.parquet')
    rates[rates.eom.isin(dates)].to_csv(ref / 'rates.csv', index=False, float_format='%.17g')
    metadata = []
    for d in dates:
        folder = ref / f'risk/{d:%Y-%m-%d}'; folder.mkdir(parents=True, exist_ok=True)
        with np.load(PREP / f'risk/{d:%Y-%m-%d}.npz') as z:
            ids, x, factor, specific = z['ids'], z['loadings'], z['factor_cov'], z['specific_variance']
        for name, array in [('loadings',x),('factor',factor),('specific',specific)]:
            write_matrix(folder / f'{name}.bin', array)
        save_json(folder / 'ids.json', ids.tolist())
        metadata.append({'date':str(d.date()),'n':len(ids),'k':x.shape[1]})
    samples = [d for d in DATES if d.month in [12,6] or d == DATES[-1]]
    assert len(samples) == 81
    save_json(ref / 'metadata.json', {'decision_dates':[str(d.date()) for d in DATES],
        'sample_input_dates':[str(d.date()) for d in samples], 'risk':metadata,
        'input_rows':len(data),'full_active_rows':int((data.valid.eq(1)&data.eom.isin(DATES)).sum()),
        'feature_count':len(FEATURES),'moment_months':len(MOMENT_DATES),
        'source_function_checks':'All fits and holdings; sampled independent input rebuilds'})
    save_json(ref / 'export_manifest.json', {'status':'EXPORTED',
        'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(ref.rglob('*')) if p.is_file()}})
    print('Complete R interchange export ready.', flush=True)


if __name__ == '__main__': main()
