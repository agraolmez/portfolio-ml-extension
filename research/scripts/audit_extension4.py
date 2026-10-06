"""Read-only parent/raw-source audit; no state selection or economic fitting."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from extension4.parent import ParentCatalog,PARENT


def main():
    out=ROOT/'output/extension4_implementation_20260924'
    destination=out/'audit.json'
    if destination.exists():raise FileExistsError('Preserve the completed audit')
    cat=ParentCatalog(ROOT)
    for rel in cat.inventory:cat.path(rel)
    print('Accepted Extension-3 inventory verified',len(cat.consumed),flush=True)
    records=[]
    for year in range(1981,2026):
        for seed in (1,2,3):
            p=cat.load(year,seed)
            records.append(dict(year=year,seed=seed,path=p.model_path,sha256=cat.consumed[p.model_path],
                fit_cutoff=p.model.cutoff,selection_available_at=p.selection_available_at,
                nonlinear_features=p.model.feature_map.parent.columns-1,base_coefficients=len(p.model.beta),
                ridge=p.settings.ridge,temporal_penalty=p.receipt['rho'],
                parent_scale_status=p.receipt['parent_scale_optimizer_status']))
    pd.DataFrame(records).to_csv(out/'selected_parents.csv',index=False)
    raw_rel='data/prepared/step3_author_v3/source_characteristics.parquet'
    rank_rel='data/prepared/step3_author_v3/monthly_inputs.parquet'
    fields=['be_me','op_at','at_gr1']
    raw=pd.read_parquet(cat.path(raw_rel),columns=['id','eom','me',*fields])
    ranks=pd.read_parquet(cat.path(rank_rel),columns=['id','eom','valid',*fields])
    if raw.duplicated(['id','eom']).any() or ranks.duplicated(['id','eom']).any():
        raise ValueError('Duplicate source keys')
    active=ranks.loc[ranks.valid,['id','eom']].merge(raw,on=['id','eom'],validate='one_to_one',indicator=True)
    if not active._merge.eq('both').all():raise ValueError('Missing raw source keys for parent universe')
    rows=[]
    for day,group in active.groupby('eom',sort=True):
        for field in fields:
            values=group[field].to_numpy()
            finite=np.isfinite(values)
            # Positive book-to-market is a diagnostic screen proposed for later approval.
            valid=finite & ((values>0) if field=='be_me' else True) & (group.me.to_numpy()>0)
            rows.append(dict(eom=str(day.date()),field=field,universe=len(group),finite_count=int(finite.sum()),
                proposed_valid_count=int(valid.sum()),proposed_coverage=float(valid.mean()),
                raw_min=float(values[finite].min()) if finite.any() else None,
                raw_max=float(values[finite].max()) if finite.any() else None,
                nonpositive_count=int((finite & (values<=0)).sum())))
    monthly=pd.DataFrame(rows);monthly.to_csv(out/'raw_source_coverage.csv',index=False)
    summaries=[]
    for field in fields:
        g=monthly[monthly.field.eq(field)];f=g[g.eom.between('1980-12-31','2025-11-30')]
        summaries.append(dict(field=field,minimum_valid_count=int(f.proposed_valid_count.min()),
            minimum_coverage=float(f.proposed_coverage.min()),months=len(f),
            below_proposed_100_count_or_70_percent=int(((f.proposed_valid_count<100)|(f.proposed_coverage<.7)).sum()),
            source_min=float(g.raw_min.min()),source_max=float(g.raw_max.max())))
    sas=ROOT/'output/step3_20260910/jkp_source/GlobalFactors/main.sas'
    schema=pq.ParquetFile(ROOT/raw_rel).schema_arrow.names
    result=dict(status='AUDITED_DECISIONS_PENDING',checked_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_manifest=PARENT+'/acceptance_manifest.json',parent_acceptance_sha256=cat.acceptance_sha256,
        parent_inventory_files_verified=len(cat.consumed),selected_parent_count=len(records),
        parent_nonlinear_features=512,parent_coefficient_count=513,constant_column_index=0,
        parent_penalty='ridge times identity on all 513 coefficients, including constant; temporal rho is separate',
        selection_available_at=cat.selection['selection_available_at'],
        state_source=dict(path=raw_rel,origin='WRDS contrib.global_factor; retained September 2026 captures',
            rows=len(raw),columns=len(schema),first=str(raw.eom.min().date()),last=str(raw.eom.max().date()),
            duplicate_security_months=0,active_rows=len(active),
            rank_provenance='scripts/step3_author_bundle.py saves this table before applying monthly empirical ranks',
            fields_profiled=summaries,publication_or_revision_date_fields=[n for n in schema if any(s in n.lower() for s in ['public','filing','available','revision','datadate','rdq'])],
            timing='latest-vintage source values with documented construction lag; not certified point-in-time',
            retained_source_lag='GlobalFactors/main.sas lines 92–93: lag_to_public=4, max_data_lag=18',
            actual_accounting_age_observable=False,source_pit_certified=False),
        conflicts=[
            'Package calls for PIT magnitudes; magnitudes exist but filing dates/vintage revisions are absent. An explicitly approved source-lag-proxy scope is needed.',
            'Selected parent rho was chosen using 2001–2010, so that period cannot be reused as legal ex-ante validation for Extension 4 without parent reselection.',
            'Parent drops disappearing IDs without separate liquidation orders/cost. Charging invented exit orders would change protected parent accounting and break neutral identity.'
        ],
        integration=dict(features='temporal_extension.features.TemporalFeatureMap.signals; constant first',
            dynamics='geometry_study.kernel.LeanBasis.operators / audited 13-slice, 12-term finite-history operator',
            coefficients='feature_geometry.training._solve; population date-averaged moments and actual ridge',
            execution='baseline.core.transition; continuous own holdings and actual modeled trades',
            new_modules=['extension4.parent','extension4.features','extension4.estimation']),
        audit_inputs_sha256={str(sas.relative_to(ROOT)):hashlib.sha256(sas.read_bytes()).hexdigest(),
            **{k:cat.consumed[k] for k in [raw_rel,rank_rel,PARENT+'/acceptance_manifest.json']}},
        financial_experiment_run=False,full_historical_run_approved=False)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','selected_parent_count','selection_available_at','conflicts']},indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
