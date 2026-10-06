"""Post-hoc, paired E4 versus fixed512 comparison from accepted saved paths.

No model fitting or scientific-setting changes. Paper utility definition;
full 1981–2020 period is descriptive, not an untouched evaluation sample.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,json,subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baseline.io import sha,save_json
from inference.core import stationary,basic_interval
OUT=ROOT/'output/extension4_baseline_comparison_20260928'

def main():
    OUT.mkdir(exist_ok=False)
    parents=[('output/feature_geometry_study_20260914/acceptance_manifest.json','output/feature_geometry_study_20260914/paths/monthly.parquet','fixed512'),
        ('output/extension4_full_study_20260924/acceptance_manifest.json','output/extension4_full_study_20260924/final_monthly/monthly.parquet','C4')]
    plan=dict(status='EXPLORATORY_SAVED_PATH_COMPARISON',period=['1981-01-31','2020-12-31'],seeds=[1,2,3],
        methods=['fixed512','C4'],utility='12*(mean(net excess)-5*sample variance(gross excess))',
        bootstrap='paired stationary circular, same dates across all methods and seeds; mean within-seed utility difference',
        expected_block_months=[12,24],draws=10000,master_seed=20260924,alpha=.05,family=1,
        interval='basic, nominal 95%, unadjusted for exploratory multiple testing',
        no_refitting=True,no_new_financial_paths=True,full_period_is_post_selected=True,
        existing_study_protocol_unchanged=True,source_sha256={p:sha(ROOT/p) for pair in parents for p in pair[:2]})
    save_json(OUT/'plan.json',plan)
    blocks=[]
    for manifest,path,method in parents:
        accepted=json.loads((ROOT/manifest).read_text())
        assert sha(ROOT/path)==accepted['artifact_sha256'][path]
        d=pd.read_parquet(ROOT/path);d.return_eom=pd.to_datetime(d.return_eom)
        d=d[d.method.eq(method)&d.return_eom.between(*plan['period'])]
        blocks.append(d)
    dates=pd.date_range(*plan['period'],freq='ME');specs=[(seed,method) for seed in plan['seeds'] for method in plan['methods']]
    frame=pd.concat(blocks,ignore_index=True);gross=[];net=[]
    for seed,method in specs:
        d=frame[frame.seed.eq(seed)&frame.method.eq(method)].sort_values('return_eom')
        assert list(d.return_eom)==list(dates)
        np.testing.assert_allclose(d.net_excess,d.gross_excess-d.impact_cost,atol=1e-15,rtol=0)
        gross.append(d.gross_excess.to_numpy());net.append(d.net_excess.to_numpy())
    gross=np.column_stack(gross);net=np.column_stack(net)
    parent=pd.read_parquet(ROOT/'output/step6_baseline_1981_2020/monthly_performance.parquet')
    np.testing.assert_allclose(gross[:,0],parent.gross_excess,atol=1e-8,rtol=1e-6)
    np.testing.assert_allclose(net[:,0],parent.net_excess,atol=1e-8,rtol=1e-6)
    utility=12*(net.mean(axis=0)-5*gross.var(axis=0,ddof=1));delta=utility[1::2]-utility[::2];observed=float(delta.mean())
    rows=[dict(seed=seed,method=method,annual_source_utility=float(utility[i]),annual_net_excess=float(12*net[:,i].mean())) for i,(seed,method) in enumerate(specs)]
    pd.DataFrame(rows).to_csv(OUT/'seed_metrics.csv',index=False)
    np.asarray(gross,dtype='<f8').tofile(OUT/'gross.bin');np.asarray(net,dtype='<f8').tofile(OUT/'net.bin')
    rcode="""args<-commandArgs(TRUE);out<-args[1];block<-args[2]
g<-matrix(readBin(file.path(out,'gross.bin'),numeric(),n=480*6,size=8,endian='little'),nrow=480,byrow=TRUE)
n<-matrix(readBin(file.path(out,'net.bin'),numeric(),n=480*6,size=8,endian='little'),nrow=480,byrow=TRUE)
idx<-matrix(readBin(file.path(out,paste0('indices_L',block,'.bin')),integer(),n=10000*480,size=4,endian='little'),nrow=10000,byrow=TRUE)
score<-function(ii){u<-12*(colMeans(n[ii,,drop=FALSE])-5*apply(g[ii,,drop=FALSE],2,var));mean(u[c(2,4,6)]-u[c(1,3,5)])}
point<-score(1:480);draw<-vapply(1:10000,function(b)score(idx[b,]),numeric(1))
writeBin(c(point,2*point-as.numeric(quantile(draw,c(.975,.025),type=7)),draw),file.path(out,paste0('R_L',block,'.bin')),size=8,endian='little')
"""
    (OUT/'verify.R').write_text(rcode);intervals=[]
    for length in plan['expected_block_months']:
        indices,_,_,_=stationary('e4_fixed512_posthoc_paper_utility_20260928',plan['master_seed'],length,plan['period'][1],480,replicas=10000)
        np.asarray(indices+1,dtype='<i4').tofile(OUT/f'indices_L{length}.bin')
        draws=np.empty(10000)
        for i in range(0,10000,250):
            idx=indices[i:i+250];u=12*(net[idx].mean(axis=1)-5*gross[idx].var(axis=1,ddof=1))
            draws[i:i+250]=(u[:,1::2]-u[:,::2]).mean(axis=1)
        lower,upper=basic_interval(observed,draws,alpha=.05,family=1)
        subprocess.run(['/usr/local/bin/Rscript',str(OUT/'verify.R'),str(OUT),str(length)],check=True)
        reference=np.fromfile(OUT/f'R_L{length}.bin',dtype='<f8')
        np.testing.assert_allclose(reference,np.r_[observed,lower,upper,draws],atol=1e-11,rtol=1e-10)
        np.savez_compressed(OUT/f'draws_L{length}.npz',utility_differences=draws)
        intervals.append(dict(block_months=length,observed=observed,lower=lower,upper=upper,excludes_zero=bool(lower>0 or upper<0),
            independent_R_max_error=float(np.max(abs(reference-np.r_[observed,lower,upper,draws])))))
    result=dict(status='PASS',months=480,seeds=3,baseline_utility=float(utility[::2].mean()),extension4_utility=float(utility[1::2].mean()),
        mean_difference=observed,seed_differences=delta.tolist(),all_seed_differences_positive=bool((delta>0).all()),intervals=intervals,
        statistically_significant_at_nominal_95=bool(all(x['excludes_zero'] for x in intervals)),
        inference_scope='Exploratory conditional inference on saved fitted paths; not a prespecified or untouched full-period evaluation',
        no_models_refitted=True,parent_hashes_verified=True,seed1_baseline_matches_accepted_step6=True,
        actual_R_draws_verified=20000)
    save_json(OUT/'result.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
