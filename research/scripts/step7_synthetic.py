"""Independent algebra/boundary checks before historical confidence fitting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from confidence.core import moving_blocks, replica_seed, fit_coefficients, state_features
from confidence.pipeline import OUT
from baseline.io import PROTOCOL, save_json


def main():
    if (OUT / 'synthetic_checks.json').exists():
        raise ValueError('Preserve the existing synthetic check receipt')
    checks = []
    def test(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
        if not condition: raise AssertionError(name)
    def rejects(name, call):
        try: call()
        except (ValueError, np.linalg.LinAlgError): test(name, True)
        else: test(name, False)
    seed, digest = replica_seed(PROTOCOL, 12, '1980-12-31', 1)
    test('Deterministic subseed replay', (seed,digest)==replica_seed(PROTOCOL,12,'1980-12-31',1))
    test('Block component separates seeds', seed != replica_seed(PROTOCOL,24,'1980-12-31',1)[0])
    test('Date and replica separate seeds', seed != replica_seed(PROTOCOL,12,'1981-12-31',1)[0] != replica_seed(PROTOCOL,12,'1980-12-31',2)[0])
    for n, length in [(12,12),(13,12),(25,24),(181,12),(649,24)]:
        st, sample, counts = moving_blocks(n,length,seed)
        independent = []
        for start in st:
            independent.extend(range(int(start),int(start)+length))
        independent = independent[:n]
        test(f'Exact concatenation/truncation {n}/{length}', np.array_equal(sample,independent))
        test(f'Counts and boundaries {n}/{length}', counts.sum()==n and sample.min()>=0 and sample.max()<n and np.array_equal(counts,np.array([independent.count(i) for i in range(n)])))
        test(f'Uniform seeded starts {n}/{length}', np.array_equal(st,np.random.Generator(np.random.PCG64(seed)).integers(0,n-length+1,size=(n+length-1)//length)))
    rejects('Insufficient block history rejected',lambda:moving_blocks(11,12,seed))
    rejects('Invalid replica ID rejected',lambda:replica_seed(PROTOCOL,12,'1980-12-31',101))
    coef,res = fit_coefficients(np.diag([2.,3.]),np.array([6.,8.]),1.)
    test('Exact ridge penalty including first coefficient',np.allclose(coef,[2.,2.]) and res<1e-12)
    rejects('Indefinite zero-penalty moments rejected',lambda:fit_coefficients(np.diag([-1.,1.]),np.ones(2),0.))
    rejects('Nonfinite moments rejected',lambda:fit_coefficients(np.diag([np.nan,1.]),np.ones(2),1.))
    rejects('Negative penalty rejected',lambda:fit_coefficients(np.eye(2),np.ones(2),-1.))
    rng=np.random.default_rng(773)
    for k in range(20):
        n=5; x=rng.normal(size=(n,n)); sigma=x.T@x+np.eye(n)
        root=np.exp(rng.normal(size=n)); mt=np.full((n,n),.04)+np.eye(n)*.3
        m=mt*root[None,:]/root[:,None]
        h=rng.normal(size=n); central=rng.normal(size=n); replicas=rng.normal(size=(n,100)); impact=np.ones(n)
        result,q0,qbar=state_features(central,replicas,h,m,sigma,impact,100.)
        q=np.column_stack([(np.eye(n)-m)@(replicas[:,b]-h) for b in range(100)])
        mean=q.sum(axis=1)/100
        num=sum(float((q[:,b]-mean)@sigma@(q[:,b]-mean)) for b in range(100))/100
        expected=num/max(float(mean@sigma@mean),1e-12)
        test(f'Independent population quadratic formula {k}',np.isclose(result['dispersion'],expected,rtol=1e-12))
        test(f'Original central aim retained {k}',np.allclose(q0,(np.eye(n)-m)@(central-h)))
        shifted,_,_=state_features(central,replicas,h+2,m,sigma,impact,100.)
        test(f'Own-state denominator changes; numerator invariant {k}',np.isclose(shifted['dispersion_numerator'],num,rtol=1e-12) and not np.isclose(shifted['mean_trade_risk'],result['mean_trade_risk']))
        scaled,_,_=state_features(3*central,3*replicas,3*h,m,sigma,impact,100.)
        test(f'Common position scale leaves D unchanged {k}',np.isclose(scaled['dispersion'],expected,rtol=1e-12))
        translated,_,_=state_features(central+2,replicas+2,h+2,m,sigma,impact,100.)
        test(f'Common aim/holding translation preserves D {k}',np.isclose(translated['dispersion'],expected,rtol=1e-12))
    zeros=np.zeros(2); eye=np.eye(2); pairs=np.tile(np.array([[1.,-1.],[2.,-2.]]),(1,50))
    f,_,_=state_features(zeros,pairs,zeros,np.zeros((2,2)),eye,np.ones(2),1.)
    test('Zero ensemble trade flags denominator floor with positive dispersion',f['denominator_floor'] and f['dispersion']==5e12 and f['alignment_floor'])
    f,_,_=state_features(np.ones(2),np.zeros((2,100)),zeros,np.zeros((2,2)),eye,np.ones(2),1.)
    test('Identical replicas yield zero D despite a different central aim',f['dispersion']==0 and f['trade_risk']==2)
    rejects('Missing replica rejected',lambda:state_features(zeros,pairs[:,:99],zeros,eye,eye,np.ones(2),1.))
    save_json(OUT/'synthetic_checks.json',dict(status='PASS',checks=checks,count=len(checks),scope='Synthetic algebra only; no historical portfolio or inference'))
    print(f'{len(checks)} synthetic checks PASS.')


if __name__=='__main__': main()
