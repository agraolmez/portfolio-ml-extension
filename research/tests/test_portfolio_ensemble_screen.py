"""Synthetic accounting and decision checks; no historical performance reads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('ensemble_screen', ROOT/'scripts/portfolio_ensemble_screen_20260928.py')
s = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(s)


def packet(ids, next_ids, holdings, targets, seed=1):
    rng = np.random.default_rng(seed)
    n = len(ids)
    return dict(ids=np.array(ids), next_ids=np.array(next_ids),
        m=np.diag(np.linspace(.1,.7,n)), targets=s.combine_targets(targets), pretrade=holdings,
        impact=np.linspace(.1,.3,n), wealth=np.array(5.), total_returns=rng.normal(0,.1,n),
        market_growth=np.array(.01), excess_returns=rng.normal(0,.05,n), methods=np.array(s.METHODS))


class ScreenTests(unittest.TestCase):
    def test_entry_exit_own_holdings_and_mixture_identity(self):
        rng = np.random.default_rng(44)
        ids=[10,20,30]; h=np.repeat(np.array([.4,-.2,.1])[:,None],9,axis=1)
        gross=[];cost=[]
        for step, nxt in enumerate(([20,30,40],[30,40,50],[30,50,60],[30,50,60])):
            p=packet(ids,nxt,h,rng.normal(0,.2,(len(ids),6)),step)
            a,rows=s.replay_packet(p)
            # Independent scalar formula, including a paid first trade and entrants.
            w=p['m']@h+(np.eye(len(ids))-p['m'])@p['targets']
            q=w-h; expected_cost=.5*float(p['wealth'])*(p['impact'][:,None]*q*q).sum(axis=0)
            np.testing.assert_allclose(a['weights'],w,atol=1e-14)
            np.testing.assert_allclose([r['impact_cost'] for r in rows],expected_cost,atol=1e-14)
            self.assertGreater(float(expected_cost.sum()),0)
            drift=w*(1+p['total_returns'][:,None])/(1+float(p['market_growth']))
            expected=np.array([drift[ids.index(x)] if x in ids else np.zeros(9) for x in nxt])
            np.testing.assert_allclose(a['next_pretrade'],expected,atol=1e-14)
            gross.append([r['gross_excess'] for r in rows]);cost.append(expected_cost)
            ids=list(nxt);h=a['next_pretrade']
        proof=s.convexity_identity(np.array(gross),np.array(cost))
        self.assertTrue(all(x['utility_gain']>=-1e-13 for x in proof))

    def test_net_return_averaging_misses_netting(self):
        components=np.array([[1.,-1.,0.,2.,-2.,0.]])
        p=packet([1],[1],np.zeros((1,9)),components)
        a,rows=s.replay_packet(p)
        costs=np.array([x['impact_cost'] for x in rows])
        self.assertGreater(costs[:6].mean(), costs[8])
        self.assertAlmostEqual(costs[8],0.)
        # Altering realized returns cannot alter frozen target construction.
        old=a['targets'].copy();p['excess_returns']*=100
        changed,_=s.replay_packet(p)
        np.testing.assert_array_equal(changed['targets'],old)
        np.testing.assert_array_equal(changed['weights'],a['weights'])

    def test_common_initial_state_is_necessary(self):
        h=np.zeros((2,9));h[0,6]=1
        with self.assertRaises(AssertionError):
            s.replay_packet(packet([1,2],[1,2],h,np.ones((2,6))))

    def test_source_risk_variance_is_sample_gross(self):
        gross=np.array([.1,-.1,.05]);cost=np.array([.01,.02,.001]);net=gross-cost
        expected=12*(net.mean()-5*np.sum((gross-gross.mean())**2)/2)
        self.assertAlmostEqual(float(s.source_utility(gross,net)),expected)
        self.assertNotAlmostEqual(float(s.source_utility(gross,net)),12*(net.mean()-5*net.var()))

    def test_gate_requires_both_controls_periods_and_lengths(self):
        metrics=[dict(period=p,method=m,source_utility=(.2 if m in ('CENS','HALF') else .1))
                 for p in ('primary','robustness') for m in s.METHODS]
        intervals=[dict(period='primary',candidate=a,control=b,block_months=l,lower=.01)
                   for a,b in s.CONTRASTS for l in (12,24)]
        yes=s.gate_decision(metrics,intervals)
        self.assertEqual(yes['candidate_for_larger_study_consideration'],'HALF')
        self.assertFalse(yes['automatic_training_authorized'])
        broken=[dict(r) for r in intervals];broken[-1]['lower']=0.
        self.assertEqual(s.gate_decision(metrics,broken)['candidate_for_larger_study_consideration'],'CENS')
        for row in metrics:
            if row['period']=='robustness' and row['method']=='CENS':row['source_utility']=.1
        self.assertFalse(s.gate_decision(metrics,broken)['exploratory_gate_passed'])
        with self.assertRaises(ValueError):s.gate_decision(metrics,intervals[:-1])

    def test_checkpoint_corruption_chain_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'arrays.npz').write_bytes(b'fixture')
            s.save_json(folder/'commit.json',dict(lock_sha256='lock',previous_commit_sha256=None,
                decision_eom='2015-12-31',files_sha256={'arrays.npz':s.sha(folder/'arrays.npz')}))
            s.verify_checkpoint(folder,'lock',None,'2015-12-31')
            with self.assertRaises(ValueError):s.verify_checkpoint(folder,'lock','wrong','2015-12-31')
            with self.assertRaises(FileExistsError):s.save_json(folder/'commit.json',{})
            (folder/'arrays.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):s.verify_checkpoint(folder,'lock',None,'2015-12-31')


if __name__=='__main__':unittest.main()
