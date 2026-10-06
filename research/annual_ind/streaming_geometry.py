"""The accepted full-objective eight-update fit, using repeatable bounded inputs."""
import hashlib
import time
import numpy as np
from feature_geometry.features import project_scales
from feature_geometry.training import _solve
from annual_ind.numerics import ordered_months
from annual_ind.inputs import check_sample


class GeometryObjective:
    def __init__(self,factory,count,fmap,settings,*,workers,guard):
        if type(count) is not int or count<=0:raise ValueError('Explicit training count required')
        self.factory,self.count,self.fmap,self.settings=factory,count,fmap,settings
        self.workers,self.guard=workers,guard;self.identity=None;self.neutral=None
        self.configuration=(fmap.fingerprint,settings.cutoff,count)

    def checked(self):
        if self.configuration!=(self.fmap.fingerprint,self.settings.cutoff,self.count):
            raise ValueError('Frozen geometry context changed')
        previous=None;count=0;digest=hashlib.sha256()
        for b in self.factory():
            self.guard();s=b.sample;m=s.market
            check_sample(s)
            if (s.available_at>self.settings.cutoff or
                    (previous is not None and m.decision_date<=previous) or
                    m.characteristics[0].shape[1]!=self.fmap.dimensions):
                raise ValueError('Invalid training chronology/dictionary')
            previous=m.decision_date;count+=1;digest.update(s.fingerprint.encode());yield b
        identity=digest.hexdigest()
        if count!=self.count or (self.identity is not None and self.identity!=identity):
            raise ValueError('Training count/source changed')
        self.identity=identity

    def evaluate(self,theta):
        p=self.fmap.columns
        if np.count_nonzero(theta)==0 and self.neutral is not None:
            # Recheck every sample even when only the fitted ridge/penalty changes.
            for _ in self.checked():pass
            score,risk,cost=self.neutral
        else:
            score,risk,cost=np.zeros(p),np.zeros((p,p)),np.zeros((p,p))
            def one(b):
                r=b.matrices(self.fmap,theta)
                return r['score'],r['risk'],r['cost']
            for ss,rr,cc in ordered_months(one,self.checked(),workers=self.workers,guard=self.guard):
                score+=ss;risk+=rr;cost+=cc
            score,risk,cost=score/self.count,risk/self.count,cost/self.count
            if np.count_nonzero(theta)==0:
                for a in (score,risk,cost):a.setflags(write=False)
                self.neutral=(score,risk,cost)
        beta,residual=_solve(risk+cost,score,self.settings.ridge)
        value=score@beta-.5*beta@(risk+cost+self.settings.ridge*np.eye(p))@beta-.5*self.settings.scale_penalty*theta@theta
        return dict(value=float(value),beta=beta,score=score,risk=risk,cost=cost,solve_residual=residual)

    def gradient(self,theta,beta):
        grad=-self.settings.scale_penalty*theta.copy();count=self.count;fmap=self.fmap
        def one(b):
            m=b.sample.market
            vector=[fmap.signals(z,v,theta)@beta for z,v in zip(m.characteristics,m.volatility)]
            v,u=b.operators(vector)
            gv=(b.sample.returns-m.gamma*(m.covariance@v))/count
            gu=-m.wealth*m.impact*u/count
            return [fmap.vjp(z,vol,theta,np.outer(adj,beta)) for z,vol,adj in zip(m.characteristics,m.volatility,b.pullback(gv,gu))]
        for pieces in ordered_months(one,self.checked(),workers=self.workers,guard=self.guard):
            for piece in pieces:grad+=piece
        return grad

    def fit(self,objective_scale,progress):
        if not np.isfinite(objective_scale) or objective_scale<=0:
            raise ValueError('Finite positive objective scale required')
        settings=self.settings;start=time.monotonic();theta=np.zeros(self.fmap.dimensions)
        current=self.evaluate(theta);evaluations=1
        trace=[dict(iteration=0,value=current['value'])];status='ITERATION_BUDGET_REACHED'
        for it in range(1,settings.max_steps+1):
            raw_grad=self.gradient(theta,current['beta']);grad=raw_grad/objective_scale
            pg=float(np.max(abs(project_scales(theta+grad,settings.bound)-theta)))
            if pg<=settings.gradient_tolerance:status='PROJECTED_GRADIENT_TOLERANCE';break
            step=settings.initial_step/max(float(np.max(abs(grad))),1e-30);accepted=False
            for bt in range(settings.max_backtracks):
                th=project_scales(theta+step*grad,settings.bound);dt=th-theta;slope=float(raw_grad@dt)
                if slope<=0:break
                trial=self.evaluate(th);evaluations+=1
                if trial['value']>=current['value']+settings.armijo*slope:accepted=True;break
                step*=.5
            if not accepted:status='LINE_SEARCH_STALLED';break
            theta,current=th,trial
            trace.append(dict(iteration=it,value=current['value'],backtracks=bt,step=step,projected_gradient_before=pg))
            progress(trace[-1])
        return dict(theta=theta,beta=current['beta'],geometry_beta=current['beta'].copy(),status=status,
            trace=trace,profile_evaluations=evaluations,seconds=time.monotonic()-start,
            solve_residual=current['solve_residual'],final_training_value=current['value'])
