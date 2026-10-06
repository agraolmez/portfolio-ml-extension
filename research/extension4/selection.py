"""Selection observes only the explicitly approved selection window."""
import numpy as np
import pandas as pd
from inference.core import utility


def candidate(method, multiplier): return method+'__m'+format(multiplier,'g')


def select(frame, *, first, last, seeds, multipliers, tie_tolerance, parent_selection_available_at):
    dates=pd.date_range(first,last,freq='ME')
    if pd.Timestamp(first)-pd.offsets.MonthEnd(1)<pd.Timestamp(parent_selection_available_at):
        raise ValueError('Inherited parent choice was unavailable before this selection period')
    result=[]
    for seed in seeds:
        for method in ('C4','CAP4','SIZE4'):
            scores=[]
            for multiplier in multipliers:
                rows=frame[(frame.seed==seed)&(frame.method==candidate(method,multiplier))].copy()
                rows['return_eom']=pd.to_datetime(rows.return_eom)
                rows=rows[rows.return_eom.between(first,last)].sort_values('return_eom')
                if list(rows.return_eom)!=list(dates):raise ValueError('Incomplete or duplicated selection months')
                scores.append((multiplier,float(utility(rows.net_excess.to_numpy()))))
            maximum=max(value for _,value in scores)
            winner=next((m,v) for m,v in scores if maximum-v<=tie_tolerance)
            result.append(dict(seed=seed,method=method,multiplier=winner[0],annual_net_utility=winner[1],
                               scores=[dict(multiplier=m,utility=v) for m,v in scores],
                               available_at=str(pd.Timestamp(last).date())))
    return result
