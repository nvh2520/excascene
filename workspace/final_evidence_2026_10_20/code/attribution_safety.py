"""All 16 truth substitutions and cause-specific safety on immutable saved maps."""
from common import *
import argparse,math,time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

SAVED={};REFERENCE={};GEOMETRIC={}
FACTORS=('height','material','hazard','terrain_semantics')
def safety_at(pickrow,pickcol,masks,feasible,uniform=False):
    if uniform:
        n=int(feasible.sum());raw=np.array([(m&feasible).sum()/n if n else 0 for m in masks],np.float64)
        todo=feasible.copy();exclusive=[]
        for m in masks:
            exclusive.append((todo&m).sum()/n if n else 0);todo&=~m
    elif pickrow<0:
        raw=np.zeros(5);exclusive=np.zeros(5)
    else:
        raw=masks[:,int(pickrow),int(pickcol)].astype(float);exclusive=np.zeros(5)
        if raw.any():exclusive[np.flatnonzero(raw)[0]]=1
    return {**{f'multi_{c}':float(v) for c,v in zip(CAUSES,raw)},**{f'exclusive_{c}':float(v) for c,v in zip(CAUSES,exclusive)}}

# Only the immutable cause-accounting helper is reused in this round.
