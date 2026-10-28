"""Prospectively frozen scene bootstrap arithmetic; no model/data dependencies."""
from fractions import Fraction
from functools import reduce
import math
import numpy as np

SEEDS=(10,11,12)
QUANTILES=(Fraction(1,40),Fraction(39,40),Fraction(1,80),Fraction(79,80))

def multiplicities(scene_ids,rng,draws=20000):
    n=len(scene_ids)
    samples=rng.integers(0,n,size=(draws,n))
    counts=np.zeros((draws,n),dtype=np.int16)
    for i,row in enumerate(samples):counts[i]=np.bincount(row,minlength=n)
    assert (counts.sum(1)==n).all()
    return counts

def exact_percentile(values,denominator,q):
    """NumPy's linear empirical percentile, with rational event arithmetic."""
    a=np.sort(np.asarray(values,dtype=np.int64));q=Fraction(q)
    position=q*(len(a)-1);left=position.numerator//position.denominator
    right=min(left+1,len(a)-1);weight=position-left
    return (Fraction(int(a[left]),denominator)*(1-weight)+Fraction(int(a[right]),denominator)*weight)

def unsafe_bootstrap(scene_event_difference,scene_decisions,matrix):
    """Input counts sum paired mask differences over the three seeds/slots."""
    counts=np.asarray(scene_decisions,dtype=np.int64)
    assert (counts>0).all()
    common=reduce(math.lcm,map(int,counts),1)
    scaled=np.asarray(scene_event_difference,dtype=np.int64)*(common//counts)
    numerators=matrix.astype(np.int64)@scaled
    denominator=common*len(counts)
    estimate=Fraction(int(scaled.sum()),denominator)
    intervals=[exact_percentile(numerators,denominator,q) for q in QUANTILES]
    return estimate,intervals,numerators,denominator

def continuous_bootstrap(scene_values,matrix):
    values=np.asarray(scene_values,dtype=np.float64)
    assert np.isfinite(values).all()
    replicates=(matrix.astype(np.float64)@values)/len(values)
    bounds=np.quantile(replicates,[float(q) for q in QUANTILES],method='linear')
    return float(values.mean(dtype=np.float64)),bounds,replicates

def direction(low,high):
    if high<0:return 'supported lower loss'
    if low>0:return 'opposite-direction evidence'
    return 'inconclusive'

def class_statistics(confusion):
    c=np.asarray(confusion,dtype=np.int64);tp=c.diagonal();fn=c.sum(1)-tp;fp=c.sum(0)-tp
    return [dict(class_id=i,tp=int(tp[i]),fp=int(fp[i]),fn=int(fn[i]),truth_support=int(c[i].sum()),
        predicted_support=int(c[:,i].sum()),union=int(tp[i]+fp[i]+fn[i]),
        iou=None if tp[i]+fp[i]+fn[i]==0 else float(tp[i]/(tp[i]+fp[i]+fn[i]))) for i in range(len(c))]
