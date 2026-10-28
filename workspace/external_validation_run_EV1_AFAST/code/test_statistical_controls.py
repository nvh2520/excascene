"""Analytic controls only: no network, historical tests, or holdout outcomes."""
import json,pathlib
from fractions import Fraction
import numpy as np
from statistical_core import *
ROOT=pathlib.Path(__file__).resolve().parents[1]
# Unequal slot counts: equal scenes yields .5; pooling frames would yield .2.
matrix=np.array([[1,1],[2,0],[0,2]],dtype=np.int16)
estimate,bounds,numerators,denom=unsafe_bootstrap([3,0],[3,12],matrix)
assert estimate==Fraction(1,2) and list(numerators)==[12,24,0] and denom==24
assert exact_percentile([0,1,2,3],3,Fraction(1,80))==Fraction(1,80)
assert exact_percentile([0,1,2,3],3,Fraction(79,80))==Fraction(79,80)
continuous,_,_=continuous_bootstrap([1.,0.],matrix);assert continuous==.5
conf=np.array([[2,1,0],[0,0,0],[0,0,0]])
stats=class_statistics(conf)
assert stats[1]['truth_support']==0 and stats[1]['fp']==1 and stats[1]['iou']==0
assert stats[2]['union']==0 and stats[2]['iou'] is None
assert direction(0,0)=='inconclusive' and direction(-.2,-.1)=='supported lower loss'
rng=np.random.default_rng(2026101303);m=multiplicities([1,2,3],rng)
assert m.shape==(20000,3) and (m.sum(1)==3).all()
receipt=dict(status='PASSED',equal_scene_not_pooled_frame=True,rational_linear_percentile=True,
    absent_class_false_positive_iou_zero=True,undefined_only_zero_union=True,zero_interval_not_equivalence=True,
    draws=20000,quantiles=['1/40','39/40','1/80','79/80'],network_forwards=0)
(ROOT/'tests/statistical_controls.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
