"""Prospectively specified scene bootstrap, four primary endpoints, full descriptives."""
from common import *
import math
from fractions import Fraction
from functools import reduce
import pandas as pd
sys.path.insert(0,str(AFAST/'code'))
from statistical_core import multiplicities,exact_percentile
CONDITIONS=['clean','iid','cam_s1','cam_s2','cam_s3','single_severe','novel','all_s2','rough_iid']
ORDINARY=[Fraction(1,40),Fraction(39,40)]
ADJUSTED=[Fraction(1,160),Fraction(159,160)]

def interval_class(low,high):return 'LOWER' if high<0 else 'HIGHER' if low>0 else 'INCONCLUSIVE'

def scene_bootstrap(frame,metric,weights,draws,integer_events=False):
    """Rows are attempts (or paired-attempt differences); seed counts stay fixed."""
    group=frame.groupby(['condition','scene_id'])[metric].agg(['sum','count','mean'])
    estimate=0.;reps=np.zeros(20000);exact_terms=[]
    for population,conditions in [('nominal',[c for c in CONDITIONS if c!='rough_iid']),('rough',['rough_iid'])]:
        ids=draws[population+'_scene_ids'];M=draws[population]
        vals=np.zeros(len(ids));fracs=[Fraction(0) for _ in ids]
        for condition in conditions:
            w=weights.get(condition,Fraction(0))
            if not w:continue
            g=group.loc[condition].reindex(ids);assert g['mean'].notna().all(),(population,condition)
            vals+=float(w)*g['mean'].to_numpy()
            if integer_events:
                sums=g['sum'].to_numpy();assert np.allclose(sums,np.rint(sums),atol=1e-10)
                for j,(s,n) in enumerate(zip(sums,g['count'])):fracs[j]+=w*Fraction(int(round(s)),int(n))
        estimate+=float(vals.mean());reps+=M.astype(np.float64)@vals/len(ids)
        if integer_events and any(fracs):
            den=reduce(math.lcm,(f.denominator for f in fracs),1)
            nums=np.array([f.numerator*(den//f.denominator) for f in fracs],np.int64)
            assert max(abs(nums),default=0)*len(ids)<np.iinfo(np.int64).max//2
            exact_terms.append((M.astype(np.int64)@nums,den*len(ids),Fraction(int(nums.sum()),den*len(ids))))
    b=np.quantile(reps,[.025,.975,1/160,159/160],method='linear')
    result=dict(estimate=estimate,lower95=float(b[0]),upper95=float(b[1]),lower98_75=float(b[2]),upper98_75=float(b[3]),
        class95=interval_class(b[0],b[1]),class98_75=interval_class(b[2],b[3]))
    if integer_events:
        den=reduce(math.lcm,(r[1] for r in exact_terms),1);nums=np.zeros(20000,np.int64);point=Fraction(0)
        for n,d,p in exact_terms:nums+=n*(den//d);point+=p
        exact=[exact_percentile(nums,den,q) for q in ORDINARY+ADJUSTED]
        assert np.allclose(np.array([float(x) for x in exact]),b,atol=1e-12)
        result.update(estimate=float(point),lower95=float(exact[0]),upper95=float(exact[1]),lower98_75=float(exact[2]),upper98_75=float(exact[3]),
            class95=interval_class(exact[0],exact[1]),class98_75=interval_class(exact[2],exact[3]),
            exact_estimate=str(point),exact_lower95=str(exact[0]),exact_upper95=str(exact[1]),exact_lower98_75=str(exact[2]),exact_upper98_75=str(exact[3]))
        # Exact event arithmetic also controls ratio-denominator classifications.
        reps=nums.astype(np.float64)/den
    return result,reps

def paired(frame,architecture,left,right,policy='direct',right_architecture=None):
    keys=['condition','scene_id','frame_id','seed'];right_architecture=right_architecture or architecture
    a=frame[(frame.architecture==architecture)&(frame.arm==left)&(frame.policy==policy)].set_index(keys)
    b=frame[(frame.architecture==right_architecture)&(frame.arm==right)&(frame.policy==policy)].set_index(keys)
    assert a.index.is_unique and b.index.is_unique and set(a.index)==set(b.index)
    b=b.reindex(a.index);out=a[['regret','unsafe','executed','abstained']]-b[['regret','unsafe','executed','abstained']]
    return out.reset_index()

def controls():
    # Small deterministic fixture: point and exact event interval agree when all
    # scenes have identical contrasts; rough and nominal counts may differ.
    ids=np.array([1,2]);draws={'nominal_scene_ids':ids,'rough_scene_ids':np.array([3]),'nominal':np.tile([1,1],(20000,1)),'rough':np.ones((20000,1),np.int16)}
    rows=[dict(condition=c,scene_id=s,unsafe=v) for c,s,v in [('clean',1,-1),('clean',2,-1),('rough_iid',3,1)]]
    r,_=scene_bootstrap(pd.DataFrame(rows),'unsafe',{'clean':Fraction(8,9),'rough_iid':Fraction(1,9)},draws,True)
    assert r['exact_estimate']=='-7/9' and r['exact_upper98_75']=='-7/9'
    assert interval_class(-1,0)=='INCONCLUSIVE' and interval_class(-2,-1)=='LOWER'
    js(ROOT/'protocol/statistical_controls.json',dict(status='PASS',scope='deterministic arithmetic fixtures; no scientific outcomes or random draws',code_sha256=sha(__file__)))
