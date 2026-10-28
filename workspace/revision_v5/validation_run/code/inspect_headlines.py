"""Standalone review-bundle check. Requires numpy/pandas, no repository or model weights.
Run: python code/inspect_headlines.py [extracted_bundle_directory]
Reconstructs four headline effects/intervals from seed/frame metrics and saved scene draws.
"""
from pathlib import Path
import sys,json,math
from fractions import Fraction
import numpy as np
import pandas as pd
def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
    metrics=pd.read_csv(root/'headline_frame_metrics.csv',float_precision='round_trip')
    reported=pd.read_csv(root/'primary_results.csv',float_precision='round_trip')
    manifest=json.loads((root/'manifests/generation_manifest.json').read_text())
    draws=np.load(root/'manifests/bootstrap_multiplicities.npz')
    comparisons=[('G','H_reachable_terrain','G3_dyn','G2_static','direct',['clean','wider_articulation']),
       ('S','regret','R2_full','R2_noaff','soft',['iid','all_s2','rough_iid']),
       ('C-regret','regret','S5A_G3_structured','S5A_G3_structured','direct',['cam_s3']),
       ('C-unsafe','unsafe','S5A_G3_structured','S5A_G3_structured','direct',['cam_s3'])]
    results=[]
    for q,ep,ma,mb,policy,conditions in comparisons:
        ref=reported[reported.question==q].iloc[0]
        if ref.status!='COMPLETE':results.append(dict(question=q,status='blocked slot retained'));continue
        a=metrics[(metrics.model==ma)&(metrics.policy==policy)&metrics.condition.isin(conditions)]
        b=metrics[(metrics.model==mb)&(metrics.policy==policy)&metrics.condition.isin(conditions)]
        if q.startswith('C-'):a=a[a['mask']];b=b[~b['mask']]
        keys=['seed','condition','scene_id','frame_id'];a=a.set_index(keys)[ep];b=b.set_index(keys)[ep]
        assert set(a.index)==set(b.index);delta=(a-b).rename('delta').reset_index()
        effects=[];rep=np.zeros(10000);fractions=[Fraction(0)]*10000 if ep=='unsafe' else None
        for condition in conditions:
            frame=delta[delta.condition==condition];family='rough' if condition=='rough_iid' else 'nominal'
            ids=manifest[family+'_scene_ids'];n=[];den=[]
            for scene in ids:
                values=frame[frame.scene_id==scene].delta.to_numpy()
                n.append(math.fsum(values));den.append(len(values))
            ns=draws[family]@np.array(n);ds=draws[family]@np.array(den)
            assert np.all(ds>0);rep+=ns/ds/len(conditions)
            effects.append(math.fsum(frame.delta)/len(frame))
            if fractions is not None:
                fractions=[x+Fraction(int(nn),int(dd)*len(conditions)) for x,nn,dd in zip(fractions,ns,ds)]
        if fractions is not None:
            fractions=sorted(fractions);bounds=[]
            for quantile in [Fraction(1,40),Fraction(39,40),Fraction(1,160),Fraction(159,160)]:
                x=(len(fractions)-1)*quantile;i=x.numerator//x.denominator;w=x-i
                bounds.append(fractions[i]*(1-w)+fractions[min(i+1,len(fractions)-1)]*w)
            assert [str(x) for x in bounds]==json.loads(ref.quantile_fractions)
            values=[math.fsum(effects)/len(effects)]+list(map(float,bounds))
        else:values=[math.fsum(effects)/len(effects)]+np.quantile(rep,[.025,.975,.00625,.99375]).tolist()
        published=[ref.mean_diff,ref.ci95_lo,ref.ci95_hi,ref.ci9875_lo,ref.ci9875_hi]
        # Forward summation error bound for consistency checking only. Never maps a reported effect to zero.
        nops=4*len(delta);eps=np.finfo(np.float64).eps;gamma=nops*eps/(1-nops*eps)
        bound=gamma*max(1,float(np.abs(delta.delta).max()))
        error=float(np.max(np.abs(np.asarray(values)-published)))
        assert error<=bound,(q,error,bound)
        results.append(dict(question=q,status='reconstructed',maximum_difference=error,summation_error_bound=bound))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
