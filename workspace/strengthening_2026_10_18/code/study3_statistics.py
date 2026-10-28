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

def main():
    from infer_study3 import verify_lock
    verify_lock();out=ROOT/'study3/analysis';out.mkdir(exist_ok=False)
    paths=sorted((ROOT/'study3/metrics').glob('*_seed*.csv'));assert len(paths)==12
    for path in paths:assert json.loads(path.with_suffix('.json').read_text())['status']=='COMPLETE'
    frame=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    ids=frame[frame.condition=='clean'].scene_id.unique();rough=frame[frame.condition=='rough_iid'].scene_id.unique();ids.sort();rough.sort()
    rng=np.random.default_rng(2026101808)
    draws={'nominal_scene_ids':ids,'rough_scene_ids':rough,'nominal':multiplicities(ids,rng),'rough':multiplicities(rough,rng)}
    np.savez_compressed(out/'bootstrap_multiplicities.npz',**draws)
    primary=[];replicates={};weights={c:Fraction(1,9) for c in CONDITIONS};per_seed=[]
    for architecture in ['KF-D','BF-D']:
        contrast=paired(frame,architecture,'gate','keep_all')
        contrast.to_csv(out/f'primary_paired_frames_{architecture}.csv',index=False)
        for metric in ['regret','unsafe']:
            result,reps=scene_bootstrap(contrast,metric,weights,draws,integer_events=metric=='unsafe')
            primary.append(dict(architecture=architecture,contrast='gate_minus_keep_all',policy='direct',metric=metric,**result,
                interval_interpretation='approximate scene-bootstrap marginal coverage; four-endpoint family adjustment'))
            replicates[architecture+'_'+metric]=reps
            for seed,sub in contrast.groupby('seed'):
                point=sub.groupby('condition')[metric].mean().reindex(CONDITIONS).mean()
                per_seed.append(dict(architecture=architecture,seed=int(seed),metric=metric,estimate=float(point)))
    csvwrite(out/'primary_results.csv',primary);csvwrite(out/'primary_per_seed.csv',per_seed);np.savez_compressed(out/'primary_replicates.npz',**replicates)
    joint={arch:all(r['upper98_75']<0 for r in primary if r['architecture']==arch) for arch in ['KF-D','BF-D']}
    js(out/'primary_results.json',dict(status='COMPLETE',endpoints=primary,joint_benefit=joint,family_size=4,adjusted_marginal_level=.9875,ordinary_level=.95,bootstrap_draws=20000,
        bootstrap_sha256=sha(out/'bootstrap_multiplicities.npz'),nominal_scenes=len(ids),rough_scenes=len(rough),all_seed_effects_retained=True))
    # Absolute behavior and all per-seed results, including zero execution and
    # undefined unsafe/execution quantities, are never filtered for favorability.
    keys=['architecture','arm','policy','condition']
    absolute=frame.groupby(keys)[['regret','unsafe','executed','abstained','height_reachable_terrain']].mean().reset_index()
    counts=frame.groupby(keys).agg(attempts=('unsafe','size'),unsafe_events=('unsafe','sum'),executions=('executed','sum'),abstentions=('abstained','sum')).reset_index()
    absolute=absolute.merge(counts,on=keys);absolute['unsafe_per_execution']=absolute.unsafe_events/absolute.executions.replace(0,np.nan)
    absolute.to_csv(out/'absolute_condition_results.csv',index=False)
    frame.groupby(keys+['seed'])[['regret','unsafe','executed','abstained','height_reachable_terrain']].mean().to_csv(out/'all_per_seed_results.csv')
    frame.groupby(keys+['scene_id'])[['regret','unsafe','executed','abstained','height_reachable_terrain']].mean().to_csv(out/'all_per_scene_results.csv')
    comparisons=[];shares=[];heavy={c:Fraction(1,2) if c=='clean' else Fraction(1,16) for c in CONDITIONS}
    for architecture in ['KF-D','BF-D']:
        for policy in ['direct','soft']:
            for left,right in [('gate','keep_all'),('always_mask','keep_all'),('oracle_mask','keep_all'),('gate','always_mask')]:
                delta=paired(frame,architecture,left,right,policy)
                for pool,w in [('equal9',weights),('clean_heavy',heavy)]+[(c,{c:Fraction(1)}) for c in CONDITIONS]:
                    for metric in ['regret','unsafe','executed','abstained']:
                        result,_=scene_bootstrap(delta,metric,w,draws,integer_events=metric!='regret')
                        comparisons.append(dict(architecture=architecture,policy=policy,contrast=left+'_minus_'+right,pool=pool,metric=metric,
                            estimate=result['estimate'],lower95=result['lower95'],upper95=result['upper95'],class95=result['class95'],inference='descriptive only'))
            gate_delta=paired(frame,architecture,'gate','keep_all',policy);oracle_delta=paired(frame,architecture,'oracle_mask','keep_all',policy)
            for pool,w in [('equal9',weights),('clean_heavy',heavy)]+[(c,{c:Fraction(1)}) for c in CONDITIONS]:
                for metric in ['regret','unsafe']:
                    a,ar=scene_bootstrap(gate_delta,metric,w,draws,metric=='unsafe');b,br=scene_bootstrap(oracle_delta,metric,w,draws,metric=='unsafe')
                    valid=br<0;ratio=None if b['estimate']>=0 else a['estimate']/b['estimate']
                    # Avoid reporting a finite unconditional ratio interval when
                    # the oracle-benefit denominator crosses zero.
                    bounds=np.quantile(ar/br,[.025,.975]) if valid.all() else [None,None]
                    shares.append(dict(architecture=architecture,policy=policy,pool=pool,metric=metric,share_recovered=ratio,
                        lower95=bounds[0],upper95=bounds[1],undefined_or_nonpositive_oracle_draws=int((~valid).sum()),
                        status='UNDEFINED_NONPOSITIVE_ORACLE_BENEFIT' if ratio is None else 'INTERVAL_UNDEFINED_DENOMINATOR' if not valid.all() else 'DEFINED'))
    for policy in ['direct','soft']:
        delta=paired(frame,'KF-D','gate','keep_all',policy,'BF-D')
        for pool,w in [('equal9',weights),('clean_heavy',heavy)]+[(c,{c:Fraction(1)}) for c in CONDITIONS]:
            for metric in ['regret','unsafe','executed','abstained']:
                r,_=scene_bootstrap(delta,metric,w,draws,metric!='regret')
                comparisons.append(dict(architecture='KF-D_vs_BF-D',policy=policy,contrast='KF_gate_minus_BF_keep_all',pool=pool,metric=metric,
                    estimate=r['estimate'],lower95=r['lower95'],upper95=r['upper95'],class95=r['class95'],inference='descriptive only'))
    csvwrite(out/'complete_secondary_comparisons.csv',comparisons);csvwrite(out/'oracle_benefit_recovery.csv',shares)
    js(out/'completion.json',dict(status='COMPLETE',primary_rows=4,secondary_rows=len(comparisons),oracle_share_rows=len(shares),metrics_files=12,seed_effects=len(per_seed),
        source_manifest_sha256=sha(ROOT/'study3/holdout/files.csv'),analysis_code_sha256=sha(__file__)))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--controls',action='store_true');a=p.parse_args()
    controls() if a.controls else main()
