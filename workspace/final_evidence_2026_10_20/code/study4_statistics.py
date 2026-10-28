"""Locked four-primary Study4family and complete prespecified descriptives."""
from common import *
from statistical_core4 import scene_bootstrap,paired,CONDITIONS,controls
from fractions import Fraction
import pandas as pd,shutil
ARCHES=['KF-D42','BF-D42','KF-D14','BF-D14'];SEEDS=[10,11,12]
def metric_paths():return [ROOT/f'study4/metrics/{a}_seed{s}.csv' for a in ARCHES for s in SEEDS]
def main():
    from infer_study4 import verify_lock
    verify_lock();out=ROOT/'study4/analysis';out.mkdir(exist_ok=False)
    for p in metric_paths():assert json.loads(p.with_suffix('.json').read_text())['status']=='COMPLETE'
    frame=pd.concat([pd.read_csv(p) for p in metric_paths()],ignore_index=True)
    draws=load_npz(ROOT/'study4/bootstrap_plan/bootstrap_multiplicities.npz')
    shutil.copy2(ROOT/'study4/bootstrap_plan/bootstrap_multiplicities.npz',out/'bootstrap_multiplicities.npz')
    assert set(frame[frame.condition=='clean'].scene_id)==set(draws['nominal_scene_ids'])
    weights={c:Fraction(1,9) for c in CONDITIONS};heavy={c:Fraction(1,2) if c=='clean' else Fraction(1,16) for c in CONDITIONS}
    pools=[('equal9',weights),('clean_heavy',heavy)]+[(c,{c:Fraction(1)}) for c in CONDITIONS]
    primary=[];replicates={};perseed=[]
    for a in ['KF-D42','BF-D42']:
        d=paired(frame,a,'gate','keep_all');d.to_csv(out/f'primary_paired_frames_{a}.csv',index=False)
        for metric in ['regret','unsafe']:
            r,b=scene_bootstrap(d,metric,weights,draws,metric=='unsafe')
            primary.append(dict(architecture=a,contrast='gate_minus_keep_all',policy='direct',metric=metric,**r,scope='Study4four-primary family; approximate98.75percent marginal coverage'))
            replicates[a+'_'+metric]=b
            for seed,g in d.groupby('seed'):perseed.append(dict(architecture=a,seed=int(seed),metric=metric,estimate=float(g.groupby('condition')[metric].mean().reindex(CONDITIONS).mean())))
    csvwrite(out/'primary_results.csv',primary);csvwrite(out/'primary_per_seed.csv',perseed);np.savez_compressed(out/'primary_replicates.npz',**replicates)
    joint={a:all(r['upper98_75']<0 for r in primary if r['architecture']==a) for a in ['KF-D42','BF-D42']}
    js(out/'primary_results.json',dict(status='COMPLETE',endpoints=primary,joint_benefit=joint,family_size=4,adjusted_marginal_level=.9875,bootstrap_draws=20000,bootstrap_sha256=sha(out/'bootstrap_multiplicities.npz'),nominal_scenes=len(draws['nominal_scene_ids']),rough_scenes=len(draws['rough_scene_ids']),new_draws_in_analysis=0))
    keys=['architecture','arm','policy','condition'];vals=['regret','unsafe','executed','abstained','height_reachable_terrain']
    absolute=frame.groupby(keys)[vals].mean().reset_index()
    counts=frame.groupby(keys).agg(attempts=('unsafe','size'),unsafe_events=('unsafe','sum'),executions=('executed','sum'),abstentions=('abstained','sum')).reset_index()
    absolute=absolute.merge(counts,on=keys);absolute['unsafe_per_execution']=absolute.unsafe_events/absolute.executions.replace(0,np.nan)
    equal_scene=frame.groupby(keys+['scene_id'])[vals].mean().groupby(keys).mean().reset_index()
    equal_scene=equal_scene.rename(columns={v:'scene_equal_'+v for v in vals})
    absolute=absolute.merge(equal_scene,on=keys,validate='one_to_one')
    absolute['estimand_note']='unprefixed means/counts are attempt-weighted; scene_equal columns match paired estimands under partial-slot scenes'
    absolute.to_csv(out/'absolute_condition_results.csv',index=False)
    frame.groupby(keys+['seed'])[vals].mean().to_csv(out/'all_per_seed_results.csv')
    frame.groupby(keys+['seed','scene_id'])[vals].mean().groupby(keys+['seed']).mean().to_csv(out/'all_per_seed_scene_equal_results.csv')
    frame.groupby(keys+['scene_id'])[vals].mean().to_csv(out/'all_per_scene_results.csv')
    comparisons=[];shares=[];relative=[]
    for a in ARCHES:
        arm_pairs=[('gate','keep_all')]
        if a.endswith('42'):arm_pairs+=[('always_mask','keep_all'),('oracle_mask','keep_all'),('gate','always_mask'),('rgb_only','keep_all'),('depth_only','keep_all'),('gate','rgb_only'),('gate','depth_only')]
        for policy in ['direct','soft']:
            for left,right in arm_pairs:
                delta=paired(frame,a,left,right,policy)
                baseline=frame[(frame.architecture==a)&(frame.arm==right)&(frame.policy==policy)]
                for pool,w in pools:
                    for metric in ['regret','unsafe','executed','abstained']:
                        r,_=scene_bootstrap(delta,metric,w,draws,metric!='regret')
                        comparisons.append(dict(architecture=a,policy=policy,contrast=left+'_minus_'+right,pool=pool,metric=metric,estimate=r['estimate'],lower95=r['lower95'],upper95=r['upper95'],class95=r['class95'],scope='descriptive only'))
                        if metric=='regret':
                            base,_=scene_bootstrap(baseline,'regret',w,draws)
                            relative.append(dict(architecture=a,policy=policy,contrast=left+'_minus_'+right,pool=pool,raw_regret_difference=r['estimate'],comparator_regret=base['estimate'],relative_percent=None if base['estimate']<=0 else 100*r['estimate']/base['estimate'],status='UNDEFINED_NONPOSITIVE_COMPARATOR' if base['estimate']<=0 else 'DESCRIPTIVE_POINT_NORMALIZATION_ONLY'))
            if not a.endswith('42'):continue
            gate=paired(frame,a,'gate','keep_all',policy);oracle=paired(frame,a,'oracle_mask','keep_all',policy)
            for pool,w in pools:
                for metric in ['regret','unsafe']:
                    r,b=scene_bootstrap(gate,metric,w,draws,metric=='unsafe');o,ob=scene_bootstrap(oracle,metric,w,draws,metric=='unsafe')
                    valid=ob<0;point=None if o['estimate']>=0 else r['estimate']/o['estimate'];bounds=np.quantile(b/ob,[.025,.975]) if valid.all() else [None,None]
                    shares.append(dict(architecture=a,policy=policy,pool=pool,metric=metric,share_recovered=point,lower95=bounds[0],upper95=bounds[1],undefined_or_nonpositive_oracle_draws=int((~valid).sum()),status='UNDEFINED_NONPOSITIVE_ORACLE_BENEFIT' if point is None else 'INTERVAL_UNDEFINED_DENOMINATOR' if not valid.all() else 'DEFINED'))
    # The paired recipe comparison retains the same scene/frame/seed keys.
    for a,new,old in [('KF','KF-D42','KF-D14'),('BF','BF-D42','BF-D14')]:
        for policy in ['direct','soft']:
            d=paired(frame,new,'keep_all','keep_all',policy,old)
            for c,w in [(c,{c:Fraction(1)}) for c in CONDITIONS]:
                for metric in ['regret','unsafe','executed','abstained']:
                    r,_=scene_bootstrap(d,metric,w,draws,metric!='regret')
                    comparisons.append(dict(architecture=a+'_42_vs_14',policy=policy,contrast='new_minus_old_keep_all',pool=c,metric=metric,estimate=r['estimate'],lower95=r['lower95'],upper95=r['upper95'],class95=r['class95'],scope='descriptive bundled recipe contrast; not an isolated epoch/pretraining/loss effect'))
    for policy in ['direct','soft']:
        d=paired(frame,'BF-D42','keep_all','keep_all',policy,'KF-D42')
        for c in ['clean','cam_s3','rough_iid']:
            for metric in ['regret','unsafe','executed','abstained']:
                r,_=scene_bootstrap(d,metric,{c:Fraction(1)},draws,metric!='regret')
                comparisons.append(dict(architecture='new_BF_minus_new_KF',policy=policy,contrast='external_minus_lightweight_keep_all',pool=c,metric=metric,estimate=r['estimate'],lower95=r['lower95'],upper95=r['upper95'],class95=r['class95'],scope='descriptive architecture comparison under prescribed42epoch recipe'))
    # Question-oriented camera-impact reading, fixed before test generation.
    impact=[]
    for a in ARCHES:
        for policy in ['direct','soft']:
            base=frame[(frame.architecture==a)&(frame.arm=='keep_all')&(frame.policy==policy)&(frame.condition=='clean')].set_index(['scene_id','frame_id','seed'])
            for c in CONDITIONS[1:-1]:
                f=frame[(frame.architecture==a)&(frame.arm=='keep_all')&(frame.policy==policy)&(frame.condition==c)].set_index(['scene_id','frame_id','seed'])
                assert set(f.index)==set(base.index)
                for metric in ['regret','unsafe']:
                    d=(f[[metric]]-base[[metric]].reindex(f.index)).reset_index();d['condition']=c
                    r,_=scene_bootstrap(d,metric,{c:Fraction(1)},draws,metric=='unsafe')
                    impact.append(dict(architecture=a,policy=policy,condition=c,contrast='condition_minus_clean_keep_all',metric=metric,estimate=r['estimate'],lower95=r['lower95'],upper95=r['upper95'],scope='prespecified descriptive context for rig versus single-camera faults'))
    csvwrite(out/'complete_secondary_comparisons.csv',comparisons);csvwrite(out/'oracle_benefit_recovery.csv',shares);csvwrite(out/'relative_regret_points.csv',relative);csvwrite(out/'sensor_impact_comparisons.csv',impact)
    js(out/'completion.json',dict(status='COMPLETE',primary_rows=4,secondary_rows=len(comparisons),oracle_share_rows=len(shares),metrics_files=12,seed_effects=12,bootstrap_sha256=sha(out/'bootstrap_multiplicities.npz'),analysis_code_sha256=sha(__file__)))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--controls',action='store_true');a=p.parse_args()
    controls() if a.controls else main()
