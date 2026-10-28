"""Complete saved-output Study 4 diagnostic tables; no new resampling."""
from common import *
import pandas as pd
from fractions import Fraction
from statistical_core4 import scene_bootstrap,CONDITIONS
DRAW=None

def reduced(kind,keys,values):
    rows=[];seeds=[]
    for p in sorted((ROOT/f'study4/diagnostics/{kind}/scenes').glob('*.csv')):
        d=pd.read_csv(p);rows.append(d.groupby(keys+['scene_id'])[values].mean().reset_index())
        if 'seed' in d:seeds.append(d.groupby(keys+['seed','scene_id'])[values].mean().reset_index())
    assert len(rows)==len(DRAW['nominal_scene_ids'])+len(DRAW['rough_scene_ids'])
    df=pd.concat(rows,ignore_index=True);df.to_csv(ROOT/f'study4/diagnostics/{kind}/per_scene.csv',index=False)
    if seeds:pd.concat(seeds).groupby(keys+['seed'])[values].mean().to_csv(ROOT/f'study4/diagnostics/{kind}/per_seed.csv')
    return df

def intervals(scene,keys,values):
    rows=[]
    for key,g in scene.groupby(keys,sort=True,dropna=False):
        key=key if isinstance(key,tuple) else (key,);identity=dict(zip(keys,key));pop='rough' if identity['condition']=='rough_iid' else 'nominal'
        ids=DRAW[pop+'_scene_ids'];f=g.set_index('scene_id').reindex(ids);assert f[values].notna().all().all()
        arr=f[values].to_numpy(float);reps=DRAW[pop].astype(float)@arr/len(ids);q=np.quantile(reps,[.025,.975],axis=0)
        for j,metric in enumerate(values):rows.append(dict(**identity,metric=metric,estimate=float(arr[:,j].mean()),lower95=float(q[0,j]),upper95=float(q[1,j]),
            scenes=len(ids),inference='prespecified descriptive diagnostic; ordinary95; same scene draws'))
    return pd.DataFrame(rows)

def attribution():
    keys=['condition','architecture','arm','subset'];vals=['regret','unsafe','executed','abstained']
    f=reduced('attribution',keys,vals);intervals(f,keys,vals).to_csv(ROOT/'study4/diagnostics/attribution/descriptive_intervals.csv',index=False)
    keys=['condition','architecture','arm','factor'];vals=['regret_loss_reduction','unsafe_loss_reduction']
    f=reduced('shapley',keys,vals);intervals(f,keys,vals).to_csv(ROOT/'study4/diagnostics/shapley/descriptive_intervals.csv',index=False)

def safety():
    keys=['condition','architecture','arm','policy','selection_kind'];vals=['multi_'+c for c in CAUSES]+['exclusive_'+c for c in CAUSES]+['unsafe','executed','abstained']
    means=[];counts=[];seedcounts=[]
    for p in sorted((ROOT/'study4/diagnostics/safety/scenes').glob('*.csv')):
        d=pd.read_csv(p);means.append(d.groupby(keys+['scene_id'])[vals].mean().reset_index())
        s=d.groupby(keys)[vals].sum();s['attempts']=d.groupby(keys).size();counts.append(s)
        s=d.groupby(keys+['seed'])[vals].sum();s['attempts']=d.groupby(keys+['seed']).size();seedcounts.append(s)
    result=pd.concat(counts).groupby(level=list(range(len(keys)))).sum()
    result['person_per_1000_attempts']=result.multi_person_buffer/result.attempts*1000
    result['unsafe_per_attempt']=result.unsafe/result.attempts;result['unsafe_per_execution']=result.unsafe/result.executed.replace(0,np.nan)
    assert np.allclose(result[['exclusive_'+c for c in CAUSES]].sum(axis=1),result.unsafe,atol=1e-8)
    result.to_csv(ROOT/'study4/diagnostics/safety/complete_cause_inventory.csv')
    pd.concat(seedcounts).groupby(level=list(range(len(keys)+1))).sum().to_csv(ROOT/'study4/diagnostics/safety/per_seed_counts.csv')
    f=pd.concat(means,ignore_index=True);f.to_csv(ROOT/'study4/diagnostics/safety/per_scene.csv',index=False)
    intervals(f,keys,vals).to_csv(ROOT/'study4/diagnostics/safety/descriptive_intervals.csv',index=False)
    pool=[]
    for key,g in f.groupby(['architecture','arm','policy','selection_kind']):
        for metric in ['multi_person_buffer','unsafe','executed','abstained']:
            r,_=scene_bootstrap(g,metric,{c:Fraction(1,9) for c in CONDITIONS},DRAW)
            pool.append(dict(zip(['architecture','arm','policy','selection_kind'],key))|dict(metric=metric,estimate=r['estimate'],lower95=r['lower95'],upper95=r['upper95']))
    csvwrite(ROOT/'study4/diagnostics/safety/equal_condition_pool.csv',pool)

def references():
    dfs=[pd.read_csv(p) for p in sorted((ROOT/'study4/diagnostics/references/scenes').glob('*.csv'))];ref=pd.concat(dfs,ignore_index=True)
    ref.to_csv(ROOT/'study4/diagnostics/references/per_frame.csv',index=False)
    keys=['condition','architecture','arm','policy','selection_kind'];vals=['regret','unsafe','executed','abstained']
    scene=ref.groupby(keys+['scene_id'])[vals].mean().reset_index();intervals(scene,keys,vals).to_csv(ROOT/'study4/diagnostics/references/descriptive_intervals.csv',index=False)
    neural=pd.concat([pd.read_csv(p,usecols=['architecture','seed','arm','policy','condition','scene_id','frame_id','regret','unsafe','executed','abstained']) for p in [ROOT/f'study4/metrics/{a}_seed{s}.csv' for a in ['KF-D42','BF-D42','KF-D14','BF-D14'] for s in [10,11,12]]],ignore_index=True)
    rows=[]
    for label,r in ref.groupby('architecture'):
        f=neural.merge(r[['condition','scene_id','frame_id']+vals],on=['condition','scene_id','frame_id'],suffixes=('_model','_reference'),validate='many_to_one')
        for metric in vals:f[metric]=f[metric+'_model']-f[metric+'_reference']
        keys=['condition','architecture','arm','policy'];scene=f.groupby(keys+['scene_id'])[vals].mean().reset_index()
        t=intervals(scene,keys,vals);t['reference']=label;rows.append(t)
    pd.concat(rows,ignore_index=True).to_csv(ROOT/'study4/diagnostics/references/complete_model_reference_comparisons.csv',index=False)

def cutoffs():
    keys=['condition','architecture','arm','policy','cutoff']
    vals=['regret','unsafe','executed','abstained','person_buffer']
    f=reduced('cutoffs',keys,vals)
    intervals(f,keys,vals).to_csv(ROOT/'study4/diagnostics/cutoffs/descriptive_intervals.csv',index=False)
    rows=[]
    for key,g in f.groupby(['architecture','arm','policy','cutoff']):
        for metric in vals:
            r,_=scene_bootstrap(g,metric,{c:Fraction(1,9) for c in CONDITIONS},DRAW)
            rows.append(dict(zip(['architecture','arm','policy','cutoff'],key))|dict(metric=metric,estimate=r['estimate'],lower95=r['lower95'],upper95=r['upper95'],
                scope='fixed prespecified cutoff curve; descriptive; no cutoff selected from holdout'))
    csvwrite(ROOT/'study4/diagnostics/cutoffs/equal_condition_pool.csv',rows)
    # Aggregate events/attempts are supplied separately from the equal-condition
    # pooled estimand; rough scenes and nominal scenes have different counts.
    counts=[]
    for p in sorted((ROOT/'study4/diagnostics/cutoffs/scenes').glob('*.csv')):
        d=pd.read_csv(p);g=d.groupby(keys)[['unsafe','executed','abstained','person_buffer']].sum();g['attempts']=d.groupby(keys).size();counts.append(g)
    t=pd.concat(counts).groupby(level=list(range(len(keys)))).sum()
    t['person_per1000_attempts']=1000*t.person_buffer/t.attempts
    t['unsafe_per_execution']=t.unsafe/t.executed.replace(0,np.nan)
    t.to_csv(ROOT/'study4/diagnostics/cutoffs/absolute_counts.csv')

if __name__=='__main__':
    from infer_study4 import verify_lock
    verify_lock();DRAW=load_npz(ROOT/'study4/bootstrap_plan/bootstrap_multiplicities.npz')
    assert json.loads((ROOT/'study4/diagnostics/scoring_receipt.json').read_text())['perception_forwards']==0
    for name,fn in [('attribution',attribution),('safety',safety),('references',references),('cutoffs',cutoffs)]:print(name,flush=True);fn()
    js(ROOT/'study4/diagnostics/table_completion.json',dict(status='COMPLETE',new_bootstrap_draws=0,bootstrap_sha256=sha(ROOT/'study4/bootstrap_plan/bootstrap_multiplicities.npz'),code_sha256=sha(__file__)))
