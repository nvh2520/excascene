"""Complete saved-output Study 3 diagnostic tables; no new resampling."""
from common import *
import pandas as pd
from fractions import Fraction
from study3_statistics import scene_bootstrap,CONDITIONS
DRAW=None

def reduced(kind,keys,values):
    rows=[];seeds=[]
    for p in sorted((ROOT/f'study3/diagnostics/{kind}/scenes').glob('*.csv')):
        d=pd.read_csv(p);rows.append(d.groupby(keys+['scene_id'])[values].mean().reset_index())
        if 'seed' in d:seeds.append(d.groupby(keys+['seed','scene_id'])[values].mean().reset_index())
    assert len(rows)==len(DRAW['nominal_scene_ids'])+len(DRAW['rough_scene_ids'])
    df=pd.concat(rows,ignore_index=True);df.to_csv(ROOT/f'study3/diagnostics/{kind}/per_scene.csv',index=False)
    if seeds:pd.concat(seeds).groupby(keys+['seed'])[values].mean().to_csv(ROOT/f'study3/diagnostics/{kind}/per_seed.csv')
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
    f=reduced('attribution',keys,vals);intervals(f,keys,vals).to_csv(ROOT/'study3/diagnostics/attribution/descriptive_intervals.csv',index=False)
    keys=['condition','architecture','arm','factor'];vals=['regret_loss_reduction','unsafe_loss_reduction']
    f=reduced('shapley',keys,vals);intervals(f,keys,vals).to_csv(ROOT/'study3/diagnostics/shapley/descriptive_intervals.csv',index=False)

def safety():
    keys=['condition','architecture','arm','policy','selection_kind'];vals=['multi_'+c for c in CAUSES]+['exclusive_'+c for c in CAUSES]+['unsafe','executed','abstained']
    means=[];counts=[];seedcounts=[]
    for p in sorted((ROOT/'study3/diagnostics/safety/scenes').glob('*.csv')):
        d=pd.read_csv(p);means.append(d.groupby(keys+['scene_id'])[vals].mean().reset_index())
        s=d.groupby(keys)[vals].sum();s['attempts']=d.groupby(keys).size();counts.append(s)
        s=d.groupby(keys+['seed'])[vals].sum();s['attempts']=d.groupby(keys+['seed']).size();seedcounts.append(s)
    result=pd.concat(counts).groupby(level=list(range(len(keys)))).sum()
    result['person_per_1000_attempts']=result.multi_person_buffer/result.attempts*1000
    result['unsafe_per_attempt']=result.unsafe/result.attempts;result['unsafe_per_execution']=result.unsafe/result.executed.replace(0,np.nan)
    assert np.allclose(result[['exclusive_'+c for c in CAUSES]].sum(axis=1),result.unsafe,atol=1e-8)
    result.to_csv(ROOT/'study3/diagnostics/safety/complete_cause_inventory.csv')
    pd.concat(seedcounts).groupby(level=list(range(len(keys)+1))).sum().to_csv(ROOT/'study3/diagnostics/safety/per_seed_counts.csv')
    f=pd.concat(means,ignore_index=True);f.to_csv(ROOT/'study3/diagnostics/safety/per_scene.csv',index=False)
    intervals(f,keys,vals).to_csv(ROOT/'study3/diagnostics/safety/descriptive_intervals.csv',index=False)
    pool=[]
    for key,g in f.groupby(['architecture','arm','policy','selection_kind']):
        for metric in ['multi_person_buffer','unsafe','executed','abstained']:
            r,_=scene_bootstrap(g,metric,{c:Fraction(1,9) for c in CONDITIONS},DRAW)
            pool.append(dict(zip(['architecture','arm','policy','selection_kind'],key))|dict(metric=metric,estimate=r['estimate'],lower95=r['lower95'],upper95=r['upper95']))
    csvwrite(ROOT/'study3/diagnostics/safety/equal_condition_pool.csv',pool)

def references():
    dfs=[pd.read_csv(p) for p in sorted((ROOT/'study3/diagnostics/references/scenes').glob('*.csv'))];ref=pd.concat(dfs,ignore_index=True)
    ref.to_csv(ROOT/'study3/diagnostics/references/per_frame.csv',index=False)
    keys=['condition','architecture','arm','policy','selection_kind'];vals=['regret','unsafe','executed','abstained']
    scene=ref.groupby(keys+['scene_id'])[vals].mean().reset_index();intervals(scene,keys,vals).to_csv(ROOT/'study3/diagnostics/references/descriptive_intervals.csv',index=False)
    neural=pd.concat([pd.read_csv(p,usecols=['architecture','seed','arm','policy','condition','scene_id','frame_id','regret','unsafe','executed','abstained']) for p in sorted((ROOT/'study3/metrics').glob('*_seed*.csv'))],ignore_index=True)
    rows=[]
    for label,r in ref.groupby('architecture'):
        f=neural.merge(r[['condition','scene_id','frame_id']+vals],on=['condition','scene_id','frame_id'],suffixes=('_model','_reference'),validate='many_to_one')
        for metric in vals:f[metric]=f[metric+'_model']-f[metric+'_reference']
        keys=['condition','architecture','arm','policy'];scene=f.groupby(keys+['scene_id'])[vals].mean().reset_index()
        t=intervals(scene,keys,vals);t['reference']=label;rows.append(t)
    pd.concat(rows,ignore_index=True).to_csv(ROOT/'study3/diagnostics/references/complete_model_reference_comparisons.csv',index=False)

def sensitivity():
    keys=['setting','condition','architecture','arm','policy'];vals=['regret','unsafe','executed','abstained'];f=reduced('sensitivity',keys,vals);rows=[]
    for (setting,policy),g in f.groupby(['setting','policy']):
        wide=g.pivot(index=['condition','scene_id'],columns=['architecture','arm'],values=vals)
        pairs=[('KF_gate_minus_keep',('KF-D','gate'),('KF-D','keep_all')),('BF_gate_minus_keep',('BF-D','gate'),('BF-D','keep_all')),
            ('KF_gate_minus_always',('KF-D','gate'),('KF-D','always_mask')),('BF_gate_minus_always',('BF-D','gate'),('BF-D','always_mask')),
            ('KF_gate_minus_BF_keep',('KF-D','gate'),('BF-D','keep_all'))]
        for label,left,right in pairs:
            delta=pd.DataFrame({v:wide[(v,*left)]-wide[(v,*right)] for v in vals}).reset_index()
            for pool,w in [('equal9',{c:Fraction(1,9) for c in CONDITIONS}),('clean_heavy',{c:Fraction(1,2) if c=='clean' else Fraction(1,16) for c in CONDITIONS})]+[(c,{c:Fraction(1)}) for c in CONDITIONS]:
                for metric in ['regret','unsafe']:
                    if metric=='unsafe':
                        # Three fixed seeds and 1..4 accepted slots imply scene
                        # event-rate denominators3,6,9,12, whose LCM is36.
                        exact_delta=delta.copy();scaled=exact_delta.unsafe.to_numpy()*36
                        assert np.max(np.abs(scaled-np.rint(scaled)))<1e-10
                        exact_delta['unsafe']=np.rint(scaled).astype(np.int64)
                        r,_=scene_bootstrap(exact_delta,metric,{c:v/Fraction(36) for c,v in w.items()},DRAW,True)
                    else:r,_=scene_bootstrap(delta,metric,w,DRAW)
                    rows.append(dict(setting=setting,policy=policy,contrast=label,pool=pool,metric=metric,**r,sign=int(np.sign(r['estimate']))))
    table=pd.DataFrame(rows);base=table[table.setting=='nominal'].set_index(['policy','contrast','pool','metric'])
    table['retains_nominal_sign']=[r['sign']==base.loc[(r['policy'],r['contrast'],r['pool'],r['metric']),'sign'] for r in rows]
    table['retains_nominal_interval_class95']=[r['class95']==base.loc[(r['policy'],r['contrast'],r['pool'],r['metric']),'class95'] for r in rows]
    table['retains_nominal_interval_class98_75']=[r['class98_75']==base.loc[(r['policy'],r['contrast'],r['pool'],r['metric']),'class98_75'] for r in rows]
    table.to_csv(ROOT/'study3/diagnostics/sensitivity/complete_comparison_intervals.csv',index=False)

if __name__=='__main__':
    DRAW=load_npz(ROOT/'study3/analysis/bootstrap_multiplicities.npz')
    assert json.loads((ROOT/'study3/diagnostics/scoring_receipt.json').read_text())['perception_forwards']==0
    for name,fn in [('attribution',attribution),('safety',safety),('references',references),('sensitivity',sensitivity)]:print(name,flush=True);fn()
    js(ROOT/'study3/diagnostics/table_completion.json',dict(status='COMPLETE',new_bootstrap_draws=0,bootstrap_sha256=sha(ROOT/'study3/analysis/bootstrap_multiplicities.npz'),code_sha256=sha(__file__)))
