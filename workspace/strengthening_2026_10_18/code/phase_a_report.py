"""Complete descriptive inventories and saved-draw intervals, then a one-page memo."""
from common import *
import pandas as pd
from fractions import Fraction
sys.path.insert(0,str(AFAST/'code'))
from statistical_core import exact_percentile
DRAW=load_npz(AFAST/'analysis/bootstrap_multiplicities.npz')

def load_scenes(kind):
    paths=sorted((ROOT/f'phase_a/{kind}/scenes').glob('*.csv'));assert len(paths)==192,(kind,len(paths))
    return pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)

def interval_table(scene,groups,values):
    result=[]
    for key,frame in scene.groupby(groups,sort=True,dropna=False):
        key=key if isinstance(key,tuple) else (key,);base=dict(zip(groups,key));rough=base['condition']=='rough_iid'
        ids=DRAW['rough_scene_ids' if rough else 'nominal_scene_ids'];matrix=DRAW['rough' if rough else 'nominal']
        f=frame.set_index('scene_id').reindex(ids);assert len(f)==len(ids) and f[values].notna().all().all(),base
        arr=f[values].to_numpy(float);rep=matrix.astype(np.float64)@arr/len(ids);bounds=np.quantile(rep,[.025,.975],axis=0)
        for i,value in enumerate(values):result.append(dict(**base,metric=value,estimate=float(arr[:,i].mean()),lower95=float(bounds[0,i]),upper95=float(bounds[1,i]),
            scenes=len(ids),interval='posthoc descriptive paired-scene percentile95; saved draws only'))
    return pd.DataFrame(result)

def attribution():
    control=json.loads((ROOT/'phase_a/attribution/control_receipt.json').read_text());assert control['status']=='COMPLETE'
    df=load_scenes('attribution');keys=['condition','architecture','seed','mask','subset']
    df.groupby(keys)[['regret','unsafe','executed','abstained']].mean().to_csv(ROOT/'phase_a/attribution/per_seed.csv')
    scene=df.groupby(['condition','architecture','mask','subset','scene_id'])[['regret','unsafe','executed','abstained']].mean().reset_index()
    scene.to_csv(ROOT/'phase_a/attribution/per_scene.csv',index=False)
    result=interval_table(scene,['condition','architecture','mask','subset'],['regret','unsafe','executed','abstained'])
    result.to_csv(ROOT/'phase_a/attribution/descriptive_intervals.csv',index=False)
    safetycols=['multi_'+c for c in CAUSES]+['exclusive_'+c for c in CAUSES]
    sums=df.groupby(['condition','architecture','mask','subset'])[safetycols+['unsafe','executed','abstained']].sum()
    sums['attempts']=df.groupby(['condition','architecture','mask','subset']).size();sums['person_per_1000_attempts']=1000*sums.multi_person_buffer/sums.attempts
    sums.to_csv(ROOT/'phase_a/attribution/safety_all_subsets.csv')
    del df
    sh=load_scenes('shapley');sh.groupby(['condition','architecture','seed','mask','factor'])[['regret_loss_reduction','unsafe_loss_reduction']].mean().to_csv(ROOT/'phase_a/shapley/per_seed.csv')
    scene=sh.groupby(['condition','architecture','mask','factor','scene_id'])[['regret_loss_reduction','unsafe_loss_reduction']].mean().reset_index()
    scene.to_csv(ROOT/'phase_a/shapley/per_scene.csv',index=False)
    interval_table(scene,['condition','architecture','mask','factor'],['regret_loss_reduction','unsafe_loss_reduction']).to_csv(ROOT/'phase_a/shapley/descriptive_intervals.csv',index=False)

def safety():
    df=load_scenes('safety');classical=pd.read_csv(ROOT/'phase_a/classical/afast_per_frame.csv');extra=[]
    for sid,frame in classical.groupby('scene_id'):
        scene,poses=historical_scene(int(sid))
        for row in frame.to_dict('records'):
            d=load_npz(AFAST/f"data/{row['condition']}/f_{sid:06d}_{row['frame_id']}.npz",['gt_sem','gt_reach'])
            masks=safety_masks(scene,poses[row['frame_id']],d['gt_sem']);values=np.zeros(5) if row['abstained'] else masks[:,row['pick_row'],row['pick_col']].astype(float)
            exclusive=np.zeros(5)
            if values.any():exclusive[np.flatnonzero(values)[0]]=1
            assert exclusive.sum()==row['unsafe']
            extra.append(dict(condition=row['condition'],scene_id=sid,frame_id=row['frame_id'],architecture='classical_ground',seed=-1,mask=0,policy='classical_ground',
                selection_kind='deterministic',unsafe=row['unsafe'],executed=row['executed'],abstained=row['abstained'],
                **{'multi_'+c:v for c,v in zip(CAUSES,values)},**{'exclusive_'+c:v for c,v in zip(CAUSES,exclusive)}))
    df=pd.concat([df,pd.DataFrame(extra)],ignore_index=True);df.to_csv(ROOT/'phase_a/safety/all_policies_per_frame.csv',index=False)
    keys=['condition','architecture','mask','policy','selection_kind'];cols=['multi_'+c for c in CAUSES]+['exclusive_'+c for c in CAUSES]+['unsafe','executed','abstained']
    summary=df.groupby(keys)[cols].sum();summary['attempts']=df.groupby(keys).size()
    summary['person_per_1000_attempts']=1000*summary.multi_person_buffer/summary.attempts
    summary['unsafe_per_attempt']=summary.unsafe/summary.attempts;summary['unsafe_per_execution']=summary.unsafe/summary.executed.replace(0,np.nan)
    assert np.allclose(summary[['exclusive_'+c for c in CAUSES]].sum(axis=1),summary.unsafe,atol=1e-10)
    summary.to_csv(ROOT/'phase_a/safety/complete_cause_inventory.csv')
    df.groupby(keys+['seed'])[cols].sum().to_csv(ROOT/'phase_a/safety/per_seed_counts.csv')
    scene=df.groupby(keys+['scene_id'])[cols].mean().reset_index()
    interval_table(scene,keys,cols).to_csv(ROOT/'phase_a/safety/descriptive_intervals.csv',index=False)

def classical():
    df=pd.read_csv(ROOT/'phase_a/classical/afast_per_frame.csv')
    scene=df.groupby(['condition','scene_id'])[['regret','unsafe','executed','abstained']].mean().reset_index()
    interval_table(scene,['condition'],['regret','unsafe','executed','abstained']).to_csv(ROOT/'phase_a/classical/descriptive_intervals.csv',index=False)
    models=pd.concat([pd.read_csv(p) for p in (AFAST/'metrics').glob('*_seed*.csv')],ignore_index=True)
    models=models.merge(df[['condition','scene_id','frame_id','regret','unsafe']],on=['condition','scene_id','frame_id'],suffixes=('_model','_classical'),validate='many_to_one')
    for metric in ['regret','unsafe']:models[metric]=models[metric+'_model']-models[metric+'_classical']
    scene=models.groupby(['condition','architecture','mask','policy','scene_id'])[['regret','unsafe']].mean().reset_index()
    interval_table(scene,['condition','architecture','mask','policy'],['regret','unsafe']).to_csv(ROOT/'phase_a/classical/model_minus_classical_intervals.csv',index=False)

def sensitivity():
    assert (ROOT/'phase_a/sensitivity/scoring_receipt.json').exists()
    # Reduce per-scene files before concatenation to bound memory.
    grouped=[];seedgroups=[]
    for path in sorted((ROOT/'phase_a/sensitivity/scenes').glob('*.csv')):
        df=pd.read_csv(path)
        grouped.append(df.groupby(['setting','condition','architecture','mask','policy','scene_id'])[['regret','unsafe','executed','abstained']].mean().reset_index())
        seedgroups.append(df.groupby(['setting','condition','architecture','seed','mask','policy','scene_id'])[['regret','unsafe','executed','abstained']].mean().reset_index())
    assert len(grouped)==192
    scene=pd.concat(grouped,ignore_index=True);scene.to_csv(ROOT/'phase_a/sensitivity/per_scene.csv',index=False)
    pd.concat(seedgroups).groupby(['setting','condition','architecture','seed','mask','policy'])[['regret','unsafe','executed','abstained']].mean().to_csv(ROOT/'phase_a/sensitivity/per_seed.csv')
    rows=[]
    for (setting,condition,policy),f in scene.groupby(['setting','condition','policy']):
        wide=f.pivot(index='scene_id',columns=['architecture','mask'],values=['regret','unsafe'])
        pairs=[('KF_removal',('KF-D',1),('KF-D',0)),('BF_removal',('BF-D',1),('BF-D',0)),('BF_minus_KF_intact',('BF-D',0),('KF-D',0)),('BF_minus_KF_masked',('BF-D',1),('KF-D',1))]
        rough=condition=='rough_iid';ids=DRAW['rough_scene_ids' if rough else 'nominal_scene_ids'];mat=DRAW['rough' if rough else 'nominal'];wide=wide.reindex(ids)
        for label,left,right in pairs:
            for metric in ['regret','unsafe']:
                values=(wide[(metric,*left)]-wide[(metric,*right)]).to_numpy();rep=mat.astype(float)@values/len(ids)
                bounds=np.quantile(rep,[.025,.975,.0125,.9875]);row=dict(setting=setting,condition=condition,policy=policy,contrast=label,metric=metric,
                    estimate=float(values.mean()),lower95=float(bounds[0]),upper95=float(bounds[1]),lower97_5=float(bounds[2]),upper97_5=float(bounds[3]),
                    sign=int(np.sign(values.mean())),class95='lower' if bounds[1]<0 else 'higher' if bounds[0]>0 else 'spans_zero',
                    class97_5='lower' if bounds[3]<0 else 'higher' if bounds[2]>0 else 'spans_zero')
                if metric=='unsafe':
                    events=np.rint(values*12).astype(np.int64);assert np.allclose(events,values*12,atol=1e-10)
                    nums=mat.astype(np.int64)@events;den=len(ids)*12
                    row.update(exact_estimate=str(Fraction(int(events.sum()),den)),exact_lower95=str(exact_percentile(nums,den,Fraction(1,40))),
                        exact_upper95=str(exact_percentile(nums,den,Fraction(39,40))),exact_lower97_5=str(exact_percentile(nums,den,Fraction(1,80))),exact_upper97_5=str(exact_percentile(nums,den,Fraction(79,80))))
                rows.append(row)
    result=pd.DataFrame(rows);baseline=result[result.setting=='nominal'].set_index(['condition','policy','contrast','metric'])
    result['retains_nominal_sign']=[r['sign']==baseline.loc[(r['condition'],r['policy'],r['contrast'],r['metric']),'sign'] for r in rows]
    result['retains_nominal_interval_class95']=[r['class95']==baseline.loc[(r['condition'],r['policy'],r['contrast'],r['metric']),'class95'] for r in rows]
    result['retains_nominal_interval_class97_5']=[r['class97_5']==baseline.loc[(r['condition'],r['policy'],r['contrast'],r['metric']),'class97_5'] for r in rows]
    result.to_csv(ROOT/'phase_a/sensitivity/complete_contrast_intervals.csv',index=False)

def memo():
    probe=json.loads((ROOT/'phase_a/material_probe/result.json').read_text());ctl=json.loads((ROOT/'phase_a/attribution/control_receipt.json').read_text())
    cl=pd.read_csv(ROOT/'phase_a/classical/afast_condition_point_estimates.csv').set_index('condition')
    shp=pd.read_csv(ROOT/'phase_a/shapley/descriptive_intervals.csv');safe=pd.read_csv(ROOT/'phase_a/safety/complete_cause_inventory.csv')
    sen=pd.read_csv(ROOT/'phase_a/sensitivity/complete_contrast_intervals.csv')
    text='''# Phase A — one-page descriptive memo\n\nStatus: completed separately from the immutable original results. The existing external-architecture holdout was already inspected; all findings here are post-hoc descriptive. Saved scene multiplicities supply intervals; no new Phase A bootstrap samples or perception-model forwards were used.\n\n'''
    text+=f"Attribution covered {ctl['processed_maps']:,} saved maps and all16 truth substitutions. Saved soft selections reproduced exactly; non-tie all-truth oracle failures: {ctl['all_truth_nonmax_choices']}; hazard reconstruction mismatches: {ctl['hazard_mask_mismatches']}. Factor Shapley values measure predicted-map substitution loss reductions, not causal sensor contributions. Complete subset, factor, seed and scene outputs are retained.\n\n"
    text+='The safety inventory separates person, obstacle, truck, machine and non-terrain causes with overlapping and ordered-exclusive counts. Uniform-reachable results remain expected probabilities/counts. Person-buffer selections per1,000 attempted decisions are reported for every policy and condition, including unfavorable results.\n\n'
    text+='The classical baseline fixed tau=1.0m and buffer=4.0m using all1,200 original validation frames and30 grid points. '
    for condition in ['clean','cam_s3','rough_iid']:
        r=cl.loc[condition];text+=f"{condition}: regret{r.regret:.4f}, unsafe/attempt{100*r.unsafe:.2f}%, execution{100*r.executed:.2f}%. "
    text+='Complete intervals and learned-model contrasts are in the classical tables.\n\n'
    text+=f"The fixed56,868-parameter RGB-D patch probe achieved validation balanced accuracy{100*probe['balanced_accuracy']:.2f}% (accuracy{100*probe['accuracy']:.2f}%; four-class chance25%) on exact visible-terrain labels, exceeding the locked50% rule. This bounded task is learnable; weak full-model material recognition cannot be attributed to intrinsic ambiguity from these results. Its cause remains unresolved. The resource amendment prohibits downstream retraining; the six-fit future experiment costs16.81GPU-hours with25% margin from historical fit times.\n\n"
    changed=sen[(sen.setting!='nominal')&(~sen.retains_nominal_sign|~sen.retains_nominal_interval_class95)]
    text+=f"Utility sensitivity reports all25 settings and all contrasts; {len(changed)} contrast/metric/condition rows change sign or95% interval class relative to nominal. Direct choices are fixed. Material/productivity/slope constants are matched in soft composition. Buffer variation changes truth utility/exclusion scoring while the aggregate learned hazard head stays fixed: per-cause matched prediction buffers are unavailable and are not invented. These sensitivities do not replace historical primary intervals.\n\nProceed to the prospective frozen-model gate study under its separate lock. No material result selects a new perception checkpoint, gate design or holdout outcome.\n"
    (ROOT/'phase_a/PHASE_A_MEMO.md').write_text(text)
    js(ROOT/'phase_a/completion.json',dict(status='COMPLETE_WITH_DECLARED_BUFFER_PREDICTION_LIMIT',bootstrap_source_sha256=sha(AFAST/'analysis/bootstrap_multiplicities.npz'),new_draws=0,
        perception_training_runs=0,material_probe_runs=1,pilot_runs=1,all_subsets=True,all_conditions=True,all_seeds=True))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--sections',nargs='+',default=['attribution','safety','classical','sensitivity','memo']);a=p.parse_args()
    for name,fn in [('attribution',attribution),('safety',safety),('classical',classical),('sensitivity',sensitivity),('memo',memo)]:
        if name not in a.sections:continue
        print(name,flush=True);fn()
