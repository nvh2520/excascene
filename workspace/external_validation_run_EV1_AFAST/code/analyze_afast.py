"""Locked, complete-case paired scene analysis. Only run after six final exports."""
import csv,json,pathlib,time
from fractions import Fraction
import numpy as np
import pandas as pd
from statistical_core import *
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
CONDITIONS=['clean','iid','all_s2','cam_s1','cam_s2','cam_s3','rough_iid']
METRICS=['regret','unsafe','executed','abstained','H_reachable_terrain','selected_utility','oracle',
    'global_feasible_gap','normalized_regret','reachable_cells','eligible_cells','exact_tie','empty_candidates','empty_reach']
COUNT_FIELDS=['attempted','unsafe','executed','abstained','exact_tie','empty_candidates','empty_reach',
    'reach_boundary','hazard_boundary','nonfinite_scores','nonfinite_reach','nonfinite_hazard']

def write_json(path,obj):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)):return None if not np.isfinite(x) else float(x)
        if isinstance(x,np.integer):return int(x)
        return x
    path.write_text(json.dumps(clean(obj),indent=2,allow_nan=False))

assert json.loads((ROOT/'records/all_six_final_checkpoints.json').read_text())['status']=='ALL_SIX_FINALS_COMPLETE'
frames=[]
for arch in ['BF-D','KF-D']:
    for seed in SEEDS:
        receipt=json.loads((ROOT/'records'/f'inference_{arch}_seed{seed}.json').read_text());assert receipt['status']=='COMPLETE'
        f=pd.read_csv(ROOT/'metrics'/f'{arch}_seed{seed}.csv');assert set(f.seed)=={seed} and set(f.architecture)=={arch}
        frames.append(f)
df=pd.concat(frames,ignore_index=True)
key=['architecture','seed','mask','condition','policy','scene_id','frame_id']
assert not df.duplicated(key).any() and set(df['policy'])=={'direct','soft'}
assert (df.attempted==1).all() and (df.unsafe<=df.executed).all() and (df.executed+df.abstained==1).all()
assert np.isfinite(df[['regret','unsafe','executed','oracle']].to_numpy()).all()
inventory=pd.read_csv(ROOT/'manifests/generated_files.csv')
expected=set(zip(inventory.condition,inventory.scene_id,inventory.frame_id))
for _,g in df.groupby(['architecture','seed','mask','policy']):assert set(zip(g.condition,g.scene_id,g.frame_id))==expected
assert len(df)==len(inventory)*24

# Equal seeds for each frame; equal eligible slots for each scene; equal scenes.
framekeys=['architecture','mask','condition','policy','scene_id','frame_id']
averaged=df.groupby(framekeys,sort=True)[METRICS].mean().reset_index()
scenekeys=['architecture','mask','condition','policy','scene_id']
scenes=averaged.groupby(scenekeys,sort=True)[METRICS].mean().reset_index()
slotcounts=averaged.groupby(scenekeys).size().rename('eligible_slots').reset_index()
scenes=scenes.merge(slotcounts,on=scenekeys,validate='one_to_one')
scenes.to_csv(OUT/'per_scene_all_arms.csv',index=False)
seedscenes=df.groupby(['architecture','seed','mask','condition','policy','scene_id'],sort=True)[METRICS].mean().reset_index()
seedscenes.to_csv(OUT/'per_seed_per_scene_all_arms.csv',index=False)
seedsummary=seedscenes.groupby(['architecture','seed','mask','condition','policy'],sort=True)[METRICS].mean().reset_index()
seedsummary.to_csv(OUT/'per_seed_condition_results.csv',index=False)

summaries=[]
for keys,g in scenes.groupby(['architecture','mask','condition','policy'],sort=True):
    arch,mask,condition,policy=keys;raw=df[(df.architecture==arch)&(df['mask']==mask)&(df.condition==condition)&(df.policy==policy)]
    row=dict(architecture=arch,mask=int(mask),condition=condition,policy=policy,scenes=len(g),
        slots=int(g.eligible_slots.sum()),fitted_seeds=3,weighting='equal fitted seeds, equal eligible slots within scene, equal scenes')
    for metric in METRICS:
        row[metric]=float(g[metric].mean());row[metric+'_undefined_scenes']=int(g[metric].isna().sum())
    for field in COUNT_FIELDS:row[field+'_count']=int(raw[field].sum())
    row['unsafe_per_execution_pooled_diagnostic']=None if row['executed_count']==0 else row['unsafe_count']/row['executed_count']
    row['unsafe_per_execution_undefined']=int(row['executed_count']==0)
    for stratum in ['zero','low','positive']:row['oracle_'+stratum+'_frame_seed_count']=int((raw.utility_stratum==stratum).sum())
    row['undefined_normalized_regret_frame_seed_count']=int(raw.normalized_regret.isna().sum())
    summaries.append(row)
summary=pd.DataFrame(summaries);summary.to_csv(OUT/'condition_results.csv',index=False)
pools=[]
for keys,g in summary.groupby(['architecture','mask','policy']):
    h=g[g.condition.isin(['iid','all_s2','rough_iid'])];assert len(h)==3
    row=dict(zip(['architecture','mask','policy'],keys));row.update(condition='fixed_one_third_iid_all_s2_rough',weights='1/3,1/3,1/3')
    for metric in METRICS:row[metric]=float(h[metric].mean()) if h[metric].notna().all() else np.nan
    pools.append(row)
pool=pd.DataFrame(pools);pool.to_csv(OUT/'fixed_weight_pool_results.csv',index=False)

# Every architectural comparison is descriptive; no superiority/equivalence test.
comparisons=[]
for table in [summary,pool]:
    for keys,g in table.groupby(['mask','condition','policy']):
        b=g[g.architecture=='BF-D'].iloc[0];k=g[g.architecture=='KF-D'].iloc[0]
        row=dict(zip(['mask','condition','policy'],keys));row['contrast']='BF-D minus KF-D';row['inference']='descriptive only'
        for metric in METRICS:row[metric]=float(b[metric]-k[metric])
        comparisons.append(row)
pd.DataFrame(comparisons).to_csv(OUT/'architecture_comparisons.csv',index=False)

pairedkeys=['architecture','seed','condition','policy','scene_id','frame_id']
pair=df[df['mask']==0].merge(df[df['mask']==1],on=pairedkeys,suffixes=('_intact','_masked'),validate='one_to_one')
assert len(pair)==len(df)//2
for metric in METRICS:pair[metric+'_effect']=pair[metric+'_masked']-pair[metric+'_intact']
pair.to_csv(OUT/'paired_frame_metrics.csv',index=False)
effects=pair.groupby(['architecture','seed','condition','policy','scene_id'])[[x+'_effect' for x in METRICS]].mean().reset_index()
effects.to_csv(OUT/'per_seed_per_scene_removal_effects.csv',index=False)
removal=effects.groupby(['architecture','condition','policy','scene_id'])[[x+'_effect' for x in METRICS]].mean().reset_index()
removal.groupby(['architecture','condition','policy'])[[x+'_effect' for x in METRICS]].mean().reset_index().to_csv(OUT/'camera_removal_costs.csv',index=False)

nominal_ids=sorted(inventory[inventory.condition!='rough_iid'].scene_id.unique());rough_ids=sorted(inventory[inventory.condition=='rough_iid'].scene_id.unique())
rng=np.random.default_rng(2026101303)
nominal=multiplicities(nominal_ids,rng);rough=multiplicities(rough_ids,rng)
np.savez_compressed(OUT/'bootstrap_multiplicities.npz',nominal_scene_ids=nominal_ids,rough_scene_ids=rough_ids,nominal=nominal,rough=rough)

primary=pair[(pair.architecture=='BF-D')&(pair.condition=='cam_s3')&(pair.policy=='direct')].copy()
primary.to_csv(OUT/'primary_paired_frames.csv',index=False)
rows=[]
for sid,g in primary.groupby('scene_id',sort=True):
    assert set(g.seed)==set(SEEDS) and len(g)%3==0
    rows.append(dict(scene_id=int(sid),eligible_slots=len(g)//3,attempted_paired_seed_slots=len(g),
        raw_regret_effect=float(g.regret_effect.mean()),unsafe_difference_count=int((g.unsafe_masked-g.unsafe_intact).sum()),
        unsafe_intact_count=int(g.unsafe_intact.sum()),unsafe_masked_count=int(g.unsafe_masked.sum()),
        executions_intact_count=int(g.executed_intact.sum()),executions_masked_count=int(g.executed_masked.sum())))
primscene=pd.DataFrame(rows);assert list(primscene.scene_id)==nominal_ids
primscene.to_csv(OUT/'primary_scene_inputs.csv',index=False)
cr,cri,crdraws=continuous_bootstrap(primscene.raw_regret_effect,nominal)
cu,cui,cudraws,cudenom=unsafe_bootstrap(primscene.unsafe_difference_count,primscene.attempted_paired_seed_slots,nominal)
np.savez_compressed(OUT/'primary_bootstrap_replicates.npz',raw_regret=crdraws,unsafe_numerators=cudraws,unsafe_denominator=np.array(cudenom))
result=dict(status='COMPLETE_STATISTICAL_ESTIMATES',planned_logical_forwards=39936,actual_logical_forwards=sum(json.loads((ROOT/'records'/f'inference_{arch}_seed{seed}.json').read_text())['completed_logical_forwards'] for arch in ['BF-D','KF-D'] for seed in SEEDS),
    accepted_records=len(inventory),nominal_scenes=len(nominal_ids),rough_scenes=len(rough_ids),
    bootstrap_draws=20000,bootstrap_seed=2026101303,adjusted_quantiles=['1/80','79/80'],ordinary_quantiles=['1/40','39/40'],
    interval_scope='Conditional environmental uncertainty across the three fitted seeds; approximate percentile coverage. Nominal Bonferroni familywise 95%, not exact coverage or a retraining-seed distribution.',
    endpoints=[dict(name='AFAST-C-R',estimate=cr,ordinary95=list(cri[:2]),adjusted97_5=list(cri[2:]),interpretation=direction(cri[2],cri[3])),
               dict(name='AFAST-C-U',estimate=float(cu),estimate_exact=str(cu),ordinary95=list(map(float,cui[:2])),adjusted97_5=list(map(float,cui[2:])),
                    ordinary95_exact=list(map(str,cui[:2])),adjusted97_5_exact=list(map(str,cui[2:])),interpretation=direction(cui[2],cui[3]),
                    bootstrap_integer_denominator=cudenom,unsafe_intact_count=int(primary.unsafe_intact.sum()),unsafe_masked_count=int(primary.unsafe_masked.sum()),
                    attempts_per_state=int(len(primary)),executions_intact=int(primary.executed_intact.sum()),executions_masked=int(primary.executed_masked.sum()))],
    joint_lower_loss=bool(cri[3]<0 and cui[3]<0),utc_unix=time.time())
write_json(OUT/'primary_results.json',result)

seedrows=[]
for seeds,label in [(list(SEEDS),'all_three')]+[([s],f'seed_{s}') for s in SEEDS]+[([s for s in SEEDS if s!=omit],f'leave_out_{omit}') for omit in SEEDS]:
    p=primary[primary.seed.isin(seeds)]
    g=p.groupby('scene_id')[['regret_effect','unsafe_effect','executed_effect']].mean()
    seedrows.append(dict(summary=label,seeds=';'.join(map(str,seeds)),scenes=len(g),raw_regret_effect=float(g.regret_effect.mean()),
        unsafe_effect=float(g.unsafe_effect.mean()),execution_effect=float(g.executed_effect.mean()),
        unsafe_difference_integer=int((p.unsafe_masked-p.unsafe_intact).sum()),attempts_per_state=len(p)))
pd.DataFrame(seedrows).to_csv(OUT/'primary_per_seed_leave_one_out.csv',index=False)

# Full pixel diagnostics reuse each prediction once, independently of selector.
confusions={}
for arch in ['BF-D','KF-D']:
    for seed in SEEDS:
        with np.load(ROOT/'metrics'/f'{arch}_seed{seed}_confusions.npz') as z:
            for i in range(len(z['seed'])):
                key=(arch,seed,int(z['mask'][i]),str(z['condition'][i]));slot=confusions.setdefault(key,[np.zeros((6,6),np.int64),np.zeros((4,4),np.int64)])
                slot[0]+=z['semantic'][i];slot[1]+=z['material'][i]
for arch in ['BF-D','KF-D']:
    for mask in [0,1]:
        for cond in CONDITIONS:
            confusions[(arch,'all_three',mask,cond)]=[sum((confusions[(arch,s,mask,cond)][k] for s in SEEDS),np.zeros((6,6) if k==0 else (4,4),np.int64)) for k in [0,1]]
diagnostics=[]
for key,cs in confusions.items():
    arch,seed,mask,condition=key
    raw=df[(df.architecture==arch)&(df['mask']==mask)&(df.condition==condition)&(df.policy=='direct')]
    if seed!='all_three':raw=raw[raw.seed==seed]
    tp,fp,fn,tn=(int(raw['hazard_'+x].sum()) for x in ['tp','fp','fn','tn'])
    diagnostics.append(dict(architecture=arch,seed=seed,mask=mask,condition=condition,pixel_weighting='pooled pixel counts; diagnostic, not primary scene estimand',
        semantic_confusion=cs[0].tolist(),semantic_classes=class_statistics(cs[0]),material_true_terrain_confusion=cs[1].tolist(),material_classes=class_statistics(cs[1]),
        hazard_tp=tp,hazard_fp=fp,hazard_fn=fn,hazard_tn=tn,hazard_precision_denominator=tp+fp,hazard_recall_denominator=tp+fn,
        hazard_precision=None if tp+fp==0 else tp/(tp+fp),hazard_recall=None if tp+fn==0 else tp/(tp+fn),
        utility_absolute_sum=float(raw.utility_absolute_sum.sum()),utility_cells=int(raw.utility_cells.sum()),
        utility_mae=float(raw.utility_absolute_sum.sum()/raw.utility_cells.sum()),material_terrain_cells=int(raw.material_terrain_cells.sum())))
write_json(OUT/'pixel_diagnostics.json',diagnostics)
write_json(OUT/'analysis_completion.json',dict(status='COMPLETE',metrics_rows=len(df),primary_joint_lower_loss=result['joint_lower_loss'],utc_unix=time.time()))
print(json.dumps(result,indent=2))
