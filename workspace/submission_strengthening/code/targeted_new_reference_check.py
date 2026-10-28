"""Narrow independent arithmetic/preservation check; never imports scoring code."""
import csv,hashlib,json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
S=ROOT/'submission_strengthening';R=S/'references';A=ROOT/'external_validation_run_EV1_AFAST'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def passed(name,**details):checks.append(dict(name=name,status='PASS',**details))

plan=json.loads((S/'records/descriptive_plan_lock.json').read_text())
lock=json.loads((R/'prior_lock.json').read_text())
score_job=json.loads((S/'resources/job_references_holdout_scoring.json').read_text())
assert plan['unix']<lock['unix_time']<score_job['start_unix']
assert lock['plan_sha256']==sha(S/'descriptive_reference_plan.md')
for name,h in lock['files'].items():assert sha(R/name)==h
train=pd.read_csv(R/'training_inventory.csv');source_train=pd.read_csv(A/'manifests/train_validation_inventory.csv')
source_train=source_train[source_train.split=='train'].reset_index(drop=True)
pd.testing.assert_frame_equal(train,source_train)
assert len(train)==12000 and train.scene_id.nunique()==3000 and set(train.split)=={'train'}
assert set(zip(train.scene_id,train.frame_id))=={(s,t) for s in range(3000) for t in range(4)}
prior=json.loads((R/'training_material_prior.json').read_text())
assert sum(prior['counts'])==prior['total_true_terrain_pixels']==100183410
assert prior['majority_class_id']==int(np.argmax(prior['counts']))==1
np.testing.assert_allclose(prior['probabilities'],np.array(prior['counts'])/sum(prior['counts']),rtol=0,atol=1e-16)
with np.load(R/'training_prior_maps.npz') as z:
    prior_u=z['mean_A_train'];prior_h=z['mean_h_train']
    assert prior_u.shape==prior_h.shape==(96,96) and np.isfinite(prior_u).all() and np.isfinite(prior_h).all()
passed('training-only inventory and pre-score prior custody',frames=12000,material_majority='soil',
       independent_restream=False,limit='checks inventory, frozen construction provenance and artifacts; does not repeat full prior stream')

refs=pd.read_csv(R/'reference_per_frame.csv',keep_default_na=False,float_precision='round_trip')
assert len(refs)==6656 and not refs.duplicated(['reference','condition','scene_id','frame_id']).any()
inventory=pd.read_csv(A/'manifests/generated_files.csv')
expected=set(zip(inventory.condition,inventory.scene_id,inventory.frame_id))
for ref,g in refs.groupby('reference'):
    assert set(zip(g.condition,g.scene_id,g.frame_id))==expected
    assert (g.attempted==1).all() and (g.executed==1).all() and (g.abstained==0).all()
    assert (g.eligible_cells>0).all() and (g.nonfinite_scores==0).all()
    if ref=='R0':
        assert (g.unsafe_event=='').all() and (g.pick_row=='').all() and (g.pick_col=='').all()
        n=g.expected_unsafe_numerator.to_numpy(np.int64);d=g.expected_unsafe_denominator.to_numpy(np.int64)
        np.testing.assert_allclose(g.unsafe.to_numpy(float),n/d,rtol=0,atol=0)
        assert (d==g.reachable_cells.to_numpy()).all() and (n>=0).all() and (n<=d).all()
    else:
        assert set(g.unsafe_event.astype(int))<={0,1}
        assert (g.unsafe_event.astype(int)==g.unsafe).all()
        assert set(zip(g.pick_row.astype(int),g.pick_col.astype(int)))=={(45,49)}
passed('reference support and event/expectation distinction',records_per_policy=3328,no_exclusions=True,no_abstentions=True,R0_integer_events_fabricated=False)

# Independently sum authoritative cells for the first clean, severe and rough records.
spot=[]
for condition in ['clean','cam_s3','rough_iid']:
    row=inventory[inventory.condition==condition].sort_values(['scene_id','frame_id']).iloc[0]
    with np.load(A/row.path) as z:
        reach=z['gt_reach'];u=z['gt_aff'].astype(np.float64);haz=z['gt_haz'];sem=z['gt_sem']
    candidates=np.flatnonzero((np.isfinite(reach)&(reach>=.05)).ravel())
    expected_utility=sum(float(u.flat[k]) for k in candidates)/len(candidates)
    expected_unsafe=Fraction(sum(int(haz.flat[k]>.5 or sem.flat[k]!=0) for k in candidates),len(candidates))
    oracle=max(float(u.flat[k]) for k in candidates)
    r0=refs[(refs.reference=='R0')&(refs.condition==condition)&(refs.scene_id==row.scene_id)&(refs.frame_id==row.frame_id)].iloc[0]
    np.testing.assert_allclose([r0.regret,r0.unsafe],[oracle-expected_utility,float(expected_unsafe)],rtol=0,atol=1e-15)
    eligible=[k for k in candidates if np.isfinite(prior_u.flat[k]) and np.isfinite(prior_h.flat[k]) and prior_h.flat[k]<=.5]
    chosen=max(eligible,key=lambda k:float(prior_u.flat[k]))  # list order preserves first row-major max
    r1=refs[(refs.reference=='R1')&(refs.condition==condition)&(refs.scene_id==row.scene_id)&(refs.frame_id==row.frame_id)].iloc[0]
    assert divmod(chosen,96)==(int(r1.pick_row),int(r1.pick_col))
    assert int(haz.flat[chosen]>.5 or sem.flat[chosen]!=0)==int(r1.unsafe_event)
    assert oracle-float(u.flat[chosen])==r1.regret
    spot.append(dict(condition=condition,scene_id=int(row.scene_id),frame_id=int(row.frame_id),R0_unsafe_exact=str(expected_unsafe),R1_selected_flat=int(chosen)))
passed('independent authoritative-cell arithmetic spots',records=spot)

models=pd.concat([pd.read_csv(A/'metrics'/f'{ar}_seed{s}.csv',float_precision='round_trip') for ar in ['BF-D','KF-D'] for s in [10,11,12]])
assert len(models)==79872 and (models.abstained==0).all()
metrics=['regret','unsafe','executed','abstained']
refscenes=refs.groupby(['reference','condition','scene_id'])[metrics].mean()
modelframes=models.groupby(['architecture','mask','policy','condition','scene_id','frame_id'])[metrics].mean()
modelscenes=modelframes.groupby(['architecture','mask','policy','condition','scene_id']).mean()
table=pd.read_csv(R/'reference_comparison_intervals.csv',keep_default_na=False,float_precision='round_trip')
assert len(table)==608
conditions=['clean','iid','all_s2','cam_s1','cam_s2','cam_s3','rough_iid']
pool='fixed_one_third_iid_all_s2_rough'
scene_values={}
def get_scene(ref,model,mask,policy,condition,metric):
    key=(ref,model,str(mask),policy,condition,metric)
    if key in scene_values:return scene_values[key]
    b=refscenes.loc[(ref,condition)][metric].sort_index()
    if model=='reference':x=b
    elif model=='R1':x=refscenes.loc[('R1',condition)][metric].sort_index()-b
    else:x=modelscenes.loc[(model,int(mask),policy,condition)][metric].sort_index()-b
    scene_values[key]=x;return x
max_point_error=0
for r in table.itertuples():
    if r.condition==pool:
        expected_point=sum(get_scene(r.reference,r.model,r.camera_mask,r.model_policy,c,r.metric).mean() for c in ['iid','all_s2','rough_iid'])/3
    else:expected_point=get_scene(r.reference,r.model,r.camera_mask,r.model_policy,r.condition,r.metric).mean()
    error=abs(r.estimate-expected_point);max_point_error=max(max_point_error,error)
    assert error<2e-15,(r,expected_point)
    scale=1 if r.metric=='regret' else 100
    np.testing.assert_allclose([r.display_estimate,r.display_lower95,r.display_upper95],np.array([r.estimate,r.lower95,r.upper95])*scale,rtol=0,atol=1e-12)
    assert r.lower95<=r.upper95 and r.bootstrap_draws==20000
    assert 'POSTHOC' in r.evidence_class and 'unadjusted' in r.interval
passed('all new reference points and display-unit conversions',rows=608,max_point_roundoff=max_point_error)

with np.load(A/'analysis/bootstrap_multiplicities.npz') as z:
    draws={'nominal':z['nominal'].astype(np.int64),'rough':z['rough'].astype(np.int64)}
    scene_ids={'nominal':z['nominal_scene_ids'],'rough':z['rough_scene_ids']}
targets=[('R0','reference','NA','NA','clean','unsafe'),('R1','reference','NA','NA','clean','regret'),
         ('R0','BF-D','0','direct','clean','regret'),('R1','BF-D','0','direct','rough_iid','unsafe'),
         ('R0','R1','NA','fixed_prior','clean','unsafe'),('R1','KF-D','1','soft',pool,'regret')]
interval_checks=[]
for ref,model,mask,policy,condition,metric in targets:
    conds=['iid','all_s2','rough_iid'] if condition==pool else [condition];parts=[]
    for cond in conds:
        stratum='rough' if cond=='rough_iid' else 'nominal'
        values=get_scene(ref,model,mask,policy,cond,metric)
        assert list(values.index)==list(scene_ids[stratum])
        parts.append(np.einsum('ij,j->i',draws[stratum],values.to_numpy(np.float64),optimize=False)/len(values))
    replicates=sum(parts)/len(parts)
    expected_ci=np.percentile(replicates,[2.5,97.5],method='linear')
    row=table[(table.reference==ref)&(table.model==model)&(table.camera_mask.astype(str)==mask)&(table.model_policy==policy)&(table.condition==condition)&(table.metric==metric)].iloc[0]
    np.testing.assert_allclose([row.lower95,row.upper95],expected_ci,rtol=0,atol=2e-14)
    interval_checks.append(dict(reference=ref,model=model,mask=mask,policy=policy,condition=condition,metric=metric,interval=list(expected_ci)))
rough=table[(table.reference=='R1')&(table.model=='BF-D')&(table.camera_mask.astype(str)=='0')&(table.model_policy=='direct')&(table.condition=='rough_iid')&(table.metric=='unsafe')].iloc[0]
assert rough.lower95<0<rough.upper95 and Fraction(rough.estimate_exact)==Fraction(-65,768)
unfavorable=table[(table.reference=='R0')&(table.model=='R1')&(table.condition=='clean')&(table.metric=='unsafe')].iloc[0]
assert unfavorable.estimate>0 and unfavorable.lower95<0<unfavorable.upper95
passed('targeted saved-draw percentile reconstruction and unfavorable results',checked=interval_checks,new_draws=0)

preservation=[]
for original,candidate in [
    ('revision_v5/validation_run/primary_results.csv','paper_submission_candidate/data/primary_results.csv'),
    ('revision_v5/validation_run/primary_results.csv','release_candidate/statistics/published_v5_2/primary_results.csv'),
    ('paper_v5_2/tables/primary.tex','paper_submission_candidate/tables/primary.tex'),
    ('paper_v5_2/tables/afast_primary.tex','paper_submission_candidate/tables/afast_primary.tex'),
    ('external_validation_run_EV1_AFAST/analysis/primary_results.json','release_candidate/afast/analysis/primary_results.json')]:
    h=sha(ROOT/original);assert h==sha(ROOT/candidate),(original,candidate)
    preservation.append(dict(original=original,candidate=candidate,sha256=h))
passed('targeted original-primary byte preservation',files=preservation,historical_statistical_campaign_rerun=False)

cross=pd.read_csv(S/'chronology_crosswalk.csv');assert len(cross)==10 and len(cross.columns)==9
passed('chronology crosswalk parse',rows=10)
receipt=dict(status='PASS_TARGETED_NEW_REFERENCE_AND_PRIMARY_PRESERVATION_CHECK',checks=checks,
    scope='Independent new reference arithmetic and selected intervals; primary copies only hash-compared. No general historical audit.',
    code_sha256=sha(Path(__file__)),new_neural_forwards=0,new_sensor_generation=0,new_bootstrap_draws=0)
(S/'records/targeted_reference_and_preservation_check.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':receipt['status'],'checks':len(checks),'max_point_roundoff':max_point_error},indent=2))
