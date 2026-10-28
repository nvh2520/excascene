"""Prospective fixed-function controls using original train inputs and analytic arrays."""
import json,pathlib,time
import numpy as np
from generate_holdout import specs,camera_latents,degrade_shared,lidar_latents,rng
from scoring import choose,evaluate
ROOT=pathlib.Path(__file__).resolve().parents[1]
rows=[]
def ok(name,**kw):rows.append(dict(test=name,passed=True,**kw))
sp=specs(0,0,False);assert set(sp)=={'clean','iid','all_s2','cam_s1','cam_s2','cam_s3'}
assert all(len(sp[c]['cam']['cab']['photo'])==5 for c in ['cam_s1','cam_s2','cam_s3'])
assert sum(x['drop'] for x in sp['cam_s3']['cam'].values())==1
assert all(not x['drop'] for c in ['clean','cam_s1','cam_s2'] for x in sp[c]['cam'].values())
L=camera_latents(0,0,0);before={k:v.copy() for k,v in L.items()}
with np.load(ROOT.parent/'data/train/f_000000_0.npz') as d: rgb=d['rgb'][0];depth=d['depth'][0].astype(np.float32)
for c in ['clean','cam_s1','cam_s2','cam_s3']:
    a=degrade_shared(rgb,depth,sp[c]['cam']['cab']['photo'],L)
    b=degrade_shared(rgb,depth,sp[c]['cam']['cab']['photo'],L)
    for x,y in zip(a,b):assert np.array_equal(x,y)
    for k in L:assert np.array_equal(L[k],before[k])
ok('original_training_camera_shared_latents_immutable_repeatable_five_compound_mechanisms')
assert not np.array_equal(rng(0,0,10).random(10),rng(0,0,40).random(10))
ok('named_pose_fault_namespaces_separate_no_intervention_key')
reach=np.array([[1.,.049],[1.,1.]],np.float32);u=np.array([[.7,.4],[0.,0.]],np.float32)
d=dict(gt_reach=reach,gt_aff=u,gt_haz=np.array([[0,0],[1,1]],np.uint8),gt_sem=np.array([[0,0],[5,5]],np.uint8),z_target=np.float32(0))
p=dict(aff=np.array([[.1,2.],[.9,.2]],np.float32),haz_prob=np.array([[.6,0],[.9,.7]],np.float32))
r=evaluate(p,d,'direct');assert r['abstained']==1 and r['executed']==0 and r['unsafe']==0 and r['regret']==float(u[0,0])
ok('strict_no_fallback_abstention_full_feasible_oracle_regret')
p['haz_prob'][0,0]=.5;r=evaluate(p,d,'direct');assert (r['pick_row'],r['pick_col'])==(0,0) and r['regret']==0
reach[0,1]=.05;r=evaluate(p,d,'direct');assert (r['pick_row'],r['pick_col'])==(0,1) and r['oracle']==float(u[0,0])
ok('inclusive_reach_hazard_boundaries_common_oracle_before_predicted_hazard_filter')
p['aff'][1,0]=3;p['haz_prob'][1,0]=.5;r=evaluate(p,d,'direct');assert r['unsafe']==1 and r['executed']==1
ok('unsafe_event_arithmetic_includes_executed_nonterrain_hazard')
score=np.ones((2,2));r=choose(score,np.ones((2,2)),np.zeros((2,2)))
assert r[0]==(0,0) and r[1]==4
ok('row_major_tie_selector')
d['gt_aff']=np.zeros((2,2),np.float32);r=evaluate(p,d,'direct');assert r['oracle']==0 and r['regret']==0 and np.isnan(r['normalized_regret'])
ok('zero_oracle_kept_raw_endpoint_normalized_quantity_undefined')
(ROOT/'tests/generator_selector_controls.json').write_text(json.dumps(dict(status='PASSED',tests=rows,new_holdout_generated=0,non_training_forwards=0,utc_unix=time.time()),indent=2))
print(json.dumps(rows,indent=2))
