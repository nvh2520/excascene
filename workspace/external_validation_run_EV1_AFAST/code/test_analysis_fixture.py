"""End-to-end scalar-only analytic fixture, isolated from real fits and holdout."""
import csv,json,pathlib,shutil,subprocess,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];fixture=ROOT/'tests/analysis_fixture'
assert not fixture.exists();fixture.mkdir()
for d in ['code','metrics','records','manifests']: (fixture/d).mkdir()
for name in ['analyze_afast.py','statistical_core.py','reconstruct_primary.py']:shutil.copy(ROOT/'code'/name,fixture/'code'/name)
(fixture/'records/all_six_final_checkpoints.json').write_text(json.dumps(dict(status='ALL_SIX_FINALS_COMPLETE',fixture_only=True)))
conditions=['clean','iid','all_s2','cam_s1','cam_s2','cam_s3','rough_iid'];inventory=[]
for condition in conditions:
    for sid in ([310000,310001] if condition=='rough_iid' else [300000,300001]):
        # Vary valid slot counts to check that the scene estimand survives.
        for frame in range(1 if sid%2==0 else 4):inventory.append(dict(condition=condition,scene_id=sid,frame_id=frame))
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csvwrite(fixture/'manifests/generated_files.csv',inventory)
for arch in ['BF-D','KF-D']:
    for seed in [10,11,12]:
        rows=[];conf=[];material=[];keys=[]
        for mask in [0,1]:
            for item in inventory:
                bad=int(not mask and item['scene_id']%2==0)
                for policy in ['direct','soft']:
                    row=dict(architecture=arch,seed=seed,mask=mask,policy=policy,**item,attempted=1,regret=float(bad),unsafe=bad,
                        executed=1,abstained=0,H_reachable_terrain=.1,selected_utility=1.-bad,oracle=1.,global_feasible_gap=0.,
                        normalized_regret=float(bad),reachable_cells=4,eligible_cells=4,exact_tie=0,empty_candidates=0,empty_reach=0,
                        reach_boundary=0,hazard_boundary=0,nonfinite_scores=0,nonfinite_reach=0,nonfinite_hazard=0,
                        utility_stratum='positive',hazard_tp=0,hazard_fp=1,hazard_fn=0,hazard_tn=3,utility_absolute_sum=.4,
                        utility_cells=4,material_terrain_cells=4)
                    rows.append(row)
                c=np.zeros((6,6),np.int64);c[0,0]=3;c[0,1]=1;conf.append(c)
                m=np.zeros((4,4),np.int64);m[0,0]=4;material.append(m);keys.append((mask,item))
        name=arch+f'_seed{seed}';csvwrite(fixture/'metrics'/f'{name}.csv',rows)
        np.savez_compressed(fixture/'metrics'/f'{name}_confusions.npz',semantic=conf,material=material,
            seed=np.full(len(keys),seed),mask=[k[0] for k in keys],scene_id=[k[1]['scene_id'] for k in keys],
            frame_id=[k[1]['frame_id'] for k in keys],condition=[k[1]['condition'] for k in keys])
        (fixture/'records'/f'inference_{name}.json').write_text(json.dumps(dict(status='COMPLETE',completed_logical_forwards=2*len(inventory),fixture_only=True)))
subprocess.run([sys.executable,str(fixture/'code/analyze_afast.py')],check=True,stdout=subprocess.DEVNULL)
subprocess.run([sys.executable,str(fixture/'code/reconstruct_primary.py')],check=True,stdout=subprocess.DEVNULL)
r=json.loads((fixture/'analysis/primary_results.json').read_text())
assert r['endpoints'][0]['estimate']==-.5 and r['endpoints'][1]['estimate_exact']=='-1/2'
diagnostics=json.loads((fixture/'analysis/pixel_diagnostics.json').read_text())
assert all(row['semantic_classes'][1]['iou']==0 and row['semantic_classes'][2]['iou'] is None and row['hazard_recall'] is None for row in diagnostics)
receipt=dict(status='PASSED',fixture_only=True,network_forwards=0,real_holdout_accessed=False,
    unequal_slots_equal_scene_effect=-.5,primary_exact_unsafe_effect='-1/2',scalar_reconstruction=True,
    undefined_zero_hazard_recall=True,absent_class_false_positive_iou_zero=True,architecture_and_pool_and_removal_tables=True)
(ROOT/'tests/analysis_fixture_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
