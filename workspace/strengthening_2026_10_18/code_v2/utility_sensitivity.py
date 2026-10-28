"""Locked utility grid; predictions frozen, direct choices fixed, saved draws only."""
from common import *
import argparse,itertools,time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
SAVED={}

def settings():
    rows=[dict(setting='nominal',material_scale=1.,productivity=1.2,slope_low=40.,slope_high=60.,buffer_scale=1.,design='baseline')]
    for i,(m,p,s,b) in enumerate(itertools.product([.75,1.25],[.9,1.5],[(35.,55.),(45.,65.)],[.75,1.25])):
        rows.append(dict(setting=f'grid_{i:02d}',material_scale=m,productivity=p,slope_low=s[0],slope_high=s[1],buffer_scale=b,design='factorial'))
    for field,values in [('material_scale',[.75,1.25]),('productivity',[.9,1.5]),('slope',[(35.,55.),(45.,65.)]),('buffer_scale',[.75,1.25])]:
        for j,value in enumerate(values):
            row=dict(rows[0],setting=f'oat_{field}_{j}',design='one_factor')
            if field=='slope':row.update(slope_low=value[0],slope_high=value[1])
            else:row[field]=value
            rows.append(row)
    return rows

def worker(sid):
    scene,poses=historical_scene(sid);rows=[];max_truth_difference=0.
    conditions=['rough_iid'] if sid>=310000 else ['clean','iid','all_s2','cam_s1','cam_s2','cam_s3']
    grid=settings();out=ROOT/f'phase_a/sensitivity/scenes/{sid}.csv';assert not out.exists()
    for frame,q in poses.items():
        for condition in conditions:
            d=load_npz(AFAST/f'data/{condition}/f_{sid:06d}_{frame}.npz',['gt_H','gt_mat','gt_sem','gt_haz','gt_reach','gt_aff','z_target'])
            feasible=np.isfinite(d['gt_reach'])&(d['gt_reach']>=.05)
            hazards={b:np.any(safety_masks(scene,q,d['gt_sem'],b)[:4],axis=0) for b in [.75,1.,1.25]}
            assert np.array_equal(hazards[1.],d['gt_haz']>.5)
            mat=np.eye(4,dtype=np.float32)[d['gt_mat']].transpose(2,0,1);terrain=(d['gt_sem']==0).astype(np.float32)
            truths={}
            for s in grid:
                haz=hazards[s['buffer_scale']]
                u=soft_score(d['gt_H'],mat,terrain,haz.astype(np.float32),d['gt_reach'],scene.z_target,
                    s['material_scale'],s['productivity'],(s['slope_low'],s['slope_high']))
                truths[s['setting']]=(u,haz|(d['gt_sem']!=0),float(u[feasible].max()) if feasible.any() else 0.)
            max_truth_difference=max(max_truth_difference,float(np.abs(truths['nominal'][0]-d['gt_aff']).max()))
            assert max_truth_difference<=2e-7,('nominal truth discrepancy',max_truth_difference)
            for arch in ['KF-D','BF-D']:
                for seed in [10,11,12]:
                    for mask in [0,1]:
                        pred=load_npz(AFAST/f'predictions/{arch}_seed{seed}/mask{mask}/{condition}/{sid}_{frame}.npz')
                        direct=SAVED[(sid,frame,condition,arch,seed,mask,'direct')]
                        for s in grid:
                            u,unsafe,oracle=truths[s['setting']]
                            score=soft_score(pred['H'],pred['mat_prob'],pred['sem_prob'][0],pred['haz_prob'],d['gt_reach'],d['z_target'],
                                s['material_scale'],s['productivity'],(s['slope_low'],s['slope_high']))
                            soft=selected_metrics(score,d,truth_utility=u,truth_unsafe=unsafe)
                            if s['setting']=='nominal':
                                old=SAVED[(sid,frame,condition,arch,seed,mask,'soft')]
                                assert (old['pick_row'],old['pick_col'])==(soft['pick_row'],soft['pick_col'])
                            pick=None if direct['abstained'] else (int(direct['pick_row']),int(direct['pick_col']))
                            dm=dict(regret=oracle-(float(u[pick]) if pick else 0),unsafe=int(pick is not None and unsafe[pick]),
                                executed=int(pick is not None),abstained=int(pick is None),pick_row=-1 if pick is None else pick[0],pick_col=-1 if pick is None else pick[1])
                            for policy,met in [('direct',dm),('soft',soft)]:
                                rows.append(dict(condition=condition,scene_id=sid,frame_id=frame,architecture=arch,seed=seed,mask=mask,
                                    setting=s['setting'],policy=policy,regret=met['regret'],unsafe=met['unsafe'],executed=met['executed'],abstained=met['abstained'],
                                    pick_row=met['pick_row'],pick_col=met['pick_col'],buffer_prediction_scope='frozen_nominal_hazard_head'))
    csvwrite(out,rows)
    return dict(scene_id=sid,rows=len(rows),max_nominal_truth_difference=max_truth_difference)

def main():
    global SAVED
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    csvwrite(ROOT/'phase_a/sensitivity/settings.csv',settings())
    for path in sorted((AFAST/'metrics').glob('*_seed*.csv')):
        for r in pd.read_csv(path).to_dict('records'):SAVED[(r['scene_id'],r['frame_id'],r['condition'],r['architecture'],r['seed'],r['mask'],r['policy'])]=r
    ids=list(range(300000,300128))+list(range(310000,310064));checks=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(worker,ids)):
            checks.append(r)
            if (i+1)%8==0:print(i+1,len(ids),flush=True)
    js(ROOT/'phase_a/sensitivity/scoring_receipt.json',dict(status='COMPLETE_WITH_EXPLICIT_BUFFER_SCOPE_LIMIT',scenes=len(ids),rows=sum(r['rows'] for r in checks),
        max_nominal_truth_difference=max(r['max_nominal_truth_difference'] for r in checks),
        buffer_scope='Changed true exclusion buffers and utility/unsafe scoring; frozen nominal aggregate hazard probabilities cannot be reparameterized into per-cause buffer predictions without a new model/assumption. Direct choices fixed; material/productivity/slope constants matched in soft composition.',
        new_bootstrap_draws=0,new_neural_forwards=0))
if __name__=='__main__':main()
