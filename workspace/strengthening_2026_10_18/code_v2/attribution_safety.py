"""All 16 truth substitutions and cause-specific safety on immutable saved maps."""
from common import *
import argparse,math,time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

SAVED={};REFERENCE={};GEOMETRIC={}
FACTORS=('height','material','hazard','terrain_semantics')
def safety_at(pickrow,pickcol,masks,feasible,uniform=False):
    if uniform:
        n=int(feasible.sum());raw=np.array([(m&feasible).sum()/n if n else 0 for m in masks],np.float64)
        todo=feasible.copy();exclusive=[]
        for m in masks:
            exclusive.append((todo&m).sum()/n if n else 0);todo&=~m
    elif pickrow<0:
        raw=np.zeros(5);exclusive=np.zeros(5)
    else:
        raw=masks[:,int(pickrow),int(pickcol)].astype(float);exclusive=np.zeros(5)
        if raw.any():exclusive[np.flatnonzero(raw)[0]]=1
    return {**{f'multi_{c}':float(v) for c,v in zip(CAUSES,raw)},**{f'exclusive_{c}':float(v) for c,v in zip(CAUSES,exclusive)}}

def worker(sid):
    start=time.time();scene,poses=historical_scene(sid)
    out=ROOT/f'phase_a/attribution/scenes/{sid}.csv';assert not out.exists()
    rows=[];safety=[];shapley=[];checks=dict(scene_id=sid,baseline_pick_mismatches=0,all_truth_nonmax_choices=0,
        all_truth_rounding_near_ties=0,all_truth_max_regret=0.,hazard_mask_mismatches=0,processed_maps=0)
    conditions=['rough_iid'] if sid>=310000 else ['clean','iid','all_s2','cam_s1','cam_s2','cam_s3']
    for frame,q in sorted(poses.items()):
        for condition in conditions:
            d=load_npz(AFAST/f'data/{condition}/f_{sid:06d}_{frame}.npz',['gt_H','gt_mat','gt_sem','gt_haz','gt_reach','gt_aff','z_target'])
            masks=safety_masks(scene,q,d['gt_sem']);haz=np.any(masks[:4],axis=0)
            checks['hazard_mask_mismatches']+=int(np.count_nonzero(haz!=(d['gt_haz']>.5)))
            assert np.array_equal(haz,d['gt_haz']>.5),('hazard reconstruction',sid,frame)
            feasible=np.isfinite(d['gt_reach'])&(d['gt_reach']>=.05)
            truth_mat=np.eye(4,dtype=np.float32)[d['gt_mat']].transpose(2,0,1)
            base=dict(condition=condition,scene_id=sid,frame_id=frame)
            for arch in ['KF-D','BF-D']:
                for seed in [10,11,12]:
                    for mask in [0,1]:
                        pred=load_npz(AFAST/f'predictions/{arch}_seed{seed}/mask{mask}/{condition}/{sid}_{frame}.npz')
                        losses=np.zeros((16,2));checks['processed_maps']+=1
                        for subset in range(16):
                            H=d['gt_H'] if subset&1 else pred['H']
                            mat=truth_mat if subset&2 else pred['mat_prob']
                            haz=d['gt_haz'].astype(np.float32) if subset&4 else pred['haz_prob']
                            terrain=(d['gt_sem']==0).astype(np.float32) if subset&8 else pred['sem_prob'][0]
                            score=soft_score(H,mat,terrain,haz,d['gt_reach'],d['z_target'])
                            met=selected_metrics(score,d)
                            losses[subset]=[met['regret'],met['unsafe']]
                            if subset==0:
                                saved=SAVED[(sid,frame,condition,arch,seed,mask,'soft')]
                                match=(met['pick_row']==saved['pick_row'] and met['pick_col']==saved['pick_col'])
                                checks['baseline_pick_mismatches']+=int(not match)
                                assert match,('baseline_pick',base,arch,seed,mask,met,saved)
                                assert abs(met['regret']-saved['regret'])<1e-12
                            if subset==15:
                                checks['all_truth_max_regret']=max(checks['all_truth_max_regret'],met['regret'])
                                checks['all_truth_nonmax_choices']+=int(met['regret']>2e-7)
                                checks['all_truth_rounding_near_ties']+=int(0<met['regret']<=2e-7)
                            row=dict(**base,architecture=arch,seed=seed,mask=mask,policy='soft_truth_substitution',subset=subset,**met,
                                **safety_at(met['pick_row'],met['pick_col'],masks,feasible))
                            assert sum(row['exclusive_'+c] for c in CAUSES)==met['unsafe']
                            rows.append(row)
                        attrs=np.zeros((4,2))
                        for i in range(4):
                            for sub in range(16):
                                if sub&(1<<i):continue
                                k=sub.bit_count();w=math.factorial(k)*math.factorial(3-k)/24
                                attrs[i]+=w*(losses[sub]-losses[sub|(1<<i)])
                        assert np.allclose(attrs.sum(0),losses[0]-losses[15],atol=1e-12,rtol=0)
                        for i,factor in enumerate(FACTORS):
                            shapley.append(dict(**base,architecture=arch,seed=seed,mask=mask,factor=factor,
                                regret_loss_reduction=attrs[i,0],unsafe_loss_reduction=attrs[i,1],
                                baseline_regret=losses[0,0],all_truth_regret=losses[15,0],baseline_unsafe=int(losses[0,1]),all_truth_unsafe=int(losses[15,1])))
                        for policy in ['direct','soft']:
                            saved=SAVED[(sid,frame,condition,arch,seed,mask,policy)]
                            detail=safety_at(saved['pick_row'],saved['pick_col'],masks,feasible)
                            assert sum(detail['exclusive_'+c] for c in CAUSES)==saved['unsafe']
                            safety.append(dict(**base,architecture=arch,seed=seed,mask=mask,policy=policy,selection_kind='deterministic',
                                unsafe=saved['unsafe'],executed=saved['executed'],abstained=saved['abstained'],**detail))
            for reference in ['R0','R1','geometric_anchor']:
                saved=(GEOMETRIC if reference=='geometric_anchor' else REFERENCE)[(sid,frame,condition,reference)]
                uniform=reference=='R0'
                detail=safety_at(-1 if uniform else saved['pick_row'],-1 if uniform else saved['pick_col'],masks,feasible,uniform)
                assert abs(sum(detail['exclusive_'+c] for c in CAUSES)-saved['unsafe'])<1e-12
                safety.append(dict(**base,architecture=reference,seed=-1,mask=0,policy=reference,
                    selection_kind='analytic_expectation' if uniform else 'deterministic',unsafe=saved['unsafe'],
                    executed=saved['executed'],abstained=saved['abstained'],**detail))
    csvwrite(out,rows);csvwrite(ROOT/f'phase_a/safety/scenes/{sid}.csv',safety)
    csvwrite(ROOT/f'phase_a/shapley/scenes/{sid}.csv',shapley)
    checks['wall_seconds']=time.time()-start;js(ROOT/f'phase_a/attribution/checks/{sid}.json',checks)
    return checks

def main():
    global SAVED,REFERENCE,GEOMETRIC
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    plan=ROOT/'protocol/phase_a_plan.md';lock=json.loads((ROOT/'protocol/phase_a_lock.json').read_text());assert sha(plan)==lock['plan_sha256']
    for path in sorted((AFAST/'metrics').glob('*_seed*.csv')):
        for r in pd.read_csv(path).to_dict('records'):
            SAVED[(r['scene_id'],r['frame_id'],r['condition'],r['architecture'],r['seed'],r['mask'],r['policy'])]=r
    for r in pd.read_csv(PROJECT/'submission_strengthening/references/reference_per_frame.csv').to_dict('records'):
        REFERENCE[(r['scene_id'],r['frame_id'],r['condition'],r['reference'])]=r
    for r in pd.read_csv(PROJECT/'submission_strengthening/references/geometric_per_frame.csv').to_dict('records'):
        GEOMETRIC[(r['scene_id'],r['frame_id'],r['condition'],'geometric_anchor')]=r
    ids=list(range(300000,300128))+list(range(310000,310064));checks=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(worker,ids)):
            checks.append(r)
            if (i+1)%8==0:print(i+1,len(ids),r,flush=True)
    totals={k:sum(c[k] for c in checks) for k in ['baseline_pick_mismatches','all_truth_nonmax_choices','all_truth_rounding_near_ties','hazard_mask_mismatches','processed_maps']}
    totals.update(status='COMPLETE' if not totals['all_truth_nonmax_choices'] else 'ATTRIBUTION_CONTROL_BLOCKED',
        all_truth_max_regret=max(c['all_truth_max_regret'] for c in checks),plan_sha256=sha(plan),scenes=len(ids))
    js(ROOT/'phase_a/attribution/control_receipt.json',totals);print(totals,flush=True)
if __name__=='__main__':main()
