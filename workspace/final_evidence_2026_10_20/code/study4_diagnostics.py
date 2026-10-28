"""Fresh Study 4 maps only: complete truth substitutions, safety and cutoffs."""
from common import *
import argparse,math,time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from attribution_safety import safety_at,FACTORS
from classical_baseline import prepare
from scipy.ndimage import distance_transform_edt
META={};FILES={};PRIOR=None;CLASSICAL=None
ARCHES=['KF-D42','BF-D42','KF-D14','BF-D14'];SEEDS=[10,11,12]
CUTOFFS=[.5,.3,.2,.1,.05]

def attribute(pred,d):
    mat=np.eye(4,dtype=np.float32)[d['gt_mat']].transpose(2,0,1);rows=[];loss=np.zeros((16,2))
    for subset in range(16):
        score=soft_score(d['gt_H'] if subset&1 else pred['H'],mat if subset&2 else pred['mat_prob'],
            (d['gt_sem']==0).astype(np.float32) if subset&8 else pred['sem_prob'][0],
            d['gt_haz'].astype(np.float32) if subset&4 else pred['haz_prob'],d['gt_reach'],d['z_target'])
        met=selected_metrics(score,d);loss[subset]=[met['regret'],met['unsafe']];rows.append(dict(subset=subset,**met))
    phi=np.zeros((4,2))
    for i in range(4):
        for s in range(16):
            if s&(1<<i):continue
            k=s.bit_count();phi[i]+=math.factorial(k)*math.factorial(3-k)/24*(loss[s]-loss[s|(1<<i)])
    assert np.allclose(phi.sum(0),loss[0]-loss[15],atol=1e-12,rtol=0)
    assert loss[15,0]<=2e-7,('all truth nonmaximal',loss[15])
    return rows,phi

def cutoff_metrics(pred,d,cutoff):
    # choose() has the historical >.5 exclusion. A value above the requested
    # cutoff is presented as excluded; no other score/candidate rule changes.
    excluded=(~np.isfinite(pred['haz_prob']))|(pred['haz_prob']>cutoff)
    hazard=np.where(excluded,1.,0.).astype(np.float32)
    return selected_metrics(pred['aff'],d,hazard)

def one(sid):
    source=ROOT/'study4/holdout';receipt=json.loads((source/f'generation/{sid}.json').read_text())
    scene=generate_scene(receipt['geometry_seed'],ood_terrain=receipt['rough'])
    poses={int(r['frame_id']):np.array(r['q']) for r in receipt['attempts'] if r['accepted']}
    attribution=[];shapley=[];safety=[];references=[];cutoffs=[]
    control=dict(scene_id=sid,frames=0,all_truth_max_regret=0.,baseline_mismatches=0,cutoff_half_mismatches=0,hazard_mismatches=0,processed_unique_maps=0)
    for item in FILES[sid]:
        condition=item['condition'];slot=int(item['frame_id']);key=(sid,condition,slot);path=source/item['path'];d=load_npz(path)
        feasible=np.isfinite(d['gt_reach'])&(d['gt_reach']>=.05);masks=safety_masks(scene,poses[slot],d['gt_sem']);control['frames']+=1
        assert np.array_equal(masks[:4].any(0),d['gt_haz']>.5),('hazard reconstruction',key)
        base=dict(scene_id=sid,frame_id=slot,condition=condition)
        n=int(feasible.sum());oracle=float(d['gt_aff'][feasible].max()) if n else 0.
        r0=dict(**base,architecture='R0',seed=-1,arm='uniform_reachable',policy='reference',
            regret=oracle-float(d['gt_aff'][feasible].astype(float).mean()) if n else 0.,
            unsafe=float(np.any(masks,axis=0)[feasible].mean()) if n else 0.,executed=int(n>0),abstained=int(n==0),pick_row=-1,pick_col=-1,
            selection_kind='analytic_expectation',expected_unsafe_numerator=int(np.any(masks,axis=0)[feasible].sum()),expected_unsafe_denominator=n)
        r1=dict(**base,architecture='R1',seed=-1,arm='training_prior',policy='reference',
            **selected_metrics(PRIOR['mean_A_train'],d,PRIOR['mean_h_train']),selection_kind='deterministic')
        _,score,residual,ego,_=prepare(path)
        detected=residual>=CLASSICAL['tau'];distance=distance_transform_edt(~detected)*BEV_RES if detected.any() else np.full(detected.shape,np.inf)
        hazard=(distance<=CLASSICAL['buffer'])|ego
        classical=dict(**base,architecture='classical_ground',seed=-1,arm='classical_ground',policy='reference',
            **selected_metrics(score,d,hazard.astype(np.float32)),selection_kind='deterministic')
        for row in [r0,r1,classical]:
            references.append(row)
            detail=safety_at(row['pick_row'],row['pick_col'],masks,feasible,row['architecture']=='R0')
            assert abs(sum(detail['exclusive_'+c] for c in CAUSES)-row['unsafe'])<1e-12
            safety.append(dict(**{k:row[k] for k in ['scene_id','frame_id','condition','architecture','seed','arm','policy','selection_kind','unsafe','executed','abstained']},**detail))
        groups={}
        for r in META[key]:groups.setdefault((r['architecture'],r['seed'],r['arm']),[]).append(r)
        cache={}
        for (arch,seed,arm),metrows in groups.items():
            identity=dict(**base,architecture=arch,seed=seed,arm=arm);assert len(metrows)==2
            for r in metrows:
                detail=safety_at(r['pick_row'],r['pick_col'],masks,feasible)
                assert sum(detail['exclusive_'+c] for c in CAUSES)==r['unsafe']
                safety.append(dict(**identity,policy=r['policy'],selection_kind='deterministic',unsafe=r['unsafe'],executed=r['executed'],abstained=r['abstained'],**detail))
            predpath=metrows[0]['prediction_path']
            if predpath not in cache:
                pred=load_npz(ROOT/predpath);att,phi=attribute(pred,d)
                curves=[dict(cutoff=c,**cutoff_metrics(pred,d,c)) for c in CUTOFFS]
                cache[predpath]=(att,phi,curves);control['processed_unique_maps']+=1
            att,phi,curves=cache[predpath]
            savedsoft=next(r for r in metrows if r['policy']=='soft');direct=next(r for r in metrows if r['policy']=='direct')
            assert (att[0]['pick_row'],att[0]['pick_col'])==(savedsoft['pick_row'],savedsoft['pick_col'])
            assert abs(att[0]['regret']-savedsoft['regret'])<1e-12
            assert (curves[0]['pick_row'],curves[0]['pick_col'])==(direct['pick_row'],direct['pick_col'])
            assert abs(curves[0]['regret']-direct['regret'])<1e-12 and curves[0]['unsafe']==direct['unsafe']
            control['all_truth_max_regret']=max(control['all_truth_max_regret'],att[15]['regret'])
            attribution.extend(dict(**identity,**r) for r in att)
            shapley.extend(dict(**identity,factor=f,regret_loss_reduction=phi[i,0],unsafe_loss_reduction=phi[i,1]) for i,f in enumerate(FACTORS))
            for r in curves:
                detail=safety_at(r['pick_row'],r['pick_col'],masks,feasible)
                cutoffs.append(dict(**identity,policy='direct',**r,person_buffer=detail['multi_person_buffer']))
    for name,rows in [('attribution',attribution),('shapley',shapley),('safety',safety),('references',references),('cutoffs',cutoffs)]:
        out=ROOT/f'study4/diagnostics/{name}/scenes/{sid}.csv';assert not out.exists();csvwrite(out,rows)
    js(ROOT/f'study4/diagnostics/controls/{sid}.json',control)
    return dict(scene_id=sid,records=control['frames'],attribution_rows=len(attribution),cutoff_rows=len(cutoffs),all_truth_max_regret=control['all_truth_max_regret'])

def main():
    global META,FILES,PRIOR,CLASSICAL
    from infer_study4 import verify_lock
    verify_lock();p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    for arch in ARCHES:
        for seed in SEEDS:
            for r in pd.read_csv(ROOT/f'study4/metrics/{arch}_seed{seed}.csv').to_dict('records'):
                META.setdefault((r['scene_id'],r['condition'],r['frame_id']),[]).append(r)
    for r in csv.DictReader((ROOT/'study4/holdout/files.csv').open()):FILES.setdefault(int(r['scene_id']),[]).append(r)
    PRIOR=load_npz(PROJECT/'submission_strengthening/references/training_prior_maps.npz')
    cfgpath=PROJECT/'strengthening_2026_10_18/phase_a/classical/validation_lock.json'
    CLASSICAL=json.loads(cfgpath.read_text());receipts=[]
    assert CLASSICAL['tau']==1. and CLASSICAL['buffer']==4.
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(one,sorted(FILES))):
            receipts.append(r)
            if (i+1)%8==0:print(i+1,len(FILES),r,flush=True)
    js(ROOT/'study4/diagnostics/scoring_receipt.json',dict(status='COMPLETE',scenes=len(FILES),
        attribution_rows=sum(r['attribution_rows'] for r in receipts),cutoff_rows=sum(r['cutoff_rows'] for r in receipts),
        all_truth_max_regret=max(r['all_truth_max_regret'] for r in receipts),classical_lock_sha256=sha(cfgpath),
        prior_sha256=sha(PROJECT/'submission_strengthening/references/training_prior_maps.npz'),perception_forwards=0,new_bootstrap_draws=0))
if __name__=='__main__':main()
