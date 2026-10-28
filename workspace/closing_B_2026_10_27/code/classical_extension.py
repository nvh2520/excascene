"""C1 only: original classical policy, added validation buffers, one selected arm."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,itertools,argparse
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parents[1];P=O.parent;F=P/'final_evidence_2026_10_20'
sys.path.insert(0,str(F/'code'))
from common import load_npz,sha,js,csvwrite,selected_metrics,BEV_RES,safety_masks,generate_scene,CAUSES
from classical_baseline import prepare,TAUS
from scipy.ndimage import distance_transform_edt
from attribution_safety import safety_at
from relative_grid import settings,soft_score
NEW_BUFFERS=[5.,6.,8.,10.,12.,16.]
def stamp():return datetime.now(timezone.utc).isoformat()
def verify():
    lock=json.loads((O/'protocol/C1_lock.json').read_text())
    for path,digest in lock['files'].items():assert sha(P/path)==digest,path
    return lock
def val_one(path):
    d,score,residual,ego,n=prepare(path);reach=np.isfinite(d['gt_reach'])&(d['gt_reach']>=.05)
    assert np.isfinite(score[reach]).all(),'Old eligible counts would not measure hazard exclusion'
    rows=[]
    for tau in TAUS:
        hit=residual>=tau;distance=distance_transform_edt(~hit)*BEV_RES if hit.any() else np.full(hit.shape,np.inf)
        for b in NEW_BUFFERS:
            hazard=(distance<=b)|ego;excluded=int((hazard&reach).sum());den=int(reach.sum())
            met=selected_metrics(score,d,hazard.astype(np.float32))
            assert met['eligible_cells']==den-excluded
            rows.append(dict(scene_id=int(d['scene_id']),frame_id=int(d['frame_id']),tau=tau,buffer=b,
                **met,reachable_cells=den,excluded_reachable_cells=excluded,excluded_reachable_fraction=excluded/den if den else None,
                source_path=str(path.relative_to(P))))
    return rows,dict(scene_id=int(d['scene_id']),frame_id=int(d['frame_id']),reachable_cells=int(reach.sum()),
        score_finite_on_reach=True,source=str(path.relative_to(P)),sha256=sha(path))
def validation(workers):
    verify();start=time.time();assert not (O/'C1/validation/selection_lock.json').exists()
    files=sorted((P/'data/val').glob('f_*.npz'));assert len(files)==1200
    rows=[];inputs=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(rr,meta) in enumerate(pool.map(val_one,files,chunksize=8)):
            rows.extend(rr);inputs.append(meta)
            if (i+1)%200==0:print('validation',i+1,len(files),flush=True)
    new=pd.DataFrame(rows);new.to_csv(O/'C1/validation/added_per_frame.csv',index=False)
    oldsrc=P/'strengthening_2026_10_18/phase_a/classical/validation_all_grid_per_frame.csv'
    old=pd.read_csv(oldsrc);den={(r['scene_id'],r['frame_id']):r['reachable_cells'] for r in inputs}
    old['reachable_cells']=[den[(s,f)] for s,f in zip(old.scene_id,old.frame_id)]
    old['excluded_reachable_cells']=old.reachable_cells-old.eligible_cells
    old['excluded_reachable_fraction']=old.excluded_reachable_cells/old.reachable_cells
    old['grid_part']='original';new['grid_part']='added'
    full=pd.concat([old,new],ignore_index=True);assert len(full)==72000
    full.to_csv(O/'C1/validation/combined_per_frame.csv',index=False)
    summary=[]
    for (tau,b),g in full.groupby(['tau','buffer']):
        assert len(g)==1200
        r=dict(tau=tau,buffer=b,grid_part=g.grid_part.iloc[0],frames=len(g))
        r.update({m:float(g[m].mean()) for m in ['regret','unsafe','executed','abstained']})
        r.update(reachable_cells=int(g.reachable_cells.sum()),excluded_reachable_cells=int(g.excluded_reachable_cells.sum()),
            excluded_reachable_share=float(g.excluded_reachable_cells.sum()/g.reachable_cells.sum()),
            mean_frame_excluded_share=float(g.excluded_reachable_fraction.mean()))
        summary.append(r)
    df=pd.DataFrame(summary).sort_values(['regret','buffer','tau'],kind='stable');assert len(df)==60
    df.to_csv(O/'C1/validation/extended_grid.csv',index=False)
    original=pd.read_csv(P/'strengthening_2026_10_18/phase_a/classical/validation_grid.csv').set_index(['tau','buffer'])
    reconstructed=df[df.grid_part=='original'].set_index(['tau','buffer']).reindex(original.index)
    error=float(np.max(np.abs(original[['regret','unsafe','executed','abstained']].to_numpy()-reconstructed[['regret','unsafe','executed','abstained']].to_numpy())))
    assert error<1e-14
    best=df.iloc[0];selection=dict(status='POSTHOC_VALIDATION_SELECTED_BEFORE_NEW_HOLDOUT_EVALUATION',utc=stamp(),
        tau=float(best.tau),buffer=float(best.buffer),grid_points=60,original_grid_points=30,added_grid_points=30,validation_frames=1200,
        selection='minimum mean validation regret; exact ties smaller buffer then smaller threshold',
        selected_validation_regret=float(best.regret),selected_validation_unsafe=float(best.unsafe),selected_validation_abstention=float(best.abstained),
        selected_excluded_reachable_share=float(best.excluded_reachable_share),largest_buffer_selected=bool(best.buffer==max(NEW_BUFFERS)),
        grid_sha256=sha(O/'C1/validation/extended_grid.csv'),code_sha256=sha(__file__),prior_grid_reconstruction_max_abs_error=error,
        original_frozen_pair=dict(tau=1.,buffer=4.),test_based_selection=False)
    js(O/'C1/validation/selection_lock.json',selection)
    js(O/'C1/validation/receipt.json',dict(status='COMPLETE',wall_seconds=time.time()-start,workers=workers,
        input_files=inputs,original_rows_source=str(oldsrc.relative_to(P)),original_rows_sha256=sha(oldsrc),
        original_excluded_share_method='Original eligible counts subtracted from reachable count; finite composed score on reach checked for every frame',
        new_neural_forwards=0,gpu_hours=0))
    print(json.dumps(selection,indent=2),flush=True)

CFG=None;FILES={}
def holdout_one(sid):
    rec=json.loads((F/f'study4/holdout/generation/{sid}.json').read_text())
    scene=generate_scene(rec['geometry_seed'],ood_terrain=rec['rough'])
    poses={int(r['frame_id']):np.asarray(r['q']) for r in rec['attempts'] if r['accepted']}
    rows=[];sens=[];sources=[];nomerr=0.
    for item in FILES[sid]:
        path=F/'study4/holdout'/item['path'];assert sha(path)==item['sha256']
        d,score,residual,ego,n=prepare(path);reach=np.isfinite(d['gt_reach'])&(d['gt_reach']>=.05)
        hit=residual>=CFG['tau'];distance=distance_transform_edt(~hit)*BEV_RES if hit.any() else np.full(hit.shape,np.inf)
        hazard=(distance<=CFG['buffer'])|ego;met=selected_metrics(score,d,hazard.astype(np.float32))
        slot=int(item['frame_id']);masks=safety_masks(scene,poses[slot],d['gt_sem'])
        assert np.array_equal(masks[:4].any(0),d['gt_haz']>.5)
        details=safety_at(met['pick_row'],met['pick_col'],masks,reach)
        assert sum(details['exclusive_'+c] for c in CAUSES)==met['unsafe']
        row=dict(scene_id=sid,frame_id=slot,condition=item['condition'],architecture='classical_posthoc',arm='classical_posthoc',policy='reference',seed=-1,
            tau=CFG['tau'],buffer=CFG['buffer'],**met,**details,reachable_cells=int(reach.sum()),
            excluded_reachable_cells=int((hazard&reach).sum()),source_path=str(path.relative_to(P)),source_sha256=item['sha256'])
        rows.append(row);sources.append(dict(path=str(path.relative_to(P)),sha256=item['sha256']))
        if item['condition']=='clean':
            mat=np.eye(4,dtype=np.float32)[d['gt_mat']].transpose(2,0,1);terrain=(d['gt_sem']==0).astype(np.float32)
            hazards={scale:safety_masks(scene,poses[slot],d['gt_sem'],scale)[:4].any(0) for scale in [.75,1.,1.25]}
            for setting in settings():
                haz=hazards[setting['buffer_scale']]
                u=soft_score(d['gt_H'],mat,terrain,haz.astype(np.float32),d['gt_reach'],d['z_target'],
                    setting['material_weights'],setting['productivity'],(setting['slope_low'],setting['slope_high']))
                oracle=float(u[reach].max());value=float(u[met['pick_row'],met['pick_col']]) if met['executed'] else 0.
                unsafe=int(met['executed'] and (haz|(d['gt_sem']!=0))[met['pick_row'],met['pick_col']])
                regret=oracle-value
                if setting['setting']=='nominal':
                    nomerr=max(nomerr,abs(regret-met['regret']));assert nomerr<3e-7 and unsafe==met['unsafe']
                sens.append(dict(scene_id=sid,frame_id=slot,setting=setting['setting'],regret=regret,unsafe=unsafe,
                    executed=met['executed'],abstained=met['abstained'],pick_row=met['pick_row'],pick_col=met['pick_col']))
    dest=O/f'C1/holdout/scenes/{sid}.csv';assert not dest.exists();csvwrite(dest,rows)
    if sens:csvwrite(O/f'C1/sensitivity/scenes/{sid}.csv',sens)
    js(O/f'C1/holdout/scenes/{sid}_receipt.json',dict(scene_id=sid,records=len(rows),sources=sources,choice_evaluations=len(rows),
        sensitivity_rows=len(sens),sensitivity_new_choices=0,max_nominal_sensitivity_difference=nomerr))
    return dict(scene_id=sid,records=len(rows),sensitivity_rows=len(sens),max_nominal_sensitivity_difference=nomerr)
def holdout(workers):
    global CFG,FILES
    verify();start=time.time();CFG=json.loads((O/'C1/validation/selection_lock.json').read_text());assert CFG['code_sha256']==sha(__file__)
    assert not (O/'C1/holdout/receipt.json').exists()
    for r in pd.read_csv(F/'study4/holdout/files.csv').to_dict('records'):
        if r['condition'] in ['clean','cam_s3','rough_iid']:FILES.setdefault(int(r['scene_id']),[]).append(r)
    assert sum(map(len,FILES.values()))==5120
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(holdout_one,sorted(FILES))):
            receipts.append(r)
            if (i+1)%64==0:print('holdout',i+1,len(FILES),flush=True)
    rows=pd.concat([pd.read_csv(O/f'C1/holdout/scenes/{sid}.csv') for sid in sorted(FILES)],ignore_index=True)
    rows.to_csv(O/'C1/holdout/per_frame.csv',index=False)
    js(O/'C1/holdout/receipt.json',dict(status='COMPLETE_ONE_SELECTED_POLICY',records=len(rows),scenes=len(FILES),
        conditions=rows.condition.value_counts().to_dict(),wall_seconds=time.time()-start,workers=workers,
        selected_pair=CFG,selection_lock_sha256=sha(O/'C1/validation/selection_lock.json'),
        selector_calls=len(rows),sensitivity_new_selector_calls=0,sensitivity_rows=sum(r['sensitivity_rows'] for r in receipts),
        max_nominal_sensitivity_difference=max(r['max_nominal_sensitivity_difference'] for r in receipts),gpu_hours=0,neural_forwards=0,new_draws=0))
    print('Holdout complete:',len(rows),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['validation','holdout']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    globals()[a.action](a.workers)
