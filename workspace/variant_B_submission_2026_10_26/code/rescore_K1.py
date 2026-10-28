"""Authorized CPU-only fixed-choice robustness; never calls a selector or model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,hashlib,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1]; PROJECT=OUT.parent; F=PROJECT/'final_evidence_2026_10_20'
sys.path.insert(0,str(F/'code'))
from common import load_npz,generate_scene,safety_masks,sha,js,csvwrite
from relative_grid import soft_score
from relative_grid import settings
GRID=settings(); META={}
COLS=['architecture','seed','arm','condition','scene_id','frame_id','policy','regret','unsafe','executed','abstained','pick_row','pick_col']

def one(sid):
    start=time.time(); rows=META[sid].copy().reset_index(drop=True)
    rec=json.loads((F/f'study4/holdout/generation/{sid}.json').read_text())
    scene=generate_scene(rec['geometry_seed'],ood_terrain=rec['rough'])
    poses={int(a['frame_id']):np.asarray(a['q']) for a in rec['attempts'] if a['accepted']}
    rg=np.zeros((len(GRID),len(rows)),np.float64);un=rg.copy();oracle_out=rg.copy()
    maxdiff=0.;truth_sources=[];nominalerr=0.;expected_fractions=[]
    for slot,frame in rows.groupby('frame_id',sort=True):
        conditions=sorted(frame.condition.unique()); base=None
        for condition in conditions:
            p=F/f'study4/holdout/data/{condition}/f_{sid}_{slot}.npz'
            d=load_npz(p,['gt_H','gt_mat','gt_sem','gt_haz','gt_reach','gt_aff','z_target'])
            truth_sources.append(dict(path=str(p.relative_to(PROJECT)),sha256=sha(p)))
            if base is None:base=d
            else:
                for k in d:assert np.array_equal(d[k],base[k],equal_nan=True),('paired truth mismatch',sid,slot,condition,k)
        d=base;reach=np.isfinite(d['gt_reach'])&(d['gt_reach']>=.05)
        hazards={b:safety_masks(scene,poses[int(slot)],d['gt_sem'],b)[:4].any(0) for b in sorted({s['buffer_scale'] for s in GRID})}
        assert np.array_equal(hazards[1.],d['gt_haz']>.5),('nominal hazard mismatch',sid,slot)
        mat=np.eye(4,dtype=np.float32)[d['gt_mat']].transpose(2,0,1);terrain=(d['gt_sem']==0).astype(np.float32)
        idx=frame.index.to_numpy(); rr=frame.pick_row.to_numpy(int);cc=frame.pick_col.to_numpy(int)
        uniform=frame.architecture.to_numpy()=='R0';active=frame.executed.to_numpy(bool)&~uniform
        assert np.all(rr[active]>=0) and np.all(cc[active]>=0)
        assert np.all(reach[rr[active],cc[active]])
        for si,s in enumerate(GRID):
            haz=hazards[s['buffer_scale']]
            u=soft_score(d['gt_H'],mat,terrain,haz.astype(np.float32),d['gt_reach'],d['z_target'],
                s['material_weights'],s['productivity'],(s['slope_low'],s['slope_high']))
            if si==0:
                maxdiff=max(maxdiff,float(np.max(np.abs(u-d['gt_aff']))));assert maxdiff<=2e-7
            unsafe=haz|(d['gt_sem']!=0);oracle=float(u[reach].max()) if reach.any() else 0.
            val=np.zeros(len(idx));ev=np.zeros(len(idx));val[active]=u[rr[active],cc[active]];ev[active]=unsafe[rr[active],cc[active]]
            if uniform.any() and reach.any():
                val[uniform]=u[reach].astype(float).mean();ev[uniform]=unsafe[reach].mean()
                expected_fractions.append(dict(scene_id=sid,frame_id=int(slot),setting=s['setting'],unsafe_cells=int(unsafe[reach].sum()),reachable_cells=int(reach.sum())))
            rg[si,idx]=oracle-val;un[si,idx]=ev;oracle_out[si,idx]=oracle
        nominalerr=max(nominalerr,float(np.max(np.abs(rg[0,idx]-frame.regret.to_numpy()))))
        assert np.max(np.abs(un[0,idx]-frame.unsafe.to_numpy()))<1e-12
    assert nominalerr<=4e-7,('nominal regret mismatch',sid,nominalerr)
    dest=OUT/f'K1/scenes/{sid}.npz';assert not dest.exists()
    np.savez_compressed(dest,regret=rg,unsafe=un,oracle=oracle_out,**{c:rows[c].to_numpy(str if c in ['architecture','arm','condition','policy'] else None) for c in COLS if c not in ['regret','unsafe']})
    csvwrite(OUT/f'K1/controls/{sid}_uniform_expected.csv',expected_fractions)
    receipt=dict(scene_id=sid,rows=len(rows),settings=len(GRID),seconds=time.time()-start,max_nominal_truth_difference=maxdiff,max_nominal_regret_difference=nominalerr,truth_sources=truth_sources,fixed_choice_columns=['pick_row','pick_col','executed','abstained'],no_selector_called=True)
    js(OUT/f'K1/controls/{sid}.json',receipt)
    return {k:v for k,v in receipt.items() if k!='truth_sources'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);a=p.parse_args();start=time.time()
    global META
    sources=[F/f'study4/metrics/{arch}_seed{seed}.csv' for arch in ['KF-D42','BF-D42','KF-D14','BF-D14'] for seed in [10,11,12]]+list(sorted((F/'study4/diagnostics/references/scenes').glob('*.csv')))
    frames=[pd.read_csv(p,usecols=COLS) for p in sources]
    f=pd.concat(frames,ignore_index=True);assert not f.duplicated(['architecture','seed','arm','condition','scene_id','frame_id','policy']).any()
    META={int(s):g for s,g in f.groupby('scene_id',sort=True)}
    csvwrite(OUT/'K1/settings.csv',GRID)
    js(OUT/'K1/source_receipt.json',dict(sources={str(p.relative_to(PROJECT)):sha(p) for p in sources},utility_source=sha(F/'code/common.py'),grid_source=sha(PROJECT/'strengthening_2026_10_18/code/utility_sensitivity.py'),choice_rows=len(f),choices='All original direct, soft, classical and training-prior selections fixed; uniform support fixed and expected values retained. Matched constants are used in the truth composition; no predicted soft map is reselected.',buffer_scope='True buffers vary; original aggregate learned hazard predictions and choices are held fixed.',gpu_hours=0,new_neural_forwards=0,new_bootstrap_draws=0))
    ids=sorted(META);ids=ids[:a.limit] if a.limit else ids;ids=[s for s in ids if not (OUT/f'K1/controls/{s}.json').exists()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(one,ids)):
            if (i+1)%8==0 or len(ids)<8:print(json.dumps(dict(done=i+1,total=len(ids),**r)),flush=True)
    receipts=[json.loads(p.read_text()) for p in sorted((OUT/'K1/controls').glob('*.json'))]
    js(OUT/'K1/scoring_receipt.json',dict(status='COMPLETE' if len(receipts)==len(META) else 'PARTIAL',scenes=len(receipts),expected_scenes=len(META),rescore_rows=sum(r['rows']*r['settings'] for r in receipts),wall_seconds_this_invocation=time.time()-start,workers=a.workers,max_nominal_truth_difference=max(r['max_nominal_truth_difference'] for r in receipts),max_nominal_regret_difference=max(r['max_nominal_regret_difference'] for r in receipts),gpu_hours=0,new_neural_forwards=0,new_bootstrap_draws=0,new_selector_calls=0))
if __name__=='__main__':main()
