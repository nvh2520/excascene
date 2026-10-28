"""K2/K4: CPU projection and counts from existing observations and decisions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,importlib.util,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parents[1];P=O.parent;F=P/'final_evidence_2026_10_20'
sys.path.insert(0,str(F/'code'))
from common import load_npz,js,csvwrite,sha
from kinafford.sim.kinematics import CAMERAS,SENSORS,SENSOR_ORDER,sensor_pose_U,camera_intrinsics,lidar_rays
from kinafford.sim.scene import BEV_N,BEV_RES,BEV_X0,BEV_Y0
spec=importlib.util.spec_from_file_location('existing_projection',P/'submission_strengthening/references/geometric_source.py')
geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
K=np.stack([camera_intrinsics(SENSORS[c]) for c in CAMERAS]).astype(np.float32)
DIR=lidar_rays(SENSORS['lidar']).astype(np.float32)
MODELS={};SOURCE_HASHES={}

def occupied(points):
    # Verbatim median_grid cell inclusion; no height filling or interpolation.
    points=np.asarray(points,dtype=np.float32).reshape(-1,3)
    points=points[np.isfinite(points).all(1)]
    ix=np.floor((points[:,0]-BEV_X0)/BEV_RES).astype(np.int64)
    iy=np.floor((points[:,1]-BEV_Y0)/BEV_RES).astype(np.int64)
    valid=(ix>=0)&(ix<BEV_N)&(iy>=0)&(iy<BEV_N)&(points[:,2]<8)
    result=np.zeros((BEV_N,BEV_N),bool);result[iy[valid],ix[valid]]=True
    return result

def project(d):
    q=d['q_rep'].astype(np.float32).astype(np.float64)
    T=np.stack([sensor_pose_U(n,q) for n in SENSOR_ORDER]).astype(np.float32)
    cams=np.stack([occupied(geo.unproject_camera(d['depth'][j].astype(np.float32),K[j],T[j],d['cam_present'][j])) for j in range(3)])
    rng=d['lidar_range'].astype(np.float32);valid=d['lidar_valid']>.5
    lid=occupied(geo.transform((rng[...,None]*DIR)[(rng>.3)&valid],T[3]))
    return cams,lid

def one(sid):
    t=time.time();rough=sid>=1210000
    safety_path=F/f'study4/diagnostics/safety/scenes/{sid}.csv'
    sf=pd.read_csv(safety_path);sf=sf[sf.policy.isin(['direct','reference'])].copy()
    flags=[];support=[];observations=[]
    for slot in range(4):
        basecond='rough_iid' if rough else 'clean'
        path=F/f'study4/holdout/data/{basecond}/f_{sid}_{slot}.npz'
        keys=['gt_reach','gt_sem','gt_haz','gt_aff']
        if not rough:keys+=['depth','cam_present','lidar_range','lidar_valid','q_rep']
        d=load_npz(path,keys);observations.append({'path':str(path.relative_to(P)),'sha256':sha(path)})
        reach=np.isfinite(d['gt_reach'])&(d['gt_reach']>=.05)
        safe=reach&(d['gt_sem']==0)&(d['gt_haz']<=.5)
        truth=np.where(reach,d['gt_aff'],-np.inf);oracle=float(np.max(truth)) if reach.any() else 0.
        flags.append(dict(scene_id=sid,frame_id=slot,reachable_cells=int(reach.sum()),safe_reachable_cells=int(safe.sum()),safe_available=int(safe.any()),zero_oracle=int(oracle==0),oracle_utility=oracle))
        if rough:continue
        oracle_index=np.unravel_index(int(np.argmax(truth)),truth.shape)
        prior_lidar=None
        for condition in ['clean','cam_s3']:
            if condition!='clean':
                path=F/f'study4/holdout/data/{condition}/f_{sid}_{slot}.npz'
                obs=load_npz(path,['depth','cam_present','lidar_range','lidar_valid','q_rep'])
                observations.append({'path':str(path.relative_to(P)),'sha256':sha(path)})
            else:obs=d
            cams,lid=project(obs)
            if prior_lidar is None:prior_lidar=lid
            else:assert np.array_equal(prior_lidar,lid),('camera intervention altered lidar support',sid,slot)
            model=MODELS[sid];model=model[(model.frame_id==slot)&(model.condition==condition)]
            def row(scope,architecture,seed,arm,policy,state,mask,at=None):
                cam=cams[~mask].any(0) if (~mask).any() else np.zeros_like(lid)
                maps={'camera':cam,'lidar':lid,'either':cam|lid}
                if at is None:den=int(reach.sum());nums={k:int((v&reach).sum()) for k,v in maps.items()}
                else:den=1;nums={k:int(v[at]) for k,v in maps.items()}
                support.append(dict(scene_id=sid,frame_id=slot,condition=condition,scope=scope,architecture=architecture,seed=seed,arm=arm,policy=policy,input_state=state,denominator=den,**{k+'_supported':v for k,v in nums.items()}))
            for arm,mask in [('keep_all',np.zeros(3,bool)),('always_mask',np.ones(3,bool))]:
                row('reachable_candidates','input',-1,arm,'truth','actual',mask)
                row('oracle_cell','input',-1,arm,'truth','actual',mask,oracle_index)
            for arch in ['KF-D42','BF-D42']:
                codes=model[(model.architecture==arch)&(model.arm=='gate')]['subset'].unique();assert len(codes)==1
                mask=(int(codes[0])&np.array([1,2,4]))!=0
                row('reachable_candidates',arch,-1,'gate','truth','actual',mask)
                row('oracle_cell',arch,-1,'gate','truth','actual',mask,oracle_index)
            for r in model.itertuples(index=False):
                assert r.executed==1 and r.pick_row>=0 and reach[r.pick_row,r.pick_col]
                mask=(int(r.subset)&np.array([1,2,4]))!=0
                row('selected_cell',r.architecture,int(r.seed),r.arm,r.policy,'actual',mask,(int(r.pick_row),int(r.pick_col)))
                if r.arm!='always_mask':
                    row('selected_cell',r.architecture,int(r.seed),r.arm,r.policy,'all_cameras_masked_support_only',np.ones(3,bool),(int(r.pick_row),int(r.pick_col)))
    joined=sf.merge(pd.DataFrame(flags),on=['scene_id','frame_id'],validate='many_to_one')
    assert len(joined)==len(sf) and joined.safe_available.notna().all()
    csvwrite(O/f'K2/scenes/{sid}.csv',joined.to_dict('records'))
    csvwrite(O/f'K2/scenes/{sid}_population.csv',flags)
    if support:csvwrite(O/f'K4/scenes/{sid}.csv',support)
    js(O/f'K2/scenes/{sid}.json',dict(seconds=time.time()-t,scene_id=sid,safety_rows=len(joined),support_rows=len(support),safety_source=str(safety_path.relative_to(P)),safety_sha256=sha(safety_path),observations=observations,neural_forwards=0,selectors_called=0))
    return sid,len(joined),len(support)

def main():
    global MODELS,SOURCE_HASHES
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();t=time.time()
    sources=[F/f'study4/metrics/{arch}_seed{s}.csv' for arch in ['KF-D42','BF-D42'] for s in [10,11,12]]
    df=pd.concat([pd.read_csv(p) for p in sources]);df=df[df.condition.isin(['clean','cam_s3'])&df.arm.isin(['keep_all','gate','always_mask'])]
    MODELS={int(s):g for s,g in df.groupby('scene_id')}
    ids=list(range(1200000,1200512))+list(range(1210000,1210256))
    assert len(MODELS)==512
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(one,ids)):
            if (i+1)%64==0:print('scenes',i+1,'/',len(ids),flush=True)
    js(O/'K4/projection_receipt.json',dict(status='COMPLETE',seconds=time.time()-t,nominal_scenes=len(MODELS),slots_per_scene=4,sources={str(p.relative_to(P)):sha(p) for p in sources},projection_source=str((P/'submission_strengthening/references/geometric_source.py').relative_to(P)),projection_sha256=sha(P/'submission_strengthening/references/geometric_source.py'),support_definition='Uninterpolated BEV cell occupancy from measured camera depth and LiDAR; nominal calibration and reported joints; same validity and grid inclusion as frozen geometric projection. Not ray-verified correct terrain/material observation.',candidate_definition='finite gt_reach >= 0.05, including nonterrain candidates',oracle='row-major argmax of gt_aff on the candidate set; tied cells use first index',all_masked='Same observations with camera contribution empty; no prediction or choice is recomputed. Actual always-mask choices are separately identified.',new_neural_forwards=0,gpu_hours=0,new_bootstrap_draws=0))
if __name__=='__main__':main()
