"""Validation-only bounded local-ground/hazard extension of the geometric anchor."""
from common import *
import argparse,importlib.util,itertools,time
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import percentile_filter,gaussian_filter,distance_transform_edt
from kinafford.sim.kinematics import SENSOR_ORDER,CAMERAS,SENSORS,sensor_pose_U,camera_intrinsics,lidar_rays
from kinafford.sim.scene import BEV_N,BEV_X0,BEV_Y0

spec=importlib.util.spec_from_file_location('historical_geometric',PROJECT/'submission_strengthening/references/geometric_source.py')
geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
K=np.stack([camera_intrinsics(SENSORS[c]) for c in CAMERAS]).astype(np.float32)
DIRECTIONS=lidar_rays(SENSORS['lidar']).astype(np.float32)
TAUS=[.25,.5,.75,1.,1.5];BUFFERS=[0.,.5,1.,1.5,2.,4.]

def prepare(path):
    d=load_npz(path);q=d['q_rep'].astype(np.float32).astype(np.float64)
    sample={k:d[k].astype(np.float32) for k in ['depth','cam_present','lidar_range','lidar_valid']}
    sample['K']=K;sample['T_dyn']=np.stack([sensor_pose_U(n,q) for n in SENSOR_ORDER]).astype(np.float32)
    height,occupied=geo.predict_geometry(sample,DIRECTIONS)
    valid=(sample['lidar_range']>.3)&(sample['lidar_valid']>.5)
    points=geo.transform((sample['lidar_range'][...,None]*DIRECTIONS)[valid],sample['T_dyn'][3])
    points=points[np.isfinite(points).all(1)]
    ix=np.floor((points[:,0]-BEV_X0)/BEV_RES).astype(int);iy=np.floor((points[:,1]-BEV_Y0)/BEV_RES).astype(int)
    keep=(ix>=0)&(ix<BEV_N)&(iy>=0)&(iy<BEV_N)&(points[:,2]<8)
    ids=(iy[keep]*BEV_N+ix[keep]);z=points[keep,2]
    ground=np.zeros(BEV_N**2,np.float32);top=np.full(BEV_N**2,-np.inf,np.float32);lidar_occ=np.zeros(BEV_N**2,bool)
    order=np.lexsort((z,ids));cells,first,n=np.unique(ids[order],return_index=True,return_counts=True);zs=z[order]
    # Fixed nearest-rank lower quartile; no data-selected ground parameters.
    ground[cells]=zs[first+np.floor((n-1)*.25).astype(int)];top[cells]=zs[first+n-1];lidar_occ[cells]=True
    ground=geo.fill_empty(ground.reshape(BEV_N,BEV_N),lidar_occ.reshape(BEV_N,BEV_N))
    ground=gaussian_filter(percentile_filter(ground,20,size=9,mode='nearest'),sigma=1,mode='nearest')
    residual=top.reshape(BEV_N,BEV_N)-ground
    X,Y=bev_grid();R=rot_z(float(q[0]));xb=R[0,0]*X+R[0,1]*Y;yb=R[1,0]*X+R[1,1]*Y
    b=MACHINE_BUFFER
    ego=((np.abs(xb)<=2.3+b)&(np.abs(yb)<=1.5+b))|((X>=-3.1-b)&(X<=1.9+b)&(np.abs(Y)<=1.35+b))
    score=geo.compose_geometry(height,d['gt_reach'],float(d['z_target']))
    return d,score,residual,ego,int(lidar_occ.sum())

def process(task):
    path,grid=task;d,score,residual,ego,n=prepare(path);rows=[]
    for tau in sorted(set(t[0] for t in grid)):
        detected=residual>=tau
        distance=distance_transform_edt(~detected)*BEV_RES if detected.any() else np.full(detected.shape,np.inf)
        for _,b in [g for g in grid if g[0]==tau]:
            hazard=(distance<=b)|ego
            r=selected_metrics(score,d,hazard.astype(np.float32));r.update(tau=tau,buffer=b,condition=str(d['suite']),
                scene_id=int(d['scene_id']),frame_id=int(d['frame_id']),lidar_occupied_cells=n,detected_cells=int(detected.sum()),
                excluded_cells=int(hazard.sum()),source_path=str(path.relative_to(PROJECT)))
            rows.append(r)
    return rows

def main():
    import pandas as pd
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tune','evaluate']);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    out=ROOT/'phase_a/classical';out.mkdir(exist_ok=True)
    lock=json.loads((ROOT/'protocol/phase_a_lock.json').read_text());assert sha(ROOT/'protocol/phase_a_plan.md')==lock['plan_sha256']
    if a.action=='tune':
        assert not (out/'validation_lock.json').exists()
        files=sorted((PROJECT/'data/val').glob('f_*.npz'));assert len(files)==1200
        grid=list(itertools.product(TAUS,BUFFERS));assert len(grid)==30
    else:
        cfg=json.loads((out/'validation_lock.json').read_text());assert cfg['code_sha256']==sha(__file__)
        files=sorted((AFAST/'data').glob('*/f_*.npz'));assert len(files)==3328
        grid=[(cfg['tau'],cfg['buffer'])]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,rs in enumerate(pool.map(process,[(p,grid) for p in files],chunksize=8)):
            rows.extend(rs)
            if (i+1)%200==0:print(a.action,i+1,len(files),flush=True)
    df=pd.DataFrame(rows)
    if a.action=='tune':
        df.to_csv(out/'validation_all_grid_per_frame.csv',index=False)
        summary=df.groupby(['tau','buffer'])[['regret','unsafe','executed','abstained']].mean().reset_index()
        summary=summary.sort_values(['regret','buffer','tau'],kind='stable');summary.to_csv(out/'validation_grid.csv',index=False)
        best=summary.iloc[0]
        js(out/'validation_lock.json',dict(status='LOCKED_BEFORE_HOLDOUT_EVALUATION',utc_unix=time.time(),tau=float(best.tau),buffer=float(best.buffer),
            validation_records=1200,grid_points=30,selection='lowest mean validation regret; ties smaller buffer then tau',
            code_sha256=sha(__file__),plan_sha256=lock['plan_sha256'],grid_sha256=sha(out/'validation_grid.csv'),
            historical_predictor_sha256=sha(PROJECT/'submission_strengthening/references/geometric_source.py')))
    else:
        df.to_csv(out/'afast_per_frame.csv',index=False)
        df.groupby('condition')[['regret','unsafe','executed','abstained']].mean().to_csv(out/'afast_condition_point_estimates.csv')
        js(out/'evaluation_receipt.json',dict(status='COMPLETE_POSTHOC',records=len(df),validation_lock_sha256=sha(out/'validation_lock.json'),per_frame_sha256=sha(out/'afast_per_frame.csv')))
if __name__=='__main__':main()
