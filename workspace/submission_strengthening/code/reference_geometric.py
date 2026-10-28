"""Conditional unchanged historical geometric predictor, explicit strict scoring."""
from __future__ import annotations
import argparse,csv,importlib.util,json,shutil,sys,time
from pathlib import Path
import numpy as np
from reference_policies import digest,load_selector,write_csv,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--afast',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--action',choices=['gate','evaluate'],required=True)
    a=p.parse_args();project=a.project.resolve();out=a.output;out.mkdir(parents=True,exist_ok=True)
    # The unchanged predictor's source imports the original pure NumPy/SciPy scene constants.
    sys.path.insert(0,str(project))
    from kinafford.sim.kinematics import SENSOR_ORDER,CAMERAS,SENSORS,sensor_pose_U,camera_intrinsics,lidar_rays
    source=project/'stage2/geometric.py';copy=out/'geometric_source.py'
    if not copy.exists():shutil.copy2(source,copy)
    assert digest(copy)==digest(source)
    spec=importlib.util.spec_from_file_location('frozen_historical_geometry',copy);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    choose=load_selector(a.afast/'source/selectors_v5_original.py')
    K=np.stack([camera_intrinsics(SENSORS[c]) for c in CAMERAS]).astype(np.float32)
    directions=lidar_rays(SENSORS['lidar']).astype(np.float32)
    def process(path):
        with np.load(path,allow_pickle=False) as z:
            sample={k:z[k].astype(np.float32) for k in ['depth','cam_present','lidar_range','lidar_valid','gt_reach','gt_H','gt_aff','gt_haz']}
            sample['gt_sem']=z['gt_sem'];sample['z_target']=float(z['z_target']);q=z['q_rep'].astype(np.float32).astype(np.float64)
        sample['K']=K;sample['T_dyn']=np.stack([sensor_pose_U(n,q) for n in SENSOR_ORDER]).astype(np.float32)
        H,occupied=mod.predict_geometry(sample,directions)
        score=mod.compose_geometry(H,sample['gt_reach'],sample['z_target'])
        pick,ties,n=choose(score,sample['gt_reach'],hazard=None)
        reach=np.isfinite(sample['gt_reach'])&(sample['gt_reach']>=.05)
        oracle=float(sample['gt_aff'][reach].max()) if reach.any() else 0.
        utility=float(sample['gt_aff'][pick]) if pick is not None else 0.
        unsafe=int(pick is not None and (sample['gt_haz'][pick]>.5 or sample['gt_sem'][pick]!=0))
        hmask=reach&(sample['gt_sem']==0)
        return dict(regret=oracle-utility,unsafe=unsafe,executed=int(pick is not None),abstained=int(pick is None),attempted=1,
            selected_utility=utility,oracle=oracle,pick_row=-1 if pick is None else int(pick[0]),pick_col=-1 if pick is None else int(pick[1]),
            reachable_cells=int(reach.sum()),eligible_cells=n,tie_count=ties,occupied_cells=int(occupied.sum()),all_empty=int(not occupied.any()),
            H_reachable_terrain=float(np.abs(H.astype(np.float64)-sample['gt_H'].astype(np.float64))[hmask].mean()) if hmask.any() else '',
            reachable_terrain_cells=int(hmask.sum()))
    if a.action=='gate':
        with (a.afast/'manifests/train_validation_inventory.csv').open() as f:
            val=next(r for r in csv.DictReader(f) if r['split']=='val')
        wall=time.perf_counter();cpu=time.process_time();v=process(project/val['path'])
        elapsed=time.perf_counter()-wall;cpusec=time.process_time()-cpu
        forecast_wall=elapsed*3328*2;forecast_cpu=cpusec*3328*2/3600
        passed=forecast_wall<=3600 and forecast_cpu<=4
        write_json(out/'geometric_gate.json',dict(status='PROCEED' if passed else 'NOT_REEVALUATED',validation_path=val['path'],
            validation_sha256=digest(project/val['path']),predictor_sha256=digest(copy),unchanged_predictor=True,
            validation_wall_seconds=elapsed,validation_cpu_seconds=cpusec,conservative_factor=2,
            forecast_3328_wall_seconds=forecast_wall,forecast_3328_cpu_core_hours=forecast_cpu,
            inference_scope='one recorded intact-sensor policy per existing AFAST record; no added camera-mask variant',
            adaptation='NumPy field loader reproducing dataset precision/extrinsics; strict v5 choose(score,reach,hazard=None), feasible oracle instead of historical driver',
            validation_plumbing=v,new_neural_forwards=0,new_random_draws=0))
        print((out/'geometric_gate.json').read_text());return
    gate=json.loads((out/'geometric_gate.json').read_text());assert gate['status']=='PROCEED' and gate['predictor_sha256']==digest(copy)
    with (a.afast/'manifests/generated_files.csv').open() as f:inventory=list(csv.DictReader(f))
    assert len(inventory)==3328;rows=[]
    for i,r in enumerate(inventory):
        row=process(a.afast/r['path']);row.update(condition=r['condition'],scene_id=int(r['scene_id']),frame_id=int(r['frame_id']),source_path=r['path'])
        rows.append(row)
        if (i+1)%512==0:print('unchanged geometric anchor',i+1,flush=True)
    write_csv(out/'geometric_per_frame.csv',rows)
    write_json(out/'geometric_scoring_provenance.json',dict(status='COMPLETE_POSTHOC_AFAST_ANCHOR',records=len(rows),
        predictor_sha256=digest(copy),source_predictor_path=str(source),legacy_driver_sha256=digest(project/'stage2/geometric_eval.py'),
        legacy_table_sha256=digest(project/'results_stage2/analysis/geometric_anchor.csv'),
        selector_sha256=digest(a.afast/'source/selectors_v5_original.py'),gate_sha256=digest(out/'geometric_gate.json'),
        inputs='measured depth/radial LiDAR, reported dynamic extrinsics, reach and supplied grade; fixed0.70 coefficient, no material/hazard prediction',
        fill='percell median; nearest occupiedcell with exact row-major tie; zeroheight if all empty',
        historical_driver_reproduced=False,new_neural_forwards=0,new_random_draws=0))

if __name__=='__main__':main()
