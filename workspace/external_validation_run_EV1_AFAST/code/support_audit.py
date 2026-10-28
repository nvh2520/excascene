"""Measured sensor support diagnostics on original development inputs only."""
import json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/excascene_snapshot'))
from kinafford.data.dataset import extrinsics
from kinafford.sim.kinematics import lidar_rays,SENSORS
dirs=lidar_rays(SENSORS['lidar']).astype(np.float32)
counts=dict(frames=0,camera_present_pixels=0,valid_optical_depth_pixels=0,depth_below_min=0,depth_at_or_above_max=0,
            valid_lidar_returns=0,lidar_range_le_point3=0,lidar_xyz_outside_xy=0,lidar_xyz_below_z_min=0,lidar_xyz_at_or_above_z_max=0,
            lidar_retained_external_support=0)
for p in sorted((ROOT.parent/'data/val').glob('*.npz')):
    with np.load(p) as d:
        assert 3000<=int(d['scene_id'])<3300
        depth=d['depth'].astype(np.float32);presence=d['cam_present']>.5
        seen=np.broadcast_to(presence[:,None,None],depth.shape);valid=seen&(depth>0)
        counts['frames']+=1;counts['camera_present_pixels']+=int(seen.sum());counts['valid_optical_depth_pixels']+=int(valid.sum())
        counts['depth_below_min']+=int((valid&(depth<.05)).sum());counts['depth_at_or_above_max']+=int((valid&(depth>=40.05)).sum())
        ranges=d['lidar_range'].astype(np.float32);lv=d['lidar_valid']>.5;T=extrinsics(d['q_rep'])[0][3]
        raw=ranges[...,None]*dirs;xyz=raw@T[:3,:3].T+T[:3,3]
        xy=(xyz[...,0]>=-8)&(xyz[...,0]<16)&(xyz[...,1]>=-12)&(xyz[...,1]<12)
        zz=(xyz[...,2]>=-8)&(xyz[...,2]<8)
        counts['valid_lidar_returns']+=int(lv.sum());counts['lidar_range_le_point3']+=int((lv&(ranges<=.3)).sum())
        counts['lidar_xyz_outside_xy']+=int((lv&~xy).sum());counts['lidar_xyz_below_z_min']+=int((lv&(xyz[...,2]<-8)).sum())
        counts['lidar_xyz_at_or_above_z_max']+=int((lv&(xyz[...,2]>=8)).sum())
        counts['lidar_retained_external_support']+=int((lv&(ranges>.3)&xy&zz).sum())
receipt=dict(status='DEVELOPMENT_DIAGNOSTIC',counts=counts,neither_training_selection_nor_holdout=True,
    camera_support='Learned depth hypotheses0.05+0.5j (j0..79); every measured RGB-D pixel remains in the camera input.',
    lidar_support='U-frame[-8,16)x[-12,12)x[-8,8), radial range>.3; sparse Z grid40 measured bins plus one empty padding bin.',utc_unix=time.time())
(ROOT/'reports/development_support_audit.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
