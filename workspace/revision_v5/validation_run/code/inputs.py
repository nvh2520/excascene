"""Explicit model input allowlist; targets cannot enter forward()."""
from common import *
from kinafford.data.dataset import extrinsics,IMAGENET_MEAN,IMAGENET_STD
from kinafford.sim.kinematics import CAMERAS,SENSORS,camera_intrinsics
K=np.stack([camera_intrinsics(SENSORS[c]) for c in CAMERAS]).astype(np.float32)
def tensorize(d,mask=False):
    rgb=(d['rgb'].astype(np.float32)/255-IMAGENET_MEAN)/IMAGENET_STD
    depth=d['depth'].astype(np.float32).copy();presence=d['cam_present'].astype(np.float32).copy()
    img=np.concatenate([rgb.transpose(0,3,1,2),(depth/20)[:,None]],axis=1)*presence[:,None,None,None]
    lr=d['lidar_range'].astype(np.float32);lv=d['lidar_valid'].astype(np.float32)
    q=d['q_rep'].astype(np.float32);td,ts=extrinsics(q)
    if mask:img[:]=0;depth[:]=0;presence[:]=0
    return dict(img=img.astype(np.float32),depth=depth,cam_present=presence,lidar_range=lr,lidar_valid=lv,q=q,
                T_dyn=td,T_static=ts,K=K.copy(),meta=np.r_[presence,lv.mean(),1].astype(np.float32))
def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}
def batch(records,mask=False):
    import torch
    ts=[tensorize(d,mask) for d in records]
    return {k:torch.from_numpy(np.stack([t[k] for t in ts])).cuda() for k in ts[0]}
def targets(d):
    return {k:d['gt_'+k].astype(np.float32) for k in ['H','sem','mat','haz','aff','reach','vis']}|{'z_target':float(d['z_target'])}
