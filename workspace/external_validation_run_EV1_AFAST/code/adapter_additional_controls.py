"""Additional analytic joint/frame, label and interface controls before fitting."""
import json,pathlib,time
import numpy as np
import torch
from models import build,neural_inputs,CFG
from data import SharedDataset,rng,EpochOrder
from kinafford.sim.kinematics import link_transforms,sensor_pose_U,camera_intrinsics,SENSORS
from kinafford.losses import compute_losses
ROOT=pathlib.Path(__file__).resolve().parents[1]
rows=[]
def ok(name,**kw):rows.append(dict(test=name,passed=True,**kw))
q=np.array([np.pi/2,np.pi/6,0.,0.]);links=link_transforms(q)
expected=np.array([.55+5.7*np.cos(np.pi/6),0.,1.95+5.7*.5])
np.testing.assert_allclose(links['arm'][:3,3],expected,atol=1e-12)
np.testing.assert_allclose(links['boom'][:3,:3]@np.array([1.,0.,0.]),[np.sqrt(3)/2,0.,.5],atol=1e-12)
dt=1e-5;qp=q.copy();qm=q.copy();qp[1]+=dt;qm[1]-=dt
derivative=(link_transforms(qp)['arm'][:3,3]-link_transforms(qm)['arm'][:3,3])/(2*dt)
np.testing.assert_allclose(derivative,[-5.7*.5,0.,5.7*np.sqrt(3)/2],atol=1e-9)
np.testing.assert_allclose(sensor_pose_U('lidar',q)[:3,3],[-.4,0.,3.35],atol=0)
ok('independent_boom_raise_sign_arm_pivot_derivative_and_upper_not_ground_frame')
for c in ['cab','boom','stick']:
    K=camera_intrinsics(SENSORS[c]);fx=128/np.tan(SENSORS[c]['hfov']/2)
    np.testing.assert_allclose(K,[[fx,0,128],[0,fx,80],[0,0,1]],atol=0)
ok('native_actual_intrinsics_and_half_pixel_sensor_convention')
ds=SharedDataset('train',40,True);ds.set_epoch(0);s0=ds[7];ds.set_epoch(1);s1=ds[7];s1b=ds[7]
for k in s1:
    if isinstance(s1[k],torch.Tensor):torch.testing.assert_close(s1[k],s1b[k],rtol=0,atol=0)
assert not np.array_equal(rng(40,'photometric',0,1,2).random(10),rng(40,'photometric',1,1,2).random(10))
assert not np.array_equal(rng(40,'photometric',0,1,2).random(10),rng(40,'dropout',0,1,2).random(10))
o=EpochOrder(ds,40);assert len(set(o.indices()))==12000
ok('epoch_aware_repeatable_shared_namespaced_augmentation_and_order')
for arch in ['BF-D','KF-D']:
    m=build(arch)
    forbidden={'q_true':torch.zeros(1),'gt_aff':torch.zeros(1),'z_target':torch.zeros(1)}
    try:m(forbidden)
    except AssertionError:ok(arch+'_privileged_oracle_fields_rejected')
    else:raise AssertionError('neural allowlist failed')
    assert not any(p.is_floating_point() and p.dtype!=torch.float32 for p in m.parameters())
    if arch=='BF-D':
        empty=torch.zeros(1,32,512);ranges=empty.clone();ranges[0,0,0]=10;valid=empty.clone();valid[0,0,0]=1
        T=torch.tensor(np.stack([sensor_pose_U(n,q) for n in ['cab','boom','stick','lidar']]),dtype=torch.float32)[None]
        p=m.lidar_points(dict(T_dyn=T,lidar_range=ranges,lidar_valid=valid))[0]
        expected=torch.tensor([[-.4-10*np.cos(np.deg2rad(28)),0.,3.35-10*np.sin(np.deg2rad(28))]],dtype=torch.float32)
        torch.testing.assert_close(p,expected,atol=1e-5,rtol=1e-5)
        ok('independent_radial_xyz_to_upper_translation_no_swing_double_rotation')
    del m
# Ground-truth terrain defines material support; semantics CE is mutually exclusive.
H=W=2;out={'H':torch.zeros(1,H,W),'H_logvar':torch.zeros(1,H,W),'sem':torch.zeros(1,6,H,W),
 'mat':torch.zeros(1,4,H,W),'haz':torch.zeros(1,H,W),'aff':torch.zeros(1,H,W),'aff_logvar':torch.zeros(1,H,W),'img_seg':[]}
b={'gt_H':torch.zeros(1,H,W),'gt_sem':torch.tensor([[[0,1],[2,5]]]),'gt_mat':torch.zeros(1,H,W,dtype=torch.long),
 'gt_haz':torch.zeros(1,H,W),'gt_aff':torch.zeros(1,H,W),'gt_reach':torch.ones(1,H,W),'gt_vis':torch.ones(1,H,W)}
loss=compute_losses(out,b,CFG);torch.testing.assert_close(loss['sem'],torch.tensor(np.log(6),dtype=torch.float32))
torch.testing.assert_close(loss['mat'],torch.tensor(np.log(4),dtype=torch.float32))
b['gt_mat'][0,0,1]=3;out['mat'][0,3,0,1]=100
torch.testing.assert_close(compute_losses(out,b,CFG)['mat'],loss['mat'])
ok('exclusive_semantic_ce_and_true_terrain_material_support')
(ROOT/'tests/adapter_additional_controls.json').write_text(json.dumps(dict(status='PASSED',tests=rows,non_training_forwards=0,utc_unix=time.time()),indent=2))
print(json.dumps(rows,indent=2))
