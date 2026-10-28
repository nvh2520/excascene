"""Asymmetric XY, native voxel boundaries and single-view masking controls."""
import json,pathlib,time
import numpy as np
import torch
from torch.utils.data import default_collate
from models import build,ExcaLSS,neural_inputs
from data import SharedDataset
ROOT=pathlib.Path(__file__).resolve().parents[1]
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest');torch.set_num_threads(4);torch.manual_seed(40)
rows=[];count=0
def ok(name,**kw):rows.append(dict(test=name,passed=True,**kw))
l=ExcaLSS(256,80,(160,256),(20,32),(-8.,16.,.125),(-12.,12.,.125),(-8.,8.,16.),(.05,40.05,.5),2).cuda()
l._cam_present=torch.ones(1,1,device='cuda')
points=torch.tensor([[-8+2.5*.125,-12+7.5*.125,0.]],device='cuda')
pooled=l.bev_pool(points[None,None,None,None],torch.full((1,1,1,1,1,80),3.,device='cuda')).transpose(-1,-2)
assert pooled.shape==(1,80,192,192) and pooled[0,0,7,2]==3 and pooled[0,0,2,7]==0
ok('asymmetric_xy_native_pool_to_row_y_col_x')
del l
m=build('BF-D').cuda().float().eval()
p=torch.tensor([[-8-.001,-12.,0.],[-8.,-12.,0.],[16.,0.,0.],[0.,12.,0.],[0.,0.,8.],[0.,0.,-8.],[-7.9,-11.8,-7.9]],device='cuda')
v,c,n=m.voxelize(p)
expected=torch.tensor([[0,0,20],[256,384,0],[3,6,0]],device='cuda',dtype=torch.int32)
assert len(c)==3
torch.testing.assert_close(c,expected,atol=0,rtol=0)
ok('native_voxel_floor_edges_support_and_xyz_coordinate_order',indices=c.cpu().tolist())
del m;torch.cuda.empty_cache()
ds=SharedDataset('train',40,False);full=default_collate([ds[0],ds[1]])
b=neural_inputs({k:v.cuda() for k,v in full.items() if isinstance(v,torch.Tensor)})
for arch in ['BF-D','KF-D']:
    m=build(arch).cuda().float().eval()
    x={k:v.clone() for k,v in b.items()};x['cam_present'][:,1]=0;x['img'][:,1]=0;x['depth'][:,1]=0;x['meta'][:,1]=0
    z={k:v.clone() for k,v in x.items()};z['img'][:,1]=1234;z['depth'][:,1]=400
    branch=[]
    layer=m.camera_backbone if arch=='BF-D' else m.encoder
    def hook(module,args,outputs):
        branch.append((args[0].detach().clone(),[v.detach().clone() for v in outputs]))
    handle=layer.register_forward_hook(hook)
    with torch.no_grad():
        count+=2;y=m(x);count+=2;baseline=m(x);count+=2;y2=m(z)
    handle.remove()
    # Input/content invariance is exact. Sparse reduction arithmetic is evaluated
    # at the independently specified 2e-5 native-vs-dense FP32 tolerance, retained
    # from native_probe.py, rather than demanding unsupported bitwise replay.
    for j in [1,2]:
        torch.testing.assert_close(branch[0][0],branch[j][0],atol=0,rtol=0)
        for first,later in zip(branch[0][1],branch[j][1]):torch.testing.assert_close(first,later,atol=0,rtol=0)
    diffs={};repeat={}
    for k in ['H','sem','mat','haz','aff','H_logvar','aff_logvar']:
        torch.testing.assert_close(y[k],y2[k],atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(y[k],baseline[k],atol=2e-5,rtol=2e-5)
        diffs[k]=float((y[k]-y2[k]).abs().max());repeat[k]=float((y[k]-baseline[k]).abs().max())
    for k in ['lidar_range','lidar_valid','q','T_dyn','K']:torch.testing.assert_close(x[k],b[k],atol=0,rtol=0)
    ok(arch+'_one_camera_removed_content_invariance_and_retained_lidar_calibration_joints_equality',
       encoder_consumed_input_and_feature_equality='bitwise_exact',full_output_precision_tolerance='prior native_probe2e-5',
       perturbed_max_absolute_differences=diffs,identical_input_replay_max_absolute_differences=repeat)
    del m;torch.cuda.empty_cache()
(ROOT/'tests/coordinate_mask_controls.json').write_text(json.dumps(dict(status='PASSED',tests=rows,non_training_forwards=count,utc_unix=time.time()),indent=2))
print(json.dumps(rows,indent=2))
