"""Constructed controls plus original training inputs; never any historical test."""
import copy, hashlib, json, pathlib, sys, time, traceback
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from models import build,neural_inputs,ExcaLSS,CFG
from data import SharedDataset,init_seed
from kinafford.losses import compute_losses
ROOT=pathlib.Path(__file__).resolve().parents[1]
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
rows=[];count=0
def record(name,**kw):
    row=dict(test=name,passed=True,**kw);rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'tests/adapter_preflight_progress.json').write_text(json.dumps(rows,indent=2))
def finite(out):
    for k,v in out.items():
        if isinstance(v,torch.Tensor):assert v.dtype==torch.float32 and torch.isfinite(v).all(),k
def same(a,b):
    for k in ['H','sem','mat','haz','aff','H_logvar','aff_logvar']:
        torch.testing.assert_close(a[k],b[k],atol=0,rtol=0)
try:
    lss=ExcaLSS(256,80,(160,256),(20,32),(-8.,16.,.125),(-12.,12.,.125),(-8.,8.,16.),(.05,40.05,.5),downsample=2).cuda()
    assert lss.frustum[0,0,0].tolist()==[4.,4.,np.float32(.05)]
    R=torch.tensor([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]],device='cuda')[None,None]
    t=torch.tensor([1.,2.,3.],device='cuda')[None,None]
    K=torch.tensor([[2.,0.,4.],[0.,2.,4.],[0.,0.,1.]],device='cuda')[None,None]
    geo=lss.get_geometry(R,t,K,torch.eye(3,device='cuda')[None,None],torch.zeros(1,1,3,device='cuda'))
    torch.testing.assert_close(geo[0,0,0,0,0],torch.tensor([1.,2.05,3.],device='cuda'))
    # Pixel x=12, optical z=.05 => optical x=.2, radial range greater than z.
    torch.testing.assert_close(geo[0,0,0,0,1],torch.tensor([1.2,2.05,3.],device='cuda'))
    T=torch.eye(4,device='cuda');T[:3,:3]=R[0,0];T[:3,3]=t[0,0]
    torch.testing.assert_close(torch.linalg.inv(T)@T,torch.eye(4,device='cuda'),atol=1e-6,rtol=1e-6)
    record('analytic_camera_axes_optical_depth_transform_inverse_and_swin_anchor')
    # Each fixture point is prescribed independently of adapter quantization.
    positions=torch.tensor([[-8.0001,-12.,0.],[-8.,-12.,0.],[15.9999,11.9999,0.],[16.,0.,0.],[0.,12.,0.]],device='cuda')
    idx=torch.floor((positions-torch.tensor([-8.,-12.,-8.],device='cuda'))/torch.tensor([.125,.125,16.],device='cuda')).long()
    assert idx.tolist()==[[-1,0,0],[0,0,0],[191,191,0],[192,96,0],[64,192,0]]
    assert (((idx>=0)&(idx<lss.nx)).all(1)).tolist()==[False,True,True,False,False]
    lss._cam_present=torch.ones(1,1,device='cuda')
    pooled=lss.bev_pool(positions[None,None,None,None],torch.ones(1,1,1,1,5,80,device='cuda'))
    assert pooled[0,0,0,0]==1 and pooled[0,0,191,191]==1 and pooled[0,0].sum()==2
    record('floor_negative_lower_exact_upper_edges_native_pool')
    del lss
    ds=SharedDataset('train',40,augment=False)
    batch=default_collate([ds[i] for i in range(4)])
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    for arch in ['BF-D','KF-D']:
        torch.manual_seed(init_seed(40,arch));m=build(arch).cuda().float()
        m.train();count+=4;out=m(neural_inputs(batch));finite(out)
        loss=compute_losses(out,batch,CFG)['total'];assert torch.isfinite(loss)
        loss.backward()
        groups= {'camera_backbone':'camera_backbone','camera_neck':'camera_neck','lift':'lift.depthnet','sparse':'sparse_encoder','fuser':'fuser','decoder_backbone':'decoder_backbone','decoder_neck':'decoder_neck','condition':'condition','image_head':'image_head'} if arch=='BF-D' else {'image':'encoder','points':'pillars','decoder':'unet','image_head':'encoder.seg'}
        groups.update({head:'heads.'+head for head in ['H','sem','mat','haz','aff']})
        norms={}
        for name,prefix in groups.items():
            grads=[p.grad for n,p in m.named_parameters() if n.startswith(prefix) and p.requires_grad and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads),(arch,name,'missing_or_nonfinite_grad')
            norms[name]=sum(float(g.abs().sum()) for g in grads);assert norms[name]>0,(arch,name,'zero_grad')
        record(arch+'_native_shapes_real_fp32_forward_backward_branch_gradients',loss=float(loss),gradient_l1=norms,peak_bytes=torch.cuda.max_memory_allocated())
        m.zero_grad(set_to_none=True);m.eval()
        b=neural_inputs(batch);snapshot={k:v.clone() for k,v in b.items()}
        with torch.no_grad():
            allmasked={k:v.clone() for k,v in b.items()};allmasked['cam_present'].zero_();allmasked['img'].zero_();allmasked['depth'].zero_();allmasked['meta'][:,:3]=0
            count+=4;y=m(allmasked);finite(y)
            arbitrary={k:v.clone() for k,v in allmasked.items()};arbitrary['img'].fill_(1234.);arbitrary['depth'].fill_(400.)
            count+=4;y2=m(arbitrary);finite(y2);same(y,y2)
            record(arch+'_all_camera_absence_and_arbitrary_removed_content_invariance')
            if arch=='BF-D':
                points=m.lidar_points(b)
                raw=b['lidar_range'][...,None]*m.rays[None]
                torch.testing.assert_close(torch.linalg.vector_norm(raw,dim=-1),b['lidar_range'],atol=1e-5,rtol=1e-5)
                record('radial_range_to_xyz_norm_agreement')
                for case in ['nonempty_empty','empty_nonempty','all_empty']:
                    empty=points[0][:0]
                    p=[points[0],empty] if case=='nonempty_empty' else [empty,points[0]] if case=='empty_nonempty' else [empty,empty]
                    count+=2;z=m.lidar_features(p)
                    assert z.shape==(2,256,96,96),case
                    absent=[1] if case=='nonempty_empty' else [0] if case=='empty_nonempty' else [0,1]
                    assert all(torch.count_nonzero(z[i])==0 for i in absent),case
                    record('logical_sparse_batch_'+case,logical_B=2,points=[len(x) for x in p],shape=list(z.shape))
            for case in ['nonempty_empty','empty_nonempty','all_empty']:
                x={k:v[:2].clone() for k,v in b.items()}
                absent=[1] if case=='nonempty_empty' else [0] if case=='empty_nonempty' else [0,1]
                x['lidar_valid'][absent]=0
                count+=2;z=m(x);finite(z);assert z['H'].shape==(2,96,96)
                record(arch+'_full_model_missing_lidar_'+case,logical_B=2)
            for k in b:torch.testing.assert_close(b[k],snapshot[k],atol=0,rtol=0)
            record(arch+'_no_input_alias_mutation')
        del m,out,loss;torch.cuda.empty_cache()
    receipt=dict(status='PASSED',tests=rows,non_training_forwards=count,utc_unix=time.time())
    (ROOT/'tests/adapter_preflight_receipt.json').write_text(json.dumps(receipt,indent=2))
except Exception as e:
    traceback.print_exc()
    (ROOT/'tests/adapter_preflight_failure.json').write_text(json.dumps(dict(status='FAILED',error=repr(e),tests=rows,non_training_forwards=count,utc_unix=time.time()),indent=2))
    raise
