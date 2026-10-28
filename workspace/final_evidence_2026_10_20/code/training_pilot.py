"""Discarded development-only forward/backward and weighted-loss controls."""
from pathlib import Path
import argparse,json,time,random
import numpy as np
import torch
from torch.utils.data import DataLoader
from models_final import initialize,compute_losses,class_weights,original_losses,neural_inputs,CFG
from data import SharedDataset,init_seed
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['KF-D','BF-D'],required=True);a=p.parse_args()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False
seed=init_seed(10,a.architecture);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
m,receipt=initialize(a.architecture);m=m.cuda().float().train();ds=SharedDataset('train',10,True)
loader=DataLoader(ds,batch_size=4,num_workers=2,pin_memory=True,drop_last=True,shuffle=False)
weights=class_weights();ones={k:torch.ones_like(v) for k,v in weights.items()}
opt=torch.optim.AdamW(m.parameters(),lr=1e-4/900,betas=(.9,.999),eps=1e-8,weight_decay=.01)
times=[];ids=[];norms=[];equal=False;opt.zero_grad(set_to_none=True)
for i,b in enumerate(loader):
    if i==64:break
    ids.extend(b['scene_id'].tolist());assert max(ids)<3000
    b={k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
    torch.cuda.synchronize();start=time.monotonic();out=m(neural_inputs(b))
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in out.values() if isinstance(v,torch.Tensor))
    if i==0:
        original=original_losses(out,b,CFG);same=compute_losses(out,b,CFG,ones)
        for k in original:torch.testing.assert_close(original[k],same[k],atol=1e-6,rtol=1e-6)
        equal=True
    loss=compute_losses(out,b,CFG,weights);assert torch.isfinite(loss['total'])
    (loss['total']/4).backward()
    if (i+1)%4==0:
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);norms.append(float(norm));opt.step();opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize();times.append(time.monotonic()-start)
    if (i+1)%16==0:print(a.architecture,'pilot microbatches',i+1,'seconds',sum(times),flush=True)
assert equal and len(times)==64 and all(np.isfinite(norms))
r=dict(status='PASS_DISCARDED_TRAINING_ONLY_PILOT',architecture=a.architecture,frames=256,optimizer_updates=16,all_weights_one_reproduces_original_losses=True,pretrained_loading=receipt,precision='actualFP32;autocast/TF32 off',seconds_per_microbatch_max_after_warmup=max(times[16:]),seconds_per_microbatch_mean_after_warmup=float(np.mean(times[16:])),finite_gradient_norms=norms,epoch_budget_selected=False,epochs=42,no_validation_or_holdout_outcomes=True)
(ROOT/f'protocol/training_pilot_{a.architecture}.json').write_text(json.dumps(r,indent=2)+'\n');print(a.architecture,r['status'],r['seconds_per_microbatch_mean_after_warmup'])
