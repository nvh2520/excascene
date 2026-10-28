"""Fixed seed-40 plumbing/timing pilot; never a candidate fit or outcome search."""
import argparse, csv, hashlib, json, pathlib, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import build,neural_inputs,CFG
from data import SharedDataset,EpochOrder,init_seed
from kinafford.losses import compute_losses
from scoring import neutral,truth,evaluate,geometry

ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--architecture',required=True,choices=['BF-D','KF-D']);p.add_argument('--attempt',default='attempt_1');a=p.parse_args()
arch=a.architecture
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.benchmark=False
directory=ROOT/'reports/pilots'/arch/a.attempt;directory.mkdir(parents=True,exist_ok=False)
counts=dict(optimizer_updates=0,training_frame_exposures=0,pilot_measurement_forwards=0,epoch_validation_forwards=0,export_forwards=0)
def save(name,obj):
    (directory/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
def seed():
    s=init_seed(40,arch);random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def gpu(b):return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
def finite(out):
    for k,v in out.items():
        if isinstance(v,torch.Tensor):assert v.dtype==torch.float32 and torch.isfinite(v).all(),(k,'nonfinite_or_wrong_precision')
def update(m,opt,batches):
    opt.zero_grad(set_to_none=True);aggregate={}
    for b in batches:
        counts['training_frame_exposures']+=len(b['q'])
        out=m(neural_inputs(b));finite(out);losses=compute_losses(out,b,CFG)
        assert torch.isfinite(losses['total'])
        (losses['total']/4).backward()
        for k,v in losses.items():aggregate[k]=aggregate.get(k,0)+float(v.detach())/4
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
    opt.step();counts['optimizer_updates']+=1
    return aggregate,float(norm)
try:
    seed();m=build(arch).cuda().float();opt=optimizer(m)
    fixed_ds=SharedDataset('train',40,False)
    fixed=gpu(next(iter(DataLoader(torch.utils.data.Subset(fixed_ds,[0,1,2,3]),batch_size=4))))
    repeated=[]
    for i in range(28):
        m.eval()
        with torch.no_grad():
            counts['pilot_measurement_forwards']+=4
            value=float(compute_losses(m(neural_inputs(fixed)),fixed,CFG)['total'])
        repeated.append(value)
        if i>=4:
            m.train();loss,norm=update(m,opt,[fixed]*4)
        save('repeated_batch_progress.json',dict(objectives=repeated,counts=counts))
    passed=bool(np.mean(repeated[-4:])<np.mean(repeated[:4]))
    save('repeated_batch_receipt.json',dict(passed=passed,first_four_mean=float(np.mean(repeated[:4])),final_four_mean=float(np.mean(repeated[-4:])),objectives=repeated,updates=24))
    assert passed,'Repeated fixed-batch objective failed to decrease under prospective procedure'
    del m,opt,fixed;torch.cuda.empty_cache()
    seed();m=build(arch).cuda().float();opt=optimizer(m);m.train()
    ds=SharedDataset('train',40,True);order=EpochOrder(ds,40)
    loader=DataLoader(ds,batch_size=4,sampler=order,num_workers=2,pin_memory=True,persistent_workers=True,drop_last=True)
    it=iter(loader);timings=[];lossrows=[];dropout_counts=np.zeros(4,dtype=np.int64)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    for step in range(120):
        begin=time.monotonic();micros=[]
        for j in range(4):
            b=next(it);dropout_counts+=np.bincount(b['dropout_branch'].numpy(),minlength=4);micros.append(gpu(b))
        # Same declared 300-step warmup. E-specific cosine horizon remains pending.
        for g in opt.param_groups:g['lr']=1e-4*(step+1)/300
        loss,norm=update(m,opt,micros);torch.cuda.synchronize();elapsed=time.monotonic()-begin
        timings.append(elapsed);lossrows.append(dict(step=step,warmup=step<20,seconds=elapsed,gradient_norm=norm,**loss))
        save('timing_progress.json',dict(architecture=arch,steps_completed=step+1,seconds=timings,counts=counts))
        if (step+1)%10==0:print(arch,'timed_effective_update',step+1,'seconds',round(elapsed,4),flush=True)
    del it,loader,micros
    measured=np.array(timings[20:]);peak=torch.cuda.max_memory_allocated()
    with (directory/'timed_updates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(lossrows[0]));w.writeheader();w.writerows(lossrows)
    m.eval();val=SharedDataset('val',40,False)
    vl=DataLoader(val,batch_size=4,num_workers=2,pin_memory=True)
    totals={};hnum=0.;hden=0;sem=np.zeros((6,6),dtype=np.int64);hz={'tp':0,'fp':0,'fn':0}
    torch.cuda.synchronize();start=time.monotonic()
    with torch.no_grad():
        for b in vl:
            b=gpu(b);counts['epoch_validation_forwards']+=len(b['q'])
            out=m(neural_inputs(b));finite(out);loss=compute_losses(out,b,CFG)
            for k,v in loss.items():totals[k]=totals.get(k,0)+float(v)*len(b['q'])
            valid=(b['gt_reach']>=.05)&(b['gt_sem']==0)
            hnum+=float((out['H']-b['gt_H']).abs()[valid].sum());hden+=int(valid.sum())
            labels=out['sem'].argmax(1)
            sem+=np.bincount((b['gt_sem']*6+labels).cpu().numpy().reshape(-1),minlength=36).reshape(6,6)
            hp=out['haz'].sigmoid()>.5;ht=b['gt_haz']>.5
            hz['tp']+=int((hp&ht).sum());hz['fp']+=int((hp&~ht).sum());hz['fn']+=int((~hp&ht).sum())
    torch.cuda.synchronize();validation_seconds=time.monotonic()-start
    # Export and both frozen policies on 32 original validation inputs.
    exports=[];times=[]
    with torch.no_grad():
        for start_i in range(0,32,4):
            b=gpu(torch.utils.data.default_collate([val[i] for i in range(start_i,start_i+4)]))
            torch.cuda.synchronize();begin=time.monotonic()
            counts['export_forwards']+=4;out=m(neural_inputs(b));finite(out)
            for j in range(4):
                pred=neutral(out,j);d=truth(b,j)
                path=directory/f'development_prediction_{start_i+j:03d}.npz'
                np.savez_compressed(path,**pred)
                for policy in ['direct','soft']:
                    exports.append(dict(validation_index=start_i+j,**evaluate(pred,d,policy),**geometry(pred,d)))
            torch.cuda.synchronize();times.append(time.monotonic()-begin)
    with (directory/'development_scoring.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(exports[0]));w.writeheader();w.writerows(exports)
    checkpoint=directory/'discarded_pilot_state.pt'
    start=time.monotonic();torch.save({'model':m.cpu().state_dict(),'seed':40,'formal_candidate':False},checkpoint);checkpoint_io_s=time.monotonic()-start
    save('pilot_receipt.json',dict(status='PASSED',architecture=arch,seed=40,counts=counts,
        updates_timed=100,seconds_per_update_mean=float(measured.mean()),seconds_per_update_p95=float(np.quantile(measured,.95)),
        seconds_per_update_conservative=float(max(measured.mean(),np.quantile(measured,.95))),
        peak_allocated_bytes=peak,parameters=sum(p.numel() for p in m.parameters()),
        validation_seconds=validation_seconds,validation_frames=1200,export_seconds=sum(times),export_frames=32,
        checkpoint_io_seconds=checkpoint_io_s,dropout_counts=dropout_counts.tolist(),
        validation_losses={k:v/1200 for k,v in totals.items()},height_absolute_sum=hnum,height_cell_count=hden,
        semantic_confusion=sem.tolist(),hazard_counts=hz,
        measured_concurrency='one A100, one fit process, 2 data workers, torch CPU threads 4',fp32=True,tf32=False,
        pilot_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),utc_unix=time.time()))
    print((directory/'pilot_receipt.json').read_text(),flush=True)
except Exception as e:
    save('pilot_failure.json',dict(error=repr(e),counts=counts,utc_unix=time.time()))
    raise
