"""Locked common-budget fit. Refuses launch without the objective gate/epoch lock."""
import argparse,csv,hashlib,json,math,pathlib,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import build,neural_inputs,CFG
from data import SharedDataset,EpochOrder,init_seed
from kinafford.losses import compute_losses
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--architecture',required=True,choices=['BF-D','KF-D']);p.add_argument('--seed',type=int,required=True,choices=[10,11,12]);a=p.parse_args()
lock=json.loads((ROOT/'manifests/execution_lock.json').read_text());assert lock['status']=='AUTHORIZED_GATES_PASSED'
E=int(lock['epochs']);assert E in [14,28,42];total=750*E
for path,h in lock['locked_files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,('frozen_file_changed',path)
clock=json.loads((ROOT/'records/start_clock.json').read_text())
assert time.time()<clock['t0_unix']+26*3600,'Final-checkpoint stage deadline reached'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False
seed=init_seed(a.seed,a.architecture);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
directory=ROOT/'checkpoints'/a.architecture/f'seed{a.seed}';directory.mkdir(parents=True,exist_ok=False)
ds=SharedDataset('train',a.seed,True);val=SharedDataset('val',a.seed,False);order=EpochOrder(ds,a.seed)
loader_rng=torch.Generator().manual_seed(700000+a.seed)
loader=DataLoader(ds,batch_size=4,sampler=order,num_workers=2,pin_memory=True,persistent_workers=True,drop_last=True,generator=loader_rng)
vl=DataLoader(val,batch_size=4,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(800000+a.seed))
m=build(a.architecture).cuda().float();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
torch.save(dict(model=m.state_dict(),architecture=a.architecture,seed=a.seed,init_seed=seed,epochs=E,pretrained=False),directory/'initial.pt')
init_hash=hashlib.sha256((directory/'initial.pt').read_bytes()).hexdigest()
counts=dict(optimizer_updates=0,training_frame_exposures=0,epoch_validation_forwards=0)
step=0;start=time.monotonic();history=[];torch.cuda.reset_peak_memory_stats()
mf=(directory/'minibatches.csv').open('w',newline='');mw=csv.DictWriter(mf,fieldnames=['epoch','microbatch','scene_slots','input_sha256','dropout_branches']);mw.writeheader()
def gpu(b):return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
def objective(b):
    out=m(neural_inputs(b))
    for k,v in out.items():
        if isinstance(v,torch.Tensor):assert v.dtype==torch.float32 and torch.isfinite(v).all(),('nonfinite_or_wrong_precision',k)
    return out,compute_losses(out,b,CFG)
def save_resume(epoch,next_micro,totals,dropouts):
    state=dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,next_micro=next_micro,step=step,
        counts=counts,history=history,epoch_loss_sums=totals,epoch_dropout_counts=dropouts.tolist(),
        random_state=random.getstate(),numpy_state=np.random.get_state(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=loader_rng.get_state(),epochs=E,architecture=a.architecture,seed=a.seed)
    temp=directory/'current.tmp';torch.save(state,temp);temp.replace(directory/'current.pt')
try:
    for epoch in range(E):
        ds.set_epoch(epoch);m.train();begin=time.monotonic();totals={};dropouts=np.zeros(4,dtype=np.int64);order_sha=order.digest()
        opt.zero_grad(set_to_none=True)
        for micro,b in enumerate(loader):
            assert time.time()<clock['t0_unix']+26*3600,'Checkpoint-stage deadline reached mid-fit'
            sha=hashlib.sha256()
            for k in ['img','depth','cam_present','lidar_range','lidar_valid','q','T_dyn','K','meta']:
                arr=b[k].numpy();sha.update(k.encode());sha.update(arr.tobytes())
            mw.writerow(dict(epoch=epoch,microbatch=micro,scene_slots=';'.join(f'{int(s)}:{int(f)}' for s,f in zip(b['scene_id'],b['frame_id'])),input_sha256=sha.hexdigest(),dropout_branches=';'.join(map(str,b['dropout_branch'].tolist()))))
            dropouts+=np.bincount(b['dropout_branch'].numpy(),minlength=4)
            b=gpu(b);counts['training_frame_exposures']+=len(b['q']);out,losses=objective(b)
            assert torch.isfinite(losses['total'])
            (losses['total']/4).backward()
            for k,v in losses.items():totals[k]=totals.get(k,0.)+float(v.detach())
            if (micro+1)%4==0:
                mult=(step+1)/300 if step<300 else .02+.98*.5*(1+math.cos(math.pi*(step+1-300)/(total-300)))
                for g in opt.param_groups:g['lr']=1e-4*mult
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True);step+=1;counts['optimizer_updates']+=1
                if step%250==0:
                    mf.flush();save_resume(epoch,micro+1,totals,dropouts)
                if step%50==0:
                    torch.cuda.synchronize()
                    progress=dict(architecture=a.architecture,seed=a.seed,epoch=epoch+1,epochs=E,step=step,total_steps=total,
                         elapsed_s=time.monotonic()-start,gradient_norm=float(norm),counts=counts,utc_unix=time.time())
                    (directory/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
        assert micro+1==3000
        torch.cuda.synchronize();train_seconds=time.monotonic()-begin;m.eval();vsum={};sem=np.zeros((6,6),np.int64)
        hazard={'tp':0,'fp':0,'fn':0,'predicted_positive':0,'true_positive_support':0,'cells':0};height_sum=0.;height_n=0;utility_sum=0.;utility_n=0
        vstart=time.monotonic()
        with torch.no_grad():
            for b in vl:
                b=gpu(b);counts['epoch_validation_forwards']+=len(b['q']);out,loss=objective(b)
                for k,v in loss.items():vsum[k]=vsum.get(k,0)+float(v)*len(b['q'])
                labels=out['sem'].argmax(1);sem+=np.bincount((b['gt_sem']*6+labels).cpu().numpy().ravel(),minlength=36).reshape(6,6)
                hp=out['haz'].sigmoid()>.5;ht=b['gt_haz']>.5
                hazard['tp']+=int((hp&ht).sum());hazard['fp']+=int((hp&~ht).sum());hazard['fn']+=int((~hp&ht).sum());hazard['cells']+=hp.numel();hazard['predicted_positive']+=int(hp.sum());hazard['true_positive_support']+=int(ht.sum())
                valid=(b['gt_reach']>=.05)&(b['gt_sem']==0);height_sum+=float((out['H']-b['gt_H']).abs()[valid].sum());height_n+=int(valid.sum())
                utility_sum+=float((out['aff'].sigmoid()-b['gt_aff']).abs().sum());utility_n+=b['gt_aff'].numel()
        torch.cuda.synchronize();validation_seconds=time.monotonic()-vstart
        row=dict(epoch=epoch+1,train_losses={k:v/3000 for k,v in totals.items()},validation_losses={k:v/1200 for k,v in vsum.items()},
            training_seconds=train_seconds,validation_seconds=validation_seconds,dropout_counts=dropouts.tolist(),order_sha256=order_sha,
            semantic_confusion=sem.tolist(),hazard_counts=hazard,height_absolute_sum=height_sum,height_cells=height_n,
            utility_absolute_sum=utility_sum,utility_cells=utility_n,counts=dict(counts),lr=opt.param_groups[0]['lr'])
        history.append(row);(directory/'history.json').write_text(json.dumps(history,indent=2));save_resume(epoch+1,0,{},np.zeros(4,dtype=np.int64))
        print('EPOCH_COMPLETE',a.architecture,a.seed,epoch+1,'train_s',round(train_seconds,2),'val_s',round(validation_seconds,2),flush=True)
    assert step==total and counts['training_frame_exposures']==12000*E and counts['epoch_validation_forwards']==1200*E
    final=directory/'final.pt';torch.save(dict(model=m.state_dict(),architecture=a.architecture,seed=a.seed,init_seed=seed,
        epochs=E,step=step,cfg=CFG,counts=counts,precision='actual_fp32',tf32=False,final_checkpoint=True,history=history),final)
    receipt=dict(status='FORMAL_FIT_COMPLETE',architecture=a.architecture,seed=a.seed,epochs=E,counts=counts,
        parameters=sum(p.numel() for p in m.parameters()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        elapsed_seconds=time.monotonic()-start,final_checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        initialization_sha256=init_hash,final_checkpoint=str(final.relative_to(ROOT)),finish_unix=time.time())
    (directory/'completion.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)
except Exception as e:
    (directory/'failure.json').write_text(json.dumps(dict(status='FAILED',error=repr(e),step=step,counts=counts,utc_unix=time.time()),indent=2))
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),counts=counts,step=step,error=repr(e)),directory/'failed_state.pt')
    raise
finally:mf.flush();mf.close()
