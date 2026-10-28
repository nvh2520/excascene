"""Training scenes only; fixed gate recipe and timed pilot, no test access."""
from common import *
import argparse,time
from concurrent.futures import ProcessPoolExecutor
from kinafford.sim.degrade import degrade_camera
KINDS=['occlusion','dust','blur','noise']

def prepare_scene(sid):
    clean=load_npz(ROOT/f'phase_a/material_replay/train/f_{sid:06d}_0.npz')
    iid=load_npz(PROJECT/f'data/train/f_{sid:06d}_0.npz',['rgb','cam_present','spec']);spec=json.loads(str(iid['spec']))
    xs=[];ys=[];details=[]
    from kinafford.sim.kinematics import CAMERAS
    for cam,name in enumerate(CAMERAS):
        variants=[('clean',[])] + [(f'full_{level}',[(k,level) for k in KINDS]+[('depth',level)]) for level in [1,2,3]]
        variants += [(f'{kind}_{level}',[(kind,level)]) for kind in KINDS for level in [1,2,3]]
        for j,(label,photo) in enumerate(variants):
            rgb=clean['rgb'][cam].copy();depth=clean['depth'][cam].copy();rng=np.random.default_rng(np.random.SeedSequence([2026101805,sid,cam,j]))
            for kind,level in photo:rgb,depth=degrade_camera(rgb,depth,kind,level,rng)
            severity=max([v for k,v in photo if k in KINDS],default=0)
            xs.append(rgb.transpose(2,0,1));ys.append(severity);details.append(dict(scene=sid,camera=cam,variant=label,severity=severity))
        if iid['cam_present'][cam]>.5:
            severity=max([v for k,v in spec['cam'][name]['photo'] if k in KINDS],default=0)
            xs.append(iid['rgb'][cam].transpose(2,0,1));ys.append(severity);details.append(dict(scene=sid,camera=cam,variant='original_iid',severity=severity))
    out=ROOT/f'study3/gate_training/{sid:06d}.npz';out.parent.mkdir(exist_ok=True,parents=True);assert not out.exists()
    np.savez_compressed(out,rgb=np.stack(xs),severity=np.array(ys,np.uint8))
    return dict(scene_id=sid,images=len(xs),counts=np.bincount(ys,minlength=4).tolist(),path=str(out.relative_to(ROOT)),sha256=sha(out))

def prepare(workers):
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(prepare_scene,range(400))):
            rows.append(r)
            if (i+1)%25==0:print(i+1,400,flush=True)
    csvwrite(ROOT/'study3/gate_training_inventory.csv',rows)
    js(ROOT/'study3/gate_training_receipt.json',dict(status='COMPLETE',scenes=400,images=sum(r['images'] for r in rows),
        counts=np.sum([r['counts'] for r in rows],axis=0).tolist(),no_validation_or_test_input=True,
        development_plan_sha256=sha(ROOT/'protocol/study3_development_plan.md'),code_sha256=sha(__file__)))

def train(pilot):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    from gate import CameraGate
    torch.set_num_threads(2);torch.manual_seed(2026101805)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    files=sorted((ROOT/'study3/gate_training').glob('*.npz'));assert len(files)==400
    images=[];labels=[]
    for f in files:
        d=load_npz(f);images.append(d['rgb']);labels.append(d['severity'])
    x=torch.from_numpy(np.concatenate(images));y=torch.from_numpy(np.concatenate(labels).astype(np.int64));del images,labels
    counts=torch.bincount(y,minlength=4);weights=1/counts.float();weights/=weights.mean()
    model=CameraGate().cuda().float();assert sum(p.numel() for p in model.parameters())<=3_000_000
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    lossfn=torch.nn.CrossEntropyLoss(weight=weights.cuda())
    loader=DataLoader(TensorDataset(x,y),batch_size=64,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(2026101805))
    start=time.time();steps=0;hist=[]
    for epoch in range(1 if pilot else 20):
        total=0.;n=0;model.train()
        for bx,by in loader:
            bx=bx.cuda(non_blocking=True).float()/255;by=by.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
            logits=model(bx);loss=lossfn(logits,by);assert torch.isfinite(loss)
            loss.backward();optimizer.step();total+=loss.item()*len(by);n+=len(by);steps+=1
            if pilot and steps==64:break
        torch.cuda.synchronize();hist.append(dict(epoch=epoch+1,training_ce=total/n,elapsed_seconds=time.time()-start));print(hist[-1],flush=True)
    if pilot:
        t=time.time()-start;js(ROOT/'study3/gate_pilot.json',dict(steps=steps,seconds=t,seconds_per_step=t/steps,training_images=len(x),steps_per_epoch=len(loader),
            forecast_fit_seconds_with_margin=1.25*(t/steps*len(loader)*20+180),pilot_weights_discarded=True,parameters=sum(p.numel() for p in model.parameters())));return
    out=ROOT/'study3/gate';out.mkdir(exist_ok=False)
    torch.save(dict(model=model.cpu().state_dict(),epochs=20,seed=2026101805,preprocess='RGBuint8/255,NCHW,160x256',class_names=['none','mild','moderate','severe'],
        class_counts=counts,architecture='CameraGate',final_checkpoint=True),out/'final.pt')
    csvwrite(out/'training_history.csv',hist)
    js(out/'receipt.json',dict(status='FINAL_FROZEN',epochs=20,parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-start,
        checkpoint_sha256=sha(out/'final.pt'),code_sha256=sha(__file__),model_code_sha256=sha(ROOT/'code/gate.py'),training_only=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','pilot','train']);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    prepare(a.workers) if a.action=='prepare' else train(a.action=='pilot')
