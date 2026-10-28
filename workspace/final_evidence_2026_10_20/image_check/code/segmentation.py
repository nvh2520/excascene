"""Bounded clean-domain SegFormer-B0 fits and paired image degradations.

Run only after the central GPU scheduler authorizes the job. Loads only the separately acquired, verified ImageNet encoder. No network retrieval during fitting. Evaluation needs the frozen shared novel-fault module used by Study 3.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, random, sys, time
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external_validation_run_EV1_AFAST/source/excascene_snapshot'
sys.path.insert(0,str(SOURCE))
from kinafford.sim.degrade import degrade_camera

CONDITIONS=[('clean',0)]+[(k,s) for k in ('occlusion','dust','blur','noise') for s in (1,2,3)]+[('mud_water',3),('sun_glare',3)]
MEAN=np.array([.485,.456,.406],dtype=np.float32);STD=np.array([.229,.224,.225],dtype=np.float32)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)

def model_for(domain):
    return SegformerForSemanticSegmentation(SegformerConfig(num_labels=7 if domain=='sim' else 64,
        depths=[2,2,2,2],sr_ratios=[8,4,2,1],hidden_sizes=[32,64,160,256],
        patch_sizes=[7,3,3,3],strides=[4,2,2,2],num_attention_heads=[1,2,5,8],
        mlp_ratios=[4,4,4,4],decoder_hidden_size=256,semantic_loss_ignore_index=255)).float()

def load_pretrained_encoder(m):
    path=OUT.parent/'pretrained/mit_b0_imagenet1k.bin'
    expected=json.loads((OUT.parent/'pretrained/acquisition_receipt.json').read_text())['assets']['mit_b0']['sha256']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    source_cfg=json.loads((OUT.parent/'pretrained/mit_b0_config.json').read_text())
    for key in ['depths','sr_ratios','hidden_sizes','patch_sizes','strides','num_attention_heads','mlp_ratios','hidden_act','layer_norm_eps','drop_path_rate','num_channels']:
        assert getattr(m.config,key)==source_cfg[key],('pretrained encoder config mismatch',key)
    original=torch.load(path,map_location='cpu',weights_only=False)
    selected={k.removeprefix('segformer.'):v for k,v in original.items() if k.startswith('segformer.')}
    assert selected and set(selected)==set(m.segformer.state_dict())
    m.segformer.load_state_dict(selected,strict=True)
    for k,v in selected.items():assert torch.equal(m.segformer.state_dict()[k],v),k
    return dict(status='VERIFIED_IMAGENET_MIT_B0_ENCODER_LOADED',source_sha256=expected,encoder_tensors=len(selected),parameter_elements=sum(x.numel() for x in m.segformer.parameters()),classifier_and_segmentation_decoder='new random initialization; ImageNet1000class classifier discarded',proof={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in selected.items()})

def load_novel(path):
    if path is None:return None
    spec=importlib.util.spec_from_file_location('study3_novel_faults',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.novel_camera

def apply_condition(rgb,image_id,condition,novel=None):
    kind,level=condition
    if not level:return rgb
    key=f"C_fault_20261018:{image_id}:{kind}:{level}"
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
    depth=np.ones(rgb.shape[:2],np.float32) # API placeholder, never measured real depth
    if kind in ('mud_water','sun_glare'):
        if novel is None:raise RuntimeError('Shared novel fault function is required; no substitute permitted')
        out,_=novel(rgb,depth,kind,rng)
    else:out,_=degrade_camera(rgb,depth,kind,level,rng)
    return out

class Images(Dataset):
    def __init__(self,domain,split,condition=('clean',0),novel=None,augment=False,resolution='parity'):
        self.domain=domain;self.rows=[e for e in json.loads((OUT/f'manifests/{domain}_cache.json').read_text()) if e['split']==split]
        self.condition=condition;self.novel=novel;self.augment=augment;self.resolution=resolution
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        e=self.rows[i]
        if self.resolution=='width512':
            assert self.domain=='real'
            with np.load(OUT/'cache/real_width512'/e['split']/(e['id']+'.npz'),allow_pickle=False) as d:rgb=d['rgb'].copy();y=d['label'].astype(np.int64)
        else:
            with np.load(ROOT/e['cache'],allow_pickle=False) as d:rgb=d['rgb'].copy();y=d['label'].astype(np.int64)
        if self.domain=='real': y[y==0]=255
        if self.augment and random.random()<.5:rgb=rgb[:,::-1].copy();y=y[:,::-1].copy()
        rgb=apply_condition(rgb,e['id'],self.condition,self.novel)
        x=((rgb.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1).copy()
        return torch.from_numpy(x),torch.from_numpy(y.copy()),i

def loader(ds,batch,shuffle=False,workers=2):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,
                      drop_last=False,persistent_workers=workers>0)

def train_step(m,opt,x,y):
    x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
    logits=m(pixel_values=x).logits
    logits=F.interpolate(logits,size=y.shape[-2:],mode='bilinear',align_corners=False)
    loss=F.cross_entropy(logits,y,ignore_index=255)
    if not torch.isfinite(loss):raise RuntimeError('Non-finite segmentation loss')
    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    return float(loss.detach())

def pilot(a):
    # Training images only; fresh random model, discarded after timing.
    m=model_for('real');load_pretrained_encoder(m);m=m.cuda().train();opt=torch.optim.AdamW(m.parameters(),lr=6e-4,weight_decay=.01)
    dl=iter(loader(Images('real','train'),a.batch,True,a.workers));times=[]
    for i in range(72):
        torch.cuda.synchronize();t=time.monotonic();x,y,_=next(dl);train_step(m,opt,x,y);torch.cuda.synchronize()
        if i>=8:times.append(time.monotonic()-t)
    m.eval();itimes=[]
    with torch.no_grad():
        for i in range(36):
            torch.cuda.synchronize();t=time.monotonic();x,y,_=next(dl)
            z=m(pixel_values=x.cuda(non_blocking=True)).logits
            F.interpolate(z,size=y.shape[-2:],mode='bilinear',align_corners=False).argmax(1).cpu()
            torch.cuda.synchronize()
            if i>=4:itimes.append(time.monotonic()-t)
    return dict(kind='discarded_training_only_pilot',batch=a.batch,updates=72,eval_batches=36,
                measured_train_seconds=times,measured_eval_seconds=itimes,
                train_seconds_per_batch_mean=float(np.mean(times)),eval_seconds_per_batch_mean=float(np.mean(itimes)),
                train_seconds_per_batch_p95=float(np.quantile(times,.95)),eval_seconds_per_batch_p95=float(np.quantile(itimes,.95)),
                parameters=sum(p.numel() for p in m.parameters()))

def fit(a):
    checkpoint=OUT/'checkpoints'/f'{a.domain}_segformer_b0_final.pt'
    if checkpoint.exists(): raise FileExistsError(checkpoint)
    m=model_for(a.domain);pretrained=load_pretrained_encoder(m);m=m.cuda().train();dl=loader(Images(a.domain,'train',augment=True),a.batch,True,a.workers)
    opt=torch.optim.AdamW(m.parameters(),lr=6e-4,weight_decay=.01);n=a.epochs*len(dl);warm=max(1,round(.05*n));step=0;logs=[]
    for epoch in range(a.epochs):
        losses=[];t=time.monotonic()
        for x,y,_ in dl:
            lr=6e-4*((step+1)/warm if step<warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,n-warm))))
            for g in opt.param_groups:g['lr']=lr
            losses.append(train_step(m,opt,x,y));step+=1
        r=dict(epoch=epoch+1,steps=step,mean_train_loss=float(np.mean(losses)),seconds=time.monotonic()-t)
        logs.append(r);print(json.dumps(r),flush=True)
    torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},config=m.config.to_dict(),
                    domain=a.domain,seed=a.seed,epochs=a.epochs,updates=step,precision='FP32_TF32_off',
                    pretrained_loading=pretrained,training_manifest_sha256=hashlib.sha256((OUT/f'manifests/{a.domain}_cache.json').read_bytes()).hexdigest()),checkpoint)
    (OUT/'protocol'/f'{a.domain}_pretrained_loading.json').write_text(json.dumps(pretrained,indent=2)+'\n')
    (OUT/'logs'/f'{a.domain}_training.json').write_text(json.dumps(logs,indent=2)+'\n')
    return dict(domain=a.domain,checkpoint=str(checkpoint.relative_to(ROOT)),epochs=a.epochs,updates=step,
                parameters=sum(p.numel() for p in m.parameters()),checkpoint_selection='fixed final only; no validation outcome inspected')

@torch.no_grad()
def evaluate(a):
    dst=OUT/'metrics'/f'{a.domain}_{a.resolution}_segmentation_confusions.npz'
    if dst.exists():raise FileExistsError(dst)
    c=torch.load(OUT/'checkpoints'/f'{a.domain}_segformer_b0_final.pt',map_location='cpu',weights_only=False)
    m=model_for(a.domain);m.load_state_dict(c['state_dict']);m=m.cuda().eval()
    novel=load_novel(a.novel_module);first=Images(a.domain,'val',resolution=a.resolution);seqs=sorted({e['sequence'] for e in first.rows});si={s:i for i,s in enumerate(seqs)}
    nc=7 if a.domain=='sim' else 64;cms=np.zeros((len(CONDITIONS),len(seqs),nc,nc),np.int64)
    imagecms=np.zeros((len(CONDITIONS),len(first),nc,nc),np.int64)
    for ci,cond in enumerate(CONDITIONS):
        ds=Images(a.domain,'val',cond,novel,resolution=a.resolution);t=time.monotonic()
        for x,y,indices in loader(ds,a.batch,False,a.workers):
            z=m(pixel_values=x.cuda(non_blocking=True)).logits
            pred=F.interpolate(z,size=y.shape[-2:],mode='bilinear',align_corners=False).argmax(1).cpu().numpy()
            for j,ix in enumerate(indices.tolist()):
                yy=y[j].numpy();ok=yy!=255
                cm=np.bincount(yy[ok]*nc+pred[j][ok],minlength=nc*nc).reshape(nc,nc)
                imagecms[ci,ix]=cm;cms[ci,si[ds.rows[ix]['sequence']]]+=cm
        print(json.dumps(dict(domain=a.domain,condition=cond,seconds=time.monotonic()-t)),flush=True)
    np.savez_compressed(dst,sequence_confusions=cms,image_confusions=imagecms,sequences=np.array(seqs),
                        image_ids=np.array([e['id'] for e in first.rows]),
                        conditions=np.array([f'{k}_s{s}' if s else 'clean' for k,s in CONDITIONS]))
    return dict(domain=a.domain,conditions=len(CONDITIONS),images_per_condition=len(first),sequences=len(seqs),forwards=len(first)*len(CONDITIONS),output=str(dst.relative_to(ROOT)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['pilot','fit','evaluate']);ap.add_argument('--domain',choices=['real','sim'],default='real')
    ap.add_argument('--resolution',choices=['parity','width512'],default='parity');ap.add_argument('--epochs',type=int,default=24);ap.add_argument('--batch',type=int,default=16);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--seed',type=int,default=2026102003);ap.add_argument('--novel-module');ap.add_argument('--resource-id',required=True)
    a=ap.parse_args()
    lock=json.loads((OUT/'protocol/image_check_lock.json').read_text())
    assert lock['status']=='LOCKED_BEFORE_IMAGE_CHECK_FITTING' and a.epochs==24
    for rel,digest in lock['files'].items():assert hashlib.sha256((OUT.parent/rel).read_bytes()).hexdigest()==digest,('image-check locked source changed',rel)
    seed_all(a.seed);assert torch.cuda.is_available();t=time.monotonic();start=time.time()
    r={'status':'FAILED'}
    try:
        r=globals()[a.action](a);r['status']='COMPLETE'
    finally:
        torch.cuda.synchronize();r.update(start_utc_unix=start,end_utc_unix=time.time(),elapsed_seconds=time.monotonic()-t,
            occupied_gpu_hours=(time.monotonic()-t)/3600,device=torch.cuda.get_device_name(),precision='actual FP32; autocast disabled; TF32 disabled',
            arguments=vars(a),peak_allocated_bytes=torch.cuda.max_memory_allocated())
        (OUT/'resources'/f'{a.resource_id}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':main()
