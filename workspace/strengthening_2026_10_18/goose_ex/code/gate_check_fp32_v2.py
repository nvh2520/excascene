"""Frozen RGB gate transfer; calibration uses unlabeled ALICE training RGB only."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from segmentation import ROOT,OUT,CONDITIONS,apply_condition,load_novel,seed_all

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class RGBOnly(Dataset):
    def __init__(self,split,condition,novel):
        self.rows=[e for e in json.loads((OUT/'manifests/real_cache.json').read_text()) if e['split']==split]
        self.condition=condition;self.novel=novel
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        e=self.rows[i]
        # Deliberately never access the cache's label field for calibration or detection.
        with np.load(ROOT/e['cache'],allow_pickle=False) as z:rgb=z['rgb'].copy()
        return apply_condition(rgb,e['id'],self.condition,self.novel),i

def infer(a):
    dst=OUT/'metrics/real_gate_probabilities.csv'
    if dst.exists():raise FileExistsError(dst)
    seed_all(20261018);gate=load_module(a.gate_module,'frozen_gate_api');novel=load_novel(a.novel_module)
    model=gate.load_gate(a.checkpoint,'cuda');rows=[]
    for split,conditions in [('train',[('clean',0)]),('val',CONDITIONS)]:
        for condition in conditions:
            ds=RGBOnly(split,condition,novel)
            dl=DataLoader(ds,batch_size=a.batch,shuffle=False,num_workers=a.workers,persistent_workers=a.workers>0)
            for rgb,ids in dl:
                probs=gate.severe_probability(model,rgb.numpy())
                assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
                for ix,p in zip(ids.tolist(),probs):
                    e=ds.rows[ix];kind,level=condition
                    rows.append(dict(split=split,image_id=e['id'],sequence=e['sequence'],mechanism=kind,severity=level,
                                     probability_severe=float(p),depth_branch='NOT_TESTED_NO_MEASURED_REAL_DEPTH'))
            print(json.dumps(dict(split=split,condition=condition,images=len(ds))),flush=True)
    with open(dst,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return dict(images=len(rows),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                gate_api_sha256=hashlib.sha256(Path(a.gate_module).read_bytes()).hexdigest(),
                novel_faults_sha256=hashlib.sha256(Path(a.novel_module).read_bytes()).hexdigest(),
                scope='RGB gate branch only; no real depth stream; all threshold variants reuse probabilities')

def summarize(a):
    dst=OUT/'metrics/real_gate_rates.csv'
    if dst.exists():raise FileExistsError(dst)
    with open(OUT/'metrics/real_gate_probabilities.csv') as f:rows=list(csv.DictReader(f))
    train=np.array([float(r['probability_severe']) for r in rows if r['split']=='train'])
    assert len(train)==2164
    k=math.floor(.05*len(train));boundary=np.sort(train)[len(train)-k-1]
    threshold=float(np.nextafter(np.float32(boundary),np.float32(np.inf)))
    calibrated_removed=int((train>=threshold).sum());assert calibrated_removed<=k
    settings=json.loads(Path(a.thresholds).read_text())
    with np.load(OUT/'metrics/sequence_bootstrap_draws.npz',allow_pickle=False) as z:
        counts=z['real_counts'];sequences=z['real_sequences'].tolist()
    ci_map={s:i for i,s in enumerate(sequences)};out=[]
    for model,t in settings.items():
        for setting,tau in [('sim_tuned',float(t['tau'])),('real_train_calibrated',threshold)]:
            for split,conditions in [('train',[('clean',0)]),('val',CONDITIONS)]:
                for mechanism,severity in conditions:
                    rr=[r for r in rows if r['split']==split and r['mechanism']==mechanism and int(r['severity'])==severity]
                    removed=np.array([float(r['probability_severe'])>=tau for r in rr]);n=len(rr);events=int(removed.sum())
                    lo=hi=None
                    if split=='val':
                        ns=np.zeros(len(sequences),np.int64);ks=np.zeros_like(ns)
                        for r,v in zip(rr,removed):j=ci_map[r['sequence']];ns[j]+=1;ks[j]+=int(v)
                        bs=(counts@ks)/(counts@ns);lo,hi=np.quantile(bs,[.025,.975])
                    out.append(dict(downstream_model=model,threshold_source=setting,threshold=tau,split=split,
                        mechanism=mechanism,severity=severity,images=n,removed=events,removal_rate=events/n,
                        ordinary_ci95_low=lo,ordinary_ci95_high=hi,
                        interpretation='false removal' if mechanism=='clean' else 'removal under injected image corruption',
                        depth_branch='NOT_TESTED',interval_scope='descriptive shared ALICE sequence bootstrap; train calibration has no test interval'))
    with open(dst,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    r=dict(calibration_split='official ALICE train; RGB probabilities only; no labels',n=len(train),allowed_removals=k,
           order_statistic_boundary=float(boundary),threshold=threshold,removed=calibrated_removed,
           empirical_false_removal_rate=calibrated_removed/len(train),
           guarantee='at most5% in these training frames only; no population or validation-set guarantee',
           comparator_thresholds_source=a.thresholds,comparator_thresholds_sha256=hashlib.sha256(Path(a.thresholds).read_bytes()).hexdigest(),
           bootstrap='reuse real sequence multiplicities saved for the segmentation ordering check; no additional draws')
    (OUT/'metrics/real_gate_calibration.json').write_text(json.dumps(r,indent=2)+'\n');return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['infer','summarize']);ap.add_argument('--gate-module',default=str(ROOT/'strengthening_2026_10_18/code/gate.py'))
    ap.add_argument('--novel-module',default=str(ROOT/'strengthening_2026_10_18/code/faults.py'))
    ap.add_argument('--checkpoint',default=str(ROOT/'strengthening_2026_10_18/study3/gate/final.pt'))
    ap.add_argument('--thresholds',default=str(ROOT/'strengthening_2026_10_18/study3/thresholds.json'))
    ap.add_argument('--batch',type=int,default=32);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--resource-id',required=True)
    a=ap.parse_args();t=time.monotonic();start=time.time();r={'status':'FAILED'}
    try:r=globals()[a.action](a);r['status']='COMPLETE'
    finally:
        r.update(action=a.action,start_utc_unix=start,finish_utc_unix=time.time(),elapsed_seconds=time.monotonic()-t,
                 occupied_gpu_hours=(time.monotonic()-t)/3600 if a.action=='infer' else 0,arguments=vars(a))
        (OUT/'resources'/f'{a.resource_id}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':main()
