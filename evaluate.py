"""One-command checkpoint/custom-model evaluation. Prepared, not executed in this pass."""
from pathlib import Path
import argparse,csv,importlib,json,sys,time

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--architecture',choices=['KF-D42','BF-D42'],default='KF-D42')
    ap.add_argument('--seed',type=int,choices=[10,11,12],default=10)
    ap.add_argument('--split',default='replication',choices=['replication','fixture'])
    ap.add_argument('--condition',default='all',choices=['all','clean','iid','cam_s1','cam_s2','cam_s3','single_severe','novel','all_s2','rough_iid'])
    ap.add_argument('--gate',action='store_true',help='Also evaluate the frozen combined monitor at the architecture-specific replication thresholds.')
    ap.add_argument('--adapter',help='New-model factory module:function; signature factory(checkpoint_path, device) -> callable(neural_inputs) -> raw task-head dictionary.')
    ap.add_argument('--checkpoint',type=Path,help='Required with --adapter; your checkpoint, never the holdout labels.')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.adapter and not a.checkpoint:ap.error('--adapter requires --checkpoint')
    if a.adapter and a.gate:ap.error('No gate threshold for an arbitrary new model is defined. Evaluate keep-all or supply your own separately specified protocol.')
    candidate=Path(__file__).resolve().parent;root=candidate/'workspace/final_evidence_2026_10_20'
    if a.output.exists():raise SystemExit('Output directory already exists; existing records will not be overwritten.')
    a.output.mkdir(parents=True);sys.path.insert(0,str(root/'code'));started=time.time()
    import numpy as np,torch
    from common import load_npz,sha
    from neural import backend,frozen,batch,neutral,decisions,ensure_fp32
    from gate import load_gate,severe_probability,camera_drop
    backend()
    if not torch.cuda.is_available():raise RuntimeError('This checkpoint runner requires a compatible installed CUDA runtime. Statistical reconstruction is separate.')
    if a.adapter:
        mod,fn=a.adapter.rsplit(':',1);factory=getattr(importlib.import_module(mod),fn)
        model=factory(a.checkpoint.resolve(),'cuda');checkpoint=dict(path=str(a.checkpoint),sha256=sha(a.checkpoint),kind='user_supplied_model')
    else:model,checkpoint=frozen(a.architecture,a.seed)
    model=model.eval().float()
    for p in model.parameters():p.requires_grad_(False)
    if a.split=='replication':
        parent=root/'study4/holdout';records=list(csv.DictReader((parent/'files.csv').open()))
        if a.condition!='all':records=[r for r in records if r['condition']==a.condition]
    else:
        parent=root;records=[dict(scene_id=r['scene_id'],frame_id=r['frame_id'],condition='clean',path=r['data']) for r in json.loads((candidate/'fixture_manifest.json').read_text())]
    gate=None
    if a.gate:
        gate=load_gate(root/'study4/gate/final.pt');th=json.loads((root/'study4/thresholds.json').read_text())[a.architecture]
    rows=[];actual=0
    for row in records:
        d=load_npz(parent/row['path']);arms={'keep_all':d['cam_present']<=.5}
        if a.gate:
            prob=severe_probability(gate,d['rgb']);tau=th.get('tau',th.get('rgb_threshold'));delta=th.get('delta',th.get('depth_threshold'))
            if tau is None or delta is None:raise ValueError('Threshold schema mismatch; inspect the supplied frozen threshold file.')
            arms['gate']=camera_drop(prob,d['depth'],d['cam_present'],tau,delta)[0]
        cache={}
        for arm,drop in arms.items():
            key=tuple(bool(x) for x in drop)
            if key not in cache:
                with torch.inference_mode():out=model(batch([d],[drop]))
                ensure_fp32(out);cache[key]=neutral(out,0);actual+=1
            for policy,met in decisions(cache[key],d).items():rows.append(dict(architecture=a.architecture if not a.adapter else a.adapter,seed=a.seed,condition=row['condition'],scene_id=int(row['scene_id']),frame_id=int(row['frame_id']),arm=arm,policy=policy,**met))
    with (a.output/'per_frame.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # No new bootstrap samples are drawn by this entry point.
    (a.output/'receipt.json').write_text(json.dumps(dict(status='COMPLETE',checkpoint=checkpoint,split=a.split,condition=a.condition,frames=len(records),decision_rows=len(rows),perception_forwards=actual,gate_camera_forwards=len(records)*3 if gate else 0,precision='FP32, TF32 off',seconds=time.time()-started,new_bootstrap_draws=0),indent=2)+'\n')
if __name__=='__main__':main()
