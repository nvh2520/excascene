"""Frozen prospective arms. Identical frame/subset inputs share one FP32 result."""
from common import *
import argparse,time

def verify_lock():
    verify_protocol()
    assert json.loads((ROOT/'study4/holdout/validity_audit.json').read_text())['status']=='PASS'

def gate_scores():
    import torch
    from gate import load_gate,severe_probability
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_gate(ROOT/'study4/gate/final.pt');source=ROOT/'study4/holdout';rows=[];start=time.time()
    files=list(csv.DictReader((source/'files.csv').open()))
    for offset in range(0,len(files),16):
        group=files[offset:offset+16];data=[load_npz(source/r['path'],['rgb','depth','cam_present']) for r in group]
        rgb=np.concatenate([d['rgb'] for d in data]);probs=severe_probability(model,rgb).reshape(len(group),3)
        for r,d,pp in zip(group,data,probs):
            invalid=np.mean((~np.isfinite(d['depth']))|(d['depth']<=.05),axis=(-2,-1))
            row=dict(condition=r['condition'],scene_id=int(r['scene_id']),frame_id=int(r['frame_id']))
            row.update({f'p{j}':float(pp[j]) for j in range(3)});row.update({f'invalid{j}':float(invalid[j]) for j in range(3)})
            row.update({f'present{j}':int(d['cam_present'][j]) for j in range(3)});rows.append(row)
        if (offset//16)%100==0:print('gate scores',offset,len(files),flush=True)
    csvwrite(source/'gate_scores.csv',rows)
    js(source/'gate_scores_receipt.json',dict(status='COMPLETE',camera_forwards=3*len(files),seconds=time.time()-start,gate_sha256=sha(ROOT/'study4/gate/final.pt'),output_sha256=sha(source/'gate_scores.csv')))

def masks(d,p,architecture,thresholds):
    from kinafford.sim.kinematics import CAMERAS
    present=d['cam_present']>.5;t=thresholds[architecture]
    rgb=np.array([p[f'p{j}']>=t['tau'] for j in range(3)])|(~present)
    depth=np.array([p[f'invalid{j}']>=t['delta'] for j in range(3)])|(~present)
    gate=rgb|depth
    if architecture.endswith('14'):return {'keep_all':~present,'gate':gate}
    spec=json.loads(str(d['spec']));oracle=~present
    for j,c in enumerate(CAMERAS):
        cs=spec['cam'][c]
        oracle[j]|=bool(cs.get('novel')) or max([int(level) for kind,level in cs['photo']],default=0)>=3
    return {'keep_all':~present,'gate':gate,'always_mask':np.ones(3,bool),'oracle_mask':oracle,'rgb_only':rgb,'depth_only':depth}

def model_run(architecture,seed):
    import pandas as pd,torch
    from neural import backend,frozen,batch,neutral,decisions,ensure_fp32
    backend();model,cp=frozen(architecture,seed);source=ROOT/'study4/holdout'
    files=sorted(list(csv.DictReader((source/'files.csv').open())),key=lambda r:(r['condition'],int(r['scene_id']),int(r['frame_id'])))
    scores=pd.read_csv(source/'gate_scores.csv').set_index(['condition','scene_id','frame_id']).to_dict('index')
    thresholds=json.loads((ROOT/'study4/thresholds.json').read_text())
    name=f'{architecture}_seed{seed}';out=ROOT/f'study4/metrics/{name}.csv';out.parent.mkdir(exist_ok=True);assert not out.exists()
    predroot=ROOT/f'study4/predictions/{name}';predroot.mkdir(parents=True,exist_ok=False)
    start=time.time();actual=0;logical=0;writer=None;confusions={};manifest=[];confusion_records=[];semantic_frames=[];material_frames=[]
    with out.open('w',newline='') as stream:
        for offset in range(0,len(files),4):
            group=files[offset:offset+4];records=[load_npz(source/r['path']) for r in group];arms=[];todo=[]
            for i,(r,d) in enumerate(zip(group,records)):
                p=scores[(r['condition'],int(r['scene_id']),int(r['frame_id']))];arm=masks(d,p,architecture,thresholds);arms.append(arm)
                codes={int(sum(int(v)*(1<<j) for j,v in enumerate(drop))):drop for drop in arm.values()}
                todo.extend((i,code,drop) for code,drop in sorted(codes.items()))
            cache={}
            for j in range(0,len(todo),4):
                sub=todo[j:j+4];b=batch([records[i] for i,_,_ in sub],[drop for _,_,drop in sub])
                with torch.inference_mode():outputs=model(b)
                ensure_fp32(outputs);actual+=len(sub)
                for t,(i,code,drop) in enumerate(sub):
                    pred=neutral(outputs,t);r=group[i];path=predroot/f"subset{code}/{r['condition']}/{r['scene_id']}_{r['frame_id']}.npz"
                    path.parent.mkdir(exist_ok=True,parents=True)
                    # Level1 compression mirrors the original inference IO cost.
                    from common_afast import npzwrite
                    npzwrite(path,**pred);cache[(i,code)]=(pred,path)
                    manifest.append(dict(condition=r['condition'],scene_id=int(r['scene_id']),frame_id=int(r['frame_id']),subset=code,path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size))
            for i,(r,d,arm) in enumerate(zip(group,records,arms)):
                for arm_name,drop in arm.items():
                    code=int(sum(int(v)*(1<<j) for j,v in enumerate(drop)));pred,path=cache[(i,code)];logical+=1
                    reachable=(d['gt_reach']>=.05)&(d['gt_sem']==0);height=float(np.abs(pred['H'].astype(float)-d['gt_H'].astype(float))[reachable].mean()) if reachable.any() else None
                    semantic=np.bincount((d['gt_sem'].astype(int)*6+pred['sem_label']).ravel(),minlength=36).reshape(6,6)
                    terrain=d['gt_sem']==0;material=np.bincount(d['gt_mat'][terrain].astype(int)*4+pred['mat_label'][terrain],minlength=16).reshape(4,4)
                    confkey=(r['condition'],arm_name)
                    if confkey not in confusions:confusions[confkey]=[np.zeros((6,6),np.int64),np.zeros((4,4),np.int64)]
                    confusions[confkey][0]+=semantic;confusions[confkey][1]+=material
                    confusion_records.append(dict(condition=r['condition'],scene_id=int(r['scene_id']),frame_id=int(r['frame_id']),arm=arm_name))
                    semantic_frames.append(semantic);material_frames.append(material)
                    for policy,met in decisions(pred,d).items():
                        row=dict(architecture=architecture,seed=seed,arm=arm_name,condition=r['condition'],scene_id=int(r['scene_id']),frame_id=int(r['frame_id']),
                            policy=policy,subset=code,dropped_present_cameras=int((drop&(d['cam_present']>.5)).sum()),present_cameras=int((d['cam_present']>.5).sum()),
                            height_reachable_terrain=height,reachable_terrain_cells=int(reachable.sum()),prediction_path=str(path.relative_to(ROOT)),**met)
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row)
            if offset%128==0:
                stream.flush();print(name,offset,len(files),'actual',actual,'logical',logical,'seconds',time.time()-start,flush=True)
        torch.cuda.synchronize()
    csvwrite(ROOT/f'study4/predictions/{name}_manifest.csv',manifest)
    np.savez_compressed(ROOT/f'study4/metrics/{name}_frame_confusions.npz',semantic=np.stack(semantic_frames),material=np.stack(material_frames))
    csvwrite(ROOT/f'study4/metrics/{name}_confusion_keys.csv',confusion_records)
    js(ROOT/f'study4/metrics/{name}_confusions.json',{str(k):{'semantic':v[0].tolist(),'material':v[1].tolist()} for k,v in confusions.items()})
    js(out.with_suffix('.json'),dict(status='COMPLETE',architecture=architecture,seed=seed,records=len(files),actual_neural_forwards=actual,
        logical_arm_forwards=logical,identical_input_reuses=logical-actual,seconds=time.time()-start,checkpoint=cp,metric_sha256=sha(out),
        precision='actual FP32; autocast/TF32 disabled',peak_allocated_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gate_scores','model']);p.add_argument('--architecture');p.add_argument('--seed',type=int);a=p.parse_args()
    verify_lock();gate_scores() if a.action=='gate_scores' else model_run(a.architecture,a.seed)
