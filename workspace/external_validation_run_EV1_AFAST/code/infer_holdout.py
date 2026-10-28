"""Final-checkpoint-only inference after explicit release; no model selection."""
import argparse,csv,hashlib,json,pathlib,time
import numpy as np
import torch
from models import build,neural_inputs
from inputs import tensorize
from scoring import neutral,evaluate,geometry
from common_afast import npzwrite,sha
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--architecture',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
release=json.loads((ROOT/'records/holdout_prediction_release.json').read_text());assert release['status']=='RELEASED_AFTER_SIX_FINALS'
for rel,h in release['analysis_files'].items():assert sha(ROOT/rel)==h,('analysis_changed',rel)
assert json.loads((ROOT/'records/sealed_holdout_audit.json').read_text())['status']=='SEALED_PHYSICAL_CONTENT_AUDIT_PASSED'
directory=ROOT/'checkpoints'/a.architecture/f'seed{a.seed}'
cp=directory/'final.pt';assert sha(cp)==release['checkpoint_files'][str(cp.relative_to(ROOT))]
state=torch.load(cp,map_location='cpu',weights_only=False);assert state['final_checkpoint'] and state['epochs']==14 and state['seed']==a.seed
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False
m=build(a.architecture).cuda().float();m.load_state_dict(state['model'],strict=True);m.eval()
files=list(csv.DictReader((ROOT/'manifests/generated_files.csv').open()));files=sorted(files,key=lambda x:(x['condition'],int(x['scene_id']),int(x['frame_id'])))
name=a.architecture+f'_seed{a.seed}';outdir=ROOT/'predictions'/name;outdir.mkdir(parents=True,exist_ok=False)
metdir=ROOT/'metrics';metdir.mkdir(exist_ok=True);metricfile=metdir/f'{name}.csv';manifestfile=ROOT/'manifests'/f'predictions_{name}.csv'
matrices=[];diagkeys=[];count=0;completed=0;begin=time.monotonic()
mf=metricfile.open('w',newline='');pf=manifestfile.open('w',newline='');mw=None;pw=None
def progress(status):
    d=dict(status=status,architecture=a.architecture,seed=a.seed,attempted_logical_forwards=count,completed_logical_forwards=completed,elapsed_s=time.monotonic()-begin,utc_unix=time.time())
    (ROOT/'records'/f'inference_{name}.json').write_text(json.dumps(d,indent=2));return d
try:
    for mask in [0,1]:
        for offset in range(0,len(files),4):
            group=files[offset:offset+4];records=[];ts=[]
            for row in group:
                path=ROOT/row['path'];assert sha(path)==row['sha256']
                with np.load(path) as z:record={k:z[k] for k in z.files}
                records.append(record);intact=tensorize(record,False);masked=tensorize(record,True)
                for k in ['lidar_range','lidar_valid','q','T_dyn','T_static','K']:
                    assert np.array_equal(intact[k],masked[k]),('mask_retained_input_mismatch',k)
                assert np.array_equal(intact['meta'][3:],masked['meta'][3:])
                ts.append(masked if mask else intact)
            b={k:torch.from_numpy(np.stack([t[k] for t in ts])).cuda() for k in ts[0]}
            count+=len(group);progress('RUNNING')
            with torch.no_grad():out=m(neural_inputs(b))
            for k,v in out.items():
                if isinstance(v,torch.Tensor):assert v.dtype==torch.float32 and torch.isfinite(v).all(),('INVALID_NONFINITE_OR_PRECISION',k)
            for j,(row,d) in enumerate(zip(group,records)):
                pred=neutral(out,j);sid=int(d['scene_id']);frame=int(d['frame_id']);condition=str(d['suite'])
                path=outdir/f'mask{mask}'/condition/f'{sid}_{frame}.npz';npzwrite(path,**pred)
                custody=dict(architecture=a.architecture,seed=a.seed,mask=mask,condition=condition,scene_id=sid,frame_id=frame,path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)
                if pw is None:pw=csv.DictWriter(pf,fieldnames=list(custody));pw.writeheader()
                pw.writerow(custody)
                labels=pred['sem_prob'].argmax(0);conf=np.bincount((d['gt_sem'].astype(np.int64)*6+labels).ravel(),minlength=36).reshape(6,6)
                terrain=d['gt_sem']==0;ml=pred['mat_prob'].argmax(0)
                material=np.bincount((d['gt_mat'][terrain].astype(np.int64)*4+ml[terrain]).ravel(),minlength=16).reshape(4,4)
                hp=pred['haz_prob']>.5;ht=d['gt_haz']>.5
                diagnostics=dict(hazard_tp=int((hp&ht).sum()),hazard_fp=int((hp&~ht).sum()),hazard_fn=int((~hp&ht).sum()),
                    hazard_tn=int((~hp&~ht).sum()),material_terrain_cells=int(terrain.sum()),
                    utility_absolute_sum=float(np.abs(pred['aff'].astype(np.float64)-d['gt_aff'].astype(np.float64)).sum()),utility_cells=pred['aff'].size)
                matrices.append((conf,material));diagkeys.append([a.seed,mask,sid,frame,condition])
                for policy in ['direct','soft']:
                    metric=dict(architecture=a.architecture,seed=a.seed,mask=mask,condition=condition,scene_id=sid,frame_id=frame,
                                attempted=1,**evaluate(pred,d,policy),**geometry(pred,d),**diagnostics)
                    if mw is None:mw=csv.DictWriter(mf,fieldnames=list(metric));mw.writeheader()
                    mw.writerow(metric)
            completed+=len(group)
            if (offset//4)%25==0:mf.flush();pf.flush();print(json.dumps(progress('RUNNING')),flush=True)
    assert completed==2*len(files)
    npzwrite(metdir/f'{name}_confusions.npz',semantic=np.stack([x[0] for x in matrices]),material=np.stack([x[1] for x in matrices]),
             seed=np.array([x[0] for x in diagkeys]),mask=np.array([x[1] for x in diagkeys]),scene_id=np.array([x[2] for x in diagkeys]),
             frame_id=np.array([x[3] for x in diagkeys]),condition=np.array([x[4] for x in diagkeys]))
    torch.cuda.synchronize();receipt=progress('COMPLETE');receipt['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    (ROOT/'records'/f'inference_{name}.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
except Exception as e:
    receipt=progress('INVALID_OR_INCOMPLETE');receipt['error']=repr(e)
    (ROOT/'records'/f'inference_{name}.json').write_text(json.dumps(receipt,indent=2));raise
finally:mf.close();pf.close()
