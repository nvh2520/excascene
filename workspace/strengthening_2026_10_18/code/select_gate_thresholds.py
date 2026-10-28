"""Gate predictions and fixed 25-point validation threshold selection."""
from common import *
import argparse,itertools,time
import pandas as pd

def predict_validation():
    import torch
    from gate import load_gate,severe_probability
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_gate(ROOT/'study3/gate/final.pt');source=ROOT/'study3/validation';rows=[]
    for r in csv.DictReader((source/'files.csv').open()):
        d=load_npz(source/r['path'],['rgb','depth','cam_present']);prob=severe_probability(model,d['rgb'])
        invalid=np.mean((~np.isfinite(d['depth']))|(d['depth']<=.05),axis=(-2,-1))
        for camera in range(3):rows.append(dict(condition=r['condition'],scene_id=int(r['scene_id']),frame_id=int(r['frame_id']),camera=camera,
            severe_probability=float(prob[camera]),invalid_fraction=float(invalid[camera]),present=int(d['cam_present'][camera])))
    csvwrite(ROOT/'study3/validation_gate_scores.csv',rows)

def select():
    probabilities=pd.read_csv(ROOT/'study3/validation_gate_scores.csv');index=['condition','scene_id','frame_id']
    p=probabilities.pivot(index=index,columns='camera',values='severe_probability');d=probabilities.pivot(index=index,columns='camera',values='invalid_fraction')
    present=probabilities.pivot(index=index,columns='camera',values='present')>.5
    rows=[];selected={}
    for architecture,seeds in [('KF-D',[10,11,12]),('BF-D',[10,11,12]),('S5A',[0,1,2])]:
        models=pd.concat([pd.read_csv(ROOT/f'study3/validation_subsets/{architecture}_seed{seed}.csv') for seed in seeds],ignore_index=True)
        models=models[models.policy=='direct'];model=models.groupby(index+['subset']).regret.mean()
        for tau,delta in itertools.product([.25,.5,.75,.9,1.01],repeat=2):
            drops=(p>=tau)|(d>=delta)|(~present)
            subset=(drops.to_numpy().astype(int)*np.array([1,2,4])[None]).sum(1)
            values=[]
            for key,s in zip(p.index,subset):values.append((*key,float(model.loc[(*key,int(s))])))
            frame=pd.DataFrame(values,columns=index+['regret']);condition_means=frame.groupby('condition').regret.mean()
            assert len(condition_means)==8
            rows.append(dict(architecture=architecture,tau=tau,delta=delta,pooled_validation_regret=float(condition_means.mean()),
                removed_fraction_present=float((drops&present).to_numpy().sum()/present.to_numpy().sum()),conditions=len(condition_means),frames=len(p),seeds=len(seeds)))
        ar=[r for r in rows if r['architecture']==architecture];best=sorted(ar,key=lambda r:(r['pooled_validation_regret'],-r['tau'],-r['delta']))[0]
        selected[architecture]={k:best[k] for k in ['tau','delta','pooled_validation_regret','removed_fraction_present']}
    csvwrite(ROOT/'study3/validation_threshold_grid.csv',rows)
    js(ROOT/'study3/thresholds.json',selected)
    js(ROOT/'study3/threshold_lock.json',dict(status='FROZEN_VALIDATION_ONLY',utc_unix=time.time(),threshold_sha256=sha(ROOT/'study3/thresholds.json'),
        grid_sha256=sha(ROOT/'study3/validation_threshold_grid.csv'),gate_sha256=sha(ROOT/'study3/gate/final.pt'),code_sha256=sha(__file__),
        no_new_test_generated=not (ROOT/'study3/holdout').exists(),objective='equal-weight eight known validation conditions, direct regret, three seeds equally averaged'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['predict','select']);a=p.parse_args()
    predict_validation() if a.action=='predict' else select()
