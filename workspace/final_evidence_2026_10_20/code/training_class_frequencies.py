"""Count original training labels only, before fitting; no test access."""
from pathlib import Path
import csv,hashlib,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
def one(path):
    with np.load(path,allow_pickle=False) as z:
        sid=int(z['scene_id']);assert 0<=sid<3000
        s=z['gt_sem'].astype(int);m=z['gt_mat'].astype(int);assert s.min()>=0 and s.max()<6 and m.min()>=0 and m.max()<4
        return np.bincount(s.ravel(),minlength=6),np.bincount(m[s==0],minlength=4),sid,int(z['frame_id'])
if __name__=='__main__':
    start=time.monotonic();paths=sorted((PROJECT/'data/train').glob('*.npz'));assert len(paths)==12000
    sem=np.zeros(6,np.int64);mat=np.zeros(4,np.int64);ident=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,(s,m,sid,fid) in enumerate(pool.map(one,paths,chunksize=16)):
            sem+=s;mat+=m;ident.append((sid,fid))
            if (i+1)%2000==0:print('training label frames',i+1,flush=True)
    assert len(set(ident))==12000 and (sem>0).all() and (mat>0).all()
    def summary(x):
        f=x/x.sum();w=1/np.sqrt(f);w/=w.mean()
        return dict(counts=x.tolist(),frequencies=f.tolist(),inverse_sqrt_mean_one_weights=w.tolist())
    out=dict(status='TRAINING_ONLY_CLASS_WEIGHTS_FROZEN',semantics=summary(sem),material=summary(mat),frames=12000,scene_ids=[0,2999],material_support='truth terrain only; original gt_sem==0',semantic_support='all BEV cells',weight_normalization='arithmetic mean over classes equals1; no clipping or smoothing; zero support is an integrity stop',unchanged='original visibility weights and terrain mask, other heads/loss coefficients and auxiliary image CE',seconds=time.monotonic()-start,identity_sha256=hashlib.sha256(np.asarray(ident,dtype='<i8').tobytes()).hexdigest(),new_test_access=False)
    (ROOT/'protocol/training_class_weights.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
