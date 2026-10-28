"""Targeted development controls, original-validation data only; no new test data."""
from common import *
import pandas as pd
from fractions import Fraction
from study3_statistics import scene_bootstrap,paired,controls
from study3_diagnostics import attribute
from faults import novel_latents,apply_novel

def main():
    assert not (ROOT/'study3/holdout').exists()
    for p in sorted((ROOT/'code').glob('*.py')):compile(p.read_text(),str(p),'exec')
    controls()
    ids=np.array([1,2]);rough=np.array([3]);M=np.tile(np.array([[2,0],[0,2],[1,1],[1,1]],np.int16),(5000,1))
    draws={'nominal_scene_ids':ids,'rough_scene_ids':rough,'nominal':M,'rough':np.ones((20000,1),np.int16)}
    fixture=pd.DataFrame([dict(condition=c,scene_id=s,unsafe=x) for c,s,values in [('clean',1,[1,0]),('clean',2,[0,0,0,0]),('rough_iid',3,[-1,-1])] for x in values])
    r,_=scene_bootstrap(fixture,'unsafe',{'clean':Fraction(1,2),'rough_iid':Fraction(1,2)},draws,True)
    assert r['exact_estimate']=='-3/8' and r['exact_lower95']=='-1/2' and r['exact_upper95']=='-1/4'
    fixture.unsafe=0;r,_=scene_bootstrap(fixture,'unsafe',{'clean':Fraction(1,2),'rough_iid':Fraction(1,2)},draws,True)
    assert r['exact_estimate']=='0' and r['class98_75']=='INCONCLUSIVE'
    rows=[]
    for arch,arm,value in [('KF-D','gate',.1),('BF-D','keep_all',.2)]:
        rows.append(dict(architecture=arch,arm=arm,policy='direct',condition='clean',scene_id=1,frame_id=0,seed=10,
            regret=value,unsafe=int(value==.2),executed=1,abstained=0))
    delta=paired(pd.DataFrame(rows),'KF-D','gate','keep_all',right_architecture='BF-D')
    assert np.allclose(delta.regret,-.1) and delta.unsafe.iloc[0]==-1
    source=ROOT/'study3/validation';files=list(csv.DictReader((source/'files.csv').open()));assert len(files)==750
    for row in files:
        path=source/row['path'];assert sha(path)==row['sha256'];d=load_npz(path)
        assert int(d['scene_id']) in range(3000,3150)
        assert np.isfinite(d['gt_aff']).all() and not np.any((d['gt_aff']>0)&((d['gt_haz']>.5)|(d['gt_sem']!=0)))
    d=load_npz(source/files[0]['path'])
    pred=dict(H=d['gt_H'],mat_prob=np.eye(4,dtype=np.float32)[d['gt_mat']].transpose(2,0,1),
        sem_prob=np.eye(6,dtype=np.float32)[d['gt_sem']].transpose(2,0,1),haz_prob=d['gt_haz'].astype(np.float32))
    rows,phi=attribute(pred,d);assert max(abs(r['regret']) for r in rows)<2e-7 and np.allclose(phi,0)
    # Mechanism code controls use constant synthetic arrays, not novel-fault
    # validation examples. No neural outputs, tuning or outcome inspection.
    rgb=np.full((160,256,3),100,np.uint8);depth=np.full((160,256),4,np.float32)
    for kind in ['mud_water','sun_glare']:
        a=novel_latents(rgb.shape[:2],kind,np.random.default_rng(7));b=novel_latents(rgb.shape[:2],kind,np.random.default_rng(7))
        assert all(np.array_equal(a[k],b[k]) for k in a)
        r,z=apply_novel(rgb,depth,a);assert r.shape==rgb.shape and r.dtype==rgb.dtype and z.shape==depth.shape and np.isfinite(z).all()
    js(ROOT/'protocol/prelock_controls.json',dict(status='PASS',validation_records=750,syntax_files=len(list((ROOT/'code').glob('*.py'))),
        checks=['scene-equal versus attempt-equal weighting','zero exact event contrast','cross-architecture pairing','validation file hashes and safe positive oracle',
            'all-truth oracle and zero Shapley fixture','deterministic novel-latent mechanism on constant arrays'],new_holdout_generated=False,code_sha256=sha(__file__)))

if __name__=='__main__':main()
