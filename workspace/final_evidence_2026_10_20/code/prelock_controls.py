"""Deterministic selector/mask/map-metric controls, no test inputs or forwards."""
from common import *
import ast
def main():
    assert not (ROOT/'study4/holdout').exists()
    from statistical_core4 import controls
    controls()
    from study4_diagnostics import attribute,cutoff_metrics
    from study4_map_metrics import metrics
    from infer_study4 import masks
    from neural import tensors
    from kinafford.sim.kinematics import CAMERAS
    n=96;X,Y=np.meshgrid(np.arange(n),np.arange(n))
    reach=np.ones((n,n),np.float32);H=(1.5+.001*X).astype(np.float32)
    mat=np.zeros((n,n),np.uint8);sem=np.zeros_like(mat);haz=np.zeros((n,n),np.float32);haz[:4]=1
    truthmat=np.eye(4,dtype=np.float32)[mat].transpose(2,0,1)
    aff=soft_score(H,truthmat,(sem==0).astype(np.float32),haz,reach,0.)
    d=dict(gt_H=H,gt_mat=mat,gt_sem=sem,gt_haz=haz,gt_reach=reach,gt_aff=aff,z_target=0.)
    pred=dict(H=H.copy(),mat_prob=truthmat.copy(),sem_prob=np.stack([np.ones_like(H)]+[np.zeros_like(H) for _ in range(5)]),haz_prob=haz.copy(),aff=aff.copy())
    att,phi=attribute(pred,d);assert att[15]['regret']<=2e-7 and np.max(np.abs(phi))==0
    for c in [.5,.3,.2,.1,.05]:assert cutoff_metrics(pred,d,c)==selected_metrics(pred['aff'],d,pred['haz_prob'])
    # Exact threshold inclusion, empty support and deterministic tie behavior.
    pred['haz_prob'].fill(.5)
    assert cutoff_metrics(pred,d,.5)['executed']==1 and cutoff_metrics(pred,d,.3)['abstained']==1
    assert selected_metrics(np.zeros_like(H),dict(d,gt_reach=np.zeros_like(H)))['abstained']==1
    S=np.zeros((6,6));T=np.eye(4)
    S[3,3]=3;S[3,0]=1;S[4,4]=1;S[4,0]=3;S[2,2]=9
    mm=metrics(S,T);assert mm['person_cell_recall']==.75 and mm['obstacle_cell_recall']==.25 and mm['material_macro_iou']==1
    sp=dict(cam={c:dict(photo=[('blur',3)] if j==1 else [],novel='sun_glare' if j==2 else '') for j,c in enumerate(CAMERAS)})
    dd=dict(cam_present=np.array([1,1,0]),spec=np.asarray(json.dumps(sp)))
    pp={**{f'p{j}':p for j,p in enumerate([.5,.2,.1])},**{f'invalid{j}':p for j,p in enumerate([.1,.5,1.])}}
    mask=masks(dd,pp,'KF-D42',{'KF-D42':dict(tau=.5,delta=.5)})
    assert np.array_equal(mask['gate'],mask['rgb_only']|mask['depth_only'])
    assert mask['gate'].all() and mask['oracle_mask'].tolist()==[False,True,True]
    assert set(masks(dd,pp,'KF-D14',{'KF-D14':dict(tau=.5,delta=.5)}))=={'keep_all','gate'}
    source=next((PROJECT/'data/train').glob('f_*.npz'));record=load_npz(source)
    baseline=tensors(record,np.zeros(3,bool));removed=tensors(record,np.array([0,1,0],bool))
    assert not removed['img'][1].any() and not removed['depth'][1].any() and not removed['cam_present'][1]
    for key in ['img','depth']:
        assert np.array_equal(removed[key][[0,2]],baseline[key][[0,2]])
    for p in (ROOT/'code').glob('*.py'):ast.parse(p.read_text())
    watched=['study4_diagnostics.py','study4_map_metrics.py','statistical_core4.py','study4_statistics.py','infer_study4.py','neural.py','prelock_controls.py']
    js(ROOT/'protocol/prelock_controls.json',dict(status='PASS',scope='deterministic fixtures and one original training input; no neural forward, scientific score or random bootstrap draw',
        all_truth_oracle=True,shapley_additivity=True,cutoff_half_identity=True,cutoff_inclusion_and_abstention=True,
        semantic_mapping_person3_obstacle4=True,branch_masks=True,retained_inputs=True,
        source_training_frame=str(source.relative_to(PROJECT)),source_sha256=sha(source),code_sha256={name:sha(ROOT/'code'/name) for name in watched}))
    print('PASS prelock deterministic controls',flush=True)
if __name__=='__main__':main()
