import copy
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from stage3.dataset import Stage3Dataset,EpochDataLoader,sensor_keep_mask,apply_sensor_dropout
from stage3.models import build_model
from stage3.selectors import soft_composition,no_threshold_metrics
from stage3.sensitivity import scene_statistic,scene_signflip
from stage3.replication import same_seed_pairs
from stage3.stats import holm_fixed,paired_values
from stage2.tests.test_interventions import fixed_batch
from kinafford.models.variants import build_model as original_build
from kinafford.data.dataset import ExcaSceneDataset

@pytest.mark.parametrize('gating',[False,True])
@pytest.mark.parametrize('dropout_flag',[0.,.15])
def test_fixed_batch_exact_default_and_dropout_model_when_present(gating,dropout_flag):
 cfg=dict(sensors=['cab','boom','stick','lidar'],gating=gating,feat_ch=64,fusion='geometric',extrinsics='dynamic',hetero=True,aff_head=True,pretrained=False)
 b=fixed_batch();torch.manual_seed(5);a=original_build(cfg).eval();torch.manual_seed(5);m=build_model(dict(cfg,modality_dropout=dropout_flag)).eval()
 with torch.no_grad():
  x,y=a(b),m(b)
  for k in x:
   if torch.is_tensor(x[k]):assert torch.equal(x[k],y[k]),k
   else:
    for (i,u),(j,v) in zip(x[k],y[k]):assert i==j and torch.equal(u,v)


def test_lidar_absence_suppresses_map_and_gate_with_original_keys():
 cfg=dict(sensors=['cab','boom','stick','lidar'],gating=True,feat_ch=64,fusion='geometric',extrinsics='dynamic',hetero=True,aff_head=True,pretrained=False,modality_dropout=.15)
 b=fixed_batch();b['lidar_present']=torch.zeros(1);b['lidar_valid'].zero_();b['meta'][:,3]=0;m=build_model(cfg).eval()
 with torch.no_grad():
  maps,_,presence=m.sensor_maps(b);out=m(b)
 assert torch.count_nonzero(maps['lidar'])==0 and presence['lidar'].item()==0
 assert out['gate'][0,3].item()==0


def test_dropout_resamples_all_absent_and_never_restores_original_absence():
 class Fake:
  def __init__(self):self.n=0
  def random(self,n):self.n+=1;return np.zeros(n) if self.n==1 else np.ones(n)
 rng=Fake();available=np.array([False,True,False,True]);assert np.array_equal(sensor_keep_mask(available,.15,rng),available);assert rng.n==2
 with pytest.raises(ValueError):sensor_keep_mask(np.zeros(4,bool),.15,np.random.default_rng(0))
 for i in range(1000):assert sensor_keep_mask(available,.15,np.random.default_rng(i)).any()


def test_dropout_updates_only_sensor_inputs_targets_and_metadata():
 b={k:v[0].clone() for k,v in fixed_batch().items()};b['gt_vis']=torch.ones(96,96);before=b['gt_vis'].clone();apply_sensor_dropout(b,[True,False,False,False])
 assert b['cam_present'].tolist()==[1,0,0] and b['meta'][:4].tolist()==[1,0,0,0]
 assert b['lidar_valid'].count_nonzero()==0 and b['lidar_present']==0 and b['rel_gt'][1:].count_nonzero()==0
 assert b['img'][1:].count_nonzero()==0 and b['depth'][1:].count_nonzero()==0
 assert torch.equal(b['gt_vis'],before)

@pytest.fixture(scope='module')
def tiny_data(tmp_path_factory):
 from kinafford.sim.scene import generate_scene
 from kinafford.sim.render import Renderer
 from kinafford.sim.generate import render_frame
 root=tmp_path_factory.mktemp('stage3_data');(root/'train').mkdir();scene=generate_scene(0);sample=render_frame(scene,Renderer(scene),0,0,'iid',False);np.savez_compressed(root/'train/f_000000_0.npz',**sample);return root


def test_dataset_default_and_epoch_zero_are_identical(tiny_data):
 kw=dict(train=True,kin_jitter_deg=1.,color_jitter=.15);a=ExcaSceneDataset(tiny_data,'train',**kw)[0]
 for fix in [False,True]:
  ds=Stage3Dataset(tiny_data,'train',epoch_rng=fix,**kw);b=ds[0];assert a.keys()==b.keys()
  for k in a:assert torch.equal(a[k],b[k]),k
  ds.set_epoch(1);c=ds[0]
  assert torch.equal(b['img'],c['img']) is (not fix)
  assert torch.equal(b['T_dyn'],c['T_dyn']) is (not fix)
  for k in ['gt_H','gt_aff','gt_vis','q']:assert torch.equal(b[k],c[k])


def test_persistent_workers_see_epoch_changes(tiny_data):
 ds=Stage3Dataset(tiny_data,'train',train=True,kin_jitter_deg=1.,color_jitter=.15,epoch_rng=True)
 loader=EpochDataLoader(ds,batch_size=1,num_workers=2,persistent_workers=True)
 x=next(iter(loader));y=next(iter(loader));assert ds.epoch.value==1
 assert not torch.equal(x['img'],y['img']) and not torch.equal(x['T_dyn'],y['T_dyn'])
 del loader


def test_soft_expected_material_semantic_hazard_and_no_threshold():
 shape=(3,3);p=dict(H=np.full(shape,1.2,np.float32),mat_prob=np.zeros((4,*shape)),sem_prob=np.zeros((6,*shape)),haz_prob=np.full(shape,.75),aff=np.ones(shape)*.8)
 p['mat_prob'][0]=.5;p['mat_prob'][3]=.5;p['sem_prob'][0]=.4;reach=np.ones(shape)
 out=soft_composition(p,reach,0);np.testing.assert_allclose(out,((1+.35)/2)*.4*.25,rtol=1e-6)
 p['sem_prob'][0]=0;assert not soft_composition(p,reach,0).any();p['sem_prob'][0]=1;p['haz_prob'][:]=1;assert not soft_composition(p,reach,0).any()
 gt=dict(H=p['H'],reach=reach,z_target=0,aff=np.ones(shape),haz=np.zeros(shape),sem=np.zeros(shape,int));m=no_threshold_metrics(p,gt);assert m['pick_row_direct_nothreshold']==m['pick_col_direct_nothreshold']==0
 gt['reach'][:]=0;m=no_threshold_metrics(p,gt);assert m['no_reachable_soft']==1 and m['pick_row_soft']==m['pick_col_soft']==0
 gt['reach'][:]=.05;m=no_threshold_metrics(p,gt);assert m['no_reachable_soft']==0


def test_fixed_eight_slot_holm_and_seed_averaged_pairing():
 p=holm_fixed([.001]+[np.nan]*7);assert p[0]==pytest.approx(.008) and np.all(p[1:]==1)
 d=pd.DataFrame([dict(tag=t,seed=s,suite='iid',scene_id=1,frame_id=0,m=v) for t,s,v in [('a',0,0),('a',1,2),('b',0,3),('b',1,5)]])
 j=paired_values(d,('id','a','b','m','m',['iid'],'lower'));assert j.a.iloc[0]==1 and j.b.iloc[0]==4
 assert same_seed_pairs(d,'a','b','m','m',['iid'],2).empty
 assert same_seed_pairs(d,'a','b','m','m',['iid'],0).a.iloc[0]==0


def test_scene_signflip_shares_sign_across_suites_and_aggregates_frames():
 d=pd.DataFrame(dict(suite=['x','x','y','y'],scene_id=[1]*4,frame_id=[0,1,0,1],a=[0,2,0,2],b=[2,2,2,2]));s=scene_statistic(d);assert len(s)==2 and np.all(s.difference==-1)
 p=scene_signflip(s);assert .48<p<.52 # one shared scene sign, not two independent suite signs
 assert p==scene_signflip(s)
 s['difference']=0;assert scene_signflip(s)==1
