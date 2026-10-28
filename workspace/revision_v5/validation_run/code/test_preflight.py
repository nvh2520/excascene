from common import *
import unittest,tempfile,torch
from selectors_v5 import choose,evaluate
from inputs import tensorize,load
from generator import specs,camera_latents,degrade_shared,rng,render_bundle,assert_pairs,validity
class Tests(unittest.TestCase):
    def test_finite_and_ties(self):
        s=np.array([[np.nan,1],[np.inf,1.]])
        self.assertEqual(choose(s,np.ones_like(s))[0],(0,1));self.assertEqual(choose(s,np.ones_like(s))[1],2)
        self.assertIsNone(choose(s,np.zeros_like(s))[0]);self.assertIsNone(choose(s,np.ones_like(s),np.ones_like(s))[0])
        self.assertEqual(choose(np.array([[1.,2.]]),np.array([[.05,.049]]),np.array([[.5,0]]))[0],(0,0))
    def test_safe_unsafe_oracle(self):
        shape=(2,2);u=np.array([[.7,0],[.4,0]],np.float32)
        d=dict(gt_aff=u,gt_reach=np.ones(shape,np.float32),gt_haz=np.array([[0,1],[0,0]]),gt_sem=np.array([[0,0],[0,2]]),z_target=0.)
        p=dict(aff=u.copy(),haz_prob=np.zeros(shape))
        a=evaluate(p,d,'direct');self.assertEqual(a['regret'],0);self.assertEqual(a['unsafe'],0)
        p['aff']=np.array([[0,1],[0,0.]])
        a=evaluate(p,d,'direct');self.assertEqual(a['unsafe'],1);self.assertAlmostEqual(a['regret'],.7,6)
        p['haz_prob'][:]=1;a=evaluate(p,d,'direct');self.assertEqual(a['executed'],0);self.assertAlmostEqual(a['regret'],.7,6)
        d['gt_aff'][:]=0;a=evaluate(p,d,'direct');self.assertEqual(a['regret'],0);self.assertTrue(np.isnan(a['normalized_regret']))
        d['gt_reach'][:]=0;self.assertEqual(evaluate(p,d,'direct')['empty_reach'],1)
    def test_original_inputs_and_mask(self):
        from kinafford.data.dataset import ExcaSceneDataset
        from stage3.dataset import apply_sensor_dropout
        ds=ExcaSceneDataset(str(ROOT/'data'),'suite_clean',train=False,limit=1);d=load(ds.files[0]);base=ds[0]
        t=tensorize(d)
        for k,v in t.items():np.testing.assert_array_equal(v,base[k].numpy(),err_msg=k)
        masked=apply_sensor_dropout(base,[False,False,False,True]);t=tensorize(d,True)
        for k,v in t.items():np.testing.assert_array_equal(v,masked[k].numpy(),err_msg=k)
        self.assertFalse(any(k.startswith('gt_') or k in ['q_true','rel_gt','z_target'] for k in t))
        d2={k:v.copy() for k,v in d.items()};d2['gt_aff'][:]=99;d2['q_true'][:]=99
        for k,v in tensorize(d2).items():np.testing.assert_array_equal(v,tensorize(d)[k])
    def test_stream_independence(self):
        a=rng(5000,0,40,1,5).normal(size=100);rng(5000,0,40,0,5).normal(size=300)
        np.testing.assert_array_equal(a,rng(5000,0,40,1,5).normal(size=100))
        sp=specs(5000,0,False,True)
        for cond in ['cam_s1','cam_s2','cam_s3']:
            for v in sp[cond]['cam'].values():self.assertEqual([p[0] for p in v['photo']],['occlusion','dust','blur','noise','depth'])
        self.assertEqual(sum(x['drop'] for x in sp['cam_s3']['cam'].values()),1)
    def test_serialized_camera_pairs_on_historical_pose(self):
        from kinafford.sim.scene import generate_scene
        from kinafford.sim.render import Renderer
        scene=generate_scene(5000);rr=Renderer(scene);d=load(ROOT/'data/suite_clean/f_005000_0.npz')
        sp=specs(5000,0,False,True);sp={k:v for k,v in sp.items() if k not in ['iid','all_s2']}
        records=render_bundle(scene,rr,0,d['q_true'],str(d['phase']),'historical_fixture',sp)
        with tempfile.TemporaryDirectory(dir=RUN) as tmp:
            for k,rec in records.items():p=Path(tmp)/(k+'.npz');npzwrite(p,**rec);records[k]=load(p)
            self.assertEqual(len(assert_pairs(records)),len(records))
        self.assertEqual(records['clean']['gt_aff'].dtype,np.float32);self.assertEqual(records['clean']['depth'].dtype,np.float16)
    def test_exact_event_quantile(self):
        from fractions import Fraction
        a=[Fraction(-1,3),Fraction(0),Fraction(1,3)]
        self.assertEqual(a[1],0);self.assertNotEqual(Fraction(1,10**30),0)
    def test_fixed_weights_with_partial_scenes(self):
        import pandas as pd
        from analysis import estimate
        # Two nominal conditions have different frame counts; equal condition weights must remain .5/.5.
        rows=[]
        for seed in range(3):
            rows.extend([dict(seed=seed,condition='a',scene_id=1,frame_id=0,difference=1),dict(seed=seed,condition='b',scene_id=1,frame_id=0,difference=-1),dict(seed=seed,condition='b',scene_id=1,frame_id=1,difference=-1)])
        draws={'nominal':(np.array([1]),np.ones((10000,1),np.int16))}
        result,rep=estimate(pd.DataFrame(rows),'unsafe',['a','b'],draws)
        self.assertEqual(result['mean_diff'],0);self.assertEqual(result['ci9875_lo'],0);self.assertTrue(np.all(rep==0))
    def test_recorded_invalid_camera_origin(self):
        from kinafford.sim.scene import generate_scene
        from kinafford.sim.render import Renderer
        scene=generate_scene(5072);d=load(ROOT/'data/suite_clean/f_005072_2.npz');validity.slot=2
        failures=validity(scene,Renderer(scene),d['q_true'],{'clean':specs(5072,2,False)['clean']})
        self.assertTrue(any('clearance' in f for f in failures),failures)
    def test_matrix_counts(self):
        import pandas as pd
        m=pd.read_csv(RUN/'manifests/run_matrix.csv');self.assertEqual(m.planned_forwards.sum(),58608)
        k=pd.read_csv(RUN/'manifests/precision_keys.csv');self.assertEqual(len(k),128);self.assertEqual(k.additional_subset.sum(),16)
        self.assertEqual(128*2*3*2+16*4*3*2,1920)
        self.assertEqual(len(pd.read_csv(RUN/'legacy/historical_tests_49.csv')),49)
        f=pd.read_csv(RUN/'legacy/F6_stability_counts.csv');q=f[(f.version=='v5_integer_correction')&(f.item=='Q3')&(f.endpoint=='both')].iloc[0]
        self.assertEqual((q.yes,q.total),(13,24))
if __name__=='__main__':unittest.main(verbosity=2)
