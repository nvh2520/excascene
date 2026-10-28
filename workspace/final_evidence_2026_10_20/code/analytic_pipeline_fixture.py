"""End-to-end arithmetic fixture; synthetic numbers, never scientific results."""
from common import *
import pandas as pd
def main():
    fixture=ROOT/'implementation_controls/analytic_fixture';assert not fixture.exists()
    (fixture/'study4/metrics').mkdir(parents=True);(fixture/'study4/bootstrap_plan').mkdir()
    js(fixture/'NOT_SCIENTIFIC_DATA.json',dict(kind='deterministic implementation fixture',scientific_data=False,neural_forwards=0,new_bootstrap_draws=0))
    from statistical_core4 import CONDITIONS
    from study4_statistics import ARCHES,SEEDS
    ids=np.array([11,12]);rid=np.array([13]);counts=np.array([[2,0],[1,1],[0,2]],np.int16)[np.arange(20000)%3]
    np.savez_compressed(fixture/'study4/bootstrap_plan/bootstrap_multiplicities.npz',nominal_scene_ids=ids,rough_scene_ids=rid,nominal=counts,rough=np.ones((20000,1),np.int16))
    for ai,a in enumerate(ARCHES):
        arms=['keep_all','gate','always_mask','oracle_mask','rgb_only','depth_only'] if a.endswith('42') else ['keep_all','gate']
        for seed in SEEDS:
            rows=[];keys=[];ss=[];tt=[]
            for c in CONDITIONS:
                scenes=rid if c=='rough_iid' else ids
                for sid in scenes:
                    for slot in range(4 if sid==11 else 2):
                        for j,arm in enumerate(arms):
                            unsafe=int((slot+seed+int(sid)+j+ai)%4==0)
                            base=dict(architecture=a,seed=seed,condition=c,scene_id=int(sid),frame_id=slot,arm=arm,
                                regret=.2+.001*slot+.0001*seed+.01*ai-.01*j,unsafe=unsafe,executed=1,abstained=0,height_reachable_terrain=.1+.01*ai,reachable_terrain_cells=10)
                            rows.extend(dict(**base,policy=p) for p in ['direct','soft'])
                            keys.append(dict(condition=c,scene_id=int(sid),frame_id=slot,arm=arm))
                            sm=np.eye(6,dtype=np.int64)*10;sm[3,0]+=ai;sm[4,0]+=ai*2
                            mt=np.eye(4,dtype=np.int64)*10;mt[1,0]+=ai
                            ss.append(sm);tt.append(mt)
            path=fixture/f'study4/metrics/{a}_seed{seed}.csv';csvwrite(path,rows)
            js(path.with_suffix('.json'),dict(status='COMPLETE',fixture_only=True))
            csvwrite(path.with_name(path.stem+'_confusion_keys.csv'),keys)
            np.savez_compressed(path.with_name(path.stem+'_frame_confusions.npz'),semantic=np.stack(ss),material=np.stack(tt))
    import infer_study4,study4_statistics,study4_map_metrics
    infer_study4.verify_lock=lambda:None
    study4_statistics.ROOT=fixture;study4_map_metrics.ROOT=fixture
    study4_statistics.main();study4_map_metrics.main()
    primary=pd.read_csv(fixture/'study4/analysis/primary_results.csv')
    assert len(primary)==4 and np.allclose(primary[primary.metric=='regret'].estimate,-.01,atol=1e-15)
    maps=pd.read_csv(fixture/'study4/map_metrics/paired_comparisons.csv')
    assert np.allclose(maps[(maps.left=='KF-D42')&(maps.right=='KF-D14')&(maps.metric=='height_frame_mae')].estimate,-.02)
    assert len(pd.read_csv(fixture/'study4/map_metrics/full_class_diagnostics.csv'))==4*3*9*10
    js(ROOT/'protocol/analytic_pipeline_control.json',dict(status='PASS',scope='deterministic, unequal-slot arithmetic fixture, not a scientific cohort',
        regret_primary_expected=-.01,paired_height_expected=-.02,exact_event_path_exercised=True,all_class_mapping_exercised=True,new_bootstrap_draws=0,neural_forwards=0,
        code_sha256={n:sha(ROOT/'code'/n) for n in ['analytic_pipeline_fixture.py','study4_statistics.py','statistical_core4.py','study4_map_metrics.py']}))
    print('PASS analytic pipeline fixture; NOT SCIENTIFIC DATA',flush=True)
if __name__=='__main__':main()
