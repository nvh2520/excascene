"""Check completeness and failure denominators before primary aggregation."""
from common import *
import pandas as pd
def main():
    locked();matrix=pd.read_csv(RUN/'manifests/run_matrix.csv');counts=[];failures=[]
    for (model,checkpoint,seed,mask),rows in matrix.groupby(['model','checkpoint','seed','mask']):
        key=f'{model}_{checkpoint}_s{seed}_mask{int(mask)}';path=RUN/f'metrics/{key}.csv'
        if not path.exists():failures.append(dict(run=key,reason='missing metrics'));continue
        t=pd.read_csv(path);assert not t.duplicated(['condition','scene_id','frame_id','policy']).any()
        assert (t.executed+t.abstained==1).all() and (t.unsafe<=t.executed).all()
        assert (t.regret>=0).all() and (t.oracle>=0).all()
        assert (t[t.abstained==1].selected_utility==0).all()
        assert (t[t.executed==1].eligible_cells>0).all()
        assert (t[t.empty_reach==1].oracle==0).all()
        for _,r in rows.iterrows():
            z=t[t.condition==r.condition];frames=z[['scene_id','frame_id']].drop_duplicates()
            assert len(frames)==r.planned_frames,(key,r.condition,len(frames),r.planned_frames)
            expected=['hard','soft','soft_hazard']+([] if model=='R2_noaff' else ['direct','risk'])
            assert set(z.policy)==set(expected)
            assert len(z)==len(frames)*len(expected)
            nonfinite_primary_height=int(z[['scene_id','frame_id','H_reachable_terrain']].drop_duplicates().H_reachable_terrain.isna().sum())
            if nonfinite_primary_height:failures.append(dict(run=key,reason='nonfinite height endpoint; G must be blocked, not drop model failures'))
            counts.append(dict(model=model,checkpoint=checkpoint,seed=int(seed),mask=bool(mask),condition=r.condition,frame_model_forwards=len(frames),metric_rows=len(z),nonfinite_height=int(z.nonfinite_height.sum()),nonfinite_score_cells=int(z.nonfinite_scores.sum()),nonfinite_hazard_cells=int(z.nonfinite_hazard.sum()),nonfinite_reach_cells=int(z.nonfinite_reach.sum()),abstained=int(z.abstained.sum()),empty_reach=int(z.empty_reach.sum()),unsafe_events=int(z.unsafe.sum())))
    csvwrite(RUN/'manifests/result_completeness.csv',counts)
    # This run has no generated slot exclusions; do not generalize the equality to an incomplete population.
    total=sum(x['frame_model_forwards'] for x in counts);assert total==58608
    counter=json.loads((RUN/'forward_counter.json').read_text());assert counter['by_phase']['fresh']==total
    assert counter['precision']==1920 and counter['total']<=120000
    js(RUN/'manifests/result_gates.json',dict(status='PASS' if not failures else 'FAILED_AFFECTED_ENDPOINT',failures=failures,fresh_forwards=total,training_steps=0,missing_metric_rows=0,nonfinite_sensor_inputs=0,nonfinite_sensor_basis='assertions on every input record before forward; RGB/presence/validity stored integer, K fixed, transforms from finite joints',parameter_buffer_updates=0,parameter_buffer_basis='state digest assertion per completed frozen run'))
    assert not failures,failures
    print('Result completeness and failure-denominator checks PASS',total,flush=True)
if __name__=='__main__':main()
