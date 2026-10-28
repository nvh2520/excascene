"""Explicitly authorized AFAST-only effect/runtime planning, before new test generation."""
from common import *
import pandas as pd
from scipy.stats import norm

def main():
    rows=[json.loads(s) for s in (AFAST/'resources/jobs.jsonl').read_text().splitlines()]
    seconds={}
    for architecture in ['BF-D','KF-D']:
        jobs=[r for r in rows if r['event']=='finish' and r.get('returncode')==0 and r['name'].startswith(architecture) and 'holdout_inference' in r['name']]
        assert len(jobs)==3
        seconds[architecture]=max(r['elapsed_seconds'] for r in jobs)/(3328*2)
    # Twelve logical forwards per architecture: three seeds x four camera arms.
    # Six secondary forwards costed conservatively at the full KF rate.
    per_record=12*seconds['BF-D']+18*seconds['KF-D']
    metrics=pd.concat([pd.read_csv(AFAST/f'metrics/KF-D_seed{s}.csv') for s in [10,11,12]],ignore_index=True)
    x=metrics[metrics.policy=='direct']
    wide=x.pivot(index=['condition','scene_id','frame_id','seed'],columns='mask',values='regret')
    wide['delta']=wide[1]-wide[0]
    effects=wide.reset_index().groupby(['condition','scene_id']).delta.mean().unstack(0)
    severe=float(wide.loc['cam_s3','delta'].mean());raw_effect=abs(severe)/2
    nominal_conditions=['clean','iid','cam_s1','cam_s2','cam_s3','cam_s3','cam_s3','all_s2']
    nom=effects.loc[range(300000,300128),nominal_conditions].sum(axis=1)/9
    rough=effects.loc[range(310000,310064),'rough_iid']/9
    fractions=x[x['mask']==0].groupby('condition').oracle.apply(lambda a:float((a>0).mean()))
    acting=(sum(fractions[c] for c in nominal_conditions)+fractions['rough_iid'])/9
    effect=raw_effect*acting
    critical=float(norm.ppf(1-1/160));curve=[]
    for n in range(64,2049,64):
        nr=n//2;records=4*(8*n+nr)
        # Include 15min fixed setup/deployment allowance before applying margin.
        forecast=1.25*(records*per_record+900)
        se=np.sqrt(nom.var(ddof=1)/n+rough.var(ddof=1)/nr)
        power=float(norm.cdf(effect/se-critical))
        curve.append(dict(nominal_scenes=n,rough_scenes=nr,planned_records=records,logical_forwards=records*30,
            forecast_gpu_hours=forecast/3600,normal_approx_power=power,standard_error_proxy=se,fits_study3_12h=forecast<=12*3600))
    fits=[r for r in curve if r['fits_study3_12h']];assert fits
    selected=fits[-1]
    csvwrite(ROOT/'study3/power_resource_curve.csv',curve)
    js(ROOT/'study3/power_resource_proposal.json',dict(status='PROPOSED_NOT_FINAL_LOCK',selected=selected,
        historical_severe_KF_removal=severe,half_effect=raw_effect,oracle_act_fraction_proxy=acting,planning_pooled_effect=-effect,
        sigma_components={'nominal_weighted_variance':float(nom.var(ddof=1)),'rough_weighted_variance':float(rough.var(ddof=1))},
        adjusted_critical_z=critical,two_sided_marginal_level=.9875,planning_variance='scene-averaged supplied-removal contrasts; single-camera severe and novel assigned severe historical covariance as a planning proxy only',
        power_caveat='Normal approximation under a fixed location shift and historical variance; not guaranteed gate power or bootstrap coverage. No bootstrap draws for planning.',
        runtime_seconds_per_forward_max_observed=seconds,per_record_seconds=per_record,margin=1.25,
        budget_plan={'probe_cap':2,'gate_cap':4,'goose_cap':6,'study3_forecast':selected['forecast_gpu_hours'],'total_max_with_study3_forecast':12+selected['forecast_gpu_hours']},
        no_new_holdout_inputs=True,all_seeds_and_arms_retained=True))
    print(json.dumps(selected,indent=2),flush=True)
if __name__=='__main__':main()
