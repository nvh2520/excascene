"""Saved-draw descriptive summaries for the unchanged measurement anchor."""
import argparse,math
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
from reference_analysis import CONDITIONS,POOL,frac_quantile
from reference_policies import write_json,write_csv,digest

METRICS=['regret','unsafe','executed','abstained','H_reachable_terrain']

def main():
    p=argparse.ArgumentParser();p.add_argument('--afast',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output
    anchor=pd.read_csv(out/'geometric_per_frame.csv');assert len(anchor)==3328
    models=pd.concat([pd.read_csv(a.afast/'metrics'/f'{ar}_seed{s}.csv') for ar in ['BF-D','KF-D'] for s in [10,11,12]])
    with np.load(a.afast/'analysis/bootstrap_multiplicities.npz') as z:
        ids={k:z[k+'_scene_ids'] for k in ['nominal','rough']};mat={k:z[k].astype(np.int64) for k in ids}
    rows=[];cache={};denoms=[];seedrows=[]
    def summarize(values,cond,metric):
        stratum='rough' if cond=='rough_iid' else 'nominal';n=len(ids[stratum]);nslot=values.shape[1]*4
        scene=values.mean(axis=(1,2));draws=mat[stratum].astype(np.float64)@scene/n
        result=dict(estimate=float(scene.mean()),lower95=float(np.quantile(draws,.025)),upper95=float(np.quantile(draws,.975)),
            exact='',lower95_exact='',upper95_exact='',draws=draws,numer=None,denom=None)
        if metric in ['unsafe','executed','abstained']:
            counts=values.sum(axis=(1,2));assert np.equal(counts,np.round(counts)).all()
            denom=n*nslot;numer=mat[stratum]@counts.astype(np.int64);exact=Fraction(int(counts.sum()),denom)
            bounds=[frac_quantile(numer,denom,q) for q in [Fraction(1,40),Fraction(39,40)]]
            result.update(estimate=float(exact),lower95=float(bounds[0]),upper95=float(bounds[1]),exact=str(exact),
                lower95_exact=str(bounds[0]),upper95_exact=str(bounds[1]),numer=numer,denom=denom)
        return result
    def append(arm,policy,cond,metric,r):
        scale=1000 if metric.startswith('H_') else 1 if metric=='regret' else 100
        row={k:v for k,v in r.items() if k not in ['draws','numer','denom']}
        row.update(arm=arm,policy=policy,condition=cond,metric=metric,contrast='anchor absolute' if arm=='geometry' else arm+' minus geometry',
            camera_state='recorded intact',display_estimate=r['estimate']*scale,display_lower95=r['lower95']*scale,display_upper95=r['upper95']*scale,
            display_unit='mm' if metric.startswith('H_') else 'raw regret' if metric=='regret' else 'percent' if arm=='geometry' else 'percentage points',
            evidence_class='NEW_POSTHOC_DESCRIPTIVE_EXISTING_ANCHOR; ordinary unadjusted95%; saved draws; approximate coverage',
            weighting='equal frame slots and scenes; models additionally equal separately scored seeds; fixed1/3 pool',
            source='geometric_per_frame.csv' if arm=='geometry' else 'geometric_per_frame.csv + original model metrics CSVs')
        rows.append(row)
    for cond in CONDITIONS:
        stratum='rough' if cond=='rough_iid' else 'nominal';n=len(ids[stratum])
        g=anchor[anchor.condition==cond].sort_values(['scene_id','frame_id']);assert len(g)==n*4
        base={m:g[m].to_numpy(np.float64).reshape(n,1,4) for m in METRICS}
        denoms.append(dict(condition=cond,attempts=len(g),unsafe_events=int(g.unsafe.sum()),executions=int(g.executed.sum()),abstentions=int(g.abstained.sum()),
            unsafe_per_execution=float(g.unsafe.sum()/g.executed.sum()) if g.executed.sum() else '',unsafe_per_execution_undefined=int(g.executed.sum()==0),
            all_empty_frames=int(g.all_empty.sum()),undefined_reachable_height=int(g.H_reachable_terrain.isna().sum()),
            reference_policy_count=1,learned_fits=0))
        for metric in METRICS:
            r=summarize(base[metric],cond,metric);cache[('geometry','geometric',cond,metric)]=r;append('geometry','geometric',cond,metric,r)
        for ar in ['BF-D','KF-D']:
            for policy in ['direct','soft']:
                m=models[(models.architecture==ar)&(models['mask']==0)&(models.policy==policy)&(models.condition==cond)].sort_values(['scene_id','seed','frame_id'])
                assert len(m)==n*12
                for metric in METRICS:
                    value=m[metric].to_numpy().reshape(n,3,4)-base[metric]
                    r=summarize(value,cond,metric);cache[(ar,policy,cond,metric)]=r;append(ar,policy,cond,metric,r)
                    for i,seed in enumerate([10,11,12]):seedrows.append(dict(model=ar,policy=policy,condition=cond,metric=metric,seed=seed,model_minus_geometry=float(value[:,i,:].mean())))
    for arm,policy in dict.fromkeys(k[:2] for k in cache):
        for metric in METRICS:
            values=[cache[(arm,policy,c,metric)] for c in ['iid','all_s2','rough_iid']]
            draws=sum(r['draws'] for r in values)/3
            r=dict(estimate=sum(r['estimate'] for r in values)/3,lower95=float(np.quantile(draws,.025)),upper95=float(np.quantile(draws,.975)),exact='',lower95_exact='',upper95_exact='')
            if all(x['numer'] is not None for x in values):
                denominator=math.lcm(*(x['denom'] for x in values))*3
                numer=sum(x['numer']*(denominator//(3*x['denom'])) for x in values)
                exact=sum((Fraction(x['exact']) for x in values),Fraction(0))/3
                bounds=[frac_quantile(numer,denominator,q) for q in [Fraction(1,40),Fraction(39,40)]]
                r.update(estimate=float(exact),lower95=float(bounds[0]),upper95=float(bounds[1]),exact=str(exact),lower95_exact=str(bounds[0]),upper95_exact=str(bounds[1]))
            append(arm,policy,POOL,metric,r)
    seed_df=pd.DataFrame(seedrows)
    seed_pool=seed_df[seed_df.condition.isin(['iid','all_s2','rough_iid'])].groupby(['model','policy','metric','seed']).model_minus_geometry.mean().reset_index()
    seed_pool['condition']=POOL
    write_csv(out/'geometric_descriptive_intervals.csv',rows);write_csv(out/'geometric_denominators.csv',denoms)
    pd.concat([seed_df,seed_pool],ignore_index=True).to_csv(out/'geometric_per_seed_comparisons.csv',index=False)
    assert len(rows)==200
    write_json(out/'geometric_analysis_provenance.json',dict(status='COMPLETE',records=3328,interval_rows=len(rows),
        source_sha256=digest(out/'geometric_per_frame.csv'),draws_sha256=digest(a.afast/'analysis/bootstrap_multiplicities.npz'),
        code_sha256=digest(Path(__file__)),new_draws=0,new_neural_forwards=0,
        limits='unchanged geometric predictor/composition, adapted loader and strict v5 selection/scoring; intact recorded sensor state only; not original-driver reproduction'))
    print(pd.DataFrame(rows)[lambda x:(x.arm=='geometry')&x.condition.isin(['clean','cam_s3','rough_iid'])][['condition','metric','display_estimate','display_lower95','display_upper95']].to_string(index=False))

if __name__=='__main__':main()
