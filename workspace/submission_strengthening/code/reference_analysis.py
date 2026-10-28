"""Complete post-hoc reference contrasts with saved scene multiplicities only."""
from __future__ import annotations
import argparse, csv, json, math
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
from reference_policies import digest,write_json,write_csv

CONDITIONS=('clean','iid','all_s2','cam_s1','cam_s2','cam_s3','rough_iid')
POOL='fixed_one_third_iid_all_s2_rough'
METRICS=('regret','unsafe','executed','abstained')
Q=(Fraction(1,40),Fraction(39,40))
EVIDENCE='NEW_POSTHOC_DESCRIPTIVE_REFERENCE; ordinary95 unadjusted; inspected AFAST holdout'
WEIGHT='frame metrics; equal fitted seeds where applicable; equal four slots within scene; equal scenes; fixed 1/3 conditions for pool'

def frac_quantile(nums,denom,q):
    a=np.sort(nums);pos=q*(len(a)-1);lo=pos.numerator//pos.denominator;wt=pos-lo
    return (int(a[lo])*(1-wt)+int(a[min(lo+1,len(a)-1)])*wt)/denom

def main():
    p=argparse.ArgumentParser();p.add_argument('--afast',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();afast=args.afast;out=args.output
    with np.load(afast/'analysis/bootstrap_multiplicities.npz',allow_pickle=False) as z:
        ids={k:z[k+'_scene_ids'].astype(np.int64) for k in ['nominal','rough']}
        matrices={k:z[k].astype(np.int64) for k in ids}
    for k,m in matrices.items():assert m.shape==(20000,len(ids[k])) and (m>=0).all() and (m.sum(1)==len(ids[k])).all()
    fmat={k:v.astype(np.float64) for k,v in matrices.items()}
    refs=pd.read_csv(out/'reference_per_frame.csv',keep_default_na=False)
    assert len(refs)==6656 and not refs.duplicated(['reference','condition','scene_id','frame_id']).any()
    model=pd.concat([pd.read_csv(afast/'metrics'/f'{arch}_seed{seed}.csv') for arch in ['BF-D','KF-D'] for seed in [10,11,12]],ignore_index=True)
    assert len(model)==79872
    for key,g in model.groupby(['architecture','seed','mask','policy','condition']):
        expected=refs[(refs.reference=='R0')&(refs.condition==key[-1])]
        joined=g.merge(expected[['scene_id','frame_id','oracle']],on=['scene_id','frame_id'],suffixes=('_model','_reference'),validate='one_to_one')
        assert len(joined)==len(expected) and (joined.oracle_model==joined.oracle_reference).all()
    reference_arrays={};model_arrays={};reference_sources={};summaries=[];denoms=[];per_seed=[];scene_rows=[]
    for ref in ['R0','R1']:
        for cond in CONDITIONS:
            stratum='rough' if cond=='rough_iid' else 'nominal'
            g=refs[(refs.reference==ref)&(refs.condition==cond)].sort_values(['scene_id','frame_id'])
            assert list(zip(g.scene_id,g.frame_id))==[(s,t) for s in ids[stratum] for t in range(4)]
            reference_arrays[(ref,cond)]={m:g[m].to_numpy(np.float64).reshape(-1,1,4) for m in METRICS}
            reference_sources[(ref,cond)]=g
    for arch in ['BF-D','KF-D']:
        for mask in [0,1]:
            for policy in ['direct','soft']:
                for cond in CONDITIONS:
                    stratum='rough' if cond=='rough_iid' else 'nominal'
                    g=model[(model.architecture==arch)&(model['mask']==mask)&(model.policy==policy)&(model.condition==cond)].sort_values(['scene_id','seed','frame_id'])
                    assert list(zip(g.scene_id,g.seed,g.frame_id))==[(s,seed,t) for s in ids[stratum] for seed in [10,11,12] for t in range(4)]
                    model_arrays[(arch,mask,policy,cond)]={m:g[m].to_numpy(np.float64).reshape(-1,3,4) for m in METRICS}
    cached={}
    def stats(values,condition,metric,reference,kind,exact_fractions=None):
        stratum='rough' if condition=='rough_iid' else 'nominal';n=len(ids[stratum]);scene=values.mean(axis=(1,2),dtype=np.float64)
        point=float(scene.mean());draws=fmat[stratum]@scene/n
        estimate_exact='';bounds_exact=['',''];integer_draws=None;denom=None
        # Exact integer event arithmetic for R1, and execution for both references.
        if metric!='regret' and not (metric=='unsafe' and reference=='R0'):
            summed=values.sum(axis=(1,2));assert np.equal(summed,np.round(summed)).all()
            counts=summed.astype(np.int64);denom=n*values.shape[1]*values.shape[2]
            integer_draws=matrices[stratum]@counts
            exact=Fraction(int(counts.sum()),denom);bounds=[frac_quantile(integer_draws,denom,q) for q in Q]
            point=float(exact);draws=integer_draws.astype(np.float64)/denom
            estimate_exact=str(exact);bounds_exact=list(map(str,bounds));low,high=map(float,bounds)
        else:
            low,high=np.quantile(draws,[.025,.975],method='linear')
            if exact_fractions is not None:
                estimate_exact=str(sum(exact_fractions,Fraction(0))/len(exact_fractions))
        return dict(point=point,low=float(low),high=float(high),estimate_exact=estimate_exact,low_exact=bounds_exact[0],high_exact=bounds_exact[1],
                    draws=draws,integer_draws=integer_draws,integer_denominator=denom,scene=scene)
    def append_result(ref,arch,mask,policy,cond,metric,kind,result):
        scale=1 if metric=='regret' else 100
        nscene=192 if cond==POOL else 64 if cond=='rough_iid' else 128
        nrecords=1280 if cond==POOL else nscene*4
        summaries.append(dict(reference=ref,model=arch,camera_mask=mask,model_policy=policy,condition=cond,metric=metric,
            estimand='reference absolute' if kind=='absolute' else 'model minus reference',estimate=result['point'],lower95=result['low'],upper95=result['high'],
            estimate_exact=result['estimate_exact'],lower95_exact=result['low_exact'],upper95_exact=result['high_exact'],
            unit='raw utility regret' if metric=='regret' else 'probability per attempted decision',
            display_estimate=result['point']*scale,display_lower95=result['low']*scale,display_upper95=result['high']*scale,
            display_unit='raw regret' if metric=='regret' else ('percent' if kind=='absolute' else 'percentage points'),
            unique_scene_identities=nscene,condition_records=nrecords,neural_fits=0 if kind=='absolute' or arch=='R1' else 3,reference_policies=1,
            bootstrap_draws=20000,interval='ordinary unadjusted 95% paired scene percentile; approximate coverage',
            evidence_class=EVIDENCE,weighting=WEIGHT,reference_unsafe_kind='expected_probability_no_integer_events' if ref=='R0' else 'deterministic_event',
            source='reference_per_frame.csv' if kind=='absolute' or arch=='R1' else 'reference_per_frame.csv + original metrics/{model}_seed{10,11,12}.csv',
            support='all existing valid AFAST bundles; no new exclusions; no undefined requested frame metrics'))
    for ref in ['R0','R1']:
        for cond in CONDITIONS:
            g=reference_sources[(ref,cond)];arr=reference_arrays[(ref,cond)]
            fractions=[Fraction(int(n),int(d)) for n,d in zip(g.expected_unsafe_numerator,g.expected_unsafe_denominator)] if ref=='R0' else None
            for metric in METRICS:
                s=stats(arr[metric],cond,metric,ref,'absolute',fractions if metric=='unsafe' else None)
                cached[(ref,'reference','NA','NA',cond,metric)]=s;append_result(ref,'reference','NA','NA',cond,metric,'absolute',s)
            denoms.append(dict(reference=ref,condition=cond,attempts=len(g),executions=int(g.executed.sum()),abstentions=int(g.abstained.sum()),
                unsafe_event_count='' if ref=='R0' else int(g.unsafe.sum()),expected_unsafe_sum=float(g.unsafe.sum()) if ref=='R0' else '',
                expected_unsafe_sum_exact=str(sum(fractions,Fraction(0))) if ref=='R0' else '',
                unsafe_per_attempt=float(g.unsafe.mean()),unsafe_per_execution=float(g.unsafe.sum()/g.executed.sum()) if g.executed.sum() else '',
                unsafe_per_execution_undefined=int(g.executed.sum()==0),empty_reach=int(g.empty_reach.sum()),empty_candidates=int(g.empty_candidates.sum()),
                nonfinite_reach=int(g.nonfinite_reach.sum()),nonfinite_scores=int(g.nonfinite_scores.sum()),
                count_note='R0 sums expectations only; these are not integer unsafe events' if ref=='R0' else 'R1 observed deterministic events'))
            for sid,index in zip(ids['rough' if cond=='rough_iid' else 'nominal'],range(len(arr['regret']))):
                scene_rows.append(dict(reference=ref,model='reference',mask='NA',policy='NA',condition=cond,scene_id=int(sid),fitted_seeds=0,slots=4,
                    **{m:float(arr[m][index].mean()) for m in METRICS}))
        for arch in ['BF-D','KF-D']:
            for mask in [0,1]:
                for policy in ['direct','soft']:
                    for cond in CONDITIONS:
                        a=model_arrays[(arch,mask,policy,cond)];b=reference_arrays[(ref,cond)]
                        for metric in METRICS:
                            delta=a[metric]-b[metric]
                            # R0 exact point preserves rational expected unsafe probabilities;
                            # bootstrap percentile arithmetic uses stable float64 as planned.
                            frac=None
                            if ref=='R0' and metric=='unsafe':
                                g=reference_sources[(ref,cond)]
                                base=[Fraction(int(n),int(d)) for n,d in zip(g.expected_unsafe_numerator,g.expected_unsafe_denominator)]
                                model_frame=a[metric].mean(1).ravel()
                                frac=[Fraction(int(a[metric][i//4,:,i%4].sum()),3)-r for i,r in enumerate(base)]
                            s=stats(delta,cond,metric,ref,'contrast',frac)
                            cached[(ref,arch,mask,policy,cond,metric)]=s;append_result(ref,arch,mask,policy,cond,metric,'contrast',s)
                        for seedidx,seed in enumerate([10,11,12]):
                            per_seed.append(dict(reference=ref,model=arch,mask=mask,policy=policy,condition=cond,seed=seed,
                                **{m:float((a[m][:,seedidx,:]-b[m][:,0,:]).mean()) for m in METRICS},
                                attempts=int(a['regret'].shape[0]*4),reference_independent_fits=0,interpretation='single fitted seed minus same deterministic/analytic policy'))
                        for index,sid in enumerate(ids['rough' if cond=='rough_iid' else 'nominal']):
                            scene_rows.append(dict(reference=ref,model=arch,mask=mask,policy=policy,condition=cond,scene_id=int(sid),fitted_seeds=3,slots=4,
                                **{m:float((a[m][index]-b[m][index]).mean()) for m in METRICS}))
    # Required R1 minus R0 descriptive context, with one policy of each kind.
    reference_pair_rows=[]
    for cond in CONDITIONS:
        a=reference_arrays[('R1',cond)];b=reference_arrays[('R0',cond)]
        g=reference_sources[('R0',cond)]
        base=[Fraction(int(n),int(d)) for n,d in zip(g.expected_unsafe_numerator,g.expected_unsafe_denominator)]
        for metric in METRICS:
            frac=[Fraction(int(a[metric].ravel()[i]))-r for i,r in enumerate(base)] if metric=='unsafe' else None
            s=stats(a[metric]-b[metric],cond,metric,'R0','contrast',frac)
            cached[('R0','R1','NA','fixed_prior',cond,metric)]=s
            append_result('R0','R1','NA','fixed_prior',cond,metric,'contrast',s)
        for i,row in enumerate(g.itertuples()):
            reference_pair_rows.append(dict(condition=cond,scene_id=int(row.scene_id),frame_id=int(row.frame_id),
                **{m+'_R1_minus_R0':float((a[m]-b[m]).ravel()[i]) for m in METRICS}))
    write_csv(out/'R1_minus_R0_per_frame.csv',reference_pair_rows)
    groups=list(dict.fromkeys(k[:4] for k in cached))
    for group in groups:
        ref,arch,mask,policy=group
        for metric in METRICS:
            constituent=[cached[(*group,c,metric)] for c in ['iid','all_s2','rough_iid']]
            draws=sum(s['draws'] for s in constituent)/3
            result=dict(point=sum(s['point'] for s in constituent)/3,
                low=float(np.quantile(draws,.025,method='linear')),high=float(np.quantile(draws,.975,method='linear')),
                estimate_exact='',low_exact='',high_exact='')
            if all(s['estimate_exact'] for s in constituent):
                exact=sum((Fraction(s['estimate_exact']) for s in constituent),Fraction(0))/3
                result.update(point=float(exact),estimate_exact=str(exact))
            if all(s['integer_draws'] is not None for s in constituent):
                denom=math.lcm(*(s['integer_denominator'] for s in constituent))*3
                numer=sum(s['integer_draws']*(denom//(3*s['integer_denominator'])) for s in constituent)
                ci=[frac_quantile(numer,denom,q) for q in Q]
                result.update(low=float(ci[0]),high=float(ci[1]),low_exact=str(ci[0]),high_exact=str(ci[1]))
            append_result(ref,arch,mask,policy,POOL,metric,'absolute' if arch=='reference' else 'contrast',result)
    seed_frame=pd.DataFrame(per_seed)
    pool=seed_frame[seed_frame.condition.isin(['iid','all_s2','rough_iid'])].groupby(['reference','model','mask','policy','seed'])[list(METRICS)].mean().reset_index()
    pool['condition']=POOL;pool['attempts']=1280;pool['reference_independent_fits']=0;pool['interpretation']='fixed1/3 per-condition effect; same reference, not three fits'
    pd.concat([seed_frame,pool],ignore_index=True).to_csv(out/'model_reference_per_seed.csv',index=False)
    write_csv(out/'reference_comparison_intervals.csv',summaries);write_csv(out/'reference_denominators.csv',denoms);write_csv(out/'reference_and_contrast_per_scene.csv',scene_rows)
    paired=model[['architecture','seed','mask','policy','condition','scene_id','frame_id',*METRICS]].merge(
        refs[['reference','condition','scene_id','frame_id',*METRICS]],on=['condition','scene_id','frame_id'],suffixes=('_model','_reference'),validate='many_to_many')
    assert len(paired)==159744
    for m in METRICS:paired[m+'_model_minus_reference']=paired[m+'_model']-paired[m+'_reference']
    paired.to_csv(out/'model_reference_per_frame.csv',index=False)
    sources=[afast/'analysis/bootstrap_multiplicities.npz',out/'reference_per_frame.csv',out/'prior_lock.json']+[afast/'metrics'/f'{arch}_seed{s}.csv' for arch in ['BF-D','KF-D'] for s in [10,11,12]]
    write_json(out/'analysis_provenance.json',dict(status='COMPLETE_POSTHOC_DESCRIPTIVE_REFERENCE_FAMILY',
        source_hashes={str(p):digest(p) for p in sources},analysis_code_sha256=digest(Path(__file__)),
        saved_draws=20000,new_draws=0,interval_quantiles=['1/40','39/40'],interval_method='linear percentile; conditional on the three existing fits; approximate coverage',
        expected_interval_rows=608,actual_interval_rows=len(summaries),paired_frame_rows=len(paired),reference_rows=6656,model_rows=79872,
        deterministic_reference_policy_count_per_reference=1,neural_ensembling=False,
        unsafe_arithmetic='R1 integer event arithmetic, exact Fraction percentiles; R0 exact frame probabilities and point fractions, float64 saved-draw percentiles; no epsilon/snapping',
        pool='iid/all_s2/rough_iid fixed1/3, nominal shared draws retained, rough separate; not pooled record weighting',
        evidence_class=EVIDENCE))
    assert len(summaries)==608
    print(json.dumps({'rows':len(summaries),'paired_frames':len(paired),'per_seed_rows':len(seed_frame)+len(pool)},indent=2))

if __name__=='__main__':main()
