"""C1 descriptive summaries from the one selected evaluation; saved draws only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import sys,json,time
from fractions import Fraction
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parents[1];P=O.parent;F=P/'final_evidence_2026_10_20'
sys.path.insert(0,str(F/'code'))
sys.path.insert(0,str(P/'external_validation_run_EV1_AFAST/code'))
from common import js,sha
from statistical_core import exact_percentile

def float_ci(values,draws):
    return dict(estimate=float(np.mean(values)),lower95=float(np.quantile(draws,.025)),upper95=float(np.quantile(draws,.975)),
                lower95_exact='',upper95_exact='',undefined_draws=0)
def event_ci(counts,den_per_scene,M):
    counts=np.asarray(counts,dtype=np.int64);den=int(den_per_scene*len(counts));draws=M@counts
    lo=exact_percentile(draws,den,Fraction(1,40));hi=exact_percentile(draws,den,Fraction(39,40))
    return dict(estimate=float(Fraction(int(counts.sum()),den)),lower95=float(lo),upper95=float(hi),
                lower95_exact=str(lo),upper95_exact=str(hi),undefined_draws=0),draws/den
def sign(v):return 'negative' if v<0 else 'positive' if v>0 else 'zero'
def interval_class(lo,hi):return 'LOWER' if hi<0 else 'HIGHER' if lo>0 else 'INCONCLUSIVE'

def main():
    start=time.time();assert not (O/'C1/analysis_receipt.json').exists()
    receipt=json.loads((O/'C1/holdout/receipt.json').read_text());assert receipt['records']==5120
    drawfile=F/'study4/analysis/bootstrap_multiplicities.npz'
    with np.load(drawfile) as z:D={k:z[k] for k in z.files}
    raw=pd.read_csv(O/'C1/holdout/per_frame.csv');existing=pd.read_csv(F/'study4/analysis/all_per_scene_results.csv')
    absrows=[];paired=[];scenerows=[]
    event_metrics=['unsafe','executed','abstained','multi_person_buffer','multi_obstacle_buffer','multi_truck_buffer','multi_machine_footprint_buffer','multi_non_terrain']
    event_metrics+=['exclusive_'+x for x in ['person_buffer','obstacle_buffer','truck_buffer','machine_footprint_buffer','non_terrain']]
    for cond in ['clean','cam_s3','rough_iid']:
        pop='rough' if cond=='rough_iid' else 'nominal';ids=D[pop+'_scene_ids'];Mi=D[pop].astype(np.int64);M=Mi.astype(float);N=len(ids)
        g=raw[raw.condition==cond];group=g.groupby('scene_id');assert (group.size()==4).all()
        s=group[['regret',*event_metrics,'excluded_reachable_cells','reachable_cells']].mean().reindex(ids);assert s.notna().all().all()
        for sid,r in s.iterrows():scenerows.append(dict(condition=cond,scene_id=int(sid),**r.to_dict()))
        vals=s.regret.to_numpy();rd=M@vals/N
        absrows.append(dict(condition=cond,metric='regret',attempts=len(g),events='',**float_ci(vals,rd)))
        for metric in event_metrics:
            counts=group[metric].sum().reindex(ids).to_numpy();assert np.array_equal(counts,np.rint(counts))
            ci,_=event_ci(counts,4,Mi)
            absrows.append(dict(condition=cond,metric=metric,attempts=len(g),events=int(counts.sum()),**ci))
        unsafe=group.unsafe.sum().reindex(ids).to_numpy(np.int64);executed=group.executed.sum().reindex(ids).to_numpy(np.int64)
        event_draw=Mi@unsafe;execution_draw=Mi@executed;undefined=int((execution_draw==0).sum())
        rate=float(unsafe.sum()/executed.sum()) if executed.sum() else None
        if undefined:lo=hi=None
        else:lo,hi=[float(x) for x in np.quantile(event_draw/execution_draw,[.025,.975])]
        absrows.append(dict(condition=cond,metric='unsafe_per_execution',attempts=int(executed.sum()),events=int(unsafe.sum()),estimate=rate,lower95=lo,upper95=hi,lower95_exact='',upper95_exact='',undefined_draws=undefined))
        excl=group.excluded_reachable_cells.sum().reindex(ids).to_numpy();reach=group.reachable_cells.sum().reindex(ids).to_numpy()
        excluded_draw=(M@excl)/(M@reach);lo,hi=np.quantile(excluded_draw,[.025,.975])
        absrows.append(dict(condition=cond,metric='excluded_reachable_share',attempts=int(reach.sum()),events=int(excl.sum()),estimate=float(excl.sum()/reach.sum()),lower95=float(lo),upper95=float(hi),lower95_exact='',upper95_exact='',undefined_draws=0))
        for arch in ['KF-D42','BF-D42']:
            learned=existing[(existing.architecture==arch)&(existing.arm=='keep_all')&(existing.policy=='direct')&(existing.condition==cond)].set_index('scene_id').reindex(ids)
            assert len(learned)==N and learned.regret.notna().all()
            lv=learned.regret.to_numpy();ld=M@lv/N;diff=lv-vals
            paired.append(dict(condition=cond,architecture=arch,contrast='learned_minus_posthoc_classical',metric='regret',**float_ci(diff,ld-rd)))
            learned_counts=np.rint(learned.unsafe.to_numpy()*12).astype(np.int64);assert np.max(np.abs(learned_counts/12-learned.unsafe))<1e-12
            cc=learned_counts-3*unsafe;ci,_=event_ci(cc,12,Mi)
            paired.append(dict(condition=cond,architecture=arch,contrast='learned_minus_posthoc_classical',metric='unsafe',**ci))
            if cond=='clean':
                undefined=int((rd<=0).sum());ratio=ld/rd if not undefined else None
                paired.append(dict(condition=cond,architecture=arch,contrast='learned_divided_by_posthoc_classical',metric='regret',estimate=float(lv.mean()/vals.mean()) if vals.mean()>0 else None,
                    lower95=float(np.quantile(ratio,.025)) if ratio is not None else None,upper95=float(np.quantile(ratio,.975)) if ratio is not None else None,
                    lower95_exact='',upper95_exact='',undefined_draws=undefined))
    pd.DataFrame(absrows).to_csv(O/'C1/holdout/condition_results.csv',index=False)
    pd.DataFrame(paired).to_csv(O/'C1/holdout/paired_comparisons.csv',index=False)
    pd.DataFrame(scenerows).to_csv(O/'C1/holdout/per_scene.csv',index=False)

    # All learned sensitivity values are read, not re-estimated or re-scored.
    aggfile=P/'variant_B_submission_2026_10_26/K1/per_scene_aggregates.npz'
    with np.load(aggfile) as z:A={k:z[k] for k in z.files}
    ids=D['nominal_scene_ids'];assert np.array_equal(A['scene_ids'][:len(ids)],ids)
    keys={tuple(k):i for i,k in enumerate(A['keys'])};settings=[str(x) for x in A['settings']]
    ss=pd.concat([pd.read_csv(O/f'C1/sensitivity/scenes/{int(s)}.csv') for s in ids],ignore_index=True)
    assert len(ss)==len(ids)*4*27
    summary=ss.groupby(['setting','scene_id'])[['regret','unsafe']].mean().reset_index()
    summary.to_csv(O/'C1/sensitivity/posthoc_classical_per_scene.csv',index=False)
    classical=summary.pivot(index='setting',columns='scene_id',values='regret').reindex(index=settings,columns=ids).to_numpy()
    assert np.isfinite(classical).all();M=D['nominal'].astype(float);N=len(ids);cboot=M@classical.T/N;out=[]
    for arch in ['KF-D42','BF-D42']:
        for policy in ['direct','soft']:
            lv=A['values'][keys[(arch,'keep_all',policy,'clean','regret')],:,:N];assert lv.shape==classical.shape
            diff=lv-classical;boot=M@diff.T/N
            for j,setting in enumerate(settings):
                ci=float_ci(diff[j],boot[:,j]);out.append(dict(setting=setting,architecture=arch,policy=policy,condition='clean',
                    contrast='learned_minus_posthoc_classical',metric='regret',learned_mean=float(lv[j].mean()),classical_mean=float(classical[j].mean()),
                    sign=sign(ci['estimate']),class95=interval_class(ci['lower95'],ci['upper95']),**ci))
    oc=pd.DataFrame(out);oc.to_csv(O/'C1/sensitivity/learned_comparisons.csv',index=False)
    compact=pd.read_csv(P/'variant_B_submission_2026_10_26/K1/compact_sensitivity_table.csv')
    for arch in ['KF-D42','BF-D42']:
        for policy in ['direct','soft']:
            rows=oc[(oc.architecture==arch)&(oc.policy==policy)].set_index('setting')
            compact[f'{arch}_{policy}_minus_posthoc_classical_clean_regret']=compact.setting.map(rows.sign)
    compact.to_csv(O/'C1/sensitivity/compact_sensitivity_table_extended.csv',index=False)
    sources=[O/'protocol/C1_lock.json',O/'C1/validation/selection_lock.json',O/'C1/holdout/per_frame.csv',drawfile,
        F/'study4/analysis/all_per_scene_results.csv',aggfile,P/'variant_B_submission_2026_10_26/K1/compact_sensitivity_table.csv']
    js(O/'C1/analysis_receipt.json',dict(status='COMPLETE_POSTHOC_DESCRIPTIVE',seconds=time.time()-start,
        saved_draws=20000,new_draws=0,interval_level=.95,quantiles=['1/40','39/40'],coverage='Approximate marginal percentile coverage; descriptive, not a new primary family',
        seed_handling='Three fixed learned seeds averaged within scene; four slots per scene for the classical policy',event_arithmetic='Integer per-scene events and integer multiplicities; fractional quantiles retained',
        sources={str(p.relative_to(P)):sha(p) for p in sources},sensitivity_choices='Fixed selected classical clean choices, original learned aggregates read unchanged',
        sensitivity_lower_point_count=int((oc.sign=='negative').sum()),sensitivity_comparisons=len(oc),gpu_hours=0,neural_forwards=0))
    print(pd.DataFrame(paired).to_string(index=False),flush=True)
if __name__=='__main__':main()
