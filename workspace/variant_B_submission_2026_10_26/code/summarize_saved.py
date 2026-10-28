"""Descriptive K2--K5 summaries. Existing observations, means and draws only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import sys,json,time,math,shutil
from fractions import Fraction
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parents[1];P=O.parent;F=P/'final_evidence_2026_10_20'
sys.path.insert(0,str(F/'code'))
from common import load_npz,js,csvwrite,sha
from statistical_core4 import CONDITIONS,exact_percentile

def k3():
    src=F/'study4/analysis/all_per_scene_results.csv';sf=pd.read_csv(src)
    D=load_npz(F/'study4/analysis/bootstrap_multiplicities.npz');ids=D['nominal_scene_ids'];M=D['nominal'].astype(float)
    rows=[];replicates={}
    for arch in ['KF-D42','BF-D42']:
        d=sf[(sf.architecture==arch)&(sf.policy=='direct')].set_index(['arm','condition','scene_id'])
        get=lambda arm,c,m:d.loc[(arm,c),m].reindex(ids).to_numpy()
        clean=get('keep_all','clean','regret');cr=float(clean.mean());crboot=M@clean/len(ids);assert (crboot>0).all()
        for metric in ['regret','unsafe']:
            delta={c:get('gate',c,metric)-get('keep_all',c,metric) for c in ['clean','cam_s3']}
            for percent in [1,5,10,25,50]:
                p=Fraction(percent,100);v=(1-float(p))*delta['clean']+float(p)*delta['cam_s3']
                point=float(v.mean());reps=M@v/len(ids);extra={}
                if metric=='unsafe':
                    c=np.rint(delta['clean']*12).astype(np.int64);s=np.rint(delta['cam_s3']*12).astype(np.int64)
                    assert np.allclose(c/12,delta['clean']) and np.allclose(s/12,delta['cam_s3'])
                    nums=(p.denominator-p.numerator)*c+p.numerator*s;den=12*p.denominator*len(ids)
                    drawnums=D['nominal'].astype(np.int64)@nums
                    exact=Fraction(int(nums.sum()),den);qs=[exact_percentile(drawnums,den,q) for q in [Fraction(1,40),Fraction(39,40)]]
                    point=float(exact);reps=drawnums/den;lo,hi=map(float,qs)
                    extra={'exact_estimate':str(exact),'exact_lower95':str(qs[0]),'exact_upper95':str(qs[1])}
                else:lo,hi=np.quantile(reps,[.025,.975],method='linear')
                rel=100*point/cr;relboot=100*reps/crboot;rlo,rhi=np.quantile(relboot,[.025,.975],method='linear')
                rows.append(dict(architecture=arch,severe_prevalence_percent=percent,metric=metric,estimate=point,lower95=lo,upper95=hi,clean_keep_regret=cr,percent_of_clean_regret=rel,percent_clean_lower95=rlo,percent_clean_upper95=rhi,normalization_note='Within-regret relative loss' if metric=='regret' else 'Unsafe-probability / clean-regret normalization; cross-unit ratio, not percent change in unsafe rate',**extra))
                replicates[f'{arch}_{metric}_{percent}']=reps
    csvwrite(O/'K3/prevalence_mixtures.csv',rows);np.savez_compressed(O/'K3/replicates_existing_draws.npz',**replicates)
    js(O/'K3/receipt.json',dict(status='COMPLETE',source=str(src.relative_to(P)),source_sha256=sha(src),multiplicities_sha256=sha(F/'study4/analysis/bootstrap_multiplicities.npz'),new_draws=0,weights='Clean 1-p; Severe camera p; same nominal scene identity and fixed seeds',interval='ordinary95 descriptive marginal percentile, approximate coverage',relative_interval='paired bootstrap mixture contrast divided by same-draw clean keep-all regret',gpu_hours=0))

def k5():
    src=P/'strengthening_2026_10_18/phase_a/classical';lock=json.loads((src/'validation_lock.json').read_text())
    assert sha(src/'validation_grid.csv')==lock['grid_sha256']
    d=pd.read_csv(src/'validation_grid.csv');assert len(d)==30
    shutil.copy2(src/'validation_grid.csv',O/'K5/validation_grid_original.csv');shutil.copy2(src/'validation_lock.json',O/'K5/validation_lock_original.json')
    d.sort_values(['tau','buffer']).to_csv(O/'K5/validation_grid.csv',index=False)
    boundary=[]
    for tau,g in d.groupby('tau'):
        g=g.sort_values('buffer');last=g.iloc[-1];prev=g.iloc[-2]
        boundary.append(dict(tau=tau,previous_buffer=float(prev.buffer),boundary_buffer=float(last.buffer),previous_regret=float(prev.regret),boundary_regret=float(last.regret),last_step_regret_change=float(last.regret-prev.regret),still_falling=bool(last.regret<prev.regret),monotone_nonincreasing_over_grid=bool((np.diff(g.regret)<=0).all())))
    csvwrite(O/'K5/boundary_trend.csv',boundary)
    js(O/'K5/receipt.json',dict(status='EXISTING_GRID_REPORTED',source=str((src/'validation_grid.csv').relative_to(P)),source_sha256=sha(src/'validation_grid.csv'),validation_frames=lock['validation_records'],grid_points=len(d),frozen_tau=lock['tau'],frozen_buffer=lock['buffer'],all_taus_still_falling=all(r['still_falling'] for r in boundary),new_scoring=False,new_tuning=False,new_test_evaluation=False))

def k2():
    files=sorted((O/'K2/scenes').glob('*.csv'));d=pd.concat([pd.read_csv(f) for f in files if not f.stem.endswith('_population')],ignore_index=True)
    pop=pd.concat([pd.read_csv(f) for f in files if f.stem.endswith('_population')],ignore_index=True)
    assert len(pop)==3072 and len(pop.drop_duplicates(['scene_id','frame_id']))==3072
    groups=['architecture','arm','policy'];cols=['safe_available','zero_oracle','unsafe','multi_person_buffer','executed','abstained']
    counts=[]
    for key,g in d.groupby(groups,sort=True):
        for pool,conditions in [('clean',['clean']),('cam_s3',['cam_s3']),('equal9',CONDITIONS)]:
            gs=g[g.condition.isin(conditions)].copy();assert set(gs.condition)==set(conditions)
            weights=gs.groupby('condition').condition.transform('size');gs['attempt_weight']=1/len(conditions)/weights
            w=gs.attempt_weight.to_numpy();assert abs(w.sum()-1)<1e-12
            unsafe=gs.unsafe.to_numpy();person=gs.multi_person_buffer.to_numpy()
            base=dict(zip(groups,key));base.update(pool=pool,attempt_rows=len(gs),selection_kind='analytic_expectation' if key[0]=='R0' else 'deterministic',unsafe_rate=float(w@unsafe),person_rate=float(w@person),execution_rate=float(w@gs.executed.to_numpy()),abstentions=int(gs.abstained.sum()))
            for stratum,flag in [('safe_present',gs.safe_available.to_numpy()==1),('no_safe',gs.safe_available.to_numpy()==0),('oracle_positive',gs.zero_oracle.to_numpy()==0),('oracle_zero',gs.zero_oracle.to_numpy()==1)]:
                den=float(w@flag);nu=float(w@(unsafe*flag));npers=float(w@(person*flag))
                base[stratum+'_attempt_share']=den;base[stratum+'_attempt_rows']=int(flag.sum())
                base[stratum+'_unsafe_rate']=nu/den if den else None;base[stratum+'_person_rate']=npers/den if den else None
                base[stratum+'_unsafe_count_or_expectation']=float(unsafe[flag].sum());base[stratum+'_person_count_or_expectation']=float(person[flag].sum())
                base[stratum+'_status']='DEFINED' if den else 'UNDEFINED_NO_ATTEMPTS'
            safe=gs.safe_available.to_numpy()==1
            base['unsafe_share_with_safe_available']=float(w@(unsafe*safe))/base['unsafe_rate'] if base['unsafe_rate'] else None
            base['unsafe_share_status']='DEFINED' if base['unsafe_rate'] else 'UNDEFINED_NO_UNSAFE_EVENTS'
            counts.append(base)
    csvwrite(O/'K2/stratified_safety.csv',counts)
    prows=[]
    for pool in ['clean','cam_s3','equal9']:
        nom=pop[pop.scene_id<1210000];rough=pop[pop.scene_id>=1210000]
        for flag in ['safe_available','zero_oracle']:
            mean=float(nom[flag].mean()) if pool!='equal9' else float(8/9*nom[flag].mean()+1/9*rough[flag].mean())
            prows.append(dict(pool=pool,flag=flag,attempt_share=mean,nominal_positive=int(nom[flag].sum()),nominal_frames=len(nom),rough_positive=int(rough[flag].sum()),rough_frames=len(rough)))
    csvwrite(O/'K2/population_support.csv',prows)
    js(O/'K2/receipt.json',dict(status='COMPLETE',scenes=768,unique_pose_frames=len(pop),saved_selection_rows=len(d),summary_rows=len(counts),pools='Clean, Severe camera, equal nine-condition pool; seed and slot averages within scenes',denominators='Rates within each stratum use weighted stratum attempts. Raw count/expectation columns are separate from equal-condition rates.',R0='Expected probabilities and fractional event counts, never fabricated integer selections',no_safe_definition='No reachable terrain cell outside the union of all buffers',zero_oracle_definition='Exactly zero maximum gt_aff on finite gt_reach >=0.05; no tolerance or chosen threshold',new_selections=0,new_draws=0,gpu_hours=0))

def k4():
    d=pd.concat([pd.read_csv(f) for f in sorted((O/'K4/scenes').glob('*.csv'))],ignore_index=True)
    keys=['condition','scope','architecture','arm','policy','input_state'];vals=['denominator','camera_supported','lidar_supported','either_supported']
    out=[]
    for seed_scope in ['all_seeds','each_seed']:
        ks=keys+(['seed'] if seed_scope=='each_seed' else [])
        for key,g in d.groupby(ks,sort=True):
            r=dict(zip(ks,key));r['seed_scope']=seed_scope
            if seed_scope=='all_seeds':r['seed']='pooled' if r['scope']=='selected_cell' else 'not_applicable'
            den=int(g.denominator.sum());r['denominator']=den;r['frame_or_attempt_rows']=len(g)
            for m in ['camera','lidar','either']:
                r[m+'_supported']=int(g[m+'_supported'].sum());r[m+'_share']=r[m+'_supported']/den if den else None
            out.append(r)
    csvwrite(O/'K4/support_summary.csv',out)
    assert not d[(d.arm=='always_mask')|(d.input_state=='all_cameras_masked_support_only')].camera_supported.any()
    masked=d[(d.arm=='always_mask')|(d.input_state=='all_cameras_masked_support_only')]
    assert np.array_equal(masked.lidar_supported,masked.either_supported)
    js(O/'K4/summary_receipt.json',dict(status='COMPLETE',raw_support_rows=len(d),summary_rows=len(out),nominal_scenes=512,slots=4,projection='Same nominal calibration and reported joints as the existing geometric projection; no interpolation or neural computation',support_estimand='Cell-weighted candidate coverage; attempt-weighted support at selected/oracle cells',conditions=['clean','cam_s3'],all_masked='Both actual always-mask decisions and support-only camera removal at unchanged keep-all/gate cells are separately labeled',intervals='Descriptive counts and fractions; no new interval family',gpu_hours=0,new_draws=0))

if __name__=='__main__':
    for action in sys.argv[1:]:globals()[action]();print(action,'COMPLETE',flush=True)
