"""Descriptive sensitivity of fixed replication choices; existing multiplicities only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import sys,json,time,math
from fractions import Fraction
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1];P=OUT.parent;F=P/'final_evidence_2026_10_20'
sys.path.insert(0,str(F/'code'))
from common import sha,load_npz,js,csvwrite
from statistical_core4 import CONDITIONS,interval_class,exact_percentile
from existing_grid import settings
GRID=settings();D=load_npz(F/'study4/analysis/bootstrap_multiplicities.npz')
IDS=np.r_[D['nominal_scene_ids'],D['rough_scene_ids']];WHERE={int(s):i for i,s in enumerate(IDS)};N=len(D['nominal_scene_ids']);R=len(D['rough_scene_ids']);A={};COUNT={}
def get(arch,arm,policy,condition,metric):
    x=A[(arch,arm,policy,condition,metric)];ids=slice(N,None) if condition=='rough_iid' else slice(None,N)
    assert np.isfinite(x[:,ids]).all();return x[:,ids]
def sign(x):return 'negative' if x<0 else 'positive' if x>0 else 'zero'
def main():
    start=time.time();assert json.loads((OUT/'R1/scoring_receipt.json').read_text())['status']=='COMPLETE'
    for i,sid in enumerate(IDS):
        d=load_npz(OUT/f'R1/scenes/{sid}.npz');frame=pd.DataFrame({k:d[k] for k in ['architecture','arm','policy','condition']})
        for key,idx in frame.groupby(list(frame.columns),sort=False).indices.items():
            COUNT.setdefault(key,np.zeros(len(IDS),int))[i]=len(idx)
            for m in ['regret','unsafe','oracle']:
                A.setdefault((*key,m),np.full((len(GRID),len(IDS)),np.nan))[:,i]=d[m][:,idx].mean(axis=1)
        if (i+1)%128==0:print('reduced',i+1,flush=True)
    keys=sorted(A);np.savez_compressed(OUT/'R1/per_scene_aggregates.npz',scene_ids=IDS,keys=np.asarray(keys),values=np.stack([A[k] for k in keys]),settings=np.array([s['setting'] for s in GRID]))
    primary=[];replicates={};nomM=D['nominal'].astype(float);roughM=D['rough'].astype(float)
    for arch in ['KF-D42','BF-D42']:
        for metric in ['regret','unsafe']:
            contrasts={c:get(arch,'gate','direct',c,metric)-get(arch,'keep_all','direct',c,metric) for c in CONDITIONS}
            nominal=sum(contrasts[c] for c in CONDITIONS if c!='rough_iid')/len(CONDITIONS);rough=contrasts['rough_iid']/len(CONDITIONS)
            point=nominal.mean(1)+rough.mean(1);reps=nomM@nominal.T/N+roughM@rough.T/R
            replicates[arch+'_'+metric]=reps
            for si,s in enumerate(GRID):
                qs=np.quantile(reps[:,si],[.025,.975,1/160,159/160],method='linear');extra={}
                if metric=='unsafe':
                    # Every retained scene has four slots and three fixed seeds.
                    for c in CONDITIONS:
                        idx=slice(N,None) if c=='rough_iid' else slice(None,N)
                        assert np.all(COUNT[(arch,'gate','direct',c)][idx]==12)
                    nn=np.rint(nominal[si]*108).astype(np.int64);nr=np.rint(rough[si]*108).astype(np.int64)
                    assert np.allclose(nn,nominal[si]*108,atol=1e-10) and np.allclose(nr,rough[si]*108,atol=1e-10)
                    den=math.lcm(108*N,108*R)
                    nums=(D['nominal'].astype(np.int64)@nn)*(den//(108*N))+(D['rough'].astype(np.int64)@nr)*(den//(108*R))
                    exactpoint=Fraction(int(nn.sum()),108*N)+Fraction(int(nr.sum()),108*R)
                    exact=[exact_percentile(nums,den,q) for q in [Fraction(1,40),Fraction(39,40),Fraction(1,160),Fraction(159,160)]]
                    assert np.allclose(qs,[float(v) for v in exact],atol=1e-12);qs=np.array([float(v) for v in exact]);point[si]=float(exactpoint)
                    extra=dict(exact_estimate=str(exactpoint),exact_lower98_75=str(exact[2]),exact_upper98_75=str(exact[3]))
                primary.append(dict(setting=s['setting'],architecture=arch,policy='direct',contrast='gate_minus_keep_all',pool='equal9',metric=metric,estimate=point[si],sign=sign(point[si]),lower95=qs[0],upper95=qs[1],lower98_75=qs[2],upper98_75=qs[3],class98_75=interval_class(qs[2],qs[3]),**extra))
    csvwrite(OUT/'R1/primary_sensitivity.csv',primary);np.savez_compressed(OUT/'R1/primary_replicates_saved_draws.npz',**replicates)
    old=pd.read_csv(F/'study4/analysis/primary_results.csv');baseline=pd.DataFrame(primary).query("setting=='nominal'")
    checks=[]
    for row in baseline.to_dict('records'):
        r=old[(old.architecture==row['architecture'])&(old.metric==row['metric'])].iloc[0]
        delta=max(abs(float(row[k])-float(r[k])) for k in ['estimate','lower98_75','upper98_75'])
        assert delta<1e-8 and row['class98_75']==r.class98_75
        checks.append(dict(architecture=row['architecture'],metric=row['metric'],max_absolute_difference=delta,class_preserved=True))
    descriptive=[]
    for policy in ['direct','soft']:
        specs=[('KF_severe_gate_minus_keep','KF-D42','gate','KF-D42','keep_all','cam_s3')]
        specs += [('BF42_minus_KF42','BF-D42','keep_all','KF-D42','keep_all',c) for c in ['clean','cam_s3','rough_iid']]
        for label,a,aa,b,ba,c in specs:
            for metric in ['regret','unsafe']:
                v=get(a,aa,policy,c,metric)-get(b,ba,policy,c,metric)
                for si,s in enumerate(GRID):descriptive.append(dict(setting=s['setting'],policy=policy,contrast=label,condition=c,metric=metric,estimate=float(v[si].mean()),sign=sign(float(v[si].mean()))))
        for arch in ['KF-D42','BF-D42']:
            v=get(arch,'keep_all',policy,'clean','regret')-get('classical_ground','classical_ground','reference','clean','regret')
            for si,s in enumerate(GRID):descriptive.append(dict(setting=s['setting'],policy=policy,contrast=arch+'_minus_classical',condition='clean',metric='regret',estimate=float(v[si].mean()),sign=sign(float(v[si].mean()))))
    csvwrite(OUT/'R1/descriptive_signs.csv',descriptive)
    pf=pd.DataFrame(primary);df=pd.DataFrame(descriptive);summary=[]
    for s in GRID:
        row=dict(s);g=pf[pf.setting==s['setting']]
        for r in g.to_dict('records'):row[r['architecture']+'_'+r['metric']]=r['sign']+'; '+r['class98_75']
        for r in df[(df.setting==s['setting'])&(df.policy=='direct')].to_dict('records'):row[r['contrast']+'_'+r['condition']+'_'+r['metric']]=r['sign']
        summary.append(row)
    csvwrite(OUT/'R1/compact_sensitivity_table.csv',summary)
    changes=[]
    for f,keys,fields,kind in [(pf,['architecture','metric'],['sign','class98_75'],'primary'),(df,['policy','contrast','condition','metric'],['sign'],'descriptive')]:
        base=f[f.setting=='nominal'].set_index(keys)
        for r in f[f.setting!='nominal'].to_dict('records'):
            b=base.loc[tuple(r[k] for k in keys)]
            for field in fields:
                if r[field]!=b[field]:changes.append(dict(kind=kind,setting=r['setting'],identity='|'.join(str(r[k]) for k in keys),field=field,nominal=b[field],changed=r[field]))
    csvwrite(OUT/'R1/sign_class_changes.csv',changes)
    js(OUT/'R1/analysis_receipt.json',dict(status='COMPLETE',settings=len(GRID),primary_rows=len(primary),descriptive_rows=len(descriptive),changed_rows=len(changes),nominal_reconstruction=checks,saved_multiplicity_sha256=sha(F/'study4/analysis/bootstrap_multiplicities.npz'),draw_count=len(D['nominal']),new_draws=0,intervals='ordinary95 and adjusted98.75 marginal percentile, approximate coverage; reporting-stage sensitivity does not replace locked primary family',wall_seconds=time.time()-start,gpu_hours=0))
    print(json.dumps(dict(primary_changes=[r for r in changes if r['kind']=='primary'],direct_changes=[r for r in changes if r['kind']=='descriptive' and r['identity'].startswith('direct')]),indent=2),flush=True)
if __name__=='__main__':main()
