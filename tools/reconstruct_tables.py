#!/usr/bin/env python3
"""Reconstruct Tables 2, 9 and 11 from released observations of decisions and saved draws.

No selector, model, random sampler or generator is called. Output is a separate receipt.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,sys,json,hashlib,re,importlib.util
from fractions import Fraction
import numpy as np,pandas as pd
REPO=Path(__file__).resolve().parents[1];W=REPO/'workspace';F=W/'final_evidence_2026_10_20';S=W/'strengthening_2026_10_18'
sys.path.insert(0,str(F/'code'))
from statistical_core4 import scene_bootstrap,paired,CONDITIONS
from statistical_core import exact_percentile

def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def parse_rows(name):
 text=(REPO/'docs/printed_tables'/name).read_text()
 return [x.strip().split('&') for x in text.splitlines() if '&' in x and not x.startswith(('Policy','Model','\\',' '))]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 sources=[];rows2=[];rows9=[];rows11=[];checks=[]
 def read(p):sources.append(p);return pd.read_csv(p)
 def draws(p):sources.append(p);return load(p)
 d4=draws(F/'study4/analysis/bootstrap_multiplicities.npz')
 model4=pd.concat([read(F/f'study4/metrics/{arch}_seed{seed}.csv') for arch in ['KF-D14','BF-D14','KF-D42','BF-D42'] for seed in [10,11,12]],ignore_index=True)
 refs=read(F/'study4/diagnostics/references/per_frame.csv');c1=read(W/'closing_B_2026_10_27/C1/holdout/per_frame.csv')
 # Table 2 lists attempt-weighted condition means, matching the table's inventory.
 policies=[('Classical local-ground',refs,'classical_ground'),('Classical (post hoc, validation-tuned)',c1,'classical_posthoc'),('KinAfford dynamic (14 epochs)',model4,'KF-D14'),('BEVFusion-Exca (14 epochs)',model4,'BF-D14'),('KinAfford dynamic (42 epochs)',model4,'KF-D42'),('BEVFusion-Exca (42 epochs)',model4,'BF-D42'),('Uniform reachable (expected)',refs,'R0'),('Fixed training prior',refs,'R1')]
 printed=parse_rows('task_value_replication.tex');table2={r[0].strip():[x.strip().removesuffix('\\\\') for x in r[1:]] for r in printed}
 for label,frame,arch in policies:
  sub=frame[frame.architecture==arch]
  if arch.startswith(('KF','BF')):sub=sub[(sub.arm=='keep_all')&(sub.policy=='direct')]
  formatted=[]
  for condition in ['clean','cam_s3','rough_iid']:
   q=sub[sub.condition==condition];assert len(q)>0,(label,condition)
   regret=q.regret.mean();unsafe=q.unsafe.mean();formatted.extend([f'{regret:.4f}',f'{100*unsafe:.2f}'])
   rows2.append(dict(policy=label,condition=condition,regret=regret,unsafe=unsafe,attempts=len(q),uniform_expected=arch=='R0'))
  assert formatted==table2[label],(label,formatted,table2[label]);checks.append(dict(table=2,row=label,printed_match=True))
 # Both four-primary families, with integer event arithmetic from the frozen code.
 spec=importlib.util.spec_from_file_location('first_study_frozen_statistics',S/'code_v2/study3_statistics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 all9=[]
 for study,root,subdir,arches,helper,pair in [('First prospective gate study',S,'study3',['KF-D','BF-D'],m.scene_bootstrap,m.paired),('Replication',F,'study4',['KF-D42','BF-D42'],scene_bootstrap,paired)]:
  frame=model4 if root==F else pd.concat([read(root/f'{subdir}/metrics/{arch}_seed{seed}.csv') for arch in arches for seed in [10,11,12]],ignore_index=True)
  D=d4 if root==F else draws(root/f'{subdir}/analysis/bootstrap_multiplicities.npz');saved=read(root/f'{subdir}/analysis/primary_results.csv');weights={c:Fraction(1,9) for c in CONDITIONS}
  for arch in arches:
   diff=pair(frame,arch,'gate','keep_all');base=frame[(frame.architecture==arch)&(frame.arm=='keep_all')&(frame.policy=='direct')]
   base_est,_=helper(base,'regret',weights,D)
   for metric in ['regret','unsafe']:
    got,_=helper(diff,metric,weights,D,metric=='unsafe');old=saved[(saved.architecture==arch)&(saved.metric==metric)].iloc[0]
    for key in ['estimate','lower95','upper95','lower98_75','upper98_75']:assert abs(got[key]-old[key])<2e-14,(study,arch,metric,key)
    if metric=='unsafe':
     for key in ['exact_estimate','exact_lower98_75','exact_upper98_75']:assert got[key]==old[key]
    relative=100*got['estimate']/base_est['estimate'] if metric=='regret' else None
    rows9.append(dict(study=study,architecture=arch,metric=metric,relative_percent=relative,**got))
    scale,prec=(1,5) if metric=='regret' else (100,3)
    all9.append(([f'{scale*got[k]:+.{prec}f}' for k in ['estimate','lower98_75','upper98_75']],relative,got['class98_75'].lower()))
 t9=[r for r in parse_rows('gate_primary_combined.tex') if r[0].strip() in ['KinAfford dynamic','BEVFusion-Exca']]
 assert len(t9)==len(all9)==8
 for i,(r,(nums,relative,klass)) in enumerate(zip(t9,all9)):
  printed_nums=re.findall(r'[+-]?\d+\.\d+',r[2]);want=[f'{float(x):+.{5 if i%2==0 else 3}f}' for x in printed_nums]
  assert nums==want,(i,nums,want)
  assert klass in r[4]
  if relative is not None:assert f'{relative:+.1f}' in r[3] or f'{relative:.1f}' in r[3]
  checks.append(dict(table=9,row=i,printed_match=True))
 # Table 11 uses scene-level overlapping cause rates already saved, not new geometry.
 safety=read(F/'study4/diagnostics/safety/per_scene.csv');ids=d4['nominal_scene_ids'];Mi=d4['nominal'].astype(np.int64);N=len(ids)
 entries=[('Classical local-ground','classical_ground','classical_ground','reference',4),('Classical (post hoc, validation-tuned)','classical_posthoc','classical_posthoc','reference',4),('Uniform (expected)','R0','uniform_reachable','reference',None),('Training prior','R1','training_prior','reference',4),('KinAfford dynamic; keep-all','KF-D42','keep_all','direct',12),('KinAfford dynamic; gate','KF-D42','gate','direct',12),('BEVFusion-Exca; keep-all','BF-D42','keep_all','direct',12),('BEVFusion-Exca; gate','BF-D42','gate','direct',12)]
 table11={r[0].strip():r[1:] for r in parse_rows('severe_person_safety.tex')}
 for label,arch,arm,policy,den in entries:
  if arch=='classical_posthoc':q=c1[c1.condition=='cam_s3'].groupby('scene_id')[['multi_person_buffer','unsafe']].mean().reindex(ids)
  else:q=safety[(safety.condition=='cam_s3')&(safety.architecture==arch)&(safety.arm==arm)&(safety.policy==policy)].set_index('scene_id').reindex(ids)
  assert q[['multi_person_buffer','unsafe']].notna().all().all(),label
  for j,(metric,scale,prec) in enumerate([('multi_person_buffer',1000,1),('unsafe',100,2)]):
   vals=q[metric].to_numpy(float)
   if den is None:
    point=vals.mean();boot=Mi.astype(float)@vals/N;lo,hi=np.quantile(boot,[.025,.975])
   else:
    counts=np.rint(vals*den).astype(np.int64);assert np.max(abs(counts/den-vals))<1e-12
    b=Mi@counts;point=float(Fraction(int(counts.sum()),den*N));lo=float(exact_percentile(b,den*N,Fraction(1,40)));hi=float(exact_percentile(b,den*N,Fraction(39,40)))
   nums=[f'{scale*x:.{prec}f}' for x in [point,lo,hi]];want=re.findall(r'\d+\.\d+',table11[label][j]);assert nums==want,(label,metric,nums,want)
   rows11.append(dict(policy=label,metric=metric,estimate=point,lower95=lo,upper95=hi,expected_probability=den is None));checks.append(dict(table=11,row=label,metric=metric,printed_match=True))
 for name,rows in [('table2',rows2),('table9',rows9),('table11',rows11)]:pd.DataFrame(rows).to_csv(a.output/(name+'.csv'),index=False)
 receipt=dict(status='PASS',tables=[2,9,11],primary_families=2,new_bootstrap_draws=0,neural_forwards=0,truth_rescoring=0,printed_checks=len(checks),checks=checks,sources=[dict(path=p.relative_to(REPO).as_posix(),sha256=sha(p)) for p in dict.fromkeys(sources)])
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks','sources']},indent=2))
if __name__=='__main__':main()
