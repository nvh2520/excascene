"""Predeclared descriptive scene-aggregate and shared-scene sign-flip sensitivities."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from stage3.common import ROOT,ANALYSIS,write_json,sha256
from stage3.stats import holm_fixed

def scene_statistic(paired):
 return paired.groupby(['suite','scene_id'],sort=True)[['a','b']].mean().assign(difference=lambda x:x.a-x.b)

def cluster_ci(scene,n_boot=2000):
 sums=scene.groupby('scene_id').difference.agg(['sum','count']);rng=np.random.default_rng(0);v=sums['sum'].to_numpy();n=sums['count'].to_numpy();values=[]
 for _ in range(n_boot):
  pick=rng.integers(len(sums),size=len(sums));values.append(v[pick].sum()/n[pick].sum())
 return tuple(np.percentile(values,[2.5,97.5]))

def scene_signflip(scene,n_permutations=10000,seed=0):
 sums=scene.groupby('scene_id').difference.sum().to_numpy();observed=float(scene.difference.mean())
 if not np.any(sums):return 1.0
 rng=np.random.default_rng(seed);count=0
 for start in range(0,n_permutations,500):
  signs=2*rng.integers(0,2,size=(min(500,n_permutations-start),len(sums)),dtype=np.int8)-1
  values=(signs@sums)/len(scene);count+=int((values<=observed).sum())
 return (count+1)/(n_permutations+1)

def run():
 source=ROOT/'results_stage2/analysis/paired_difference_histograms/paired_differences.csv';data=pd.read_csv(source);rows=[]
 for stage,path in [('Stage 1',ROOT/'results/analysis/paired_tests.csv'),('Stage 2',ROOT/'results_stage2/analysis/registered_tests.csv')]:
  registry=pd.read_csv(path)
  for r in registry.itertuples():
   paired=data[(data.stage==stage)&(data.id==r.hypothesis)]
   if len(paired)!=r.n_pairs or not np.isclose((paired.a-paired.b).mean(),r.mean_diff,rtol=1e-10,atol=1e-12):raise ValueError('Frozen pairing export mismatch')
   scene=scene_statistic(paired);ci=cluster_ci(scene);d=scene.difference.to_numpy()
   rows.append(dict(stage=stage,id=r.hypothesis,registered_supported=bool(r.supported),paired_frames=len(paired),scene_suite_pairs=len(scene),unique_scenes=scene.index.get_level_values('scene_id').nunique(),mean_a=scene.a.mean(),mean_b=scene.b.mean(),mean_diff=d.mean(),ci_lo=ci[0],ci_hi=ci[1],scene_wilcoxon_p=wilcoxon(d,alternative='less').pvalue if np.any(d!=0) else 1.0,scene_signflip_p=scene_signflip(scene),permutations=10000,permutation_seed=0,status='DESCRIPTIVE'))
 out=pd.DataFrame(rows)
 for _,idx in out.groupby('stage',sort=False).groups.items():
  for method in ['scene_wilcoxon','scene_signflip']:
   out.loc[idx,method+'_holm_p']=holm_fixed(out.loc[idx,method+'_p'])
   out.loc[idx,method+'_supported']=(out.loc[idx,method+'_holm_p']<.05)&(out.loc[idx,'ci_hi']<0)
   out.loc[idx,method+'_concordant']=out.loc[idx,method+'_supported']==out.loc[idx,'registered_supported']
 out.to_csv(ANALYSIS/'cluster_sensitivity_concordance.csv',index=False)
 write_json(ANALYSIS/'cluster_sensitivity_provenance.json',dict(source=str(source.relative_to(ROOT)),sha256=sha256(source),families=[14,19],permutations=10000,seed=0,shared_sign_unit='scene_id across suites',registered_verdicts_unchanged=True))
 print(out.groupby('stage')[['scene_wilcoxon_concordant','scene_signflip_concordant']].sum().to_string(),flush=True)
if __name__=='__main__':run()
