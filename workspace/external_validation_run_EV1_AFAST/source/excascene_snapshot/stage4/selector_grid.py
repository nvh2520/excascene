"""Read complete cached maps, retain missing coverage explicitly, never infer models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from stage4.common import ROOT,RESULTS,ANALYSIS,sha256,write_json,utc
from stage4.selectors import selector_metrics,POLICIES
SUITES=['iid','all_s2','ood_terrain','cam_s3','all_s3','test_iid']
def cache_paths(tag,seed,suite):
 old=ROOT/'results_stage3/A/cache'/f'{"S3A_R2" if tag=="R2_full" else "S3A_G3"}_s{seed}'/suite
 new=RESULTS/'B/cache'/f'{"S4B_R2_cache" if tag=="R2_full" else "S4B_G3_cache"}_s{seed}'/suite
 existing=list(old.glob('*.npz'));fresh=list(new.glob('*.npz'))
 if existing and fresh:raise ValueError('Duplicate old/new cache sources for a suite')
 return sorted(existing or fresh)
def run():
 rows=[];frame_rows=[];hashes={};replays=[]
 for tag in ['R2_full','G3_dyn']:
  for seed in range(3):
   frozen=ROOT/'results'/f'{tag}_s{seed}.csv';data=pd.read_csv(frozen);hashes[str(frozen.relative_to(ROOT))]=sha256(frozen)
   for suite in SUITES:
    files=cache_paths(tag,seed,suite);expected=2000 if suite=='test_iid' else 960;source=data[data.suite==suite]
    if len(files) not in [0,expected]:raise ValueError(f'Partial cache {tag}/{seed}/{suite}: {len(files)}/{expected}')
    if not files:
     for policy in POLICIES:
      endpoint={'direct':'direct','direct_risk_lambda1':'risk','hard_composed':'comp'}.get(policy)
      rows.append(dict(tag=tag,seed=seed,suite=suite,policy=policy,observations=len(source) if endpoint else 0,mean_regret=source['regret_'+endpoint].mean() if endpoint else np.nan,unsafe_count=int(source['unsafe_'+endpoint].sum()) if endpoint else np.nan,unsafe_fraction=source['unsafe_'+endpoint].mean() if endpoint else np.nan,fallback_count=np.nan,no_reachable_count=np.nan,tie_frame_count=np.nan,excess_tied_maxima=np.nan,status='HISTORICAL ENDPOINTS; DIAGNOSTICS UNAVAILABLE' if endpoint else 'MISSING PROBABILITY CACHE'))
     continue
    actual=[]
    for p in files:
     hashes[str(p.relative_to(ROOT))]=sha256(p)
     with np.load(p) as c:
      pred={k:c[k] for k in ['H','H_sigma','sem_prob','mat_label','haz_prob','aff','aff_sigma','mat_prob']};gt={k:c['gt_'+k] for k in ['H','sem','mat','haz','aff','reach','vis']};gt['z_target']=float(c['z_target']);scene=int(c['scene_id']);frame=int(c['frame_id'])
      actual.extend(dict(tag=tag,seed=seed,suite=suite,scene_id=scene,frame_id=frame,**r) for r in selector_metrics(pred,gt))
    d=pd.DataFrame(actual);assert not d.duplicated(['scene_id','frame_id','policy']).any();frame_rows.extend(actual)
    for policy,g in d.groupby('policy'):
     rows.append(dict(tag=tag,seed=seed,suite=suite,policy=policy,observations=len(g),mean_regret=g.regret.mean(),unsafe_count=int(g.unsafe.sum()),unsafe_fraction=g.unsafe.mean(),fallback_count=int(g.fallback.sum()),no_reachable_count=int(g.no_reachable.sum()),tie_frame_count=int(g.tie_frame.sum()),excess_tied_maxima=int(g.excess_tied_maxima.sum()),status='COMPLETE CACHED MAPS'))
     endpoint={'direct':'direct','direct_risk_lambda1':'risk','hard_composed':'comp'}.get(policy)
     if endpoint:
      j=g.merge(source,on=['scene_id','frame_id'],suffixes=('_grid','_frozen'),validate='one_to_one');assert len(j)==expected
      for metric,frozen_column in [('regret','regret_'+endpoint),('unsafe','unsafe_'+endpoint)]:
       delta=np.abs(j[metric]-j[frozen_column]);replays.append(dict(tag=tag,seed=seed,suite=suite,policy=policy,endpoint=metric,frames=len(j),max_abs_deviation=float(delta.max()),unequal_values=int((delta!=0).sum())))
    print(f'Selector grid {tag} seed {seed} {suite} complete',flush=True)
 result=pd.DataFrame(rows);result.to_csv(ANALYSIS/'selector_grid_per_seed.csv',index=False)
 overall=[]
 for keys,g in result.groupby(['tag','suite','policy'],sort=False):
  sums={k:g[k].sum(min_count=3) for k in ['observations','unsafe_count','fallback_count','no_reachable_count','tie_frame_count','excess_tied_maxima']}
  overall.append(dict(zip(['tag','suite','policy'],keys))|dict(mean_regret=g.mean_regret.mean(),unsafe_fraction=g.unsafe_fraction.mean(),**sums,seeds=','.join(map(str,sorted(g.seed))),status='COMPLETE CACHED MAPS' if (g.status=='COMPLETE CACHED MAPS').all() else '; '.join(sorted(g.status.unique()))))
 pd.DataFrame(overall).to_csv(ANALYSIS/'selector_grid.csv',index=False);pd.DataFrame(frame_rows).to_csv(ANALYSIS/'selector_grid_per_frame.csv',index=False);pd.DataFrame(replays).to_csv(ANALYSIS/'selector_replay.csv',index=False)
 write_json(RESULTS/'audit/selector_input_hashes.json',hashes)
 missing=result[result.status!='COMPLETE CACHED MAPS'][['tag','seed','suite','policy','status']].to_dict('records');write_json(ANALYSIS/'selector_grid_status.json',dict(time=utc(),rows=len(result),aggregate_rows=len(overall),missing=missing,status='COMPLETE' if not missing else 'INCOMPLETE MISSING CACHES',registered_tests=0))
 print('SELECTOR GRID '+('COMPLETE' if not missing else 'INCOMPLETE: MISSING CACHES'),flush=True)
if __name__=='__main__':run()
