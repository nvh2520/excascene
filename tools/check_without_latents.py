#!/usr/bin/env python3
"""CPU evaluator replay: four fixture records plus one additional released holdout map."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,argparse,json,csv
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];F=R/'workspace/final_evidence_2026_10_20'
sys.path.insert(0,str(F/'code'))
from common import load_npz,sha
from neural import decisions

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 assert not list((R/'data').glob('**/latents/*.npz')),'This check requires a latent-free extracted deposit'
 fixture=json.loads((R/'fixture_manifest.json').read_text());expected=pd.read_csv(R/'expected_decisions.csv');rows=[]
 for x in fixture:
  e=expected[(expected.architecture=='KF-D42')&(expected.scene_id==x['scene_id'])&(expected.frame_id==x['frame_id'])&(expected.arm=='keep_all')]
  d=load_npz(F/x['data']);pred=load_npz(F/e.iloc[0].prediction_path)
  for policy,met in decisions(pred,d).items():
   want=e[e.policy==policy].iloc[0]
   for k in ['pick_row','pick_col','unsafe','executed','abstained']:assert met[k]==want[k],(x,policy,k)
   assert abs(met['regret']-want.regret)<2e-5
   rows.append(dict(kind='fixture',scene_id=x['scene_id'],frame_id=x['frame_id'],policy=policy,exact_selection_and_events=True))
 metrics=pd.read_csv(F/'study4/metrics/KF-D42_seed10.csv');fixturekeys={(x['scene_id'],x['frame_id']) for x in fixture};sample=None
 for rel in sorted((F/'study4/predictions/KF-D42_seed10').rglob('*.npz')):
  q=metrics[(metrics.prediction_path==rel.relative_to(F).as_posix())&(metrics.arm=='keep_all')]
  if len(q) and (int(q.iloc[0].scene_id),int(q.iloc[0].frame_id)) not in fixturekeys:sample=(rel,q);break
 assert sample is not None,'Need an existing figure-frame prediction beyond the four-record fixture'
 rel,q=sample;e=q.iloc[0];files=pd.read_csv(F/'study4/holdout/files.csv');r=files[(files.condition==e.condition)&(files.scene_id==e.scene_id)&(files.frame_id==e.frame_id)].iloc[0]
 for policy,met in decisions(load_npz(rel),load_npz(F/'study4/holdout'/r.path)).items():
  want=q[q.policy==policy].iloc[0]
  for k in ['pick_row','pick_col','unsafe','executed','abstained']:assert met[k]==want[k]
  assert abs(met['regret']-want.regret)<2e-5
  rows.append(dict(kind='additional_holdout_sample',scene_id=int(e.scene_id),frame_id=int(e.frame_id),condition=e.condition,policy=policy,exact_selection_and_events=True))
 receipt=dict(status='PASS',neural_forwards=0,latent_files_present=False,fixture_records=4,additional_holdout_records=1,comparisons=rows)
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
