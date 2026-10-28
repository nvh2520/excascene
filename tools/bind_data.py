#!/usr/bin/env python3
"""Link extracted deposit artifacts into the unchanged historical source layout.

All links stay inside the checkout. Scientific files are never overwritten.
"""
from pathlib import Path
import argparse,hashlib,json
REPO=Path(__file__).resolve().parents[1]
PREFIXES={
 'holdout/replication':'final_evidence_2026_10_20/study4/holdout',
 'holdout/first_gate_study':'strengthening_2026_10_18/study3/holdout',
 'holdout/external_study':'external_validation_run_EV1_AFAST',
 'holdout/reference_study':'revision_v5/validation_run',
 'validation/original':'data/val','validation/first_gate':'strengthening_2026_10_18/study3/validation',
 'validation/replication':'final_evidence_2026_10_20/study4/validation',
 'training_split/observations':'data/train',
 'training_split/material_probe_replay':'strengthening_2026_10_18/phase_a/material_replay'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=REPO/'data');a=p.parse_args();data=a.data.resolve()
 assert data.is_relative_to(REPO),'Extract inside the fresh checkout, e.g. data/'
 n=0
 for src in sorted(data.rglob('*')):
  if not src.is_file():continue
  rel=src.relative_to(data).as_posix();dst=None
  for pre in ['checkpoints/workspace/','analysis/workspace/','protocols/workspace/','decisions/workspace/','real_image_check/workspace/','fixtures/workspace/']:
   if rel.startswith(pre):dst=REPO/'workspace'/rel[len(pre):];break
  if dst is None:
   for pre,target in PREFIXES.items():
    if rel.startswith(pre+'/'):dst=REPO/'workspace'/target/rel[len(pre)+1:];break
  if dst is None and rel.startswith('fixtures/') and '/' not in rel[len('fixtures/'):]:dst=REPO/rel[len('fixtures/'):]
  if dst is None:continue
  assert '..' not in dst.relative_to(REPO).parts
  if dst.exists():
   if dst.resolve()==src:continue
   assert sha(dst)==sha(src),('Conflicting released artifact',dst)
   continue
  dst.parent.mkdir(parents=True,exist_ok=True)
  import os
  dst.symlink_to(os.path.relpath(src,dst.parent));n+=1
 print(json.dumps({'linked_files':n,'data':str(data.relative_to(REPO)),'latent_files_required':False}))
if __name__=='__main__':main()
