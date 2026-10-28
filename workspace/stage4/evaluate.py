"""Frozen evaluation arithmetic, isolated outputs and gated missing-probability caches."""
import argparse,json,sys
import numpy as np
from stage4.common import ROOT,RESULTS,output_path,sha256
def main():
 import torch
 import kinafford.evaluate as original
 from stage4.models import build_model
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--cache-probabilities',action='store_true');parser.add_argument('--frozen-calibration');args,rest=parser.parse_known_args()
 if '--out' not in rest:raise SystemExit('Stage-4 --out required')
 out=output_path(rest[rest.index('--out')+1],('results_stage4',));tag=rest[rest.index('--tag')+1];seed=rest[rest.index('--seed')+1]
 original.build_model=build_model
 if args.frozen_calibration:
  original.calibrate=lambda *a,**k:json.loads((ROOT/args.frozen_calibration).read_text())['conformal']
 if args.cache_probabilities:
  permission=RESULTS/'audit/cache_inference_authorization.json'
  if 'logs_stage4/preflight' not in str(ROOT):
   if not permission.exists() or not json.loads(permission.read_text()).get('authorized'):raise RuntimeError('Missing explicit authorization for new frozen-checkpoint inference')
  predictor,dataset=original.predict,original.ExcaSceneDataset
  class WithSuite(dataset):
   def __init__(self,root,split,*a,**kw):super().__init__(root,split,*a,**kw);self.suite=split.replace('suite_','')
   def __getitem__(self,i):s=super().__getitem__(i);s['stage4_suite']=self.suite;return s
  def predict(models,batch,amp=True):
   probs=[]
   def capture(module,inputs,outputs):probs.append(torch.softmax(outputs['mat'].float(),1).detach())
   hooks=[model.register_forward_hook(capture) for model in models]
   try:p=predictor(models,batch,amp)
   finally:
    for hook in hooks:hook.remove()
   p['mat_prob']=torch.stack(probs).mean(0).cpu().numpy()
   for i,suite in enumerate(batch['stage4_suite']):
    if suite not in ['cam_s3','all_s3','test_iid']:continue
    path=out/'cache'/f'{tag}_s{seed}'/suite/f'{int(batch["scene_id"][i])}_{int(batch["frame_id"][i])}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError(f'Refusing cache overwrite: {path}')
    values={k:v[i] for k,v in p.items() if v is not None}
    for k in ['gt_H','gt_sem','gt_mat','gt_haz','gt_aff','gt_reach','gt_vis','z_target','scene_id','frame_id']:values[k]=batch[k][i].detach().cpu().numpy()
    np.savez_compressed(path,**values)
   return p
  original.ExcaSceneDataset=WithSuite;original.predict=predict
 sys.argv=[sys.argv[0]]+rest;original.main()
if __name__=='__main__':main()
