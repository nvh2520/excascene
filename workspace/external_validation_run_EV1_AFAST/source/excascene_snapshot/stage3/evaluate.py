"""Original evaluation arithmetic plus opt-in material probabilities and Stage-3 endpoints."""
import argparse,sys
import numpy as np
from stage3.common import output_path

def main():
 import torch
 import kinafford.evaluate as original
 from stage3.models import build_model
 from stage3.selectors import no_threshold_metrics
 ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--soft',action='store_true');args,rest=ap.parse_known_args()
 if '--out' not in rest:raise SystemExit('Stage-3 --out required')
 out=output_path(rest[rest.index('--out')+1],('results_stage3',));tag=rest[rest.index('--tag')+1];seed=rest[rest.index('--seed')+1]
 original.build_model=build_model
 if args.soft:
  predictor,metric,dataset=original.predict,original.frame_metrics,original.ExcaSceneDataset
  class WithSuite(dataset):
   def __init__(self,root,split,*a,**kw):super().__init__(root,split,*a,**kw);self.suite=split.replace('suite_','')
   def __getitem__(self,i):s=super().__getitem__(i);s['stage3_suite']=self.suite;return s
  def predict(models,batch,amp=True):
   probs=[]
   def capture(module,inputs,output):probs.append(torch.softmax(output['mat'].float(),1).detach())
   handles=[m.register_forward_hook(capture) for m in models]
   try:p=predictor(models,batch,amp)
   finally:
    for h in handles:h.remove()
   p['mat_prob']=torch.stack(probs).mean(0).cpu().numpy()
   for i,suite in enumerate(batch['stage3_suite']):
    if suite not in ['iid','all_s2','ood_terrain']:continue
    path=out/'cache'/f'{tag}_s{seed}'/suite/f'{int(batch["scene_id"][i])}_{int(batch["frame_id"][i])}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError(f'Refusing cache overwrite: {path}')
    values={k:v[i] for k,v in p.items() if v is not None}
    for k in ['gt_H','gt_sem','gt_mat','gt_haz','gt_aff','gt_reach','gt_vis','z_target','scene_id','frame_id','rel_gt','cam_present']:values[k]=batch[k][i].detach().cpu().numpy()
    np.savez_compressed(path,**values)
   return p
  def frame_metrics(pred,gt,conf,lam=1.0):
   values=metric(pred,gt,conf,lam);values.update(no_threshold_metrics(pred,gt));return values
  original.ExcaSceneDataset=WithSuite;original.predict=predict;original.frame_metrics=frame_metrics
 sys.argv=[sys.argv[0]]+rest;original.main()
if __name__=='__main__':main()
