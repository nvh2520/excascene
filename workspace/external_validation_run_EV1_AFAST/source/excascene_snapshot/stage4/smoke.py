import json,subprocess,sys
from stage4.common import ROOT
def main():
 if 'logs_stage4/preflight' not in str(ROOT) or ROOT.name!='scratch':raise RuntimeError('Scratch required')
 tag='S4A_G3_structured';run=ROOT/'runs_stage4'/tag/'seed0'
 subprocess.run([sys.executable,'-m','stage4.train','--variant',tag,'--seed','0','--out',str(run),'--config','configs_stage4/default.yaml','--variants','configs_stage4/variants.yaml','--set','data_root=data_smoke','--set','epochs=2','--set','batch_size=4','--set','train_limit=32','--set','val_limit=16','--set','num_workers=2'],check=True)
 subprocess.run([sys.executable,'-m','stage4.evaluate','--run',str(run),'--tag',tag,'--seed','0','--data','data_smoke','--out','results_stage4','--suites','test_iid','--limit','4','--save_preds','0','--num_workers','2','--cache-probabilities'],check=True)
 assert json.loads((run/'done.json').read_text())['steps']>0
 exposure=json.loads((run/'dropout_exposures.json').read_text());assert sum(x['observations'] for x in exposure['epochs'])==24
 import pandas as pd,numpy as np
 assert len(pd.read_csv(ROOT/'results_stage4'/f'{tag}_s0.csv'))==4
 from stage4.selectors import selector_metrics
 caches=sorted((ROOT/'results_stage4/cache'/f'{tag}_s0/test_iid').glob('*.npz'));assert len(caches)==4
 for file in caches:
  with np.load(file) as c:
   gt={k:c['gt_'+k] for k in ['H','sem','mat','haz','aff','reach','vis']};gt['z_target']=float(c['z_target']);assert len(selector_metrics(c,gt))==5
 print('STAGE-4 WRAPPER SMOKE PASSED: nonzero updates, observed masks, four caches, five selector policies',flush=True)
if __name__=='__main__':main()
