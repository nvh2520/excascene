"""Tiny wrapper integration in copied preflight scratch only; no study outputs."""
import json,subprocess,sys
from stage3.common import ROOT

def main():
 if 'logs_stage3/preflight' not in str(ROOT) or ROOT.name!='scratch':raise RuntimeError('Wrapper smoke requires isolated scratch')
 for tag in ['S3B_R2_dropout','S3C_G3_jitter_epoch']:
  run=ROOT/'runs_stage3'/tag/'seed0';subprocess.run([sys.executable,'-m','stage3.train','--variant',tag,'--seed','0','--out',str(run),'--config','configs_stage3/default.yaml','--variants','configs_stage3/variants.yaml','--set','data_root=data_smoke','--set','epochs=2','--set','batch_size=4','--set','train_limit=32','--set','val_limit=16','--set','num_workers=2'],check=True)
  subprocess.run([sys.executable,'-m','stage3.evaluate','--run',str(run),'--tag',tag,'--seed','0','--data','data_smoke','--out','results_stage3','--suites','suite_iid','--limit','4','--save_preds','0','--num_workers','2','--soft'],check=True)
  assert json.loads((run/'done.json').read_text())['steps']>0
  import pandas as pd
  assert len(pd.read_csv(ROOT/'results_stage3'/f'{tag}_s0.csv'))==4
 print('STAGE-3 WRAPPER SMOKE PASSED',flush=True)
if __name__=='__main__':main()
