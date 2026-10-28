"""Committed, preflight-gated sequential Stage-3 orchestration."""
import argparse,json,sys
from stage3.common import ROOT,RESULTS,ANALYSIS,SUITES,protocol_gate,source_hashes,write_json,sha256,utc,deviation
from stage3.jobs import Runner,Job
from stage3.registry import BLOCK_TAGS

def gate():
 protocol_gate();receipt=json.loads((RESULTS/'audit/preflight_pass.json').read_text())
 if receipt['source_hashes']!=source_hashes():raise RuntimeError('Implementation changed since preflight; repeat pytest then isolated smoke')
 for name,digest in receipt['config_hashes'].items():
  if sha256(ROOT/name)!=digest:raise RuntimeError('Configuration changed since preflight')

def train_job(tag,seed):
 out=ROOT/'runs_stage3'/tag/f'seed{seed}'
 command=[sys.executable,'-m','stage3.train','--variant',tag,'--seed',str(seed),'--out',str(out),'--config','configs_stage3/default.yaml','--variants','configs_stage3/variants.yaml']
 return Job(f'train/{tag}/seed{seed}',command,sentinel=str(out/'done.json'),archive=[str(out)],train_dir=str(out))

def eval_job(tag,seed,original=None):
 run=ROOT/('runs' if original else 'runs_stage3')/(original or tag)/f'seed{seed}';out=RESULTS/('A' if original else '')
 provenance=RESULTS/'audit/checkpoints'/f'{tag}_s{seed}.json'
 receipt=dict(tag=tag,seed=seed,run=str(run.relative_to(ROOT)),checkpoint_sha256=sha256(run/'best.pt'),config_sha256=sha256(run/'config.json'))
 if provenance.exists():assert json.loads(provenance.read_text())==receipt
 else:write_json(provenance,receipt)
 command=[sys.executable,'-m','stage3.evaluate','--run',str(run),'--tag',tag,'--seed',str(seed),'--out',str(out),'--data','data','--batch_size','16','--num_workers','6','--save_preds','0','--suites']+(['iid','all_s2','ood_terrain'] if original else SUITES)
 command=[('suite_'+x if x in SUITES and x!='test_iid' else x) for x in command]
 if original:command+=['--soft']
 return Job(f'eval/{tag}/seed{seed}',command,sentinel=str(out/f'{tag}_s{seed}.json'),archive=[str(out/f'{tag}_s{seed}.csv'),str(out/f'{tag}_s{seed}.json'),str(out/'cache'/f'{tag}_s{seed}')])

def task(runner,key,module,block='analysis',args=(),sentinel=None):
 failed=runner.run([Job(key,[sys.executable,'-m',module,*args],sentinel=str(sentinel) if sentinel else None)],block,parallel=1)
 if failed:
  deviation(f'Incomplete analysis job(s): {failed}; controller stopped for reconciliation.','INCOMPLETE')
  raise RuntimeError(f'Analysis prerequisite failed: {failed}')
 return not failed

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--through',choices=list('ABCDEF'),default='F');ap.add_argument('--finalize',action='store_true');args=ap.parse_args();gate();runner=Runner()
 if not args.finalize:
  for block in 'ABCDEF':
   if any(e['kind']=='block_finish' and e.get('block')==block for e in runner.state['events']):continue
   gate();runner.begin_block(block);failed=[]
   if block=='A':
    jobs=[eval_job(tag,seed,original) for tag,original in [('S3A_R2','R2_full'),('S3A_G3','G3_dyn')] for seed in range(3)];failed=runner.run(jobs,block)
    task(runner,'A/replay','stage3.replay',block)
   elif block in 'BCD':
    seeds=[3,4] if block=='D' else list(range(3));tags=BLOCK_TAGS[block]
    failed=runner.run([train_job(tag,s) for tag in tags for s in seeds],block)
    jobs=[eval_job(tag,s) for tag in tags for s in seeds if (ROOT/'runs_stage3'/tag/f'seed{s}/done.json').exists()];failed+=runner.run(jobs,block)
    if block=='D':task(runner,'D/replication','stage3.replication',block)
   elif block=='E':task(runner,'E/sensitivity','stage3.sensitivity',block)
   elif block=='F':task(runner,'F/feasibility','stage3.feasibility',block)
   if failed:deviation(f'S3-{block} unavailable jobs: {failed}','INCOMPLETE')
   task(runner,f'report/{block}','stage3.analyse',args=['--block',block]);runner.event('block_finish',block=block,failed=failed)
   if block==args.through:break
 if args.finalize or args.through=='F':
  task(runner,'final/analysis','stage3.analyse',args=['--final'])
  task(runner,'final/protection','stage3.common',args=['final'])
  task(runner,'final/audit','stage3.final_audit')
  task(runner,'final/summary','stage3.summary')
 runner.save();print('STAGE-3 CONTROLLER FINISHED; REAL EXPERIMENTS NOT AUTHORIZED',flush=True)
if __name__=='__main__':main()
