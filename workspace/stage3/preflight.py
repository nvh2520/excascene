"""Copy all code to scratch, then pytest and released smoke; frozen trees stay untouched."""
import hashlib,json,shutil,sys
from stage3.common import ROOT,RESULTS,source_hashes,sha256,write_json,utc,protocol_gate,inventory
from stage3.jobs import Runner,Job

def main():
 protocol_gate();hashes=source_hashes();configs={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'configs_stage3').glob('*.yaml'))};fingerprint=hashlib.sha256(json.dumps([hashes,configs],sort_keys=True).encode()).hexdigest()[:12]
 scratch=ROOT/'logs_stage3/preflight'/fingerprint/'scratch';scratch.mkdir(parents=True,exist_ok=True)
 for name in ['kinafford','scripts','configs','tests','stage2','stage3','configs_stage3']:
  shutil.copytree(ROOT/name,scratch/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for src,name in [('stage2/tests/test_interventions.py','test_stage2_interventions.py'),('stage3/tests/test_stage3.py','test_stage3.py')]:shutil.copy2(ROOT/src,scratch/'tests'/name)
 runner=Runner()
 for label,cmd in [('pytest',[sys.executable,'-m','pytest','-q','tests']),('smoke',['bash','scripts/smoke_test.sh']),('stage3_wrappers',[sys.executable,'-m','stage3.smoke'])]:
  failed=runner.run([Job(f'preflight/{label}/{fingerprint}',cmd,cwd=scratch)],'analysis',parallel=1)
  if failed:raise SystemExit(f'Preflight failed: {failed}')
 protected=inventory('after_preflight',full=False)
 if hashes!=source_hashes():raise SystemExit('Checks passed for copied source, but workspace source changed; repeat preflight')
 write_json(RESULTS/'audit/preflight_pass.json',dict(time=utc(),source_hashes=hashes,config_hashes=configs,scratch=str(scratch),protected=protected,commands=['python -m pytest -q tests','bash scripts/smoke_test.sh','python -m stage3.smoke']))
 print('STAGE-3 PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
