from __future__ import annotations
import hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results_stage3';ANALYSIS=RESULTS/'analysis'
CAPS={'A':1,'B':6,'C':6,'D':7,'E':1,'F':1,'analysis':2}
SUITES=['test_iid','clean','iid','cam_s1','cam_s2','cam_s3','lidar_s1','lidar_s2','lidar_s3','ext_s1','ext_s2','ext_s3','joint_s1','joint_s2','joint_s3','all_s2','all_s3','ood_artic','ood_terrain']
def utc():return datetime.now(timezone.utc).isoformat()
def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def write_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n');os.replace(tmp,path)
def output_path(path,namespaces=('results_stage3','runs_stage3','data_stage3','logs_stage3')):
 path=Path(path).resolve()
 if not any(path.is_relative_to(ROOT/n) for n in namespaces):raise ValueError(f'Output outside Stage 3: {path}')
 return path
def deviation(reason,finding='EXECUTION'):
 RESULTS.mkdir(exist_ok=True)
 with (RESULTS/'DEVIATIONS.md').open('a') as f:f.write(f'\n- {utc()} **{finding}** — {reason}\n')
def protocol_gate():
 a=json.loads((RESULTS/'PREREGISTRATION_ATTESTATION.json').read_text());p=ROOT/a['protocol']
 if sha256(p)!=a['sha256']:raise RuntimeError('Registered protocol changed; require a separately dated amendment')
 committed=subprocess.check_output(['git','show',a['commit']+':'+a['protocol']],cwd=ROOT)
 if hashlib.sha256(committed).hexdigest()!=a['sha256']:raise RuntimeError('Protocol not in attested commit')
 return a
PROTECTED=['results','paper','runs','configs','kinafford','results_stage2','paper_v2','runs_stage2','configs_stage2','stage2','data','data_stage2']
def protected_paths():
 for root in PROTECTED:
  for p in (ROOT/root).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':yield p
 for name in ['PREREGISTRATION.md','docs/stage2_preregistration.md','docs/stage2_audit.md']:
  if (ROOT/name).exists():yield ROOT/name
 # Freeze original orchestration, while new work goes under stage3/ exclusively.
 for p in (ROOT/'scripts').rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':yield p
def inventory(label='before',full=True):
 paths=sorted(set(protected_paths()));dest=RESULTS/'audit/protected_before.json'
 if label=='before':
  if dest.exists():raise RuntimeError('Initial inventory already frozen')
  rows={}
  for i,p in enumerate(paths):
   st=p.stat();rows[str(p.relative_to(ROOT))]={'sha256':sha256(p),'size':st.st_size,'mtime_ns':st.st_mtime_ns}
   if (i+1)%10000==0:print(f'Inventory {i+1}/{len(paths)}',flush=True)
  write_json(dest,dict(time=utc(),files=len(rows),records=rows));return
 base=json.loads(dest.read_text())['records'];now={str(p.relative_to(ROOT)):p for p in paths};changed=[];hashed=0
 for name in base.keys()|now.keys():
  if name not in base or name not in now:changed.append(name);continue
  p=now[name];st=p.stat();row=base[name]
  if full or st.st_size!=row['size'] or st.st_mtime_ns!=row['mtime_ns']:
   hashed+=1
   if sha256(p)!=row['sha256']:changed.append(name)
 report=dict(time=utc(),files=len(base),full_sha256=full,hashed_files=hashed,changed=sorted(changed),pass_check=not changed)
 write_json(RESULTS/f'audit/protected_{label}.json',report)
 if changed:raise RuntimeError(f'Frozen files changed: {changed[:20]}')
 return report
def source_hashes():return {str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'stage3').rglob('*.py'))}
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('label');ap.add_argument('--quick',action='store_true');args=ap.parse_args();inventory(args.label,not args.quick)
