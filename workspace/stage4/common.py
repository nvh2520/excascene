from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results_stage4';ANALYSIS=RESULTS/'analysis'
CAPS={'A':4,'B':1,'reporting':1,'q2':.25,'analysis':1.75}
from stage3.common import SUITES
def utc():return datetime.now(timezone.utc).isoformat()
def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def write_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n');os.replace(tmp,path)
def output_path(path,namespaces=('results_stage4','runs_stage4','logs_stage4')):
 p=Path(path).resolve()
 if not any(p.is_relative_to(ROOT/n) for n in namespaces):raise ValueError(f'Output outside authorized namespace: {p}')
 return p
def deviation(reason,finding='EXECUTION'):
 with (RESULTS/'DEVIATIONS.md').open('a') as f:f.write(f'\n- {utc()} **{finding}** — {reason}\n')
def protocol_gate():
 a=json.loads((RESULTS/'PREREGISTRATION_ATTESTATION.json').read_text())
 for key,hashkey in [('protocol','sha256'),('q2_rubric','q2_rubric_sha256')]:
  if sha256(ROOT/a[key])!=a[hashkey]:raise RuntimeError('Registered document changed')
  data=subprocess.check_output(['git','show',a['commit']+':'+a[key]],cwd=ROOT)
  if hashlib.sha256(data).hexdigest()!=a[hashkey]:raise RuntimeError('Document not in attested commit')
 return a
def source_hashes():return {str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'stage4').rglob('*.py'))}
def protected_paths():
 from stage3.common import protected_paths as previous
 yield from previous()
 for name in ['stage3','configs_stage3','runs_stage3','results_stage3','logs_stage3']:
  for p in (ROOT/name).rglob('*'):
   if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
   relative=str(p.relative_to(ROOT))
   if relative=='results_stage3/REPORT.md' or relative.startswith(('results_stage3/report_completion/','results_stage3/q2/')):continue
   yield p
 for p in (ROOT/'audit').glob('stage3*'):
  if p.is_file():yield p
def inventory(label='before',full=True):
 dest=RESULTS/'audit/protected_before.json';paths={str(p.relative_to(ROOT)):p for p in sorted(set(protected_paths()))}
 if label=='before':
  if dest.exists():raise RuntimeError('Initial protection inventory already exists')
  prior=json.loads((ROOT/'results_stage3/audit/protected_before.json').read_text())['records']
  closing=json.loads((ROOT/'results_stage3/audit/protected_final.json').read_text());assert closing['full_sha256'] and closing['pass_check']
  rows={};reused=0;hashed=0
  for i,(name,p) in enumerate(paths.items()):
   st=p.stat();old=prior.get(name)
   if old and old['size']==st.st_size and old['mtime_ns']==st.st_mtime_ns:digest=old['sha256'];reused+=1
   else:digest=sha256(p);hashed+=1
   rows[name]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns,sha256=digest,mode='prefix' if name=='results_stage3/DEVIATIONS.md' else 'full')
   if (i+1)%10000==0:print(f'Inventory {i+1}/{len(paths)}',flush=True)
  write_json(dest,dict(time=utc(),files=len(rows),hashed_files=hashed,reused_from_verified_stage3_custody=reused,prior_closing_receipt_sha256=sha256(ROOT/'results_stage3/audit/protected_final.json'),records=rows));return
 base=json.loads(dest.read_text())['records'];changed=[];hashed=0
 for name in base.keys()|paths.keys():
  if name not in base or name not in paths:changed.append(name);continue
  p=paths[name];r=base[name];st=p.stat()
  if r['mode']=='prefix':
   with p.open('rb') as f:digest=hashlib.sha256(f.read(r['size'])).hexdigest()
   hashed+=1
   if st.st_size<r['size'] or digest!=r['sha256']:changed.append(name)
  elif full or st.st_size!=r['size'] or st.st_mtime_ns!=r['mtime_ns']:
   hashed+=1
   if sha256(p)!=r['sha256']:changed.append(name)
 report=dict(time=utc(),files=len(base),full_sha256=full,hashed_files=hashed,changed=sorted(changed),pass_check=not changed,allowed_append_only='results_stage3/DEVIATIONS.md')
 write_json(RESULTS/f'audit/protected_{label}.json',report)
 if changed:raise RuntimeError(f'Frozen content changed: {changed[:20]}')
 return report
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--quick',action='store_true');a=p.parse_args();inventory(a.label,not a.quick)
