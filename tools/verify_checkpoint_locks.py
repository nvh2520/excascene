#!/usr/bin/env python3
"""Verify every deposited checkpoint against its existing historical digest."""
from pathlib import Path
import csv,hashlib,json,argparse
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 rows=list(csv.DictReader((R/'docs/checkpoint_inventory.csv').open()))
 for r in rows:
  file=R/'workspace'/r['source'];assert file.resolve().is_relative_to(R/'data'),r['source']
  actual=sha(file);assert actual==r['sha256'],r['source'];r['verified_from_extracted_deposit']=True
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='PASS',files=len(rows),checkpoints=rows),indent=2)+'\n');print('PASS',len(rows),'checkpoint files from extracted deposit')
if __name__=='__main__':main()
