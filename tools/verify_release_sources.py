#!/usr/bin/env python3
"""Verify the released source and historical weights without equating rebuilt binaries to old ones."""
from pathlib import Path
import csv,hashlib,json
REPO=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def verify():
 rows=list(csv.DictReader((REPO/'docs/source_hashes.csv').open()))
 for r in rows:assert sha(REPO/r['path'])==r['released_sha256'],r['path']
 root=REPO/'workspace/final_evidence_2026_10_20'
 lock=json.loads((root/'protocol/study4_final_lock.json').read_text())
 expected=json.loads((REPO/'docs/lock_file_hashes.json').read_text())
 for rel,h in expected.items():assert sha(REPO/'workspace'/rel)==h,rel
 weights=json.loads((root/'protocol/checkpoint_inventory.json').read_text())['checkpoints']
 for r in weights:
  if r['architecture'] in ['KF-D42','BF-D42'] and r['seed']==10:assert sha(REPO/'workspace'/r['path'])==r['sha256'],r['path']
 assert sha(root/'study4/gate/final.pt')==lock['files']['study4/gate/final.pt']
 restored=json.loads((REPO/'audit_receipt_resolution.json').read_text())['restored']
 assert sha(REPO/restored['path'])==restored['sha256']
 build=json.loads((REPO/'native_build_receipt.json').read_text())
 assert build['status']=='PASS' and build['fresh_compilation']
 for r in build['binaries']:assert sha(REPO/r['path'])==r['sha256'],r['path']
 return {'source_files':len(rows),'historical_checkpoint_locks':'exact for both fixture checkpoints and gate','native_binaries':'verified against this fresh compilation receipt, not the historical binary hashes','original_study4_protocol':expected['final_evidence_2026_10_20/protocol/study4_final_lock.json']}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
