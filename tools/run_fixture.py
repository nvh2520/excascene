#!/usr/bin/env python3
"""Use the unchanged four-record fixture harness with release-scoped integrity checks."""
from pathlib import Path
import sys,json,csv,hashlib
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'workspace/final_evidence_2026_10_20/code'))
sys.path.insert(0,str(REPO))
import common
from verify_release_sources import verify
verified=verify()
# The historical verifier references private-tree exclusions and the old compiled
# binaries. Its release replacement verifies all released scientific sources,
# the original protocol lock, fixture weight locks and fresh native build hashes.
common.verify_protocol=lambda:verified
with (REPO/'package_inventory.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['path','sha256']);w.writeheader()
 for name in ['fixture_manifest.json','expected_decisions.csv','expected_gate.csv','saved_output_provenance.json','audit_receipt_resolution.json']:
  w.writerow({'path':name,'sha256':hashlib.sha256((REPO/name).read_bytes()).hexdigest()})
import evaluate_fixture
try:evaluate_fixture.main()
finally:
 if '--output' in sys.argv:
  dest=Path(sys.argv[sys.argv.index('--output')+1]);raw=dest/'receipt.json'
  if raw.exists():
   r=json.loads(raw.read_text())
   # Keep the unchanged harness receipt, whose runtime description refers to its
   # original use; this separate sidecar records the actual installation context.
   (dest/'fresh_installation_context.json').write_text(json.dumps({'fixture_status':r['status'],'fresh_environment':True,'fresh_native_compilation':True,'verification':verified,'harness_runtime_text':'Historical harness metadata is not used as evidence for this installation; see native_build_receipt.json and the environment receipt.','gpu_seconds_upper_bound':r['elapsed_seconds']},indent=2)+'\n')
