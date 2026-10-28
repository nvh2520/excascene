"""Verify the final private package after extraction without replaying experiments."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'AFAST_reporting_closure'
receipt=json.loads((C/'records/private_review_package_receipt.json').read_text())
archive=ROOT/receipt['path']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
out=C/'package_verification/extracted'
out.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        target=(out/name).resolve()
        assert target.is_relative_to(out.resolve())
    assert z.testzip() is None
    z.extractall(out)
run=subprocess.run([sys.executable,str(out/'AFAST_reporting_closure/code/verify_review_bundle.py'),'--root',str(out)],capture_output=True,text=True)
assert run.returncode==0,run.stdout+run.stderr
verified=json.loads(run.stdout)
assert verified['status']=='AVAILABLE_PAYLOAD_VERIFIED'
record=dict(status='PRIVATE_DELIVERY_VERIFIED',package_sha256=receipt['sha256'],package_bytes=archive.stat().st_size,
            extraction='AFAST_reporting_closure/package_verification/extracted',portable_verifier=verified,
            new_neural_forwards=0,new_bootstrap_draws=0,statistical_recalculation_in_this_check=False,
            independently_reproduced_training_or_inference=False)
(C/'records/private_delivery_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
