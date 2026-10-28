"""Extract the new local handoff once and verify included identities only."""
from pathlib import Path
import csv,datetime,hashlib,json,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'submission_strengthening'
r=json.loads((S/'records/private_handoff_receipt.json').read_text());archive=ROOT/r['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['sha256']
target=S/'private_handoff_extracted';assert not target.exists();target.mkdir()
with zipfile.ZipFile(archive) as z:
    for n in z.namelist():
        p=Path(n);assert not p.is_absolute() and '..' not in p.parts,n
    z.extractall(target)
rows=list(csv.DictReader((target/'submission_strengthening/private_handoff_manifest.csv').open()))
for item in rows:
    p=target/item['path'];assert p.is_file() and p.stat().st_size==int(item['bytes']),item
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
result=subprocess.run([sys.executable,str(target/'release_candidate/scripts/verify_candidate.py'),'--root',str(target/'release_candidate'),'--output',str(S/'records/extracted_release_payload_verification.json')],capture_output=True,text=True,check=True)
receipt=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='EXTRACTED_PRIVATE_HANDOFF_VERIFIED',archive_sha256=r['sha256'],payload_files=len(rows),extraction=str(target.relative_to(ROOT)),release_candidate=json.loads(result.stdout),neural_forwards=0,generation=0,training=0,excluded_checkpoints_or_runtime_independently_reproduced=False)
(S/'records/private_handoff_delivery_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
