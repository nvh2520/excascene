"""Create one private handoff from the explicit authorized output trees."""
from pathlib import Path
import csv,datetime,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'submission_strengthening'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
for name in ('main_page_inspection.json','appendix_page_inspection.json','targeted_candidate_check.json'):
    assert (S/'records'/name).is_file(),name
archive=S/'ExcaScene_SUBMISSION_CANDIDATE_PRIVATE.zip'
assert not archive.exists(),'Never rewrite a sealed archive'
files=set()
for dirname in ('release_candidate','paper_submission_candidate'):
    for p in (ROOT/dirname).rglob('*'):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
        if p.suffix in ('.aux','.out','.toc','.spl','.blg'):continue
        files.add(p)
for name in ('task_value_report.md','material_and_support_diagnostics.md','reproduction_report.md','descriptive_reference_plan.md','publication_numbers_submission_candidate.json','registry_source_resolution.csv','claim_change_memo.md','chronology_crosswalk.csv','access_rights_inventory.md','scientific_readiness.md','FINAL_REPORT.md'):
    files.add(S/name)
for dirname in ('code','references','resources'):
    for p in (S/dirname).rglob('*'):
        if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:files.add(p)
for p in (S/'material/complete_v1').iterdir():
    if p.is_file():files.add(p)
for name in ('source_review.md','source_review_hashes.json','convention_clarification_v1.json'):
    files.add(S/'material'/name)
# These are newly authored/actual technical receipts, not private author accounts.
for p in (S/'records').iterdir():
    if p.is_file() and p.suffix in ('.json','.csv') and p.name!='private_handoff_receipt.json':files.add(p)
for p in (S/'reproduction').glob('*.json'):files.add(p)
for rel in ('checkpoint_smoke64_recovery1/checkpoint_reproduction.json','generator_smoke7/generator_reproduction.json'):
    files.add(S/'reproduction'/rel)
files.add(S/'reproduction/package_finalize.py')
old=[]
for rel,expected in [
    ('external_validation_run_EV1_AFAST/EV1_AFAST_evidence.zip','0f5e2a4ebd3bd415efb5c153b88e68485e91c4df481e28121616a434e4846ec4'),
    ('AFAST_reporting_closure/ExcaScene_v5_2_PRIVATE_author_review.zip','14b5d0a02952c4c2cc4e8bce77118ef80200990483b03f82fcb3762ce8d68712')]:
    path=ROOT/rel;actual=digest(path);assert actual==expected,rel
    old.append(dict(path=rel,bytes=path.stat().st_size,sha256=actual,original_archive_unchanged=True,new_package_contains_archive=False))
identity=S/'records/preserved_archive_identity.json';identity.write_text(json.dumps(old,indent=2)+'\n');files.add(identity)
manifest=[]
for p in sorted(files):
    assert p.is_file() and not p.is_symlink(),p
    manifest.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)))
mp=S/'private_handoff_manifest.csv'
with mp.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(manifest)
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
    z.write(mp,str(mp.relative_to(ROOT)))
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=digest(archive),payload_files=len(files),payload_bytes=sum(r['bytes'] for r in manifest),manifest=str(mp.relative_to(ROOT)),status='LOCAL_PRIVATE_HANDOFF_CREATED',public_release_authorized=False,full_checkpoints_included=False,full_datasets_included=False,original_archives=old,resource_note='Pre-seal totals included; post-delivery totals and extraction receipt are separate sidecars so the archive remains immutable.')
(S/'records/private_handoff_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
