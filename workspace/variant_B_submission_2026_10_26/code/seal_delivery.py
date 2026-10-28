"""Package this completed reporting overlay once; no external action or replay."""
from pathlib import Path
import csv,json,hashlib,shutil,zipfile,time
from datetime import datetime,timezone
O=Path(__file__).resolve().parents[1];P=O.parent;R=O/'release'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def js(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
start=time.time();archive=R/'ExcaScene_variant_B_reporting_overlay.zip'
assert not archive.exists(),'Do not rewrite a sealed archive'
base=P/'final_pre_submission_2026_10_26/release'
expected='3a1f6dcbf47f07e7b5c7431a407dcac195bdaf58cdc12ae772ef6cdcfc3c6d2e'
assert sha(base/'asset_inventory.csv')==expected
shutil.copy2(base/'asset_inventory.csv',R/'base_asset_inventory_UNCHANGED.csv')
prior=R/'prior_receipts_UNCHANGED';prior.mkdir(exist_ok=True)
originals=[base/'final_preparation_receipt.json',base/'source_archive_receipt.json',P/'submission_readiness_2026_10_22/records/C2_job_receipt.json',P/'submission_readiness_2026_10_22/C2/C2_execution_report.md']
copied=[]
for p in originals:
    target=prior/p.name;shutil.copy2(p,target)
    copied.append(dict(source=str(p.relative_to(P)),copy=str(target.relative_to(O)),sha256=sha(target),meaning='Original historical receipt, copied unchanged; not a new reproduction'))
js(R/'base_payload_reference.json',dict(selected_release='B',retired_current_variants=['A','C'],
    base_source='final_pre_submission_2026_10_26/release/github_candidate',
    base_data='final_pre_submission_2026_10_26/release/zenodo_candidate',
    base_inventory_sha256=expected,source_bytes=37193167,data_bytes=60204643254,
    existing_assets_rehashed_this_pass=False,inventory_digest_verified=True,
    intended_code_terms='Apache-2.0, with applicable upstream notices and third-party terms retained',
    intended_data_checkpoint_terms='CC BY 4.0',license_grant_performed=False,
    reviewer_locator='[LINK]',public_locator='[REPOSITORY/DOI]',public_release_timing='on acceptance',
    uploaded=False,contains_redistributed_GOOSE_images=False,prior_receipts=copied))

def include(p):
    rel=p.relative_to(O);parts=rel.parts
    if p.name in ['SHA256SUMS','overlay_file_inventory.csv','current_output_sizes.json','zip_delivery_receipt.json'] or p.suffix=='.zip':return False
    if any('page_renders' in x for x in parts):return False
    if p.suffix in ['.aux','.out','.spl','.blg','.synctex','.pyc']:return False
    if '__pycache__' in parts:return False
    if parts[:2]==('manuscript','code') and p.name!='build_and_render.py':return False
    if parts[0]=='manuscript' and p.suffix=='.log':return False
    if parts[0]=='manuscript' and p.name=='outline.md':return False
    return True
files=sorted(p for p in O.rglob('*') if p.is_file() and include(p))
rows=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
with (R/'overlay_file_inventory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader();w.writerows(rows)
groups={}
for r in rows:
    key=Path(r['path']).parts[0];groups[key]=groups.get(key,0)+r['bytes']
js(R/'current_output_sizes.json',dict(measurement_utc=datetime.now(timezone.utc).isoformat(),
    reporting_overlay_payload_bytes=sum(r['bytes'] for r in rows),payload_files=len(rows),bytes_by_component=groups,
    local_tree_before_archive_bytes=sum(p.stat().st_size for p in O.rglob('*') if p.is_file()),
    scope='Measured uncompressed payload; excludes final inventory/checksum/receipt self-reference, intermediate page-render history and build byproducts. Compression is measured in zip_delivery_receipt.json.'))
files += [R/'overlay_file_inventory.csv',R/'current_output_sizes.json']
checksums=''.join(f'{sha(p)}  {p.relative_to(O)}\n' for p in files)
(R/'SHA256SUMS').write_text(checksums);files.append(R/'SHA256SUMS')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
    for p in files:z.write(p,arcname=str(Path(O.name)/p.relative_to(O)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.infolist())==len(files)
    for name in ['main_submission.pdf','supplement_submission.pdf','cover_letter.pdf']:
        member=f'{O.name}/manuscript/{name}'
        assert hashlib.sha256(z.read(member)).hexdigest()==sha(O/'manuscript'/name)
    uncompressed=sum(i.file_size for i in z.infolist())
receipt=dict(status='LOCAL_VARIANT_B_REPORTING_OVERLAY_SEALED_AND_VERIFIED',created_utc=datetime.now(timezone.utc).isoformat(),
    archive=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
    zip_members=len(files),uncompressed_member_bytes=uncompressed,zip_crc_check='PASS',
    final_pdf_hash_checks='PASS',elapsed_packaging_seconds=time.time()-start,
    base_payload='Required separately; exact existing source/data inventory and original receipts accompany this overlay',
    exclusions='Large base observations/checkpoints, GOOSE-Ex images, intermediate render history; final page PNGs remain locally available and are hashed in report/page_inspection.csv',
    new_neural_forwards=0,new_bootstrap_draws=0,gpu_hours=0,independent_reviews=0,
    uploads=0,external_communications=0,license_grants=0,submissions=0,
    author_fields='Only [LINK], [REPOSITORY/DOI] and the explicit author declarations listed in change_memo.md')
js(R/'zip_delivery_receipt.json',receipt)
print(json.dumps(receipt,indent=2))
