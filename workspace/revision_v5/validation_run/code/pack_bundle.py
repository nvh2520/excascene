"""Build compact review ZIP after all scientific jobs and inspections finish.

The metered package job is sealed afterwards with its final ledger/resource snapshot.
"""
from common import *
import zipfile,shutil

def main():
    locked()
    target=RUN/'evidence_bundle.zip'
    assert not target.exists(), 'Preserve earlier bundles; do not silently overwrite'
    dest=RUN/'legacy/original_F6';dest.mkdir(exist_ok=True)
    for p in (ROOT/'revision_v5/inputs/original_F6').iterdir():
        if p.is_file():shutil.copyfile(p,dest/p.name)
    readme='''# ExcaScene / KinAfford-Ex bounded validation evidence

Start with validation_results.md and manuscript_impact_memo.md. The four questions,
models, populations, selectors, seeds and statistics are fixed in
validation_protocol_amended.md, manifests/analysis_lock.json and the preserved
controlling instruction under original_prompt1/.

To independently reconstruct all four headline effects and interval endpoints:

    python code/inspect_headlines.py .

This requires Python, NumPy and pandas only. It reads headline_frame_metrics.csv,
primary_results.csv, manifests/generation_manifest.json and the saved scene
bootstrap multiplicities. No workstation paths, model weights, repository imports,
GPU or network are required for that check. logs/inspect_headlines.log records the
same check performed during execution. This verifies statistical reconstruction,
not independent model inference or physical validity.

The ZIP includes paired scalar effects, integer event counts, per-seed results,
bootstrap replicates, secondary/mechanism results, precision summaries and replay,
all 49 original tests/families, original and corrected F.6 exports, source/checkpoint
and input hashes, preflight tests, run manifests, author reply and conditional
blocked reports. Ledger and resource_summary.json are final accounting snapshots.

Use claim_evidence_matrix_corrected_v2.csv and both matrix change logs for corrected
historical attribution. claim_evidence_matrix_results.csv adds current outcomes and
actions. The original locked matrix remains unchanged. Read all four errata in
reports/: legacy comparator, rough visibility reference, manuscript references,
and resource-monitor recovery of the unchanged inference process.
The original preflight/report language describes its creation time and is superseded
only by these explicit versioned corrections; it is not retroactively rewritten.

Large sensor/latent arrays, full prediction/precision maps, model weights and the
historical repository remain preserved locally and are omitted from this compact
bundle. Their digests and source versions are recorded. The full inference scripts
require those artifacts and their original repository layout; only the headline
inspection script is deliberately standalone. The bundle is an author review
artifact, not a public data/model release or a manuscript submission.

The conditional PP-LiteSeg and raw-geometry branches are BLOCKED on documented
records. No replacement model, inferred calibration, training or new annotation
was added. See author_reply_record.json for the actual user's limited confirmation;
historical human-account questions remain open.

bundle_file_hashes.csv lists SHA-256 for the included scientific files. The final
ledger/resource snapshot and container digest are recorded at sealing; no recursive
self-hash is implied. An external .sha256 file records the finished ZIP digest.
'''
    (RUN/'EVIDENCE_README.md').write_text(readme)
    excluded={'ledger.jsonl','resource_summary.json','live_usage.json','bundle_file_hashes.csv'}
    paths=[p for p in RUN.iterdir() if p.is_file() and p.suffix in ['.md','.csv','.json','.npz'] and p.name not in excluded]
    for directory in ['code','manifests','logs','legacy','reports','conditional','original_prompt1','generation']:
        paths += [p for p in (RUN/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p != RUN/'logs/bundle.log']
    paths += [p for p in (RUN/'precision').iterdir() if p.is_file()]
    paths=sorted(set(paths)); rows=[dict(path=str(p.relative_to(RUN)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
    csvwrite(RUN/'bundle_file_hashes.csv',rows)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,str(p.relative_to(RUN)))
        z.write(RUN/'bundle_file_hashes.csv','bundle_file_hashes.csv')
    # Zip integrity checks compressed members without extracting large copies.
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    event('evidence_bundle_prepared',path=target.name,scientific_files=len(paths),bytes=target.stat().st_size,final_ledger_snapshot_pending=True)
    print('Evidence bundle prepared',len(paths),'files',target.stat().st_size,'bytes',flush=True)

if __name__=='__main__':main()
