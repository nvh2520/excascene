"""Verify the delivered archive and close actual job costs; no neural forwards.

Run after the scientific closure job has finished. The ZIP is an immutable
snapshot; this verification receipt and post-closure costs remain beside it.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def elapsed():
    return max(time.time() - clock['t0_unix'],
               time.monotonic() - clock['t0_monotonic']) / 3600


clock = read(ROOT / 'records/start_clock.json')
assert read(ROOT / 'final_status.json')['status'] == 'COMPLETE'
assert elapsed() < 36
events = [json.loads(line) for line in (ROOT / 'resources/jobs.jsonl').read_text().splitlines()]
finished = [e for e in events if e['event'] == 'finish']
closure = [e for e in finished if e['name'] == 'final_scientific_report_and_evidence']
assert len(closure) == 1 and closure[0]['returncode'] == 0, 'Closure job has not finished'
receipt = read(ROOT / 'records/evidence_zip_receipt.json')
archive = ROOT / receipt['zip_path']
assert sha(archive) == receipt['sha256']
assert archive.stat().st_size == receipt['zip_bytes']

verified_locked_files = 0
for lock_name, field in [('execution_lock.json', 'locked_files'),
                         ('analysis_lock.json', 'files')]:
    for path, digest in read(ROOT / 'manifests' / lock_name)[field].items():
        assert sha(ROOT / path) == digest, ('FROZEN_FILE_CHANGED', path)
        verified_locked_files += 1
protected = read(ROOT / 'manifests/protected_before.json')
for row in protected:
    assert sha(ROOT.parent / row['path']) == row['sha256'], ('PROTECTED_EVIDENCE_CHANGED', row['path'])
fits = read(ROOT / 'records/all_six_final_checkpoints.json')['fits']
assert len(fits) == 6
for fit in fits:
    completed = read(ROOT / 'checkpoints' / fit['architecture'] / f'seed{fit["seed"]}' / 'completion.json')
    # Use the recorded checkpoint hash field, without selecting a new checkpoint.
    expected = completed['final_checkpoint_sha256']
    if expected is None:
        raise AssertionError(('Missing recorded final checkpoint hash', fit['architecture'], fit['seed']))
    assert sha(ROOT / 'checkpoints' / fit['architecture'] / f'seed{fit["seed"]}' / 'final.pt') == expected

extraction = ROOT / 'tests/evidence_reconstruction'
extraction.mkdir(parents=True, exist_ok=True)
(extraction / 'records').mkdir(exist_ok=True)
scalar_files = [
    'code/reconstruct_primary.py', 'code/statistical_core.py',
    'analysis/primary_scene_inputs.csv', 'analysis/primary_paired_frames.csv',
    'analysis/bootstrap_multiplicities.npz', 'analysis/primary_results.json',
]
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    manifest = json.loads(z.read('manifests/evidence_package_files.json'))
    for row in manifest:
        content = z.read(row['path'])
        assert len(content) == row['bytes']
        assert hashlib.sha256(content).hexdigest() == row['sha256']
    for path in scalar_files:
        z.extract(path, extraction)
reconstruction = subprocess.run(
    [sys.executable, str(extraction / 'code/reconstruct_primary.py'), '--root', str(extraction)],
    check=True, capture_output=True, text=True,
)
assert read(extraction / 'records/scalar_reconstruction.json')['status'] == 'SCALAR_RECONSTRUCTION_MATCHED'

snapshot = read(ROOT / 'resources/final_resource_snapshot.json')
# Include the completed closure job. The verification job's finish record is
# written by run_job.py after this script exits; refresh these totals once then.
metered_gpu = sum(e['occupied_gpu_device_hours'] for e in finished)
metered_cpu = sum(e['cpu_core_hours'] for e in finished)
charged_gpu = metered_gpu + snapshot['initial_gpu_device_hour_upper_allowance']
charged_cpu = (metered_cpu + snapshot['initial_cpu_core_hour_upper_allowance'] +
               snapshot['other_unwrapped_scalar_shell_and_reporting_cpu_core_hour_allowance'])
disk = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
cache_allowance = snapshot['outside_run_dependency_cache_disk_allowance_bytes']
assert charged_gpu < 64 and charged_cpu < 240 and disk + cache_allowance < 120e9 and elapsed() < 36
write(ROOT / 'records/delivery_verification.json', {
    'status': 'DELIVERY_VERIFIED', 'archive_sha256': receipt['sha256'],
    'archive_manifest_files_verified': len(manifest),
    'frozen_files_verified': verified_locked_files, 'protected_files_verified': len(protected),
    'formal_final_checkpoints_verified': len(fits),
    'standalone_reconstruction_from_archive': 'SCALAR_RECONSTRUCTION_MATCHED',
    'standalone_dependencies': 'NumPy and Python standard library; no Torch/checkpoints',
    'new_neural_forwards': 0, 'elapsed_hours': elapsed(), 'utc_unix': time.time(),
    'resource_totals_snapshot_excludes_running_verification_job': True,
    'reconstruction_stdout': reconstruction.stdout,
})
write(ROOT / 'resources/post_closure_resource_totals.json', {
    'status': 'POST_CLOSURE_TOTALS_PENDING_VERIFICATION_JOB_FINISH',
    'metered_occupied_gpu_device_hours': metered_gpu,
    'metered_launched_job_cpu_core_hours': metered_cpu,
    'charged_gpu_device_hours': charged_gpu, 'charged_cpu_core_hours': charged_cpu,
    'initial_gpu_device_hour_upper_allowance': snapshot['initial_gpu_device_hour_upper_allowance'],
    'initial_cpu_core_hour_upper_allowance': snapshot['initial_cpu_core_hour_upper_allowance'],
    'other_unwrapped_cpu_core_hour_allowance': snapshot['other_unwrapped_scalar_shell_and_reporting_cpu_core_hour_allowance'],
    'incremental_run_disk_bytes': disk,
    'outside_run_dependency_cache_disk_allowance_bytes': cache_allowance,
    'charged_disk_bytes': disk + cache_allowance,
    'evidence_zip_sha256': receipt['sha256'],
    'completed_jobs': len(finished), 'closure_job_included': True,
    'new_neural_forwards': 0, 'verification_elapsed_hours': elapsed(),
    'network_accounting': snapshot['network_measurement'],
    'no_purchases_or_externally_billed_services': True,
    'electricity_and_hardware_amortization_currency_cost': None,
})
print('DELIVERY_VERIFIED', receipt['sha256'], 'elapsed_hours', elapsed(), flush=True)
