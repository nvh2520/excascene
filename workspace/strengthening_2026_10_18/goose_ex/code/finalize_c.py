"""Close Phase C from saved outputs only; no models, image replay, or new draws."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
STRENGTHENING = OUT.parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    probabilities = load_csv(OUT / 'metrics/real_gate_probabilities.csv')
    rates = load_csv(OUT / 'metrics/real_gate_rates.csv')
    calibration = json.loads((OUT / 'metrics/real_gate_calibration.json').read_text())
    lock = json.loads((OUT / 'protocol/gate_transfer_lock_v2.json').read_text())
    for entry in lock['files']:
        assert digest(ROOT / entry['path']) == entry['sha256'], entry['path']
    assert len(probabilities) == 5044 and len(rates) == 96
    assert len({(r['split'], r['image_id'], r['mechanism'], r['severity'])
                for r in probabilities}) == 5044
    train = np.array([float(r['probability_severe']) for r in probabilities
                      if r['split'] == 'train'], dtype=np.float32)
    assert len(train) == 2164
    assert np.isfinite(train).all() and ((train >= 0) & (train <= 1)).all()
    boundary = np.sort(train)[len(train) - math.floor(.05 * len(train)) - 1]
    tau = np.nextafter(boundary, np.float32(np.inf))
    assert float(tau) == calibration['threshold']
    assert int((train >= tau).sum()) == calibration['removed'] == 108
    assert calibration['removed'] <= math.floor(.05 * len(train))
    checked = 0
    for row in rates:
        matching = [r for r in probabilities if
                    (r['split'], r['mechanism'], r['severity']) ==
                    (row['split'], row['mechanism'], row['severity'])]
        score64 = np.array([float(r['probability_severe']) for r in matching])
        score32 = score64.astype(np.float32)
        threshold = float(row['threshold'])
        removed64 = score64 >= threshold
        removed32 = score32 >= np.float32(threshold)
        assert np.array_equal(removed64, removed32), (row, 'threshold representation mismatch')
        assert int(removed64.sum()) == int(row['removed'])
        assert len(matching) == int(row['images'])
        assert float(row['removal_rate']) == int(row['removed']) / int(row['images'])
        if row['split'] == 'val':
            assert int(row['images']) == 192
            assert 0 <= float(row['ordinary_ci95_low']) <= float(row['ordinary_ci95_high']) <= 1
        else:
            assert int(row['images']) == 2164
            assert row['ordinary_ci95_low'] == row['ordinary_ci95_high'] == ''
        if row['downstream_model'] == 'BF-D' and row['threshold_source'] == 'sim_tuned':
            assert threshold > 1 and int(row['removed']) == 0
        checked += 1

    draws_path = OUT / 'metrics/sequence_bootstrap_draws.npz'
    with np.load(draws_path, allow_pickle=False) as z:
        assert z['real_counts'].shape == (20000, 10)
        assert (z['real_counts'].sum(1) == 10).all()
    targeted = dict(status='PASS', utc_unix=time.time(), probability_rows=len(probabilities),
                    rate_rows_checked=checked, train_calibration_removals=108,
                    train_calibration_images=2164, threshold=float(tau),
                    float32_float64_decisions_identical_for_all_reported_thresholds=True,
                    frozen_inputs_hashes_unchanged=True,
                    saved_sequence_draws_sha256=digest(draws_path),
                    new_model_forwards=0, new_bootstrap_draws=0,
                    scope='Saved gate counts, denominators, FP32 boundary and frozen provenance only')
    write_json(OUT / 'resources/gate_targeted_check.json', targeted)

    # Actual central-meter receipts include process startup and the discarded pilot.
    jobs = []
    for p in sorted((STRENGTHENING / 'resources').glob('job_C*.json')):
        row = json.loads(p.read_text())
        assert row['phase'] == 'goose'
        jobs.append(dict(path=str(p.relative_to(ROOT)), sha256=digest(p), **row))
    assert len(jobs) == 8 and all(j['returncode'] == 0 for j in jobs)
    gpu_hours = sum(j['gpu_device_hours'] for j in jobs)
    cpu_hours = sum(j['cpu_core_hours'] for j in jobs)
    resources = dict(status='COMPLETE', utc_unix=time.time(),
                     occupied_gpu_hours=gpu_hours, phase_gpu_cap_hours=6,
                     metered_launched_job_cpu_core_hours=cpu_hours,
                     metered_job_count=len(jobs), failed_jobs=0,
                     jobs=jobs,
                     preparation_wall_seconds={name: json.loads((OUT / 'resources' / name).read_text())['elapsed_seconds']
                                               for name in ['prepare_inventory.json', 'prepare_sim-cache.json', 'prepare_real-cache.json']},
                     cpu_coverage='CPU core-hours cover central-meter jobs only. Earlier CPU cache preparation has recorded elapsed time, not a historical CPU-use meter; no missing historical receipt is manufactured.',
                     no_further_gpu_job_required=True)
    write_json(OUT / 'resources/C_final_resource_totals.json', resources)

    display = {'clean': 'Original images', 'occlusion': 'Occlusion', 'dust': 'Haze',
               'blur': 'Blur/exposure', 'noise': 'Noise', 'mud_water': 'Mud/water droplets',
               'sun_glare': 'Glare/saturation'}
    levels = {'1': 'mild', '2': 'moderate', '3': 'severe'}
    conditions = [(r['mechanism'], r['severity']) for r in rates if
                  r['downstream_model'] == 'KF-D' and r['threshold_source'] == 'sim_tuned' and r['split'] == 'val']
    compact = []
    lines = ['| Real image condition | Lightweight simulation threshold | BEVFusion-Exca simulation threshold | Utility-specialist simulation threshold | Real-training calibrated threshold |',
             '|---|---:|---:|---:|---:|']
    for mechanism, severity in conditions:
        label = display[mechanism]
        if mechanism not in ('clean', 'mud_water', 'sun_glare'):
            label += ', ' + levels[severity]
        entry = dict(mechanism=mechanism, severity=severity, images=192)
        formatted = []
        for name, model, setting in [('lightweight_sim_tau', 'KF-D', 'sim_tuned'),
                                     ('bevfusion_sim_tau', 'BF-D', 'sim_tuned'),
                                     ('utility_specialist_sim_tau', 'S5A', 'sim_tuned'),
                                     ('real_train_calibrated', 'KF-D', 'real_train_calibrated')]:
            match, = [r for r in rates if (r['mechanism'], r['severity'], r['split'], r['downstream_model'], r['threshold_source']) ==
                       (mechanism, severity, 'val', model, setting)]
            for key in ['removed', 'removal_rate', 'ordinary_ci95_low', 'ordinary_ci95_high']:
                entry[name + '_' + key] = match[key]
            lo = float(match['ordinary_ci95_low']) * 100
            hi = float(match['ordinary_ci95_high']) * 100
            formatted.append(f"{int(match['removed'])}/192; {100 * float(match['removal_rate']):.1f}% [{lo:.1f}, {hi:.1f}]")
        compact.append(entry)
        lines.append('| ' + label + ' | ' + ' | '.join(formatted) + ' |')
    with (OUT / 'metrics/real_gate_compact.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(compact[0])); w.writeheader(); w.writerows(compact)
    (OUT / 'C_gate_result_table.md').write_text(
        '# Phase C: real-image gate removal rates\n\n'
        'Each cell gives removed images/192; percentage [ordinary descriptive 95% sequence-bootstrap interval]. '
        'The 20,000 multiplicities are reused from the segmentation check. The depth branch is untested. '
        'The calibrated threshold is identical across all three downstream labels because the same gate scores '
        'and the same unlabeled training-only rule are used. All 96 uncollapsed rows, including training denominators, '
        'are retained in `metrics/real_gate_rates.csv`.\n\n' + '\n'.join(lines) + '\n\n'
        'Zero-width intervals at zero or one reflect the finite observed sequence outcomes; they do not establish '
        'population certainty. Original images have no added synthetic corruption, but their natural fault status '
        'has not been independently established. Thus clean false removal is an operational image-removal measure.\n')

    report = f'''# Phase C complete: real-image perception and gate transfer

The fixed image check finds similar degradation ordering across simulation and GOOSE-Ex ALICE: Spearman correlation **0.8813 [0.7978, 0.9253]**, with an ordinary descriptive 95% sequence-bootstrap interval. **Real clean recognition remains weak:** native-label mIoU is **0.1093 [0.0798, 0.1229]**, and **20 of 42 supported classes have zero clean true positives**. Simulation clean mIoU is 0.8511 [0.8387, 0.8636]; the native ontologies differ, so these absolute scores are not a cross-domain architectural comparison. All 15 conditions and all 1,065 class/condition rows, including undefined ratios, are retained. Mild real blur/exposure has a relative-mIoU interval spanning one.

**The simulation-tuned RGB gate transfers poorly to the original real images.** The lightweight model's threshold 0.25 removes **73/192 (38.02%; 95% interval [21.13%, 56.42%])**; the utility-specialist threshold 0.9 removes **18/192 (9.38%; [2.40%, 18.23%])**. The BEVFusion-Exca threshold 1.01 disables RGB removal, yielding zero for every condition. Its real depth branch is **NOT TESTED**, so these zeros do not establish successful transfer of the full gate. Original images mean no added corruption, not independently verified fault-free imagery.

Unlabeled official-training-only calibration selects the next FP32 value above the fixed order-statistic boundary: **tau = {calibration['threshold']:.16f}**. It removes **108/2,164 (4.991%)** training images and **7/192 (3.65%; [0%, 8.22%])** held-out original images. The same recalibrated threshold detects **184/192 severe occlusions**, **192/192 severe noise cases**, but **0/192 haze cases at every severity**, **0/192 mud/water cases**, and **4/192 glare cases**. Lower clean removal therefore comes with substantial mechanism-specific misses. Calibration does not replace the prospective simulation thresholds; its at-most-5% statement applies only to these training frames.

Both segmentation models were random initialized and trained for 24 fixed epochs with the same SegFormer-B0 recipe, final checkpoints, actual FP32 and TF32 disabled. Native heads have 7/64 outputs. Evaluation used 300 simulation images (100 scenes) and 192 ALICE images (10 official validation sequences); training used 1,200/2,164 images. All inputs are 256 by 160, with declared real-image aspect-ratio change. The 20,000 saved scene/sequence draws are paired across conditions; gate intervals reuse them without new draws. One fit per domain, weak real recognition, ten real sequences, reduced resolution and RGB-only gating constrain interpretation. This check supports conditional image-degradation ordering, not reliable scene understanding, real excavator decisions, physical safety, or physically calibrated faults.

The complete degradation table is `C_segmentation_results.md`; gate counts and intervals are `C_gate_result_table.md` and `metrics/real_gate_rates.csv`. Both targeted checks passed. The pre-execution FP32 threshold-boundary correction is versioned; the original script remains preserved. Metered Phase C use, including the discarded pilot, both fits, all evaluations and gate inference, is **{gpu_hours:.6f} occupied GPU-hours** out of six, with **{cpu_hours:.6f} launched-job CPU core-hours** and zero failed jobs. Earlier CPU cache preparation has separate elapsed-time records, without a manufactured historical CPU-use receipt.

Official [GOOSE-Ex setup/license](https://goose-dataset.de/docs/setup/) and the local archive license establish CC BY-SA 4.0 terms. Existing official labels alone were used. Real original, resized and corrupted images are excluded from the local release candidate; no image redistribution, upload or license grant occurred. The final inventory records code, numerical outputs, checkpoint hashes, notices and these exclusions. No further Phase C computation is required.
'''
    (OUT / 'C_results.md').write_text(report)
    result = dict(status='COMPLETE', segmentation=dict(spearman=0.8813186813186813,
                  ci95=[0.7978021978021979, 0.9252747252747253],
                  real_clean_miou=0.1092545770267667, real_supported_zero_tp=20,
                  real_supported_classes=42), gate_calibration=calibration,
                  gate_full_rows=96, gate_probability_rows=5044,
                  gate_depth_branch='NOT_TESTED_NO_MEASURED_REAL_DEPTH',
                  occupied_gpu_hours=gpu_hours, report='C_results.md',
                  no_further_gpu_job_required=True)
    write_json(OUT / 'protocol/C_final_results.json', result)

    # Exact private-candidate inventory; image caches are never eligible for inclusion.
    inventory_path = OUT / 'protocol/release_inventory_final.csv'
    inventory = []
    excluded_bytes = excluded_files = 0
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p == inventory_path:
            continue
        rel = p.relative_to(OUT)
        image = rel.parts[0] == 'cache'
        if image:
            inclusion = 'NO'; scope = 'DO_NOT_REDISTRIBUTE_IMAGES'
            role = 'private image/label cache'; sha = 'stored in cache manifests'
            excluded_bytes += p.stat().st_size; excluded_files += 1
        elif rel.suffix == '.html':
            inclusion = 'NO'; scope = 'private official-page verification copy'
            role = 'source verification copy'; sha = digest(p)
        elif rel.parts[0] == 'checkpoints':
            inclusion = 'LOCAL_PRIVATE_CANDIDATE_ONLY'; scope = 'public distribution terms unassigned; no license grant'
            role = 'fixed final image model'; sha = digest(p)
        else:
            inclusion = 'YES_LOCAL_CANDIDATE'; scope = 'prepare only; no public release or license grant'
            role = 'code, numerical output, provenance, notice or report'; sha = digest(p)
        inventory.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                              sha256=sha, role=role, local_candidate_inclusion=inclusion,
                              distribution_scope=scope))
    with inventory_path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(inventory[0])); w.writeheader(); w.writerows(inventory)
    write_json(OUT / 'protocol/release_exclusions_final.json', dict(
        status='COMPLETE', utc_unix=time.time(), inventory=str(inventory_path.relative_to(ROOT)),
        inventory_sha256=digest(inventory_path), private_image_cache_files=excluded_files,
        private_image_cache_bytes=excluded_bytes,
        exclusion_patterns=['goose_ex/cache/**', 'goose_ex/evidence/*.html'],
        no_real_images_for_release=True,
        no_public_license_grant=True,
        source_images='Remain in original local dataset tree; never copied into release candidate',
        resource_receipts='Canonical job_C*.json stay under strengthening_2026_10_18/resources; exact hashes and contents are in C_final_resource_totals.json'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
