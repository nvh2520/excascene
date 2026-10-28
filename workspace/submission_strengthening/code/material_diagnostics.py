"""Post-hoc pooled-pixel diagnostics from saved AFAST outputs, without model execution.

The global plan must already be locked. Training prior is an input constructed by
the separate train-only stream. CLEAN coverage is an explicitly approximate depth
projection; exact render hits/image-material identities were not serialized.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SEM = ['terrain', 'rock', 'truck', 'person', 'obstacle', 'machine']
MAT = ['sand', 'soil', 'gravel', 'rocky']
CONDITIONS = ['clean', 'iid', 'all_s2', 'cam_s1', 'cam_s2', 'cam_s3', 'rough_iid']
CONFUSION_KEYS = {'semantic':'semantic_confusion', 'material':'material_true_terrain_confusion'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def csvwrite(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(matrix, names, key):
    c = np.asarray(matrix, dtype=np.int64)
    assert c.shape == (len(names), len(names)) and np.all(c >= 0)
    truth = c.sum(1); pred = c.sum(0); tp = c.diagonal()
    union = truth + pred - tp
    rows = []
    for i, name in enumerate(names):
        rows.append(dict(key, class_id=i, class_name=name, tp=int(tp[i]),
            fp=int(pred[i]-tp[i]), fn=int(truth[i]-tp[i]), truth_support=int(truth[i]),
            predicted_support=int(pred[i]), union=int(union[i]),
            precision=None if not pred[i] else float(tp[i]/pred[i]),
            recall=None if not truth[i] else float(tp[i]/truth[i]),
            iou=None if not union[i] else float(tp[i]/union[i]),
            precision_undefined=bool(not pred[i]), recall_undefined=bool(not truth[i]),
            iou_undefined=bool(not union[i])))
    nonzero = union > 0
    summary = dict(key, cells=int(c.sum()), correct=int(tp.sum()),
        accuracy=None if not c.sum() else float(tp.sum()/c.sum()),
        macro_iou=None if not nonzero.any() else float(np.mean(tp[nonzero]/union[nonzero])),
        macro_iou_defined_classes=int(nonzero.sum()),
        truth_counts=truth.tolist(), prediction_counts=pred.tolist(), confusion=c.tolist())
    return rows, summary


def majority_matrix(truth, cls):
    result = np.zeros((4, 4), dtype=np.int64)
    result[:, cls] = truth
    return result


def load_kinematics(path):
    spec = importlib.util.spec_from_file_location('saved_kinematics_numpy_only', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def projected_terrain_coverage(record, kin):
    """Union of projected saved terrain-pixel depth hits, intersected with BEV terrain.

    Optical Z depth is saved float16. Pixel centres use +0.5 exactly as original
    camera_rays. U-from-camera uses q_true and saved true drift. No dilation,
    depth threshold tuning, LiDAR or gt_vis. A rasterized hit and BEV cell centre
    can differ in material; this procedure cannot establish exact correspondence.
    """
    shape = record['gt_sem'].shape
    assert shape == (96, 96)
    covered = np.zeros(shape, dtype=bool)
    info = dict(present_cameras=0, valid_terrain_depth_pixels=0,
        projected_pixels_inside_grid=0, projected_pixels_onto_true_terrain=0)
    for i, camera in enumerate(kin.CAMERAS):
        if record['cam_present'][i] <= .5:
            continue
        info['present_cameras'] += 1
        depth = record['depth'][i].astype(np.float64)
        good = np.isfinite(depth) & (depth > .05) & (record['img_sem'][i] == 0)
        info['valid_terrain_depth_pixels'] += int(good.sum())
        K = kin.camera_intrinsics(kin.SENSORS[camera])
        v, u = np.nonzero(good)
        z = depth[v, u]
        points = np.stack([(u+.5-K[0, 2])*z/K[0, 0],
                           (v+.5-K[1, 2])*z/K[1, 1], z], axis=1)
        transform = kin.sensor_pose_U(camera, record['q_true'].astype(np.float64),
                                      drift=record['drift'][i].astype(np.float64))
        upper = points @ transform[:3, :3].T + transform[:3, 3]
        cols = np.floor((upper[:, 0]+8)/.25).astype(np.int64)
        rows = np.floor((upper[:, 1]+12)/.25).astype(np.int64)
        inside = (cols >= 0) & (cols < 96) & (rows >= 0) & (rows < 96)
        rr, cc = rows[inside], cols[inside]
        info['projected_pixels_inside_grid'] += int(inside.sum())
        terrain = record['gt_sem'][rr, cc] == 0
        info['projected_pixels_onto_true_terrain'] += int(terrain.sum())
        covered[rr[terrain], cc[terrain]] = True
    return covered, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--diagnostics', type=Path, required=True)
    p.add_argument('--training-prior', type=Path, required=True)
    p.add_argument('--plan-lock', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert a.plan_lock.is_file(), 'Global descriptive plan must be locked first'
    lock = json.loads(a.plan_lock.read_text())
    plan_path = a.plan_lock.parent.parent/'descriptive_reference_plan.md'
    assert sha(plan_path) == lock['plan_sha256'], 'Global plan changed after lock'
    if a.out.exists():
        assert not list(a.out.iterdir()), 'Never overwrite an existing diagnostic output'
    else:
        a.out.mkdir(parents=True)
    train = json.loads(a.training_prior.read_text())
    # The construction script supplies this before any new test comparisons.
    majority = int(train['majority_class_id'])
    assert 0 <= majority < 4
    prior_digest = sha(a.training_prior)
    full = json.loads(a.diagnostics.read_text())
    assert len(full) == 112
    classrows = []; summaries = []; hazards = []; matrices = []
    full_index = {}
    for item in full:
        key = dict(architecture=item['architecture'], seed=item['seed'], mask=item['mask'],
            condition=item['condition'], stratum='full_official_support',
            evidence_class='posthoc_pooled_pixel_diagnostic')
        for kind, names in [('semantic', SEM), ('material', MAT)]:
            key2 = dict(key, task=kind)
            rows, summary = stats(item[CONFUSION_KEYS[kind]], names, key2)
            classrows.extend(rows); summaries.append(summary)
            matrices.append(dict(key2, confusion=summary['confusion']))
            full_index[(item['architecture'], str(item['seed']), int(item['mask']), item['condition'], kind)] = summary
        hp = {k: int(item['hazard_'+k]) for k in ['tp', 'fp', 'fn', 'tn']}
        hazards.append(dict(key, **hp, precision_denominator=hp['tp']+hp['fp'],
            recall_denominator=hp['tp']+hp['fn'], cells=sum(hp.values()),
            precision=None if hp['tp']+hp['fp'] == 0 else hp['tp']/(hp['tp']+hp['fp']),
            recall=None if hp['tp']+hp['fn'] == 0 else hp['tp']/(hp['tp']+hp['fn'])))
    # One deterministic train-majority policy, not three independently trained models.
    for condition in CONDITIONS:
        source = full_index[('BF-D', '10', 0, condition, 'material')]
        truth = source['truth_counts']
        for item in full:
            if item['condition'] == condition and str(item['seed']) in ['10', '11', '12']:
                assert np.array_equal(np.asarray(item[CONFUSION_KEYS['material']]).sum(1), truth)
        key = dict(architecture='TRAIN_MAJORITY', seed='not_applicable', mask='not_applicable',
            condition=condition, stratum='full_official_support', task='material',
            evidence_class='posthoc_training_selected_pixel_reference')
        rows, summary = stats(majority_matrix(truth, majority), MAT, key)
        classrows.extend(rows); summaries.append(summary)
        matrices.append(dict(key, confusion=summary['confusion']))

    # Read only the 512 existing CLEAN records and corresponding saved predictions.
    manifest = list(csv.DictReader((a.run/'manifests/generated_files.csv').open()))
    clean = sorted([r for r in manifest if r['condition'] == 'clean'],
                   key=lambda r: (int(r['scene_id']), int(r['frame_id'])))
    assert len(clean) == 512
    kinpath = a.run/'source/excascene_snapshot/kinafford/sim/kinematics.py'
    kin = load_kinematics(kinpath)
    coverage_rows = []; masks = []; identifiers = []; stored = {}
    source_records = []
    for row in clean:
        path = a.run/row['path']
        assert sha(path) == row['sha256']
        with np.load(path, allow_pickle=False) as z:
            assert not any('hits' in k for k in z.files)
            record = {k:z[k] for k in ['gt_sem', 'gt_mat', 'depth', 'img_sem', 'cam_present', 'q_true', 'drift']}
        assert np.isin(record['gt_sem'], np.arange(6)).all()
        assert np.isin(record['gt_mat'], np.arange(4)).all()
        covered, info = projected_terrain_coverage(record, kin)
        terrain = record['gt_sem'] == 0
        assert not (covered & ~terrain).any()
        sid, frame = int(row['scene_id']), int(row['frame_id'])
        coverage_rows.append(dict(scene_id=sid, frame_id=frame,
            terrain_cells=int(terrain.sum()), projected_depth_covered=int(covered.sum()),
            not_projected_depth_covered=int((terrain & ~covered).sum()), **info))
        masks.append(covered); identifiers.append((sid, frame))
        stored[(sid, frame)] = (record['gt_sem'], record['gt_mat'], terrain, covered)
        source_records.append(dict(path=row['path'], sha256=row['sha256'], role='CLEAN_saved_sensor_and_truth'))
    csvwrite(a.out/'camera_support_per_frame.csv', coverage_rows)
    np.savez_compressed(a.out/'camera_projected_depth_masks.npz',
        scene_id=np.asarray([x[0] for x in identifiers], dtype=np.int32),
        frame_id=np.asarray([x[1] for x in identifiers], dtype=np.int16),
        projected_depth_covered=np.stack(masks))
    strata = ['projected_depth_covered_terrain', 'not_projected_depth_covered_terrain']
    stratum_matrices = []; storage_ties = []
    for architecture in ['BF-D', 'KF-D']:
        for seed in [10, 11, 12]:
            predicted = list(csv.DictReader((a.run/f'manifests/predictions_{architecture}_seed{seed}.csv').open()))
            predicted = [r for r in predicted if r['condition'] == 'clean']
            assert len(predicted) == 1024
            accum = {(mask, stratum, kind): np.zeros((n, n), dtype=np.int64)
                     for mask in [0, 1] for stratum in strata for kind, n in [('material', 4), ('semantic', 6)]}
            clean_full = {(mask, kind):np.zeros((n, n), dtype=np.int64)
                          for mask in [0, 1] for kind, n in [('material', 4), ('semantic', 6)]}
            for row in predicted:
                sid, frame, mask = int(row['scene_id']), int(row['frame_id']), int(row['mask'])
                sem, mat, terrain, covered = stored[(sid, frame)]
                path = a.run/row['path']
                assert sha(path) == row['sha256']
                with np.load(path, allow_pickle=False) as z:
                    sem_pred = z['sem_prob'].argmax(0)
                    probabilities = z['mat_prob']
                    # The original official confusion uses probability argmax;
                    # pre-softmax logit argmax is also saved, and can differ when
                    # floating softmax rounds near-equal logits to an exact tie.
                    mat_pred = probabilities.argmax(0)
                    stored_label = z['mat_label']
                    mismatch = np.argwhere(stored_label != mat_pred)
                    for rr, cc in mismatch:
                        stored_class = int(stored_label[rr, cc])
                        assert probabilities[stored_class, rr, cc] == probabilities[:, rr, cc].max(), ('non_tie_material_label_mismatch',str(path),int(rr),int(cc))
                        storage_ties.append(dict(architecture=architecture, seed=seed, mask=mask,
                            scene_id=sid, frame_id=frame, row=int(rr), col=int(cc),
                            stored_logit_argmax=stored_class, official_probability_argmax=int(mat_pred[rr,cc]),
                            tied_probability=float(probabilities[stored_class,rr,cc]), true_terrain=bool(terrain[rr,cc]),
                            source_path=row['path']))
                source_records.append(dict(path=row['path'], sha256=row['sha256'], role='saved_CLEAN_prediction'))
                for kind, truth, pred, n in [('material', mat, mat_pred, 4), ('semantic', sem, sem_pred, 6)]:
                    official = terrain if kind == 'material' else np.ones_like(terrain)
                    clean_full[(mask, kind)] += np.bincount((truth[official].astype(np.int64)*n+pred[official]).ravel(), minlength=n*n).reshape(n,n)
                    for stratum, support in [(strata[0], covered), (strata[1], terrain & ~covered)]:
                        accum[(mask, stratum, kind)] += np.bincount((truth[support].astype(np.int64)*n+pred[support]).ravel(), minlength=n*n).reshape(n,n)
            for (mask, kind), matrix in clean_full.items():
                assert np.array_equal(matrix, full_index[(architecture,str(seed),mask,'clean',kind)]['confusion']), 'Saved per-cell outputs must reproduce original CLEAN confusions'
            for (mask, stratum, kind), matrix in accum.items():
                stratum_matrices.append(dict(architecture=architecture, seed=seed, mask=mask, condition='clean',
                    stratum=stratum, task=kind, evidence_class='posthoc_approximate_coverage_pixel_diagnostic', confusion=matrix.tolist()))
            print(f'Completed saved prediction diagnostics {architecture} seed {seed}', flush=True)
    for architecture in ['BF-D','KF-D']:
        for mask in [0,1]:
            for stratum in strata:
                for kind in ['material','semantic']:
                    chosen = [x for x in stratum_matrices if x['architecture']==architecture and x['mask']==mask and x['stratum']==stratum and x['task']==kind]
                    assert len(chosen)==3
                    item = dict(chosen[0]); item['seed']='pooled_three_fitted_seeds'
                    item['confusion'] = np.sum([x['confusion'] for x in chosen],axis=0).tolist()
                    stratum_matrices.append(item)
    for stratum in strata:
        chosen = next(x for x in stratum_matrices if x['architecture']=='BF-D' and x['seed']==10 and x['mask']==0 and x['task']=='material' and x['stratum']==stratum)
        item=dict(chosen, architecture='TRAIN_MAJORITY',seed='not_applicable',mask='not_applicable',
            evidence_class='posthoc_training_selected_pixel_reference',
            confusion=majority_matrix(np.asarray(chosen['confusion']).sum(1),majority).tolist())
        stratum_matrices.append(item)
    for item in stratum_matrices:
        key = {k:v for k,v in item.items() if k!='confusion'}
        rows, summary = stats(item['confusion'], MAT if item['task']=='material' else SEM, key)
        classrows.extend(rows); summaries.append(summary)
    matrices.extend(stratum_matrices)
    csvwrite(a.out/'class_metrics.csv', classrows)
    csvwrite(a.out/'summary_metrics.csv', [{k:v for k,v in row.items() if k not in ['truth_counts','prediction_counts','confusion']} for row in summaries])
    csvwrite(a.out/'hazard_head_metrics.csv', hazards)
    dump(a.out/'full_confusions_and_support.json', summaries)
    csvwrite(a.out/'source_records.csv', source_records)
    csvwrite(a.out/'stored_label_probability_argmax_ties.csv', storage_ties)
    history_rows = []
    for architecture in ['BF-D', 'KF-D']:
        for seed in [10, 11, 12]:
            history_path = a.run/f'checkpoints/{architecture}/seed{seed}/history.json'
            history = json.loads(history_path.read_text())
            assert len(history) == 14
            for entry in history:
                assert np.isfinite(entry['train_losses']['mat']) and np.isfinite(entry['validation_losses']['mat'])
                history_rows.append(dict(architecture=architecture, seed=seed, epoch=entry['epoch'],
                    training_material_loss=entry['train_losses']['mat'],
                    validation_material_loss=entry['validation_losses']['mat'],
                    source_sha256=sha(history_path)))
    csvwrite(a.out/'saved_material_loss_history.csv', history_rows)
    coverage = {key:sum(x[key] for x in coverage_rows) for key in ['terrain_cells','projected_depth_covered','not_projected_depth_covered']}
    dump(a.out/'diagnostic_receipt.json', dict(status='COMPLETE_POSTHOC_DESCRIPTIVE_DIAGNOSTICS',
        training_prior_sha256=prior_digest, majority_class_id=majority, majority_class_name=MAT[majority],
        diagnostics_source_sha256=sha(a.diagnostics), plan_lock_sha256=sha(a.plan_lock),
        kinematics_source_sha256=sha(kinpath), generated_manifest_sha256=sha(a.run/'manifests/generated_files.csv'),
        clean_records=512, saved_clean_predictions=6144, fullmap_groups=112,
        coverage=coverage, exact_camera_material_observation='NOT_RECOVERABLE_FROM_SAVED_ARTIFACTS',
        missing=['unquantized raw CLEAN hits_B keyed by camera/pixel', 'per-hit image material identity or retained Scene material raster for exact correspondence'],
        implemented_support='approximate projected float16 CLEAN optical-depth terrain hits in canonical U grid; true q/drift; no LiDAR or gt_vis',
        camera_mask_support='strata defined by original CLEAN available-camera support in both model states, not observed pixels after removal',
        weighting='pooled pixels, pooled-three-seed counts are repeated same support; separate from scene-primary estimands',
        stored_label_probability_argmax_ties=len(storage_ties),
        confusion_argmax='saved mat_prob argmax with lowest-index ties, identical to original official inference diagnostic',
        new_scene_records=0, model_forwards=0, bootstrap_draws=0, source_defect_identified=False))
    assert 'torch' not in sys.modules, 'This diagnostic must not import a neural runtime'


if __name__ == '__main__':
    main()
