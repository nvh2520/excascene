"""Preprediction physical/content audit. Never reads model outputs or tunes fits."""
import csv,hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/excascene_snapshot'))
from kinafford.sim.scene import generate_scene
status=json.loads((ROOT/'generation_status.json').read_text());assert status['complete'] and status['overall_invalid_fraction']<=.10
old=json.loads((ROOT/'manifests/historical_scene_pose_fingerprints.json').read_text())
inventory=list(csv.DictReader((ROOT/'manifests/generated_files.csv').open()));rows=[];new=set();fingerprints={};scene_hashes={}
assert len(inventory)==status['records']
for row in inventory:
    p=ROOT/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    with np.load(p) as z:
        d={k:z[k] for k in z.files}
    sid=int(d['scene_id']);slot=int(d['frame_id']);key=(sid,slot,str(d['suite']))
    assert key not in new;new.add(key)
    for k in ['gt_H','gt_reach','gt_aff']:
        assert d[k].dtype==np.float32 and np.isfinite(d[k]).all(),(key,k,'authoritative_FP32')
    assert np.isin(d['gt_sem'],np.arange(6)).all() and np.isin(d['gt_mat'],np.arange(4)).all()
    feasible=np.isfinite(d['gt_reach'])&(d['gt_reach']>=.05)
    assert np.isfinite(d['gt_aff'][feasible]).all()
    assert ((d['gt_aff']>=0)&(d['gt_aff']<=1)).all()
    assert not np.any((d['gt_aff']>0)&((d['gt_haz']>.5)|(d['gt_sem']!=0)))
    h=hashlib.sha256()
    for k,dt in [('q_true','<f4'),('gt_H','<f2'),('gt_sem','u1')]:h.update(k.encode());h.update(np.ascontiguousarray(d[k],dtype=dt).tobytes())
    digest=h.hexdigest();assert digest not in old,('exact_historical_scene_pose_content_duplicate',key,old.get(digest))
    if (sid,slot) in fingerprints:assert fingerprints[(sid,slot)]==digest,'unshared truth across condition bundle'
    fingerprints[(sid,slot)]=digest
    rows.append(dict(scene_id=sid,frame_id=slot,condition=str(d['suite']),scene_pose_fingerprint=digest,
                     gt_H_sha256=hashlib.sha256(d['gt_H'].tobytes()).hexdigest(),gt_aff_sha256=hashlib.sha256(d['gt_aff'].tobytes()).hexdigest(),
                     fp32_oracles=True,finite_oracles=True,positive_oracle_safe=True))
# Distinct fresh terrain geometry and derived original-generator seeds. Scalar
# metadata from historical inventories is used solely for seed/content collision.
historical_ids=set(json.loads((ROOT/'manifests/collision_audit_receipt.json').read_text())['actual_scene_ids'])
historical_geometry=json.loads((ROOT/'manifests/historical_source_geometry_hashes.json').read_text())
for p in sorted((ROOT/'generation').glob('*.json')):
    g=json.loads(p.read_text());sid=int(p.stem);seed=int(g['geometry_seed'])
    assert seed not in historical_ids,('geometry_seed_reuses_historical_identity',sid,seed)
    scene=generate_scene(seed,ood_terrain=sid>=310000)
    h=hashlib.sha256()
    for a in [scene.h,scene.mat,scene.rock]:h.update(np.ascontiguousarray(a).tobytes())
    digest=h.hexdigest();assert digest not in scene_hashes,('exact_duplicate_fresh_scene_geometry',sid,scene_hashes.get(digest))
    assert digest not in historical_geometry,('exact_historical_source_geometry_duplicate',sid,historical_geometry.get(digest))
    scene_hashes[digest]=sid
with (ROOT/'manifests/holdout_content_validity.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
receipt=dict(status='SEALED_PHYSICAL_CONTENT_AUDIT_PASSED',records=len(rows),base_slots=len(fingerprints),scenes=len(scene_hashes),
    historical_content_duplicates=0,fresh_scene_geometry_duplicates=0,fp32_oracles=True,whole_bundle_status=status,
    historical_source_geometry_duplicates=0,historical_geometry_reference='source-compatible internal reconstruction, with both original population flags; not independent physical ground truth',
    historical_scores_accessed=False,model_outputs_accessed=False,independent_custody=False,
    held_data_for_training_or_adapter_selection=False,utc_unix=time.time())
(ROOT/'records/sealed_holdout_audit.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
