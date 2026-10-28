"""Pre-inference retained population, entire-bundle, collision and input checks."""
from common import *
import pandas as pd
def main():
    verify_protocol();source=ROOT/'study4/holdout';cfg=json.loads((ROOT/'protocol/holdout_manifest.json').read_text())
    assert not (ROOT/'study4/metrics').exists() and not (source/'gate_scores.csv').exists()
    files=pd.read_csv(source/'files.csv');slots=pd.read_csv(source/'slots.csv')
    nominal=set(cfg['nominal_scene_ids']);rough=set(cfg['rough_scene_ids']);assert len(slots)==3072 and set(slots.scene_id)==nominal|rough
    empty=sorted(int(s) for s,g in slots.groupby('scene_id') if not g.accepted.any())
    assert set(files.scene_id)==(nominal|rough)-set(empty)
    assert set(map(tuple,files[['scene_id','frame_id']].drop_duplicates().to_numpy()))==set(map(tuple,slots.loc[slots.accepted,['scene_id','frame_id']].to_numpy()))
    historical=set(json.loads((AFAST/'manifests/historical_scene_pose_fingerprints.json').read_text()))
    historical.update(pd.read_csv(AFAST/'manifests/holdout_content_validity.csv').scene_pose_fingerprint)
    historical.update(pd.read_csv(PROJECT/'strengthening_2026_10_18/study3/holdout/scene_pose_fingerprints.csv').scene_pose_fingerprint)
    geometry=set(json.loads((AFAST/'manifests/historical_source_geometry_hashes.json').read_text()));newgeometry=set()
    for folder in [PROJECT/'strengthening_2026_10_18/study3/holdout/generation']:
        for p in folder.glob('*.json'):geometry.add(json.loads(p.read_text())['source_geometry_sha256'])
    for p in sorted((source/'generation').glob('*.json')):
        r=json.loads(p.read_text());digest=r['source_geometry_sha256']
        assert digest not in geometry and digest not in newgeometry;newgeometry.add(digest)
        assert r['geometry_seed']==cfg['geometry_seeds'][str(r['scene_id'])]
    fingerprints=set();rows=[];accepted=[]
    for (sid,slot),g in files.groupby(['scene_id','frame_id']):
        expected=set(['rough_iid'] if sid in rough else cfg['nominal_conditions']);assert set(g.condition)==expected and len(g)==len(expected)
        ds={}
        for r in g.to_dict('records'):
            path=source/r['path'];assert sha(path)==r['sha256'];d=load_npz(path);ds[r['condition']]=d
            for k in ['rgb','depth','lidar_range','q_true','q_rep','gt_H','gt_aff','gt_reach']:assert np.isfinite(d[k]).all(),(sid,slot,k)
            assert np.isin(d['gt_sem'],range(6)).all() and np.isin(d['gt_mat'],range(4)).all()
            assert not np.any((d['gt_aff']>0)&((d['gt_haz']>.5)|(d['gt_sem']!=0))) and np.any(d['gt_reach']>=.05)
        base=next(iter(ds.values()))
        for d in ds.values():
            for k in ['gt_H','gt_aff','gt_reach','gt_sem','gt_haz','gt_mat','q_true']:assert np.array_equal(base[k],d[k])
        if 'clean' in ds:
            base=ds['clean']
            for c in ['cam_s1','cam_s2','cam_s3','single_severe','novel']:
                for k in ['lidar_range','lidar_valid','q_rep','q_true','drift']:assert np.array_equal(base[k],ds[c][k])
            assert ds['single_severe']['cam_present'].all() and ds['novel']['cam_present'].all()
        h=hashlib.sha256()
        for k,dt in [('q_true','<f4'),('gt_H','<f2'),('gt_sem','u1')]:h.update(k.encode());h.update(np.ascontiguousarray(base[k],dtype=dt).tobytes())
        fingerprint=h.hexdigest();assert fingerprint not in historical and fingerprint not in fingerprints
        fingerprints.add(fingerprint);rows.append(dict(scene_id=int(sid),frame_id=int(slot),scene_pose_fingerprint=fingerprint));accepted.append(dict(scene_id=int(sid),frame_id=int(slot),conditions=len(g),whole_bundle=True))
    failures=[]
    for label,ids in [('nominal',nominal),('rough',rough)]:
        if 1-float(slots[slots.scene_id.isin(ids)].accepted.mean())>.10:failures.append(label+':more_than10percent_invalid_slots')
    csvwrite(source/'scene_pose_fingerprints.csv',rows);csvwrite(source/'accepted_bundles.csv',accepted)
    exclusions=[]
    for sid in empty:
        r=json.loads((source/f'generation/{sid}.json').read_text());assert len(r['attempts'])==200 and not any(x['accepted'] for x in r['attempts'])
        reasons={}
        for a in r['attempts']:
            for reason in a['reasons']:reasons[reason]=reasons.get(reason,0)+1
        exclusions.append(dict(scene_id=sid,attempts=200,reasons=reasons,rule='prespecified whole-scene exclusion without replacement; not a protocol deviation'))
    record=dict(status='PASS' if not failures else 'BLOCKED',records=len(files),planned_records=17408,nominal_scenes=len(nominal-set(empty)),rough_scenes=len(rough-set(empty)),excluded_scenes=empty,exclusion_details=exclusions,
        accepted_bundles=len(accepted),excluded_bundles=int((~slots.accepted).sum()),failures=failures,scene_pose_collisions=0,source_geometry_collisions=0,
        source_geometry_comparison_scope='Available original historical source-geometry hashes and stored Study3 source hashes; all earlier cohorts additionally screened by identity/geometry-seed and saved scene-pose fingerprints. Earlier external fresh source hashes were not retained; no earlier scene is regenerated here.',
        manifest_sha256=sha(source/'files.csv'),test_predictions_before_exclusion=0,test_gate_scores_before_exclusion=0,no_replacement=True)
    js(source/'validity_audit.json',record);assert not failures,failures
    # Create the only20,000scene draws on the retained population, before inference.
    from statistical_core4 import multiplicities
    ids=np.array(sorted(set(files[files.condition=='clean'].scene_id)),np.int64);rid=np.array(sorted(set(files[files.condition=='rough_iid'].scene_id)),np.int64)
    rng=np.random.default_rng(cfg['bootstrap_seed']);dest=ROOT/'study4/bootstrap_plan';dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'bootstrap_multiplicities.npz',nominal=multiplicities(ids,rng,20000),rough=multiplicities(rid,rng,20000),nominal_scene_ids=ids,rough_scene_ids=rid)
    js(dest/'receipt.json',dict(status='CREATED_ONCE_BEFORE_INFERENCE',seed=cfg['bootstrap_seed'],draws=20000,nominal=len(ids),rough=len(rid),sha256=sha(dest/'bootstrap_multiplicities.npz'),excluded_scenes=empty))
    print(record['status'],record['records'],record['nominal_scenes'],record['rough_scenes'],'excluded',empty,flush=True)
if __name__=='__main__':main()
