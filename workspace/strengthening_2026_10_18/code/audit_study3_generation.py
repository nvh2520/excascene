"""Prospectively fixed whole-bundle and input validity checks; no outcome selection."""
from common import *
import pandas as pd
def main():
    verify_protocol()
    source=ROOT/'study3/holdout';cfg=json.loads((ROOT/'protocol/holdout_manifest.json').read_text());files=pd.read_csv(source/'files.csv')
    slots=pd.read_csv(source/'slots.csv');nom=set(cfg['nominal_scene_ids']);rough=set(cfg['rough_scene_ids'])
    assert not nom&rough
    assert set(slots.scene_id)==nom|rough and len(slots)==4*(len(nom)+len(rough))
    assert set(files.scene_id)==nom|rough,'A planned scene has no accepted slot; no silent scene replacement'
    historical=set(json.loads((AFAST/'manifests/historical_scene_pose_fingerprints.json').read_text()))
    historical.update(pd.read_csv(AFAST/'manifests/holdout_content_validity.csv').scene_pose_fingerprint)
    old_geometry=set(json.loads((AFAST/'manifests/historical_source_geometry_hashes.json').read_text()))
    geometry=set();fingerprints=set();failures=[];counts=[];fingerprint_rows=[]
    for path in sorted((source/'generation').glob('*.json')):
        receipt=json.loads(path.read_text());digest=receipt['source_geometry_sha256']
        assert digest not in old_geometry and digest not in geometry,('source_geometry_collision',receipt['scene_id'])
        geometry.add(digest)
        assert receipt['geometry_seed']==cfg['geometry_seeds'][str(receipt['scene_id'])]
    for (sid,slot),group in files.groupby(['scene_id','frame_id']):
        expected=set(['rough_iid'] if sid in rough else cfg['nominal_conditions'])
        assert set(group.condition)==expected and len(group)==len(expected)
        first=None
        for r in group.to_dict('records'):
            path=source/r['path'];assert sha(path)==r['sha256'];d=load_npz(path)
            for key in ['rgb','depth','lidar_range','gt_H','gt_aff','gt_reach','q_rep','q_true']:assert np.isfinite(d[key]).all(),(sid,slot,key)
            assert np.isin(d['gt_sem'],range(6)).all() and np.isin(d['gt_mat'],range(4)).all()
            assert not np.any((d['gt_aff']>0)&((d['gt_haz']>.5)|(d['gt_sem']!=0)))
            assert np.any(d['gt_reach']>=.05)
            if first is not None:
                for k in ['gt_H','gt_aff','gt_reach','gt_sem','gt_haz','gt_mat','q_true']:assert np.array_equal(first[k],d[k]),(sid,slot,k)
            first=d
        h=hashlib.sha256()
        for k,dt in [('q_true','<f4'),('gt_H','<f2'),('gt_sem','u1')]:
            h.update(k.encode());h.update(np.ascontiguousarray(first[k],dtype=dt).tobytes())
        digest=h.hexdigest();assert digest not in historical and digest not in fingerprints,('scene_pose_collision',sid,slot)
        fingerprints.add(digest);fingerprint_rows.append(dict(scene_id=int(sid),frame_id=int(slot),scene_pose_fingerprint=digest))
        counts.append(dict(scene_id=int(sid),frame_id=int(slot),conditions=len(group),whole_bundle=True))
    for population,ids in [('nominal',nom),('rough',rough)]:
        s=slots[slots.scene_id.isin(ids)];invalid=1-float(s.accepted.mean())
        if invalid>.10:failures.append(f'{population}:more_than_10_percent_invalid_slots')
    js(source/'validity_audit.json',dict(status='PASS' if not failures else 'BLOCKED',records=len(files),planned_records=cfg['planned_records'],accepted_bundles=len(counts),
        excluded_bundles=int((~slots.accepted).sum()),failures=failures,whole_bundle_validity=True,manifest_sha256=sha(source/'files.csv'),
        historical_or_new_scene_pose_collisions=0,source_geometry_collisions=0,geometry_seed_collisions=0,
        content_scope='Canonical stored-pose/height/semantic fingerprints; source-compatible historical geometry hashes; distinct frozen geometry seeds. No independent physical reference.'))
    csvwrite(source/'accepted_bundles.csv',counts)
    csvwrite(source/'scene_pose_fingerprints.csv',fingerprint_rows)
    assert not failures,failures
if __name__=='__main__':main()
