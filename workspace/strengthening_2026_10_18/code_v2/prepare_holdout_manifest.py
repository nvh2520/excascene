"""Identity-only collision check and prospective manifest; no scene generation."""
from common import *
import time
def main():
    assert not (ROOT/'study3/holdout').exists()
    old=json.loads((AFAST/'manifests/collision_audit_receipt.json').read_text())
    ids=set(old['actual_scene_ids']);old_seeds=set(ids)
    for source in [AFAST/'generation',PROJECT/'revision_v5/validation_run/generation']:
        for p in source.glob('*.json'):
            d=json.loads(p.read_text());ids.add(int(d.get('scene_id',p.stem)))
            if 'geometry_seed' in d:old_seeds.add(int(d['geometry_seed']))
    for p in [AFAST/'manifests/generation_manifest.json',PROJECT/'revision_v5/validation_run/manifests/generation_manifest.json']:
        d=json.loads(p.read_text());ids.update(d['nominal_scene_ids']);ids.update(d['rough_scene_ids'])
    nominal=list(range(800000,800512));rough=list(range(810000,810256));new=nominal+rough
    assert len(set(new))==768 and not set(new)&ids
    seeds={str(s):int(np.random.default_rng(np.random.SeedSequence([2026101807,s,0,1,0,0,0])).integers(0,2**31-1)) for s in new}
    assert len(set(seeds.values()))==len(new) and not set(seeds.values())&(old_seeds|ids|set(new))
    cfg=dict(status='PROSPECTIVE_IDENTITIES_NO_GENERATION',root_seed=2026101807,bootstrap_seed=2026101808,
        nominal_scene_ids=nominal,rough_scene_ids=rough,geometry_seeds=seeds,slots=4,
        nominal_conditions=['clean','iid','cam_s1','cam_s2','cam_s3','single_severe','novel','all_s2'],rough_conditions=['rough_iid'],
        planned_records=17408,planned_logical_neural_forwards=522240,expected_power=.803017100248717,
        acceptance='original pose validity and all-condition sensor-clearance;50 attempts/slot;exclude whole bundle;stop if population invalid-slot fraction>10%',
        camera_assignment='one common uniform camera index per scene-slot;single severe has no absence;novel alternates mud_water/sun_glare by(scene_id+slot)%2',
        unique_geometry_seed_count=768,old_identity_count=len(ids),old_geometry_seed_count=len(old_seeds),collisions=0,
        id_inventory_basis='prior exhaustive identity receipt plus later sealed reference and external-study manifests/geometry-seed receipts',
        no_model_scores_read=True,utc_unix=time.time())
    js(ROOT/'protocol/holdout_manifest.json',cfg)
    print(json.dumps({k:cfg[k] for k in ['planned_records','planned_logical_neural_forwards','unique_geometry_seed_count','collisions']}))
if __name__=='__main__':main()
