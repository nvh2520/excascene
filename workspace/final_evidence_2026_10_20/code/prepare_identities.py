"""Prospective Study4identities, distinct from every historical scene/seed."""
from common import *
import datetime
def main():
    assert not (ROOT/'study4/holdout').exists()
    ids=set(json.loads((AFAST/'manifests/collision_audit_receipt.json').read_text())['actual_scene_ids']);seeds=set(ids)
    for folder in [AFAST/'generation',PROJECT/'revision_v5/validation_run/generation',PROJECT/'strengthening_2026_10_18/study3/holdout/generation']:
        for p in folder.glob('*.json'):
            r=json.loads(p.read_text());ids.add(int(r.get('scene_id',p.stem)))
            if 'geometry_seed' in r:seeds.add(int(r['geometry_seed']))
    nominal=list(range(1200000,1200512));rough=list(range(1210000,1210256));new=nominal+rough
    assert len(set(new))==768 and not set(new)&ids
    geom={str(s):int(np.random.default_rng(np.random.SeedSequence([2026102007,s,0,1,0,0,0])).integers(0,2**31-1)) for s in new}
    assert len(set(geom.values()))==768 and not set(geom.values())&(seeds|ids|set(new))
    cfg=dict(status='PROSPECTIVE_IDENTITIES_NO_GENERATION',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_seed=2026102007,bootstrap_seed=2026102008,nominal_scene_ids=nominal,rough_scene_ids=rough,geometry_seeds=geom,slots=4,
        nominal_conditions=['clean','iid','cam_s1','cam_s2','cam_s3','single_severe','novel','all_s2'],rough_conditions=['rough_iid'],planned_records=17408,planned_logical_neural_forwards=835584,
        acceptance='Original compound pose and all-condition sensor-clearance;50attempts/slot; whole-bundle exclusion. A zero-accepted-slot scene is excluded whole without replacement before inference, without triggering a scene-empty stop. Other validity controls remain.',
        camera_assignment='shared uniform index per scene-slot;single-camera severe and novel retain all cameras; novel type alternates by(scene_id+slot)%2',old_identity_count=len(ids),old_geometry_seed_count=len(seeds),identity_collisions=0,geometry_seed_collisions=0,
        source='Prior exhaustive collision inventory plus all later reference/external/Study3generation receipts, including excluded scenes',no_model_outcomes_read=True,power='Population size prescribed by execution brief; no new power claim for the changed training recipe')
    js(ROOT/'protocol/holdout_manifest.json',cfg);print('Fresh identity/seed collisions0;512+256planned')
if __name__=='__main__':main()
