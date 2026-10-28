"""Source-provenance geometry collision screen; no historical metrics or fits."""
import hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/excascene_snapshot'))
from kinafford.sim.scene import generate_scene
import numpy as np
def digest(scene):
    h=hashlib.sha256()
    for a in [scene.h,scene.mat,scene.rock]:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
source=ROOT/'source/excascene_snapshot/kinafford/sim/scene.py'
v51=ROOT.parent/'revision_v5/readiness_v5_1/source/kinafford/sim/scene.py'
assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(v51.read_bytes()).digest()
ids=json.loads((ROOT/'manifests/collision_audit_receipt.json').read_text())['actual_scene_ids']
seen={};start=time.monotonic()
for i,sid in enumerate(ids):
    for rough in [False,True]:
        s=generate_scene(sid,ood_terrain=rough);seen.setdefault(digest(s),dict(scene_id=sid,rough=rough))
    if (i+1)%500==0:print('historical_source_geometry_screen',i+1,len(ids),flush=True)
(ROOT/'manifests/historical_source_geometry_hashes.json').write_text(json.dumps(seen,indent=2))
receipt=dict(status='SOURCE_COMPATIBLE_GEOMETRY_HASH_SCREEN_COMPLETE',identities=len(ids),
    reconstructions=2*len(ids),unique_geometry_hashes=len(seen),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    source_matches_completed_v51=True,flags='both nominal and rough reconstructed conservatively for every actual recorded identity',
    evidence_type='internal source reconstruction for duplicate screening; not independent physical ground truth',
    historical_model_scores_accessed=False,elapsed_s=time.monotonic()-start,utc_unix=time.time())
(ROOT/'records/historical_source_geometry_screen.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
