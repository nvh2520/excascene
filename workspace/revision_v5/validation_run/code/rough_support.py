"""CPU-only clean visibility references at existing locked rough poses; no new records/forwards."""
from common import *
from kinafford.sim.scene import generate_scene
from kinafford.sim.render import Renderer,visibility_mask
from kinafford.sim.kinematics import CAMERAS
def scene_job(scene_id):
    records=json.loads((RUN/f'generation/{scene_id}.json').read_text());scene=generate_scene(scene_id,ood_terrain=True);renderer=Renderer(scene)
    rows=[]
    for a in records['attempts']:
        if not a['accepted']:continue
        q=np.array(a['q'],dtype=np.float64);slot=int(a['frame_id']);mb=renderer._machine_boxes(q);st=renderer._static_objects()
        hits=[renderer.render_camera(c,q,np.random.default_rng(0),machine_boxes=mb,static=st)['hits_B'] for c in CAMERAS]
        hits.append(renderer.render_lidar(q,None,machine_boxes=mb,static=st)['hits_B'])
        mask=visibility_mask(hits,q);path=RUN/f'support_reference/rough_iid/{scene_id}_{slot}.npz';assert not path.exists()
        npzwrite(path,gt_vis_clean=mask,q_render=q)
        rows.append(dict(scene_id=scene_id,frame_id=slot,path=str(path.relative_to(RUN)),sha256=sha(path),reference='all sensors present; nominal mounts; no faults; exact accepted float64 rendering pose',visible_cells=int(mask.sum())))
    return rows
def main():
    locked();gm=json.loads((RUN/'manifests/generation_manifest.json').read_text())
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=6) as p:rows=[r for group in p.map(scene_job,gm['rough_scene_ids']) for r in group]
    assert len(rows)==256;csvwrite(RUN/'manifests/rough_support_references.csv',rows)
    event('rough_visibility_reference_complete',existing_poses=256,fresh_frame_records_added=0,neural_forwards=0,workers=6)
    print('Derived256 clean reference visibility masks on existing locked rough poses;0 new evaluation records/forwards',flush=True)
if __name__=='__main__':main()
