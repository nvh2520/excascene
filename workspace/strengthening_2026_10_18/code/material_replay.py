"""Original train/validation scenes only: exact visible-terrain material labels."""
from common import *
import argparse,time
from concurrent.futures import ProcessPoolExecutor
from kinafford.sim.scene import sample_frame,SEM_TERRAIN
from kinafford.sim.render import Renderer
from kinafford.sim.generate import frame_rng
from kinafford.sim.kinematics import CAMERAS

class LabelRenderer(Renderer):
    def cast(self,*args,**kwargs):
        hit=super().cast(*args,**kwargs);self.last_hit=hit;return hit

def one(task):
    split,sid=task;out=ROOT/f'phase_a/material_replay/{split}/f_{sid:06d}_0.npz'
    assert not out.exists(), str(out)
    start=time.time();scene=generate_scene(sid);q,phase=sample_frame(scene,0)
    original=load_npz(PROJECT/f'data/{split}/f_{sid:06d}_0.npz',['q_true'])
    assert np.array_equal(q.astype(np.float32),original['q_true'])
    renderer=LabelRenderer(scene);mb=renderer._machine_boxes(q);st=renderer._static_objects()
    rng=frame_rng('clean',sid,0);records=[];patches=[];targets=[];camids=[];centers=[]
    for cam,name in enumerate(CAMERAS):
        d=renderer.render_camera(name,q,rng,machine_boxes=mb,static=st)
        hit=renderer.last_hit;terrain=hit['cls']==SEM_TERRAIN
        mat=np.full(terrain.shape,255,np.uint8)
        mat[terrain]=scene.material_at(hit['P'][terrain,0],hit['P'][terrain,1])
        mat=mat.reshape(d['sem'].shape)
        assert np.array_equal(mat<4,d['sem']==SEM_TERRAIN)
        records.append(dict(rgb=d['rgb'],depth=d['depth'],img_sem=d['sem'],img_mat=mat))
        coords=np.column_stack(np.where(mat<4))
        prng=np.random.default_rng(np.random.SeedSequence([2026101804,sid,cam]))
        if len(coords)>128:coords=coords[prng.choice(len(coords),128,replace=False)]
        img=np.concatenate([d['rgb'].astype(np.float32)/255,np.clip(d['depth'][...,None],0,20)/20],axis=-1)
        padded=np.pad(img,((4,4),(4,4),(0,0)),mode='edge')
        for y,x in coords:
            patches.append(padded[y:y+9,x:x+9].transpose(2,0,1).astype(np.float16));targets.append(mat[y,x]);camids.append(cam);centers.append((y,x))
    payload={k:np.stack([r[k] for r in records]) for k in records[0]}
    payload.update(camera_present=np.ones(3,np.uint8),scene_id=np.int32(sid),frame_id=np.int32(0),q_true=q,phase=np.str_(phase))
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**payload)
    patchpath=ROOT/f'phase_a/material_patches/{split}/f_{sid:06d}_0.npz';patchpath.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(patchpath,x=np.stack(patches) if patches else np.empty((0,4,9,9),np.float16),y=np.array(targets,np.uint8),camera=np.array(camids,np.uint8),center=np.array(centers,np.int16))
    return dict(split=split,scene_id=sid,frame_id=0,cameras=3,terrain_pixels=sum(int((r['img_mat']<4).sum()) for r in records),
        patches=len(targets),class_counts=np.bincount(targets,minlength=4).tolist(),original_pose_exact=True,
        image_path=str(out.relative_to(ROOT)),image_sha256=sha(out),patch_path=str(patchpath.relative_to(ROOT)),patch_sha256=sha(patchpath),wall_seconds=time.time()-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    tasks=[('train',s) for s in range(400)]+[('val',s) for s in range(3000,3100)];rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(one,tasks)):
            rows.append(r)
            if (i+1)%25==0:print(i+1,len(tasks),flush=True)
    csvwrite(ROOT/'phase_a/material_replay_inventory.csv',rows)
    js(ROOT/'phase_a/material_replay_receipt.json',dict(status='COMPLETE',records=len(rows),train_frames=400,validation_frames=100,
        exact_ray_hit_labels=True,terrain_rule='hit class == terrain, not combined LiDAR/camera visibility',
        original_pose_exact=all(r['original_pose_exact'] for r in rows),inventory_sha256=sha(ROOT/'phase_a/material_replay_inventory.csv')))
if __name__=='__main__':main()
