"""Paired validated bundles in a new namespace; test generation requires a final lock."""
from common import *
import argparse,copy,importlib.util,os,time
from concurrent.futures import ProcessPoolExecutor
from faults import novel_latents,apply_novel
sys.path.insert(0,str(AFAST/'code'))
spec=importlib.util.spec_from_file_location('frozen_pair_generator',AFAST/'code/generate_holdout.py')
gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
from kinafford.sim.kinematics import CAMERAS
from kinafford.sim.render import Renderer
from kinafford.sim.degrade import sample_spec,CAM_PHOTO_KINDS
OUT=None;MODE=None
NOMINAL=['clean','iid','cam_s1','cam_s2','cam_s3','single_severe','novel','all_s2']

def phase_rng(scene,slot,namespace,sensor=0,identity=0,attempt=0):
    return np.random.default_rng(np.random.SeedSequence([2026102007,scene,slot,namespace,sensor,identity,attempt]))

def specs(sid,slot,rough):
    if rough:
        sp=sample_spec('iid',phase_rng(sid,slot,21));sp['suite']='rough_iid';return {'rough_iid':sp}
    mix=sample_spec('iid',phase_rng(sid,slot,21));chosen=int(phase_rng(sid,slot,20).integers(3));out={}
    for condition in NOMINAL:
        if condition=='iid':sp=copy.deepcopy(mix)
        else:
            level=int(condition[-1]) if condition.startswith('cam_s') else 2 if condition=='all_s2' else 0
            sp=dict(cam={},lidar=dict(level=2 if condition=='all_s2' else 0,ext=2 if condition=='all_s2' else 0),joint=2 if condition=='all_s2' else 0)
            for i,c in enumerate(CAMERAS):
                l=3 if condition=='single_severe' and i==chosen else level
                sp['cam'][c]=dict(drop=bool(condition=='cam_s3' and i==chosen),photo=[(k,l) for k in CAM_PHOTO_KINDS] if l else [],ext=2 if condition=='all_s2' else 0)
            if condition=='novel':
                kind='mud_water' if (sid+slot)%2==0 else 'sun_glare'
                sp['cam'][CAMERAS[chosen]]['novel']=kind
        sp['suite']=condition;out[condition]=sp
    return out

def one(task):
    sid,rough,slots=task;start=time.time();gen.RUN=OUT;gen.rng=phase_rng
    geometry_seed=int(phase_rng(sid,0,1).integers(0,2**31-1))
    scene=generate_scene(geometry_seed,ood_terrain=rough);scene.seed=sid;renderer=Renderer(scene)
    geometry_hash=hashlib.sha256()
    for arr in [scene.h,scene.mat,scene.rock]:geometry_hash.update(np.ascontiguousarray(arr).tobytes())
    attempts=[];slotrows=[];files=[];pairs=[]
    for slot in range(slots):
        sp=specs(sid,slot,rough);accepted=False
        for attempt in range(50):
            q,component,phase=gen.pose(scene,slot,attempt,False);gen.validity.slot=slot
            reasons=gen.validity(scene,renderer,q,sp)
            attempts.append(dict(scene_id=sid,frame_id=slot,attempt=attempt,q=q.tolist(),reasons=reasons,accepted=not reasons,rough=rough))
            if not reasons:accepted=True;break
        slotrows.append(dict(scene_id=sid,frame_id=slot,rough=rough,accepted=accepted,attempts=attempt+1))
        if not accepted:continue # entire paired bundle omitted, never condition-specific.
        novel={}
        if 'novel' in sp:
            for cam,c in enumerate(CAMERAS):
                if 'novel' in sp['novel']['cam'][c]:
                    kind=sp['novel']['cam'][c]['novel'];L=novel_latents((160,256),kind,phase_rng(sid,slot,60,cam));novel[cam]=L
                    path=OUT/f'latents/novel_{sid}_{slot}_{cam}.npz';path.parent.mkdir(exist_ok=True,parents=True)
                    np.savez_compressed(path,**L)
        records=gen.render_bundle(scene,renderer,slot,q,phase,component,sp)
        if 'novel' in records:
            for cam,L in novel.items():
                d=records['novel'];d['rgb'][cam],depth=apply_novel(d['rgb'][cam],d['depth'][cam].astype(np.float32),L);d['depth'][cam]=depth.astype(np.float16)
        if 'clean' in records:
            base=records['clean']
            for condition in ['cam_s1','cam_s2','cam_s3','single_severe','novel']:
                if condition not in records:continue
                d=records[condition]
                for key in ['lidar_range','lidar_valid','q_rep','q_true','gt_H','gt_aff','gt_haz','gt_reach','drift']:
                    assert np.array_equal(base[key],d[key]),('retained input',sid,slot,condition,key)
                pairs.append(dict(scene_id=sid,frame_id=slot,condition=condition,retained_input_equal=True))
        for condition,d in records.items():
            d['camera_fault_severity']=np.asarray([max([int(level) for _,level in sp[condition]['cam'][c]['photo']],default=0) for c in CAMERAS],np.uint8)
            d['camera_novel_fault']=np.asarray([bool(sp[condition]['cam'][c].get('novel')) for c in CAMERAS],np.uint8)
            path=OUT/f'data/{condition}/f_{sid:06d}_{slot}.npz';assert not path.exists();gen.npzwrite(path,**d)
            files.append(dict(scene_id=sid,frame_id=slot,condition=condition,path=str(path.relative_to(OUT)),sha256=sha(path),bytes=path.stat().st_size))
    js(OUT/f'generation/{sid}.json',dict(geometry_seed=geometry_seed,source_geometry_sha256=geometry_hash.hexdigest(),scene_id=sid,rough=rough,attempts=attempts,slots=slotrows,files=files,pairs=pairs,seconds=time.time()-start))
    return dict(scene_id=sid,files=len(files),seconds=time.time()-start)

def main():
    global MODE,OUT
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['holdout']);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    MODE=a.mode;OUT=ROOT/'study4/holdout'
    assert not OUT.exists(), 'Use an explicit recovery directory, never overwrite a generation run'
    verify_protocol()
    config=json.loads((ROOT/'protocol/holdout_manifest.json').read_text())
    tasks=[(s,False,4) for s in config['nominal_scene_ids']]+[(s,True,4) for s in config['rough_scene_ids']]
    OUT.mkdir(parents=True);rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,r in enumerate(pool.map(one,tasks)):
            rows.append(r)
            if (i+1)%10==0:print(MODE,i+1,len(tasks),flush=True)
    records=[json.loads(f.read_text()) for f in sorted((OUT/'generation').glob('*.json'))]
    for key in ['files','slots','attempts','pairs']:csvwrite(OUT/f'{key}.csv',[r for rec in records for r in rec[key]])
    js(OUT/'generation_receipt.json',dict(status='GENERATION_COMPLETE_PENDING_VALIDITY_AUDIT',mode=MODE,scenes=len(tasks),records=sum(r['files'] for r in rows),
        file_manifest_sha256=sha(OUT/'files.csv'),code_sha256=sha(__file__),whole_bundle_exclusion=True))
if __name__=='__main__':main()
