"""Locked paired-condition generator. Historical source is imported without modification."""
import copy,hashlib,json,time,os
from common_afast import *
from kinafford.sim.scene import generate_scene,bev_ground_truth,_pose_ok,SEM_SKY
from kinafford.sim.kinematics import CAMERAS,SENSORS,sample_joint_config,T_B_from_U,sensor_pose_U
from kinafford.sim.render import Renderer,visibility_mask
from kinafford.sim.degrade import SEV,CAM_PHOTO_KINDS,sample_spec,reliability_targets,drift_transform,joint_bias
from scipy.ndimage import gaussian_filter
NOMINAL=['clean','iid','all_s2','cam_s1','cam_s2','cam_s3']
CAMERA_ONLY=['clean','cam_s1','cam_s2','cam_s3','appearance_only','depth_only','absence_only','combined_present']
ROOT_SEED=2026101302
def rng(scene,slot,namespace,sensor=0,identity=0,attempt=0):
    family=int(scene>=310000)
    return np.random.default_rng(np.random.SeedSequence([ROOT_SEED,family,scene,slot,namespace,sensor,identity,attempt]))

def pose(scene,slot,attempt,wide):
    r=rng(scene.seed,slot,11 if wide else 10,attempt=attempt)
    q3,component=sample_joint_config(r,ood_artic=wide);phase=component
    if r.uniform()<.7:th=scene.theta_work+r.normal(0,np.deg2rad(25))
    else:th=scene.theta_truck+r.normal(0,np.deg2rad(12));phase='dump'
    return np.r_[np.arctan2(np.sin(th),np.cos(th)),q3],component,phase

def specs(scene,slot,rough,extra=False):
    conditions=['rough_iid'] if rough else [c for c in NOMINAL if c!='wider_articulation']+(['combined_present'] if extra else [])
    out={};dropped=int(rng(scene,slot,20).integers(3))
    mix=sample_spec('iid',rng(scene,slot,21))
    for condition in conditions:
        if condition in ['iid','rough_iid']:sp=copy.deepcopy(mix)
        else:
            level=2 if condition=='all_s2' else int(condition[-1]) if condition.startswith('cam_s') else 3 if condition in ['appearance_only','depth_only','combined_present'] else 0
            sp=dict(cam={},lidar=dict(level=2 if condition=='all_s2' else 0,ext=2 if condition=='all_s2' else 0),joint=2 if condition=='all_s2' else 0)
            for i,c in enumerate(CAMERAS):
                sp['cam'][c]=dict(drop=bool(i==dropped and condition in ['cam_s3','absence_only']),
                    photo=[(k,level) for k in CAM_PHOTO_KINDS] if level else [],ext=2 if condition=='all_s2' else 0)
        sp['suite']=condition;out[condition]=sp
    return out

def drifts(scene,slot,sp):
    return {c:drift_transform(sp['cam'][c]['ext'],rng(scene,slot,30,i)) for i,c in enumerate(CAMERAS)}|{
        'lidar':drift_transform(sp['lidar']['ext'],rng(scene,slot,30,3))}

def validity(scene,renderer,q,specs_by_condition):
    reasons=[]
    if not _pose_ok(scene,q):reasons.append('original_pose_check')
    boxes,cyls=renderer._static_objects();boxes=renderer._machine_boxes(q)+boxes
    TB=T_B_from_U(q)
    for name,sp in specs_by_condition.items():
        ds=drifts(scene.seed,validity.slot,sp)
        for c in CAMERAS:
            if sp['cam'][c]['drop']:continue
            p=(TB@sensor_pose_U(c,q,drift=ds[c]))[:3,3]
            if p[2]<float(scene.height_at([p[0]],[p[1]])[0])+.001:reasons.append(f'{name}:{c}:terrain_clearance')
            for i,(T,half,_) in enumerate(boxes):
                local=(p-T[:3,3])@T[:3,:3]
                distance=np.linalg.norm(np.maximum(np.abs(local)-half,0))
                if distance<.001:reasons.append(f'{name}:{c}:box_{i}_clearance')
            for i,(x,y,r,z0,z1,_) in enumerate(cyls):
                distance=np.hypot(max(np.hypot(p[0]-x,p[1]-y)-r,0),max(z0-p[2],p[2]-z1,0))
                if distance<.001:reasons.append(f'{name}:{c}:cylinder_{i}_clearance')
    return sorted(set(reasons))

def camera_latents(scene,slot,i):
    H,W=SENSORS[CAMERAS[i]]['H'],SENSORS[CAMERAS[i]]['W'];shape=(H,W)
    yy,xx=np.mgrid[:H,:W];mask=np.zeros(shape,bool);masks=[];ellipses=[];r=rng(scene,slot,40,i,0)
    for fraction in SEV['occl_frac']:
        while mask.sum()<fraction*H*W:
            cy,cx=r.uniform(0,H),r.uniform(0,W);ry,rx=r.uniform(8,.35*H),r.uniform(8,.35*W);th=r.uniform(0,np.pi)
            dx,dy=xx-cx,yy-cy;lx=np.cos(th)*dx+np.sin(th)*dy;ly=-np.sin(th)*dx+np.cos(th)*dy
            mask|=(lx/rx)**2+(ly/ry)**2<=1;ellipses.append([cy,cx,ry,rx,th])
        masks.append(mask.copy())
    return dict(occlusion_masks=np.stack(masks),ellipses=np.array(ellipses),
        blob=gaussian_filter(rng(scene,slot,40,i,1).normal(0,12,(H,W,3))+[70,55,40],(3,3,0)),
        haze=np.array([200,190,170])+gaussian_filter(rng(scene,slot,40,i,2).normal(0,25,(H,W,3)),(6,6,0)),
        dust_u=rng(scene,slot,40,i,3).uniform(size=shape),exposure_u=rng(scene,slot,40,i,4).uniform(size=()),
        noise_z=rng(scene,slot,40,i,5).normal(size=(H,W,3)),depth_z=rng(scene,slot,40,i,6).normal(size=shape),
        depth_u=rng(scene,slot,40,i,7).uniform(size=shape),shade_z=rng(scene,slot,50,i).normal(size=(H*W,3)))

def degrade_shared(rgb,depth,photo,latent):
    r,d=rgb.copy(),depth.copy()
    for kind,level in photo:
        j=level-1;img=r.astype(np.float32)
        if kind=='occlusion':
            m=latent['occlusion_masks'][j];img[m]=latent['blob'][m];d[m]=0
        elif kind=='dust':
            a=SEV['dust_alpha'][j];img=(1-a)*img+a*latent['haze'];d[latent['dust_u']<SEV['dust_holes'][j]]=0
        elif kind=='blur':
            lo,hi=SEV['exposure'][j];img=gaussian_filter(img,(SEV['blur_sigma'][j],SEV['blur_sigma'][j],0))
            img*=lo+(hi-lo)*latent['exposure_u']
        elif kind=='noise':img=img+SEV['noise_sigma'][j]*latent['noise_z']
        elif kind=='depth':
            valid=d>0;d[valid]*=1+SEV['depth_rel'][j]*latent['depth_z'][valid]
            d[latent['depth_u']<SEV['depth_holes'][j]]=0
        else:raise ValueError(kind)
        r=np.clip(img,0,255).astype(np.uint8);d=d.astype(np.float32)
    return r,d

class Shade:
    def __init__(self,z):self.z=z
    def normal(self,loc,scale,size):
        assert tuple(size)==self.z.shape
        return loc+scale*self.z

def lidar_latents(scene,slot):
    sh=(SENSORS['lidar']['rings'],SENSORS['lidar']['azimuths'])
    return dict(drop_u=rng(scene,slot,41,3,0).uniform(size=sh),noise_z=rng(scene,slot,41,3,1).normal(size=sh),
                spur_u=rng(scene,slot,41,3,2).uniform(size=sh),range_u=rng(scene,slot,41,3,3).uniform(size=sh))
def lidar_shared(lo,level,L):
    rr=lo['range'].copy();vv=lo['valid'].copy();ss=lo['sem'].copy()
    if not level:return rr,vv,ss
    j=level-1;keep=SEV['lidar_ring_keep'][j]
    if keep>1:vv[np.arange(vv.shape[0])%keep!=0]=False
    drop=L['drop_u']<SEV['lidar_drop'][j];vv&=~drop
    rr[vv]+=SEV['lidar_range_sigma'][j]*L['noise_z'][vv]
    spur=(L['spur_u']<SEV['lidar_spurious'][j])&~drop
    rr[spur]=.5+2*L['range_u'][spur];vv[spur]=True;ss[spur]=SEM_SKY;rr[~vv]=0
    return rr,vv,ss

def render_bundle(scene,renderer,slot,q,phase,component,sp_by,wide=False):
    mb=renderer._machine_boxes(q);st=renderer._static_objects();cache={};gt=bev_ground_truth(scene,q)
    L=[camera_latents(scene.seed,slot,i) for i in range(3)];LL=lidar_latents(scene.seed,slot)
    latent_arrays={f'camera{i}_{k}':v for i,l in enumerate(L) for k,v in l.items()}|{'lidar_'+k:v for k,v in LL.items()}
    # Wider pose uses the same photometric sensor-noise latent field, not a new condition-specific redraw.
    if not wide:npzwrite(RUN/f'latents/{scene.seed}_{slot}.npz',**latent_arrays)
    records={}
    for name,sp in sp_by.items():
        ds=drifts(scene.seed,slot,sp);rgb=[];depth=[];sem=[];present=[];hits=[]
        for i,c in enumerate(CAMERAS):
            key=(c,ds[c].tobytes())
            if key not in cache:cache[key]=renderer.render_camera(c,q,Shade(L[i]['shade_z']),drift=ds[c],machine_boxes=mb,static=st)
            base=cache[key];r,d=degrade_shared(base['rgb'],base['depth'],sp['cam'][c]['photo'],L[i])
            if name=='appearance_only':d=base['depth'].copy()
            if name=='depth_only':r=base['rgb'].copy()
            if sp['cam'][c]['drop']:
                rgb.append(np.zeros_like(r));depth.append(np.zeros_like(d));sem.append(np.zeros_like(base['sem']));present.append(0)
            else:
                rgb.append(r);depth.append(d);sem.append(base['sem']);present.append(1);hits.append(base['hits_B'])
        key=('lidar',ds['lidar'].tobytes())
        if key not in cache:cache[key]=renderer.render_lidar(q,None,drift=ds['lidar'],machine_boxes=mb,static=st)
        lo=cache[key];lr,lv,ls=lidar_shared(lo,sp['lidar']['level'],LL);hits.append(lo['hits_B'])
        rec=dict(rgb=np.stack(rgb),depth=np.stack(depth).astype(np.float16),img_sem=np.stack(sem),cam_present=np.array(present,np.uint8),
          lidar_range=lr.astype(np.float16),lidar_valid=lv.astype(np.uint8),lidar_sem=ls,
          q_true=q.astype(np.float32),q_rep=(q+joint_bias(sp['joint'],rng(scene.seed,slot,31))).astype(np.float32),
          drift=np.stack([ds[c] for c in CAMERAS]+[ds['lidar']]).astype(np.float32),
          gt_H=gt['H'],gt_sem=gt['sem'],gt_mat=gt['mat'],gt_haz=gt['haz'],gt_aff=gt['aff'],gt_reach=gt['reach'],
          gt_aff_fp16=gt['aff'].astype(np.float16),gt_vis=visibility_mask(hits,q),rel_gt=reliability_targets(sp),
          z_target=np.float32(scene.z_target),scene_id=np.int32(scene.seed),frame_id=np.int32(slot),
          phase=np.str_(phase),component=np.str_(component),spec=np.str_(json.dumps(sp)),suite=np.str_(name))
        records[name]=rec
    if 'clean' in records:
        for rec in records.values():rec['gt_vis_clean']=records['clean']['gt_vis']
    else:
        for rec in records.values():rec['gt_vis_clean']=rec['gt_vis']
    return records

def assert_pairs(records):
    if 'clean' not in records:return []
    from inputs import tensorize
    clean=records['clean'];rows=[];tensors={k:tensorize(v) for k,v in records.items()}
    unchanged=['lidar_range','lidar_valid','q','T_dyn','T_static','K']
    for name in CAMERA_ONLY:
        if name not in records:continue
        for field in unchanged:assert np.array_equal(tensors['clean'][field],tensors[name][field]),(name,field)
        for field in ['gt_H','gt_aff','gt_reach','gt_sem','gt_haz','z_target','q_true']:
            assert np.array_equal(clean[field],records[name][field]),(name,field)
    if 'appearance_only' in records:
        for f in ['depth','cam_present','meta']:assert np.array_equal(tensors['clean'][f],tensors['appearance_only'][f]),f
        assert np.array_equal(tensors['clean']['img'][:,3],tensors['appearance_only']['img'][:,3])
    if 'depth_only' in records:
        for f in ['cam_present','meta']:assert np.array_equal(tensors['clean'][f],tensors['depth_only'][f]),f
        assert np.array_equal(tensors['clean']['img'][:,:3],tensors['depth_only']['img'][:,:3])
    if 'combined_present' in records:
        ad=tensors['combined_present'];a=tensors['appearance_only'];d=tensors['depth_only']
        assert np.array_equal(ad['img'][:,:3],a['img'][:,:3]);assert np.array_equal(ad['depth'],d['depth'])
        assert np.array_equal(ad['img'][:,3],d['img'][:,3])
        keep=tensors['cam_s3']['cam_present']>.5
        for f in ['img','depth','cam_present']:
            assert np.array_equal(tensors['cam_s3'][f][keep],ad[f][keep]),f
            assert not np.any(tensors['cam_s3'][f][~keep]),f
    if 'absence_only' in records:
        keep=tensors['absence_only']['cam_present']>.5
        for f in ['img','depth','cam_present']:
            assert np.array_equal(tensors['absence_only'][f][keep],tensors['clean'][f][keep]);assert not np.any(tensors['absence_only'][f][~keep])
    for name,t in tensors.items():
        assert np.array_equal(t['img'][:,3],t['depth']/20*t['cam_present'][:,None,None])
        rows.append(dict(condition=name,scene_id=int(records[name]['scene_id']),frame_id=int(records[name]['frame_id']),
            input_tensor_sha256=hashlib.sha256(''.join(array_sha(t[k]) for k in sorted(t)).encode()).hexdigest(),pairing_pass=True))
    return rows

def scene_job(scene_id):
    start=time.process_time();manifest=json.loads((RUN/'manifests/generation_manifest.json').read_text())
    rough=scene_id>=310000;geometry_seed=int(rng(scene_id,0,1).integers(0,2**31-1));scene=generate_scene(geometry_seed,ood_terrain=rough);scene.seed=scene_id;renderer=Renderer(scene)
    extra=False;attempts=[];slots=[];pairs=[];files=[]
    for slot in range(4):
        for wide in [False]:
            sp=specs(scene_id,slot,rough,extra)
            if wide:sp={'wider_articulation':copy.deepcopy(sp['clean'])};sp['wider_articulation']['suite']='wider_articulation'
            accepted=False
            for attempt in range(50):
                q,component,phase=pose(scene,slot,attempt,wide);validity.slot=slot
                failures=validity(scene,renderer,q,sp)
                attempts.append(dict(scene_id=scene_id,frame_id=slot,wide=wide,attempt=attempt,component=component,phase=phase,
                    q=q.tolist(),reasons=failures,accepted=not failures))
                if not failures:accepted=True;break
            slots.append(dict(scene_id=scene_id,frame_id=slot,population='rough' if rough else 'wide' if wide else 'nominal',
                accepted=accepted,attempts=attempt+1,component=component,phase=phase,grade=scene.z_target))
            if not accepted:continue
            records=render_bundle(scene,renderer,slot,q,phase,component,sp,wide)
            for name,rec in records.items():
                path=RUN/f'data/{name}/f_{scene_id:06d}_{slot}.npz';assert not path.exists()
                npzwrite(path,**rec)
                # Identities are tested after lossless serialization and canonical tensor construction.
                with np.load(path) as z:records[name]={k:z[k] for k in z.files}
                files.append(dict(scene_id=scene_id,frame_id=slot,condition=name,path=str(path.relative_to(RUN)),sha256=sha(path),bytes=path.stat().st_size))
            pairs.extend(assert_pairs(records))
    js(RUN/f'generation/{scene_id}.json',dict(slots=slots,attempts=attempts,pairs=pairs,files=files,cpu_seconds=time.process_time()-start,geometry_seed=geometry_seed))
    return scene_id,len(files),time.process_time()-start

def main():
    locked();m=json.loads((RUN/'manifests/generation_manifest.json').read_text());(RUN/'generation').mkdir(exist_ok=False)
    from concurrent.futures import ProcessPoolExecutor
    ids=m['nominal_scene_ids']+m['rough_scene_ids'];print('generation',len(ids),'scenes',flush=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,(scene,n,cpu) in enumerate(pool.map(scene_job,ids)):
            print(i+1,scene,n,'cpu_s',round(cpu,2),flush=True)
    results=[json.loads(p.read_text()) for p in sorted((RUN/'generation').glob('*.json'))]
    for k,name in [('slots','generation_slots.csv'),('attempts','generation_attempts.csv'),('pairs','pairing_identities.csv'),('files','generated_files.csv')]:
        csvwrite(RUN/'manifests'/name,[r for x in results for r in x[k]])
    import pandas as pd
    slots=pd.read_csv(RUN/'manifests/generation_slots.csv');summary=slots.groupby('population').accepted.agg(['count','sum'])
    summary['invalid_fraction']=1-summary['sum']/summary['count'];summary.to_csv(RUN/'generation_validity_summary.csv')
    total=int(sum(len(r['files']) for r in results));assert total<=3328
    js(RUN/'generation_status.json',dict(records=total,planned=3328,populations=summary.reset_index().to_dict('records'),
        overall_invalid_fraction=float(1-slots.accepted.mean()),complete=True))
    assert float(1-slots.accepted.mean())<=.10,'Whole-bundle exhaustion exceeded10%' 
if __name__=='__main__':main()
