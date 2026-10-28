"""New-phase helpers: immutable historical inputs, separate output root."""
from pathlib import Path
import ast,csv,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
AFAST=PROJECT/'external_validation_run_EV1_AFAST'
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(AFAST/'source/excascene_snapshot'))
from kinafford.sim.scene import (generate_scene,bev_grid,dist_to_rect,rot_z,BEV_RES,
    MAT_DIGGABILITY,smoothstep,PERSON_EXCL,OBSTACLE_EXCL,TRUCK_BUFFER,MACHINE_BUFFER)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def js(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def csvwrite(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        if not rows:return
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
def load_npz(path,keys=None):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in (keys or z.files)}

def verify_protocol():
    lock=json.loads((ROOT/'protocol/study4_final_lock.json').read_text())
    assert lock['status']=='LOCKED_BEFORE_STUDY4_HOLDOUT_GENERATION'
    for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,('changed locked final-round file',rel)
    for rel,h in lock['external_files'].items():assert sha(PROJECT/rel)==h,('changed historical dependency',rel)
    return lock

tree=ast.parse((AFAST/'source/selectors_v5_original.py').read_text())
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='choose'],type_ignores=[]),'<frozen choose>','exec'),globals())

CAUSES=('person_buffer','obstacle_buffer','truck_buffer','machine_footprint_buffer','non_terrain')
def safety_masks(scene,q,sem,scale=1.):
    X,Y=bev_grid();R=rot_z(float(q[0]))[:3,:3]
    xb=R[0,0]*X+R[0,1]*Y;yb=R[1,0]*X+R[1,1]*Y
    person=np.zeros(X.shape,bool);obstacle=person.copy();truck=person.copy()
    for px,py in scene.persons:person|=np.hypot(xb-px,yb-py)<=PERSON_EXCL*scale
    for px,py in scene.obstacles:obstacle|=np.hypot(xb-px,yb-py)<=OBSTACLE_EXCL*scale
    if scene.truck is not None:
        t=scene.truck
        truck=dist_to_rect(xb,yb,*t['center'],t['yaw'],t['half'][0]+2*t['cab']['half'][0],t['half'][1])<=TRUCK_BUFFER*scale
    b=MACHINE_BUFFER*scale
    machine=((np.abs(xb)<=2.30+b)&(np.abs(yb)<=1.50+b))|((X>=-3.10-b)&(X<=1.90+b)&(np.abs(Y)<=1.35+b))
    return np.stack([person,obstacle,truck,machine,sem!=0])

def historical_scene(scene_id):
    rec=json.loads((AFAST/f'generation/{scene_id}.json').read_text())
    scene=generate_scene(rec['geometry_seed'],ood_terrain=scene_id>=310000)
    poses={int(r['frame_id']):np.asarray(r['q'],np.float64) for r in rec['attempts'] if r['accepted'] and not r['wide']}
    return scene,poses

def soft_score(H,mat,terrain,haz,reach,z,material_scale=1.,productivity=1.2,slope_band=(40.,60.)):
    dig=(mat*(MAT_DIGGABILITY*material_scale)[:,None,None]).sum(0)
    gy,gx=np.gradient(H,BEV_RES);slope=np.degrees(np.arctan(np.hypot(gx,gy)))
    return (dig*(1-smoothstep(slope,*slope_band))*reach*np.clip((H-z)/productivity,0,1)*(1-haz)*terrain).astype(np.float32)

def selected_metrics(score,d,hazard=None,truth_utility=None,truth_unsafe=None):
    reach=d['gt_reach'];u=d['gt_aff'] if truth_utility is None else truth_utility
    support=np.isfinite(reach)&(reach>=.05)
    oracle=float(u[support].max()) if support.any() else 0.
    pick,ties,n=choose(score,reach,hazard=hazard)
    unsafe=(d['gt_haz']>.5)|(d['gt_sem']!=0) if truth_unsafe is None else truth_unsafe
    return dict(regret=oracle-(float(u[pick]) if pick is not None else 0.),unsafe=int(pick is not None and unsafe[pick]),
        executed=int(pick is not None),abstained=int(pick is None),pick_row=-1 if pick is None else int(pick[0]),
        pick_col=-1 if pick is None else int(pick[1]),oracle=oracle,tie_count=ties,eligible_cells=n)
