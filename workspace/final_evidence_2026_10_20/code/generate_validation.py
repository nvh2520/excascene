"""Full original nominal validation pose/target bank and separate rough replay."""
from common import *
import copy,importlib.util,time
from concurrent.futures import ProcessPoolExecutor
from kinafford.sim.kinematics import CAMERAS
from kinafford.sim.render import Renderer
from kinafford.sim.degrade import sample_spec,CAM_PHOTO_KINDS
sys.path.insert(0,str(AFAST/'code'))
spec=importlib.util.spec_from_file_location('development_pair_generator',AFAST/'code/generate_holdout.py')
gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
OUT=ROOT/'study4/validation';POP=None
CONDITIONS=['clean','iid','cam_s1','cam_s2','cam_s3','single_severe','all_s2']
def rng(scene,slot,namespace,sensor=0,identity=0,attempt=0):
    return np.random.default_rng(np.random.SeedSequence([2026102006,int(POP=='rough'),scene,slot,namespace,sensor,identity,attempt]))
def specs(sid,slot,rough):
    if rough:
        s=sample_spec('iid',rng(sid,slot,21));s['suite']='rough_iid';return {'rough_iid':s}
    chosen=int(rng(sid,slot,20).integers(3));rows={}
    for c in CONDITIONS:
        if c=='iid':s=sample_spec('iid',rng(sid,slot,21))
        else:
            level=int(c[-1]) if c.startswith('cam_s') else 2 if c=='all_s2' else 0
            s=dict(cam={},lidar=dict(level=2 if c=='all_s2' else 0,ext=2 if c=='all_s2' else 0),joint=2 if c=='all_s2' else 0)
            for j,name in enumerate(CAMERAS):
                l=3 if c=='single_severe' and j==chosen else level
                s['cam'][name]=dict(drop=bool(c=='cam_s3' and j==chosen),photo=[(k,l) for k in CAM_PHOTO_KINDS] if l else [],ext=2 if c=='all_s2' else 0)
        s['suite']=c;rows[c]=s
    return rows
def one(task):
    global POP
    sid,rough=task;POP='rough' if rough else 'nominal';dest=OUT/POP
    gen.RUN=dest;gen.rng=rng;scene=generate_scene(sid,ood_terrain=rough);scene.seed=sid;renderer=Renderer(scene)
    files=[];slots=[];attempts=[];original=[]
    for slot in range(4):
        source=PROJECT/f'data/val/f_{sid:06d}_{slot}.npz';base=load_npz(source)
        assert int(base['scene_id'])==sid and int(base['frame_id'])==slot
        sp=specs(sid,slot,rough)
        if not rough:
            q=base['q_true'].astype(np.float64);phase=str(base['phase']);component=str(base['component']) if 'component' in base else 'UNRECORDED_ORIGINAL_VALIDATION_COMPONENT';accepted=True
            attempts.append(dict(population=POP,scene_id=sid,frame_id=slot,attempt=0,accepted=True,reasons=[],q=q.tolist(),source='exact saved original validation pose; no additional rejection'))
            original.append(dict(scene_id=sid,frame_id=slot,source_path=str(source.relative_to(PROJECT)),source_sha256=sha(source),pose_equal=True))
        else:
            accepted=False
            for attempt in range(50):
                q,component,phase=gen.pose(scene,slot,attempt,False);gen.validity.slot=slot;reasons=gen.validity(scene,renderer,q,sp)
                attempts.append(dict(population=POP,scene_id=sid,frame_id=slot,attempt=attempt,accepted=not reasons,reasons=reasons,q=q.tolist(),source='original generator validity procedure, rough validation replay'))
                if not reasons:accepted=True;break
        slots.append(dict(population=POP,scene_id=sid,frame_id=slot,accepted=accepted))
        if not accepted:continue
        records=gen.render_bundle(scene,renderer,slot,q,phase,component,sp)
        for condition,d in records.items():
            if not rough:
                # Preserve the full original nominal target bank, including its
                # recorded precision; standardized sensor-fault inputs are a
                # declared development replay, not original byte-identical RGB.
                for key in ['gt_H','gt_sem','gt_mat','gt_haz','gt_aff','gt_reach','z_target']:d[key]=base[key].copy()
                assert np.array_equal(d['q_true'],base['q_true'])
            path=dest/f'data/{condition}/f_{sid:06d}_{slot}.npz';assert not path.exists();gen.npzwrite(path,**d)
            files.append(dict(condition=condition,scene_id=sid,frame_id=slot,population=POP,path=str(path.relative_to(OUT)),sha256=sha(path),bytes=path.stat().st_size))
    js(OUT/f'generation/{POP}_{sid}.json',dict(population=POP,scene_id=sid,geometry_seed=sid,slots=slots,attempts=attempts,original=original,files=files))
    return dict(scene_id=sid,population=POP,records=len(files))
if __name__=='__main__':
    assert not OUT.exists();OUT.mkdir(parents=True)
    tasks=[(sid,rough) for rough in [False,True] for sid in range(3000,3300)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(one,tasks)):
            if (i+1)%20==0:print('validation scenes',i+1,len(tasks),flush=True)
    receipts=[json.loads(p.read_text()) for p in sorted((OUT/'generation').glob('*.json'))]
    for name in ['files','slots','attempts','original']:csvwrite(OUT/f'{name}.csv',[r for rec in receipts for r in rec[name]])
    nominal=[r for rec in receipts for r in rec['files'] if r['population']=='nominal'];assert len(nominal)==8400
    js(OUT/'completion.json',dict(status='COMPLETE_DEVELOPMENT_ONLY',nominal_source_frames=1200,nominal_records=len(nominal),rough_records=sum(len(r['files']) for r in receipts if r['population']=='rough'),
        validation_pools=8,selection_objective='equal eight-condition direct regret; all three seeds; full25pair grid; no novel faults',source='all original nominal IDs3000–3299, four poses/targets each, standardized known-fault input replay; rough geometry replay uses these original development geometry seeds and valid poses',no_holdout_access=True))
