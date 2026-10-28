"""Frozen inference only. No optimizer, training, calibration or checkpoint selection."""
from common import *
import copy,pandas as pd,torch
from inputs import load,batch
from selectors_v5 import evaluate,geometry
from stage4.models import build_model
from kinafford.evaluate import predict as original_predict

def state_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(array_sha(v.detach().cpu().numpy()).encode())
    return h.hexdigest()
def frozen(tag,seed,ckpt):
    inv=pd.read_csv(RUN/'manifests/checkpoint_manifest.csv');row=inv[(inv.model==tag)&(inv.seed==seed)&(inv.checkpoint==ckpt)].iloc[0]
    path=ROOT/row.path;assert sha(path)==row.file_sha256
    z=torch.load(path,map_location='cpu',weights_only=False);cfg=copy.deepcopy(z['cfg']['model']);cfg['pretrained']=False
    model=build_model(cfg);model.load_state_dict(z['model'],strict=True);model.float().cuda().eval()
    for p in model.parameters():p.requires_grad_(False)
    digest=state_digest(model);assert digest==row.weight_sha256
    return model,digest
def backend(mode):
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=mode=='bf16'
    return dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(0),
      matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,autocast=mode=='bf16',batch_size=16,torch_threads=torch.get_num_threads(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
@torch.inference_mode()
def forward(model,records,mask,mode,phase,capture_logits=False):
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    b=batch(records,mask);assert all(v.dtype==torch.float32 for v in b.values())
    logits={};dtypes={}
    def capture(m,args,out):
        for k,v in out.items():
            if torch.is_tensor(v):
                dtypes[k]=str(v.dtype)
                if mode=='fp32' and v.is_floating_point():assert v.dtype==torch.float32,(k,v.dtype)
                if capture_logits:logits['logit_'+k]=v.detach().float().cpu().numpy()
        logits['mat_prob']=torch.softmax(out['mat'].float(),1).detach().cpu().numpy()
    hook=model.register_forward_hook(capture)
    reserve_forwards(len(records),phase)
    try:pred=original_predict([model],b,amp=mode=='bf16')
    finally:hook.remove()
    pred['mat_prob']=logits.pop('mat_prob')
    return [{k:v[i] for k,v in pred.items() if v is not None}|{k:v[i] for k,v in logits.items()} for i in range(len(records))],dtypes
def policies(pred):return ['hard','soft','soft_hazard']+(['direct','risk'] if 'aff' in pred else [])
def rows_for(pred,d,identity):
    geo=geometry(pred,d)
    return [identity|geo|evaluate(pred,d,policy) for policy in policies(pred)]

def fresh():
    locked();assert (RUN/'generation_gates.json').exists()
    gates=json.loads((RUN/'generation_gates.json').read_text());pg=json.loads((RUN/'precision/gates.json').read_text())
    matrix=pd.read_csv(RUN/'manifests/run_matrix.csv');files=pd.read_csv(RUN/'manifests/generated_files.csv')
    status=[];dtype=[];runtime_start=time.monotonic()
    for (tag,ckpt,seed,mask),group in matrix.groupby(['model','checkpoint','seed','mask'],sort=False):
        key=f'{tag}_{ckpt}_s{seed}_mask{int(mask)}';out=RUN/f'metrics/{key}.csv'
        assert not out.exists(),('do not overwrite',out)
        if tag in pg['blocked_models']:
            status.append(dict(run=key,status='BLOCKED_PRECISION_FIDELITY',frames=0));continue
        model,digest=frozen(tag,seed,ckpt);settings=backend('bf16');event('fresh_model_start',run=key,backend=settings)
        rows=[];nforward=0
        for _,r in group.iterrows():
            pop='rough' if r.condition=='rough_iid' else 'wide' if r.condition=='wider_articulation' else 'nominal'
            if pop in gates['blocked_populations']:
                status.append(dict(run=key+'/'+r.condition,status='BLOCKED_POPULATION',frames=0));continue
            fs=files[files.condition==r.condition].sort_values(['scene_id','frame_id'])
            # G paired intersection is fixed from generation only; other endpoints use their complete shared bundles.
            if tag in ['G2_static','G3_dyn'] and r.condition in ['clean','wider_articulation']:
                allowed=set(map(tuple,gates['G_keys']));fs=fs[[tuple(x) in allowed for x in fs[['scene_id','frame_id']].to_numpy()]]
            for offset in range(0,len(fs),16):
                part=fs.iloc[offset:offset+16];records=[load(RUN/p) for p in part.path]
                for d in records:
                    assert all(np.isfinite(d[k]).all() for k in ['depth','lidar_range','q_rep','gt_H','gt_aff','gt_reach'])
                pp,dt=forward(model,records,bool(mask),'bf16','fresh');dtype.append(dict(run=key,condition=r.condition,offset=offset,head_dtypes=json.dumps(dt)))
                for (_,f),p,d in zip(part.iterrows(),pp,records):
                    ident=dict(model=tag,checkpoint=ckpt,seed=int(seed),mask=bool(mask),condition=r.condition,scene_id=int(f.scene_id),frame_id=int(f.frame_id))
                    path=RUN/f'predictions/{key}/{r.condition}/{f.scene_id}_{f.frame_id}.npz';assert not path.exists();npzwrite(path,**p)
                    rows+=rows_for(p,d,ident)
                nforward+=len(records)
            print(key,r.condition,len(fs),'frames',flush=True)
        assert state_digest(model)==digest,'parameter or normalization-buffer mutation'
        csvwrite(out,rows);status.append(dict(run=key,status='COMPLETE',frames=nforward));del model;torch.cuda.empty_cache()
        csvwrite(RUN/'manifests/fresh_run_status.csv',status);csvwrite(RUN/'manifests/fresh_head_dtypes.csv',dtype)
        event('fresh_model_end',run=key,frames=nforward,state_unchanged=True)
    js(RUN/'fresh_status.json',dict(complete=True,elapsed_seconds=time.monotonic()-runtime_start))

def legacy_pick(p,d,policy):
    from kinafford.decision import select_dig_point,composed_affordance
    from stage3.selectors import soft_composition
    if policy=='soft':score=soft_composition(p,d['gt_reach'].astype(np.float32),float(d['z_target']));haz=None;sigma=None
    elif policy=='hard':score=composed_affordance(p|{'sem_label':p['sem_prob'].argmax(0)},d['gt_reach'].astype(np.float32),float(d['z_target']));haz=None;sigma=None
    else:score=p['aff'];haz=p['haz_prob'];sigma=p.get('aff_sigma') if policy=='risk' else None
    return select_dig_point(score,d['gt_reach'].astype(np.float32),haz,sigma,1 if policy=='risk' else 0)[0]

def precision():
    locked();assert (RUN/'legacy/camera_valid_only.csv').exists()
    keys=pd.read_csv(RUN/'manifests/precision_keys.csv');results=[];replay=[];dtype=[];blocked=[];metrics=[]
    versions=[('R2_full','best',False),('G3_dyn','best',False),('R2_noaff','last',True),('G2_static','last',True),('S5A_G3_structured','best',True),('S2E_G6_lidar','best',True)]
    for tag,ckpt,small in versions:
      kk=keys[keys.additional_subset] if small else keys
      for seed in range(3):
        key=f'{tag}_{ckpt}_s{seed}';model,digest=frozen(tag,seed,ckpt)
        # Preserve original contiguous batch16 membership for authoritative cache replay.
        if not small:
          backend('bf16')
          for suite,sk in kk.groupby('suite',sort=False):
            original_files=sorted((ROOT/f'data/suite_{suite}').glob('*.npz'))
            indices={str(p.relative_to(ROOT)):i for i,p in enumerate(original_files)}
            batches=sorted({indices[p]//16 for p in sk.input_path});wanted=set(sk.input_path)
            for j in batches:
              paths=original_files[j*16:(j+1)*16];records=[load(p) for p in paths]
              pp,dt=forward(model,records,False,'bf16','cache_replay');dtype.append(dict(run=key,mode='cache_replay_bf16',head_dtypes=json.dumps(dt)))
              for path,p,d in zip(paths,pp,records):
                if str(path.relative_to(ROOT)) not in wanted:continue
                old=load(cache_path(tag,seed,suite,int(d['scene_id']),int(d['frame_id'])))
                for field in ['gt_H','gt_aff','gt_reach','gt_sem','gt_haz']:
                    assert np.array_equal(old[field],d[field].astype(old[field].dtype)),('cache target custody',field,path)
                for policy in ['direct','risk','hard','soft']:
                    a=legacy_pick(p,d,policy);b=legacy_pick(old,d,policy);ok=a==b
                    replay.append(dict(model=tag,checkpoint=ckpt,seed=seed,suite=suite,scene_id=int(d['scene_id']),frame_id=int(d['frame_id']),policy=policy,original_row=b[0],original_col=b[1],replayed_row=a[0],replayed_col=a[1],pick_equal=ok,max_H_difference=float(np.max(np.abs(p['H']-old['H'])))))
                    if not ok and tag not in blocked:blocked.append(tag)
        for offset in range(0,len(kk),16):
          part=kk.iloc[offset:offset+16];records=[load(ROOT/p) for p in part.input_path];both={}
          for mode in ['bf16','fp32']:
            settings=backend(mode);pp,dt=forward(model,records,False,mode,'precision',True)
            dtype.append(dict(run=key,mode=mode,head_dtypes=json.dumps(dt)));both[mode]=pp
            event('precision_batch',run=key,offset=offset,frames=len(records),mode=mode,backend=settings,head_dtypes=dt)
            for (_,k),p,d in zip(part.iterrows(),pp,records):
                npzwrite(RUN/f'precision/maps/{key}/{mode}/{k.suite}_{k.scene_id}_{k.frame_id}.npz',**p)
                metrics+=rows_for(p,d,dict(model=tag,checkpoint=ckpt,seed=seed,mask=False,condition=k.suite,scene_id=int(k.scene_id),frame_id=int(k.frame_id),mode=mode))
          for (_,k),a,b,d in zip(part.iterrows(),both['bf16'],both['fp32'],records):
            for policy in policies(a):
                ma=evaluate(a,d,policy);mb=evaluate(b,d,policy)
                results.append(dict(model=tag,checkpoint=ckpt,seed=seed,suite=k.suite,scene_id=int(k.scene_id),frame_id=int(k.frame_id),policy=policy,
                  pick_changed=(ma['pick_row'],ma['pick_col'])!=(mb['pick_row'],mb['pick_col']),regret_bf16=ma['regret'],regret_fp32=mb['regret'],unsafe_bf16=ma['unsafe'],unsafe_fp32=mb['unsafe'],
                  tie_bf16=ma['exact_tie'],tie_fp32=mb['exact_tie'],max_H_difference=float(np.max(np.abs(a['H']-b['H']))),max_hazard_difference=float(np.max(np.abs(a['haz_prob']-b['haz_prob']))),hazard_crossings=int(((a['haz_prob']<=.5)!=(b['haz_prob']<=.5)).sum())))
        assert state_digest(model)==digest,'frozen state mutation';del model;torch.cuda.empty_cache()
        csvwrite(RUN/'precision/legacy_replay.csv',replay);csvwrite(RUN/'precision/differences.csv',results);csvwrite(RUN/'precision/metrics.csv',metrics);csvwrite(RUN/'precision/head_dtypes.csv',dtype)
        print('precision complete',key,flush=True)
    js(RUN/'precision/gates.json',dict(blocked_models=blocked,reason='Any historical original-convention pick mismatch blocks affected model interpretations; no precision-mode switch',precision_forwards=json.loads((RUN/'forward_counter.json').read_text())['precision'],complete=True))
if __name__=='__main__':
    torch.set_num_threads(1)
    if sys.argv[1]=='precision':precision()
    elif sys.argv[1]=='fresh':fresh()
    else:raise ValueError(sys.argv[1])
