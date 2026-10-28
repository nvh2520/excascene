"""Read-only frozen checkpoint adapters; no training or network weight retrieval."""
from common import *
import copy,torch
sys.path.insert(0,str(AFAST/'code'))
from models import build as build_afast,neural_inputs
from inputs import tensorize

def backend():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False

def checkpoint_rows():
    rows=[]
    for arch in ['KF-D','BF-D']:
        for seed in [10,11,12]:
            path=AFAST/f'checkpoints/{arch}/seed{seed}/final.pt'
            rows.append(dict(architecture=arch,seed=seed,path=str(path.relative_to(PROJECT)),sha256=sha(path),selection='frozen final14',training_precision='FP32'))
    with (PROJECT/'revision_v5/validation_run/manifests/checkpoint_manifest.csv').open() as f:old=list(csv.DictReader(f))
    for arch,tag in [('S5A','S5A_G3_structured'),('G6','S2E_G6_lidar')]:
        for seed in [0,1,2]:
            r=next(r for r in old if r['model']==tag and int(r['seed'])==seed and r['checkpoint']=='best')
            path=PROJECT/r['path'];assert sha(path)==r['file_sha256']
            rows.append(dict(architecture=arch,seed=seed,path=r['path'],sha256=r['file_sha256'],selection=f"historically selected best epoch{r['epoch_one_based']}",training_precision='historical AMP/BF16',original_model=tag))
    return rows

def frozen(architecture,seed):
    inv=json.loads((ROOT/'protocol/checkpoint_inventory.json').read_text())['checkpoints']
    row=next(r for r in inv if r['architecture']==architecture and r['seed']==seed)
    path=PROJECT/row['path'];assert sha(path)==row['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    if architecture in ['KF-D','BF-D']:model=build_afast(architecture)
    else:
        from stage4.models import build_model
        cfg=copy.deepcopy(state['cfg']['model']);cfg['pretrained']=False
        model=build_model(cfg)
    model.load_state_dict(state['model'],strict=True);model=model.cuda().float().eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,row

def tensors(d,drop):
    intact=tensorize(d,False);out={k:v.copy() for k,v in intact.items()}
    drop=np.asarray(drop,bool);out['img'][drop]=0;out['depth'][drop]=0;out['cam_present'][drop]=0;out['meta'][:3]=out['cam_present']
    for k in ['lidar_range','lidar_valid','q','T_dyn','T_static','K']:assert np.array_equal(out[k],intact[k]),k
    assert np.array_equal(out['meta'][3:],intact['meta'][3:])
    return out

def batch(records,drops):
    ts=[tensors(d,dr) for d,dr in zip(records,drops)]
    b={k:torch.from_numpy(np.stack([t[k] for t in ts])).cuda() for k in ts[0]}
    assert all(t.dtype==torch.float32 for t in b.values())
    return neural_inputs(b)

def neutral(out,index):
    result=dict(H=out['H'][index].detach().cpu().numpy().astype(np.float32),
        sem_prob=out['sem'][index].softmax(0).detach().cpu().numpy().astype(np.float32),
        sem_label=out['sem'][index].argmax(0).detach().cpu().numpy().astype(np.int16),
        mat_prob=out['mat'][index].softmax(0).detach().cpu().numpy().astype(np.float32),
        mat_label=out['mat'][index].argmax(0).detach().cpu().numpy().astype(np.int16),
        haz_prob=out['haz'][index].sigmoid().detach().cpu().numpy().astype(np.float32),
        aff=out['aff'][index].sigmoid().detach().cpu().numpy().astype(np.float32))
    assert all(np.isfinite(a).all() for a in result.values())
    return result

def decisions(pred,d):
    score=soft_score(pred['H'],pred['mat_prob'],pred['sem_prob'][0],pred['haz_prob'],d['gt_reach'],d['z_target'])
    return {'direct':selected_metrics(pred['aff'],d,pred['haz_prob']),'soft':selected_metrics(score,d)}

def ensure_fp32(out):
    for k,v in out.items():
        if isinstance(v,torch.Tensor):assert v.dtype==torch.float32 and torch.isfinite(v).all(),(k,v.dtype)

if __name__=='__main__':
    js(ROOT/'protocol/checkpoint_inventory.json',dict(status='EXISTING_FROZEN_CHECKPOINTS_ONLY',evaluation_precision='actual FP32',checkpoints=checkpoint_rows()))
