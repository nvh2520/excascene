"""Study4 read-only epoch42/old14 checkpoint adapter; unchanged original graphs."""
from common import *
import copy,torch
sys.path.insert(0,str(AFAST/'code'))
from models import build as build_afast,neural_inputs
from inputs import tensorize

def backend():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False

def frozen(architecture,seed):
    inv=json.loads((ROOT/'protocol/checkpoint_inventory.json').read_text())['checkpoints']
    row=next(r for r in inv if r['architecture']==architecture and r['seed']==seed)
    path=PROJECT/row['path'];assert sha(path)==row['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    base={'KF-D42':'KF-D','BF-D42':'BF-D','KF-D14':'KF-D','BF-D14':'BF-D'}[architecture]
    model=build_afast(base)
    model.load_state_dict(state['model'],strict=True);model=model.cuda().float().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
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
    raise SystemExit('Import-only adapter. Build the inventory with checkpoint_inventory.py after the six prescribed fits.')
