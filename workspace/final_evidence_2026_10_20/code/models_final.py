"""Unchanged task graphs, explicit ImageNet initialization with tensor receipts."""
from pathlib import Path
import hashlib,json,sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;AFAST=PROJECT/'external_validation_run_EV1_AFAST'
sys.path.insert(0,str(AFAST/'code'))
from models import build as original_build,neural_inputs,CFG
from kinafford.losses import compute_losses as original_losses,DEFAULT_WEIGHTS

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def initialize(architecture):
    model=original_build(architecture)
    acquisition=json.loads((ROOT/'pretrained/acquisition_receipt.json').read_text())['assets']
    kind='resnet18' if architecture=='KF-D' else 'swin_t'
    row=acquisition[kind];path=ROOT/row['path'];assert sha(path)==row['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    encoder=model.encoder if architecture=='KF-D' else model.camera_backbone
    before={k:tensor_sha(v) for k,v in encoder.state_dict().items()};integer_counters=set()
    if architecture=='KF-D':
        source=state;wanted={}
        for k,v in source.items():
            if k=='conv1.weight':wanted['stem.0.weight']=torch.cat([v,v.mean(1,keepdim=True)],1)
            elif k.startswith('bn1.'):wanted['stem.1.'+k[4:]]=v
            elif k.startswith(('layer1.','layer2.','layer3.')):wanted['l'+k[5:]]=v
        allowed_missing={k for k in encoder.state_dict() if k.startswith(('lat1.','lat2.','lat3.','smooth.','seg.'))}
        integer_counters={k for k in encoder.state_dict() if k.endswith('num_batches_tracked')}
        # The official legacy ImageNet file predates these bookkeeping buffers.
        for k in integer_counters:wanted[k]=encoder.state_dict()[k].clone()
        depthkey='stem.0.weight';rgbsource=state['conv1.weight']
        ignored=[k for k in state if k.startswith(('layer4.','fc.'))]
    else:
        source=state['model'];from mmdet.models.utils.ckpt_convert import swin_converter
        converted=swin_converter(source);wanted={}
        for k,v in converted.items():
            k=k.removeprefix('backbone.')
            if k.startswith('norm.'):k='norm3.'+k[5:]
            if k=='patch_embed.projection.weight':v=torch.cat([v,v.mean(1,keepdim=True)],1)
            if k in encoder.state_dict():wanted[k]=v
        allowed_missing={'norm1.weight','norm1.bias','norm2.weight','norm2.bias'}
        depthkey='patch_embed.projection.weight';rgbsource=source['patch_embed.proj.weight']
        ignored=[k for k in source if k.startswith('head.') or 'attn_mask' in k]
    assert set(encoder.state_dict())-set(wanted)==allowed_missing,('unexpected unloaded encoder state',set(encoder.state_dict())-set(wanted),allowed_missing)
    incompatible=encoder.load_state_dict(wanted,strict=False)
    assert set(incompatible.missing_keys)==allowed_missing and not incompatible.unexpected_keys
    actual=encoder.state_dict();proof=[]
    for k,v in wanted.items():
        assert torch.equal(actual[k],v),(architecture,k)
        proof.append(dict(key=k,shape=list(v.shape),tensor_sha256=tensor_sha(actual[k]),changed_from_random=before[k]!=tensor_sha(actual[k]),source_kind='default integer bookkeeping buffer, not a pretrained tensor' if k in integer_counters else 'converted ImageNet tensor'))
    assert torch.equal(actual[depthkey][:,:3],rgbsource)
    assert torch.equal(actual[depthkey][:,3:],rgbsource.mean(1,keepdim=True))
    params=dict(encoder.named_parameters());loaded_parameters=sum(params[k].numel() for k in wanted if k in params)
    receipt=dict(status='VERIFIED_PRETRAINED_TENSORS_LOADED',architecture=architecture,source=kind,weight_path=str(path.relative_to(ROOT)),weight_sha256=row['sha256'],loaded_state_tensors=len(proof)-len(integer_counters),default_integer_bookkeeping_buffers=sorted(integer_counters),loaded_parameter_elements=loaded_parameters,
        encoder_parameter_elements=sum(x.numel() for x in params.values()),unloaded_new_task_layers=sorted(allowed_missing),ignored_classifier_or_unused_stage_keys=ignored,
        depth_stem='fourth channel is exact arithmetic RGB-channel mean; original RGB channels unchanged',depth_stem_check=True,loaded_tensor_proof=proof,
        scope='Graph, task adapter and all non-image initialization unchanged. KF retains only ResNet stages1–3. BF intermediate output norms1/2 are newly initialized; final ImageNet norm maps to output norm3. No architecture replacement or task-trained pretraining.')
    return model,receipt

def compute_losses(out,batch,cfg,weights):
    losses=original_losses(out,batch,cfg);w=.3+.7*batch['gt_vis']
    semw=weights['semantics'][batch['gt_sem']]*w
    matw=weights['material'][batch['gt_mat']]*(batch['gt_sem']==0).float()*w
    losses['sem']=(semw*F.cross_entropy(out['sem'],batch['gt_sem'],reduction='none')).sum()/semw.sum().clamp(min=1.)
    losses['mat']=(matw*F.cross_entropy(out['mat'],batch['gt_mat'],reduction='none')).sum()/matw.sum().clamp(min=1.)
    losses['total']=sum(DEFAULT_WEIGHTS[k]*v for k,v in losses.items() if k!='total')
    return losses

def class_weights(device='cuda'):
    r=json.loads((ROOT/'protocol/training_class_weights.json').read_text())
    return {k:torch.tensor(r[k]['inverse_sqrt_mean_one_weights'],device=device,dtype=torch.float32) for k in ['semantics','material']}

if __name__=='__main__':
    for architecture in ['KF-D','BF-D']:
        torch.manual_seed(2026102000);m,r=initialize(architecture)
        out=ROOT/f'protocol/pretrained_loading_{architecture}.json';out.write_text(json.dumps(r,indent=2)+'\n')
        print(architecture,r['status'],r['loaded_state_tensors'],r['loaded_parameter_elements'],r['unloaded_new_task_layers'],flush=True)
        del m
