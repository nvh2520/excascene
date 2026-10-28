"""Fixed small RGB-D patch probe; validation is opened only after final training."""
from common import *
import argparse,time,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,4))
    def forward(self,x):return self.net(x)

def dataset(split):
    rows=[load_npz(p) for p in sorted((ROOT/f'phase_a/material_patches/{split}').glob('*.npz'))]
    assert len(rows)==(400 if split=='train' else 100)
    return TensorDataset(torch.from_numpy(np.concatenate([r['x'] for r in rows]).astype(np.float32)),torch.from_numpy(np.concatenate([r['y'] for r in rows]).astype(np.int64)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026101804);np.random.seed(2026101804)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    train=dataset('train');counts=torch.bincount(train.tensors[1],minlength=4);assert (counts>0).all()
    weight=(1/counts.float());weight/=weight.mean();m=PatchCNN().cuda().float();optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    loader=DataLoader(train,batch_size=512,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(2026101804))
    criterion=nn.CrossEntropyLoss(weight=weight.cuda());start=time.time();steps=0;history=[]
    for epoch in range(1 if a.pilot else 20):
        m.train();total=0.;n=0
        for x,y in loader:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
            logits=m(x);loss=criterion(logits,y);assert torch.isfinite(loss)
            loss.backward();optimizer.step();total+=loss.item()*len(y);n+=len(y);steps+=1
            if a.pilot and steps==64:break
        torch.cuda.synchronize();history.append(dict(epoch=epoch+1,training_weighted_ce=total/n,seconds=time.time()-start))
        print(history[-1],flush=True)
    if a.pilot:
        seconds=time.time()-start;js(ROOT/'phase_a/material_probe_pilot.json',dict(steps=steps,seconds=seconds,seconds_per_step=seconds/steps,training_patches=len(train),steps_per_epoch=len(loader),epochs=20,
            forecast_fit_seconds_with_margin=1.25*(seconds/steps*len(loader)*20+120),parameters=sum(p.numel() for p in m.parameters()),pilot_weights_discarded=True));return
    out=ROOT/'phase_a/material_probe';out.mkdir(exist_ok=False)
    torch.save(dict(model=m.cpu().state_dict(),architecture='PatchCNN',epochs=20,seed=2026101804,train_class_counts=counts,preprocessing='RGB/255; clipped depth/20; 9x9 patch'),out/'final.pt');m.cuda().eval()
    validation=dataset('val');conf=np.zeros((4,4),np.int64)
    with torch.no_grad():
        for x,y in DataLoader(validation,batch_size=512,shuffle=False):
            pred=m(x.cuda()).argmax(1).cpu().numpy();truth=y.numpy();conf+=np.bincount(truth*4+pred,minlength=16).reshape(4,4)
    support=conf.sum(1);recall=np.divide(conf.diagonal(),support,out=np.full(4,np.nan),where=support>0)
    balanced=float(recall.mean()) if (support>0).all() else None
    js(out/'result.json',dict(status='COMPLETE',training_patches=len(train),validation_patches=len(validation),class_names=['sand','soil','gravel','rocky'],
        train_counts=counts.tolist(),confusion=conf.tolist(),validation_support=support.tolist(),recall=[None if np.isnan(x) else float(x) for x in recall],
        balanced_accuracy=balanced,accuracy=float(conf.trace()/conf.sum()),criterion=0.5,chance=0.25,
        meets_locked_learnability_rule=None if balanced is None else bool(balanced>=.5),no_downstream_retraining=True,
        checkpoint_sha256=sha(out/'final.pt'),parameters=sum(p.numel() for p in m.parameters()),actual_precision='FP32, autocast and TF32 disabled',seconds=time.time()-start))
    csvwrite(out/'training_history.csv',history)
if __name__=='__main__':main()
