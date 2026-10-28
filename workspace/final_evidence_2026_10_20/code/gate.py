"""Small RGB-only severity gate and explicit missing-depth rule."""
import numpy as np
import torch
from torch import nn

class CameraGate(nn.Module):
    def __init__(self):
        super().__init__();blocks=[];ch=3
        for out in [32,64,96,128,192]:
            blocks.extend([nn.Conv2d(ch,out,3,stride=2,padding=1),nn.ReLU()]);ch=out
        self.net=nn.Sequential(*blocks,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(192,4))
    def forward(self,x):return self.net(x)

def load_gate(checkpoint,device='cuda'):
    model=CameraGate().to(device).float()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True);return model.eval()

def severe_probability(model,rgb):
    """RGB uint8 [N,H,W,3] or[H,W,3] -> probability array; no depth input."""
    arr=np.asarray(rgb);arr=arr[None] if arr.ndim==3 else arr
    assert arr.dtype==np.uint8 and arr.shape[-1]==3
    x=torch.from_numpy(np.ascontiguousarray(arr.transpose(0,3,1,2))).to(next(model.parameters()).device).float()/255
    with torch.no_grad():return model(x).softmax(1)[:,3].cpu().numpy()

def camera_drop(probability,depth,present,tau,delta):
    invalid=np.mean((~np.isfinite(depth))|(depth<=.05),axis=(-2,-1))
    return (np.asarray(present)<=.5)|((np.asarray(probability)>=tau)|(invalid>=delta)),invalid
