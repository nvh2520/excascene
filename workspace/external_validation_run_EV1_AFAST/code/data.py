"""Prospective epoch/frame keyed shared input exposure. Holdout is never used here."""
import hashlib, multiprocessing, pathlib, sys
import numpy as np
import torch
from torch.utils.data import Sampler
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/excascene_snapshot'))
from kinafford.data.dataset import ExcaSceneDataset,IMAGENET_MEAN,IMAGENET_STD
from stage4.dataset import structured_keep_mask
from stage3.dataset import apply_sensor_dropout
NAMES={'initialization':101,'ordering':102,'photometric':103,'dropout':104}
ROOT_SEED=2026101301
def rng(seed,namespace,epoch=0,scene=0,slot=0,architecture=0):
    return np.random.default_rng(np.random.SeedSequence([ROOT_SEED,NAMES[namespace],seed,epoch,scene,slot,architecture]))
def init_seed(seed,architecture):
    return int(rng(seed,'initialization',architecture=1 if architecture=='BF-D' else 2).integers(0,2**31))

class SharedDataset(ExcaSceneDataset):
    def __init__(self,split,seed,augment=False):
        assert split in ['train','val']
        super().__init__(str(ROOT.parent/'data'),split,train=False,kin_jitter_deg=0,color_jitter=0)
        self.seed=seed;self.augment=augment;self.epoch=multiprocessing.Value('i',0)
    def set_epoch(self,epoch): self.epoch.value=epoch
    def __getitem__(self,i):
        s=super().__getitem__(i); ep=int(self.epoch.value);scene=int(s['scene_id']);slot=int(s['frame_id'])
        assert (0<=scene<3000 if self.augment else 0<=scene<3300),('forbidden_development_id',scene)
        if self.augment:
            r=rng(self.seed,'photometric',ep,scene,slot)
            # Same original per-view gain/offset definition, now epoch aware.
            gains=r.uniform(.85,1.15,size=(3,1,1,1)).astype(np.float32)
            offsets=(r.uniform(-.15,.15,size=(3,1,1,1))*.3).astype(np.float32)
            mean=torch.tensor(IMAGENET_MEAN)[None,:,None,None];std=torch.tensor(IMAGENET_STD)[None,:,None,None]
            rgb=(s['img'][:,:3]*std+mean)*torch.from_numpy(gains)+torch.from_numpy(offsets)
            s['img'][:,:3]=(rgb.clamp(0,1)-mean)/std*s['cam_present'][:,None,None,None]
            available=np.r_[s['cam_present'].numpy()>.5,True]
            keep,branch,proposals=structured_keep_mask(available,rng(self.seed,'dropout',ep,scene,slot))
            apply_sensor_dropout(s,keep)
            s['dropout_branch']=torch.tensor(branch);s['dropout_proposed']=torch.tensor(int(proposals.argmax()))
        return s

class EpochOrder(Sampler):
    def __init__(self,dataset,seed):self.dataset=dataset;self.seed=seed
    def indices(self): return rng(self.seed,'ordering',int(self.dataset.epoch.value)).permutation(len(self.dataset)).tolist()
    def __iter__(self):return iter(self.indices())
    def __len__(self):return len(self.dataset)
    def digest(self):return hashlib.sha256(np.asarray(self.indices(),dtype='<i4').tobytes()).hexdigest()
