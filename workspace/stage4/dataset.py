"""Epoch-seeded structured dropout; frozen augmentation and targets otherwise unchanged."""
import multiprocessing
import numpy as np
from stage3.dataset import Stage3Dataset,EpochDataLoader,apply_sensor_dropout

BRANCHES=['lidar_only','drop_lidar','drop_one_camera','nominal']
PRESENCE=['lidar_only','camera_only','all_four_present','mixed_camera_lidar']
def presence_class(keep):
 cams=bool(np.asarray(keep)[:3].any());lidar=bool(keep[3])
 if lidar and not cams:return 0
 if cams and not lidar:return 1
 if np.asarray(keep).all():return 2
 if cams and lidar:return 3
 raise ValueError('Empty sensor mask')
def structured_keep_mask(available,rng):
 available=np.asarray(available,dtype=bool)
 if available.shape!=(4,) or not available.any():raise ValueError('Need at least one available sensor')
 if not available[3]:raise ValueError('Registered inputs always have LiDAR before artificial dropout')
 proposals=np.zeros(4,dtype=np.int64)
 u=float(rng.random());branch=0 if u<.30 else 1 if u<.45 else 2 if u<.65 else 3
 proposals[branch]+=1;keep=available.copy()
 camera=int(rng.integers(3)) if branch==2 else None
 if not available[:3].any():return keep,0,proposals
 if branch==0:keep[:3]=False
 elif branch==1:keep[3]=False
 elif branch==2:keep[camera]=False
 assert keep.any()
 return keep,branch,proposals

class Stage4Dataset(Stage3Dataset):
 def __init__(self,*a,structured_dropout=False,epochs=14,**kw):
  super().__init__(*a,**kw);self.structured_dropout=bool(structured_dropout);self.epochs=int(epochs)
  self.exposures=multiprocessing.Array('q',self.epochs*17,lock=True)
 def __getitem__(self,i):
  sample=super().__getitem__(i)
  if self.train and self.structured_dropout:
   ep=int(self.epoch.value)
   if not 0<=ep<self.epochs:raise ValueError('Epoch outside registered training schedule')
   available=np.r_[sample['cam_present'].numpy()>.5,True]
   rng=np.random.default_rng(np.random.SeedSequence([self.training_seed,ep,i,int(sample['scene_id']),int(sample['frame_id']),4104]))
   keep,branch,proposals=structured_keep_mask(available,rng);apply_sensor_dropout(sample,keep)
   increment=np.zeros(17,dtype=np.int64);increment[branch]=1;increment[4:8]=proposals
   increment[8+presence_class(keep)]=1;increment[12]=int(np.array_equal(available,keep));increment[13]=int(not available[:3].any());increment[14]=int(branch!=int(proposals.argmax()));increment[15]=0;increment[16]=1
   with self.exposures.get_lock():
    for j,value in enumerate(increment):self.exposures[ep*17+j]+=int(value)
  return sample
 def exposure_records(self):
  with self.exposures.get_lock():counts=np.asarray(self.exposures[:],dtype=np.int64).reshape(self.epochs,17)
  return [dict(epoch=i,epoch_one_based=i+1,accepted_branches=dict(zip(BRANCHES,map(int,r[:4]))),proposed_branches=dict(zip(BRANCHES,map(int,r[4:8]))),presence=dict(zip(PRESENCE,map(int,r[8:12]))),no_additional_drop=int(r[12]),original_no_cameras=int(r[13]),precedence_changed_branch=int(r[14]),redraws=int(r[15]),observations=int(r[16])) for i,r in enumerate(counts)]
