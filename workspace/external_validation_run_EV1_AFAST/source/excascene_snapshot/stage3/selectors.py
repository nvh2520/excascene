"""Prespecified probabilistic composition and common no-threshold selection."""
import numpy as np
from kinafford.sim.scene import MAT_DIGGABILITY,BEV_RES,SEM_TERRAIN,smoothstep
from kinafford.decision import select_dig_point,regret,unsafe_pick

def soft_composition(pred,reach,z_target):
 mat=np.asarray(pred['mat_prob']);sem=np.asarray(pred['sem_prob'])
 if mat.shape[0]!=len(MAT_DIGGABILITY) or mat.shape[1:]!=reach.shape:raise ValueError('Material probability shape mismatch')
 if not np.isfinite(mat).all() or not np.isfinite(sem).all():raise ValueError('Nonfinite probabilities')
 dig=(mat*MAT_DIGGABILITY[:,None,None]).sum(axis=0)
 gy,gx=np.gradient(pred['H'],BEV_RES)
 slope=np.degrees(np.arctan(np.hypot(gx,gy)))
 return (dig*(1-smoothstep(slope,40.0,60.0))*reach*np.clip((pred['H']-z_target)/1.2,0,1)*(1-pred['haz_prob'])*sem[SEM_TERRAIN]).astype(np.float32)

def no_threshold_metrics(pred,gt):
 out={}
 scores={'soft':soft_composition(pred,gt['reach'],gt['z_target']),'direct_nothreshold':pred['aff']}
 for name,score in scores.items():
  masked=np.where(gt['reach']<0.05,-np.inf,score)
  fallback=not np.isfinite(masked).any()
  pick,_=select_dig_point(score,gt['reach'])
  out.update({f'regret_{name}':regret(gt['aff'],pick)[0],f'unsafe_{name}':int(unsafe_pick(gt['haz'],gt['sem'],pick)),f'fallback_{name}':int(fallback),f'no_reachable_{name}':int(not (gt['reach']>=0.05).any()),f'pick_row_{name}':int(pick[0]),f'pick_col_{name}':int(pick[1])})
 return out
