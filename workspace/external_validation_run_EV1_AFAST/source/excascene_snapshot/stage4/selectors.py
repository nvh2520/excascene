"""The five fixed decision policies; no learned selector or threshold search."""
import numpy as np
from kinafford.decision import composed_affordance,select_dig_point,regret,unsafe_pick
from stage3.selectors import soft_composition
POLICIES=['direct','direct_risk_lambda1','hard_composed','soft_composed','soft_hazard_0.5']
def select_with_diagnostics(score,reach,hazard=None,sigma=None,lam=0):
 provisional=np.asarray(score,dtype=np.float64).copy()
 if sigma is not None and lam>0:provisional-=lam*sigma
 provisional[reach<.05]=-np.inf
 if hazard is not None:provisional[hazard>.5]=-np.inf
 fallback=not np.isfinite(provisional).any();pick,final=select_dig_point(score,reach,hazard,sigma,lam)
 finite=np.isfinite(final);ties=int((final==final[finite].max()).sum()) if finite.any() else 0
 return pick,dict(fallback=int(fallback),no_reachable=int(not (reach>=.05).any()),tie_frame=int(ties>1),excess_tied_maxima=max(0,ties-1),tied_maxima=ties,pick_row=int(pick[0]),pick_col=int(pick[1]))
def selector_metrics(pred,gt):
 p=dict(pred);p['sem_label']=p['sem_prob'].argmax(0)
 hard=composed_affordance(p,gt['reach'],gt['z_target']);soft=soft_composition(p,gt['reach'],gt['z_target'])
 if p.get('aff_sigma') is None:raise ValueError('Risk policy requires cached predicted affordance sigma')
 settings=[(p['aff'],p['haz_prob'],None,0),(p['aff'],p['haz_prob'],p['aff_sigma'],1),(hard,None,None,0),(soft,None,None,0),(soft,p['haz_prob'],None,0)]
 rows=[]
 for name,(score,hazard,sigma,lam) in zip(POLICIES,settings):
  pick,diag=select_with_diagnostics(score,gt['reach'],hazard,sigma,lam)
  rows.append(dict(policy=name,regret=regret(gt['aff'],pick)[0],unsafe=int(unsafe_pick(gt['haz'],gt['sem'],pick)),**diag))
 return rows
