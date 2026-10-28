"""Strict v5 candidate handling, with historical hard score convention explicit."""
from common import *
from stage3.selectors import soft_composition
from kinafford.decision import compose_affordance
def choose(score,reach,hazard=None,sigma=None):
    score=np.asarray(score,dtype=np.float64).copy();reach=np.asarray(reach)
    if sigma is not None:score-=sigma
    feasible=np.isfinite(reach)&(reach>=.05)
    eligible=feasible&np.isfinite(score)
    if hazard is not None:eligible&=np.isfinite(hazard)&(hazard<=.5)
    if not eligible.any():return None,0,int(eligible.sum())
    masked=np.where(eligible,score,-np.inf);pick=int(masked.argmax())
    return np.unravel_index(pick,score.shape),int(np.sum(masked==masked.flat[pick])),int(eligible.sum())
def evaluate(pred,d,policy):
    reach=d['gt_reach'];u=d['gt_aff'];feasible=np.isfinite(reach)&(reach>=.05)
    if not np.isfinite(u[feasible]).all():raise ValueError('nonfinite authoritative oracle target')
    optimum=float(u[feasible].max()) if feasible.any() else 0.
    if optimum>0:
        oracle=choose(u,reach)[0]
        assert not (d['gt_haz'][oracle]>.5 or d['gt_sem'][oracle]!=0),'positive oracle must be safe'
    hazard=None;sigma=None
    if policy in ['soft','soft_hazard']:
        try:score=soft_composition(pred,reach,float(d['z_target']))
        except ValueError:score=np.full_like(u,np.nan)
        if policy=='soft_hazard':hazard=pred['haz_prob']
    elif policy=='hard':score=compose_affordance(pred['H'],pred['sem_prob'].argmax(0),pred['mat_label'],pred['haz_prob'],reach,float(d['z_target']))
    else:
        score=pred['aff'];hazard=pred['haz_prob']
        if policy=='risk':sigma=pred['aff_sigma']
    pick,ties,n=choose(score,reach,hazard,sigma)
    executed=int(pick is not None);utility=float(u[pick]) if executed else 0.
    unsafe=int(executed and (d['gt_haz'][pick]>.5 or d['gt_sem'][pick]!=0))
    q=u.astype(np.float16).astype(np.float32);qopt=float(q[feasible].max()) if feasible.any() else 0.
    return dict(policy=policy,regret=optimum-utility,unsafe=unsafe,executed=executed,abstained=1-executed,
      pick_row=-1 if pick is None else int(pick[0]),pick_col=-1 if pick is None else int(pick[1]),
      selected_utility=utility,oracle=optimum,global_oracle=float(u.max()),global_feasible_gap=float(u.max())-optimum,
      regret_fp16_target=qopt-(float(q[pick]) if executed else 0.),
      normalized_regret=(optimum-utility)/optimum if optimum>.05 else np.nan,
      utility_stratum='zero' if optimum==0 else 'low' if optimum<=.05 else 'positive',
      reachable_cells=int(feasible.sum()),eligible_cells=n,empty_reach=int(not feasible.any()),empty_candidates=int(n==0),
      tie_count=ties,exact_tie=int(ties>1),reach_boundary=int((reach==.05).sum()),hazard_boundary=int((pred['haz_prob']==.5).sum()),
      nonfinite_scores=int((~np.isfinite(score)).sum()),nonfinite_reach=int((~np.isfinite(reach)).sum()),
      nonfinite_hazard=int((~np.isfinite(pred['haz_prob'])).sum()))
def geometry(pred,d):
    reach=np.isfinite(d['gt_reach'])&(d['gt_reach']>=.05)&(d['gt_sem']==0)
    vis=d['gt_vis']>.5;clean=d.get('gt_vis_clean',d['gt_vis'])>.5
    out={};err=np.abs(pred['H'].astype(np.float64)-d['gt_H'].astype(np.float64))
    for name,mask in [('reachable_terrain',reach),('suite_visible',vis),('clean_fixed',clean),('intersection',vis&clean)]:
        out['H_'+name]=float(err[mask].mean()) if mask.any() else np.nan;out['n_'+name]=int(mask.sum())
    out.update(visibility_lost=int((clean&~vis).sum()),visibility_gained=int((vis&~clean).sum()),nonfinite_height=int((~np.isfinite(pred['H'])).sum()))
    return out
