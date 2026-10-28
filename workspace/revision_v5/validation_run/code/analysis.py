"""Locked four-primary environmental bootstrap; descriptive controls, no model selection."""
from common import *
import pandas as pd
from fractions import Fraction
KEY=['seed','condition','scene_id','frame_id']
QUANT=[Fraction(1,40),Fraction(39,40),Fraction(1,160),Fraction(159,160)]
def exact_quantile(a,q):
    a=sorted(a);x=Fraction(len(a)-1)*q;i=x.numerator//x.denominator;w=x-i
    return a[i]*(1-w)+a[min(i+1,len(a)-1)]*w
def multiplicities(n,rng):
    draws=rng.choice(n,size=(10000,n),replace=True);return np.stack([np.bincount(x,minlength=n) for x in draws]).astype(np.int16)
def weights_for(conditions):return {c:1/len(conditions) for c in conditions}
def estimate(frame,endpoint,conditions,draws,subset=False):
    # Data contains all three seed-specific metric differences, then aggregation is performed.
    assert not frame.duplicated(KEY).any();counts=frame.groupby(['condition','scene_id','frame_id']).seed.nunique();assert (counts==3).all()
    reps=np.zeros(10000,np.float64);exact=[Fraction(0)]*10000 if endpoint=='unsafe' else None
    means=[];denoms=[];perseed={s:[] for s in range(3)}
    for c in conditions:
        z=frame[frame.condition==c];family='subset' if subset else 'rough' if c=='rough_iid' else 'nominal'
        ids,mult=draws[family]
        byframe=z.groupby(['scene_id','frame_id']).difference.mean()
        sc=byframe.groupby(level='scene_id').agg(['sum','count']).reindex(ids).fillna(0)
        numerator=mult@sc['sum'].to_numpy(np.float64);denominator=mult@sc['count'].to_numpy(np.int64)
        with np.errstate(divide='ignore',invalid='ignore'):reps+=numerator/denominator/len(conditions)
        means.append(Fraction(int(z.difference.sum()),len(z)) if endpoint=='unsafe' else float(byframe.mean()));denoms.append(dict(condition=c,frames=len(byframe),scenes=byframe.index.get_level_values('scene_id').nunique(),attempted_seed_decisions=len(z)))
        for seed in range(3):perseed[seed].append(float(z[z.seed==seed].difference.mean()))
        if exact is not None:
            integer=z.groupby('scene_id').difference.sum().reindex(ids).fillna(0).to_numpy(np.int64)
            ns=mult@integer
            for i,(n,d) in enumerate(zip(ns,denominator)):
                exact[i]=None if d==0 or exact[i] is None else exact[i]+Fraction(int(n),int(d)*3*len(conditions))
    undefined=int((~np.isfinite(reps)).sum())
    if undefined:bounds=[np.nan]*4;rational=['undefined']*4
    elif exact is not None:
        bounds=[float(exact_quantile(exact,q)) for q in QUANT];rational=[str(exact_quantile(exact,q)) for q in QUANT];reps=np.array(list(map(float,exact)))
    else:bounds=np.quantile(reps,list(map(float,QUANT))).tolist();rational=['continuous']*4
    seedvals={s:float(np.mean(x)) for s,x in perseed.items()}
    return dict(mean_diff=float(sum(means)/len(means)),ci95_lo=bounds[0],ci95_hi=bounds[1],ci9875_lo=bounds[2],ci9875_hi=bounds[3],undefined_replicates=undefined,
        quantile_fractions=json.dumps(rational),per_seed=json.dumps(seedvals),leave_one_seed_out=json.dumps({s:float(np.mean([v for k,v in seedvals.items() if k!=s])) for s in seedvals}),denominators=json.dumps(denoms),degenerate_zero_interval=bool(bounds[0]==bounds[1]==0)),reps
def select(all,tag,ckpt,mask,policy):return all[(all.model==tag)&(all.checkpoint==ckpt)&(all['mask']==mask)&(all.policy==policy)]
def difference(a,b,ep,conditions):
    a=a[a.condition.isin(conditions)][KEY+[ep]];b=b[b.condition.isin(conditions)][KEY+[ep]]
    assert set(map(tuple,a[KEY].values))==set(map(tuple,b[KEY].values)),'unequal model/seed keys'
    z=a.merge(b,on=KEY,validate='one_to_one',suffixes=('_a','_b'));z['difference']=z[ep+'_a']-z[ep+'_b'];return z
def main():
    locked();paths=sorted((RUN/'metrics').glob('*.csv'));all=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True) if paths else pd.DataFrame()
    r=np.random.default_rng(20261011);gm=json.loads((RUN/'manifests/generation_manifest.json').read_text())
    draws={f:(np.array(ids),multiplicities(len(ids),r)) for f,ids in [('nominal',gm['nominal_scene_ids']),('rough',gm['rough_scene_ids']),('subset',gm['mechanism_subset'])]}
    npzwrite(RUN/'manifests/bootstrap_multiplicities.npz',**{f:ms for f,(ids,ms) in draws.items()})
    primary=[];second=[];replicates={};paired=[]
    specs=[('G','H_reachable_terrain',('G3_dyn','last',False,'direct'),('G2_static','last',False,'direct'),['clean','wider_articulation']),
       ('S','regret',('R2_full','last',False,'soft'),('R2_noaff','last',False,'soft'),['iid','all_s2','rough_iid']),
       ('C-regret','regret',('S5A_G3_structured','best',True,'direct'),('S5A_G3_structured','best',False,'direct'),['cam_s3']),
       ('C-unsafe','unsafe',('S5A_G3_structured','best',True,'direct'),('S5A_G3_structured','best',False,'direct'),['cam_s3'])]
    def calculate(name,ep,aa,bb,conds,primary_flag=False,subset=False):
        row=dict(question=name,endpoint=ep,contrast=str(aa)+' minus '+str(bb),conditions=';'.join(conds),status='COMPLETE')
        try:
            if all.empty:raise ValueError('No completed inference rows')
            a=select(all,*aa);b=select(all,*bb)
            if subset:a=a[a.scene_id.isin(gm['mechanism_subset'])];b=b[b.scene_id.isin(gm['mechanism_subset'])]
            if set(a.condition)&set(conds)!=set(conds) or set(b.condition)&set(conds)!=set(conds):raise ValueError('Required model/condition rows missing; see gates and run status')
            z=difference(a,b,ep,conds);result,rep=estimate(z,ep,conds,draws,subset);row.update(result)
            row['conclusion']='lower loss supported within locked comparison' if row['ci9875_hi']<0 and primary_flag else 'not established' if primary_flag else 'descriptive estimate'
            replicates[name]=rep;z['question']=name;z['endpoint']=ep;paired.append(z)
        except (ValueError,AssertionError) as e:
            row.update(status='BLOCKED_OR_INVALID',reason=str(e),conclusion='No primary conclusion' if primary_flag else 'Unavailable')
        return row
    for name,ep,aa,bb,conds in specs:primary.append(calculate(name,ep,aa,bb,conds,True))
    for ckpt in ['last','best']:
      for pol in ['soft','soft_hazard','hard']:
       for ep in ['regret','unsafe']:
        second.append(calculate(f'S_{ckpt}_{pol}_{ep}',ep,('R2_full',ckpt,False,pol),('R2_noaff',ckpt,False,pol),['iid','all_s2','rough_iid']))
    for tag in ['G3_dyn','R2_full']:
      for pol in ['soft','soft_hazard','hard','risk']:
       for ep in ['regret','unsafe']:
        second.append(calculate(f'context_{tag}_{pol}_{ep}',ep,(tag,'last',False,pol),(tag,'last',False,'direct'),['iid','all_s2','rough_iid']))
    for cond in ['clean','cam_s1','cam_s2','cam_s3','appearance_only','depth_only','absence_only','combined_present']:
      for contrast,aa,bb in [('mask_unmask',('S5A_G3_structured','best',True,'direct'),('S5A_G3_structured','best',False,'direct')),('mask_specialist',('S5A_G3_structured','best',True,'direct'),('S2E_G6_lidar','best',False,'direct'))]:
       for ep in ['regret','unsafe','H_reachable_terrain']:
        second.append(calculate(f'C_{contrast}_{cond}_{ep}',ep,aa,bb,[cond],subset=cond=='combined_present'))
    # Camera mechanisms use the prospectively fixed 28-scene complete block for every term.
    mechanisms=[]
    for tag,mask in [('S5A_G3_structured',False),('S5A_G3_structured',True),('S2E_G6_lidar',False)]:
      if all.empty:continue
      base=select(all,tag,'best',mask,'direct');base=base[base.scene_id.isin(gm['mechanism_subset'])]
      cells=['clean','appearance_only','depth_only','combined_present','cam_s3','absence_only']
      if set(base.condition)&set(cells)!=set(cells):continue
      for ep in ['regret','unsafe']:
        wide=base[base.condition.isin(cells)].pivot(index=['seed','scene_id','frame_id'],columns='condition',values=ep)
        assert not wide.isna().any().any(),'incomplete mechanism block'
        for label,terms in [('A-clean',{'appearance_only':1,'clean':-1}),('D-clean',{'depth_only':1,'clean':-1}),('AD-A',{'combined_present':1,'appearance_only':-1}),('AD-D',{'combined_present':1,'depth_only':-1}),('interaction',{'combined_present':1,'appearance_only':-1,'depth_only':-1,'clean':1}),('cam_s3-AD',{'cam_s3':1,'combined_present':-1}),('absence-clean',{'absence_only':1,'clean':-1})]:
            z=sum(wide[k]*v for k,v in terms.items()).rename('difference').reset_index();z['condition']='mechanism'
            result,rep=estimate(z,ep,['mechanism'],draws,True);name=f'mechanism_{tag}_mask{mask}_{ep}_{label}'
            mechanisms.append(dict(model=tag,mask=mask,endpoint=ep,contrast=label,cell_means=json.dumps({c:float(wide[c].mean()) for c in cells}),**result));replicates[name]=rep
    csvwrite(RUN/'primary_results.csv',primary);csvwrite(RUN/'secondary_results.csv',second);csvwrite(RUN/'camera_mechanism_results.csv',mechanisms)
    if paired:pd.concat(paired,ignore_index=True).to_csv(RUN/'paired_effects.csv',index=False)
    npzwrite(RUN/'bootstrap_replicates.npz',**replicates)
    if not all.empty:
      groups=['model','checkpoint','seed','mask','condition','policy']
      summary=all.groupby(groups).agg(attempted=('unsafe','size'),unsafe_events=('unsafe','sum'),executed=('executed','sum'),abstained=('abstained','sum'),empty_reach=('empty_reach','sum'),empty_candidates=('empty_candidates','sum'),exact_ties=('exact_tie','sum'),nonfinite_scores=('nonfinite_scores','sum'),nonfinite_hazard=('nonfinite_hazard','sum'),mean_regret=('regret','mean'),mean_regret_fp16_target=('regret_fp16_target','mean'),H_reachable_terrain=('H_reachable_terrain','mean'),H_suite_visible=('H_suite_visible','mean'),H_clean_fixed=('H_clean_fixed','mean'),H_intersection=('H_intersection','mean'),n_reachable_terrain=('n_reachable_terrain','sum'),n_suite_visible=('n_suite_visible','sum'),n_clean_fixed=('n_clean_fixed','sum'),n_intersection=('n_intersection','sum'),visibility_lost=('visibility_lost','sum'),visibility_gained=('visibility_gained','sum'))
      summary['unsafe_per_attempt']=summary.unsafe_events/summary.attempted;summary['unsafe_per_executed']=summary.unsafe_events/summary.executed.replace(0,np.nan);summary.to_csv(RUN/'decision_support_summary.csv')
      all.groupby(groups+['utility_stratum']).agg(attempted=('unsafe','size'),executed=('executed','sum'),unsafe_events=('unsafe','sum'),mean_regret=('regret','mean')).to_csv(RUN/'utility_strata.csv')
    event('analysis_complete',primary_slots=4,primary_complete=sum(x['status']=='COMPLETE' for x in primary))
    print(pd.DataFrame(primary).to_string(index=False))
if __name__=='__main__':main()
