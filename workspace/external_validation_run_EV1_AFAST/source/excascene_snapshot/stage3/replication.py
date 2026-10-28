"""Same-seed paired descriptive replication; no additional support tests."""
import numpy as np
import pandas as pd
from stage3.common import ROOT,RESULTS,ANALYSIS
from kinafford.analysis.aggregate import HYPOTHESES
CONTRASTS=[('H1a-H',0),('H1a-R',1),('H2',4),('H4b',12)]

def same_seed_pairs(frame,a,b,ma,mb,suites,seed):
 keys=['suite','scene_id','frame_id'];x=frame[(frame.tag==a)&(frame.seed==seed)&frame.suite.isin(suites)];y=frame[(frame.tag==b)&(frame.seed==seed)&frame.suite.isin(suites)]
 if ma not in x or mb not in y:return pd.DataFrame(columns=['a','b'])
 return pd.concat([x.set_index(keys)[ma].rename('a'),y.set_index(keys)[mb].rename('b')],axis=1).dropna()

def run():
 frames=[]
 for tag in ['G2_static','G3_dyn','R2_full','G6_lidar']:
  for s in range(5):
   path=(ROOT/'results'/f'{tag}_s{s}.csv') if s<3 else (RESULTS/f'S3D_{tag}_s{s}.csv')
   if path.exists():
    d=pd.read_csv(path);d['tag']=tag;frames.append(d)
 data=pd.concat(frames,ignore_index=True);registered=pd.read_csv(ROOT/'results/analysis/paired_tests.csv');rows=[];paired=[]
 for short,index in CONTRASTS:
  name,a,b,ma,mb,suites,_=HYPOTHESES[index];original=registered.iloc[index];allpairs=[];signs=[];available=[]
  for seed in range(5):
   j=same_seed_pairs(data,a,b,ma,mb,suites,seed)
   if len(j):
    d=j.a-j.b;sign=int(np.sign(d.mean()));signs.append(sign);available.append(seed);allpairs.append(j.rename(columns={'a':f'a{seed}','b':f'b{seed}'}));paired.append(j.reset_index().assign(contrast=short,seed=seed,difference=d.to_numpy()))
    rows.append(dict(id=short,seed=str(seed),paired_frames=len(j),a_mean=j.a.mean(),b_mean=j.b.mean(),mean_diff=d.mean(),median_diff=d.median(),sign=sign,agrees_with_stage1_sign=bool(sign==np.sign(original.mean_diff)),stage1_supported=bool(original.supported),status='DESCRIPTIVE'))
   else:rows.append(dict(id=short,seed=str(seed),paired_frames=0,a_mean=np.nan,b_mean=np.nan,mean_diff=np.nan,median_diff=np.nan,sign=np.nan,agrees_with_stage1_sign=False,stage1_supported=bool(original.supported),status='MISSING PAIR'))
  if len(available)==5:
   joined=pd.concat(allpairs,axis=1).dropna();av=joined[[f'a{s}' for s in range(5)]].mean(axis=1);bv=joined[[f'b{s}' for s in range(5)]].mean(axis=1);difference=av-bv
   rows.append(dict(id=short,seed='five_seed_average',paired_frames=len(joined),a_mean=av.mean(),b_mean=bv.mean(),mean_diff=difference.mean(),median_diff=difference.median(),sign=int(np.sign(difference.mean())),agrees_with_stage1_sign=bool(np.sign(difference.mean())==np.sign(original.mean_diff)),stage1_supported=bool(original.supported),status='DESCRIPTIVE'))
   paired.append(pd.DataFrame({'a':av,'b':bv,'difference':difference}).reset_index().assign(contrast=short,seed='five_seed_average'))
  else:rows.append(dict(id=short,seed='five_seed_average',paired_frames=0,status='INCOMPLETE FIVE-SEED COMPARISON'))
  rows.append(dict(id=short,seed='seed_agreement',paired_frames=0,sign_agreement_numerator=sum(s==np.sign(original.mean_diff) for s in signs),sign_agreement_denominator=len(signs),sign_agreement_fraction=np.mean([s==np.sign(original.mean_diff) for s in signs]) if signs else np.nan,available_seeds=','.join(map(str,available)),missing_seeds=','.join(str(s) for s in range(5) if s not in available),stage1_supported=bool(original.supported),status='DESCRIPTIVE'))
 pd.DataFrame(rows).to_csv(ANALYSIS/'seed_replication.csv',index=False)
 pd.concat(paired,ignore_index=True).to_csv(ANALYSIS/'replication_paired_values.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':run()
