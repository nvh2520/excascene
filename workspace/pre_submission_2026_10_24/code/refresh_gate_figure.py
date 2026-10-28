from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
O=Path(__file__).resolve().parents[1];R=O.parent;P=O/'manuscript'
paths=[R/'strengthening_2026_10_18/study3/analysis/complete_secondary_comparisons.csv',R/'final_evidence_2026_10_20/study4/analysis/complete_secondary_comparisons.csv']
conds=['clean','iid','cam_s1','cam_s2','cam_s3','single_severe','novel','all_s2','rough_iid']
names=['Clean','Independent faults','Mild camera','Moderate camera','Severe camera','One-camera severe','Novel camera','Moderate all-sensor','Rough terrain']
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(7.3,5.3),sharex=True,sharey=True);lim=[]
for i,path in enumerate(paths):
 d=pd.read_csv(path)
 for j,a in enumerate(['KF-D','BF-D']):
  q=d[(d.architecture==(a if i==0 else a+'42'))&(d.policy=='direct')&(d.contrast=='gate_minus_keep_all')&(d.metric=='regret')].set_index('pool').reindex(conds)
  ax=axes[i,j];ax.axvline(0,color='.5',lw=.7);ax.grid(axis='x',color='.92',lw=.6);color=['#0072B2','#D55E00'][j]
  ax.hlines(np.arange(9),q.lower95,q.upper95,color=color,lw=1);ax.plot(q.estimate,np.arange(9),'o',ms=3,color=color)
  ax.set_yticks(np.arange(9),names);ax.xaxis.set_major_locator(MaxNLocator(4))
  ax.set_title(('First prospective gate study' if i==0 else 'Replication')+'\n'+['KinAfford dynamic','BEVFusion-Exca'][j],fontsize=9)
  if i==1:ax.set_xlabel('Gate minus keep-all raw regret')
  lim.extend([q.lower95.min(),q.upper95.max()])
axes[0,0].invert_yaxis();axes[0,0].set_xlim(min(lim)-.003,max(lim)+.003);fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(P/f'figures/gate_condition_effects_final.{ext}',bbox_inches='tight',dpi=240)
plt.close(fig)
for f in list((P/'tables').glob('*.tex'))+list((P/'code').glob('*.py')):
 s=f.read_text().replace('Iid mixture','Independent faults').replace("'Iid'","'Independent faults'").replace(' & Iid & ',' & Independent faults & ').replace('Iid faults','Independent faults');f.write_text(s)
(O/'visualizations/gate_label_update.json').write_text(json.dumps(dict(scope='Label repair only, plotting stored points and intervals with shared axes',source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},new_estimates=0,new_draws=0,new_metrics=0,gpu_hours=0),indent=2)+'\n')
