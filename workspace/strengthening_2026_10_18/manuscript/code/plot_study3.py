"""Plot completed descriptive intervals; no scoring, fitting or new draws."""
from pathlib import Path
import hashlib,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT.parent/'study3'
p=B/'analysis/complete_secondary_comparisons.csv';d=pd.read_csv(p)
conditions=['clean','iid','cam_s1','cam_s2','cam_s3','single_severe','novel','all_s2','rough_iid']
labels=['Clean','Iid faults','Mild camera','Moderate camera','Severe camera','One-camera severe','One-camera novel','Moderate all sensors','Rough iid']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(8.7,4.5),sharey=True,layout='constrained')
for ax,metric,scale,title in zip(axes,['regret','unsafe'],[1,100],['Raw-regret difference','Unsafe difference (percentage points)']):
    for a,label,color,offset,marker in [('KF-D','KinAfford dynamic','#0072B2',-.14,'o'),('BF-D','BEVFusion-Exca','#D55E00',.14,'s')]:
        t=d[(d.architecture==a)&(d.policy=='direct')&(d.contrast=='gate_minus_keep_all')&(d.metric==metric)].set_index('pool').loc[conditions]
        x=t.estimate.to_numpy()*scale;lo=t.lower95.to_numpy()*scale;hi=t.upper95.to_numpy()*scale;y=np.arange(len(conditions))+offset
        ax.errorbar(x,y,xerr=np.vstack([x-lo,hi-x]),fmt=marker,color=color,label=label,markersize=4,capsize=2,elinewidth=1)
    ax.axvline(0,color='.4',lw=.7,ls='--');ax.set_xlabel(title);ax.grid(axis='x',color='.92',linewidth=.5);ax.set_axisbelow(True)
axes[0].set_yticks(np.arange(len(labels)),labels);axes[0].invert_yaxis();axes[1].legend(loc='best',frameon=False,fontsize=8)
for ext in ['pdf','png']:fig.savefig(ROOT/f'figures/gate_condition_effects.{ext}',dpi=220,bbox_inches='tight')
(ROOT/'data/gate_figure_sources.json').write_text(json.dumps(dict(source=str(p.relative_to(ROOT.parent)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='all nine conditions, direct gate-minus-keep, ordinary95 descriptive intervals; no new computation of endpoints'),indent=2)+'\n')
