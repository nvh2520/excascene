"""Reproducible plots of saved estimates/intervals, without recomputation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];NEW=ROOT.parent;PROJECT=NEW.parent
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7'];sources={}
def read(path):
 p=PROJECT/path;sources[str(path)]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def save(fig,name):
 fig.savefig(ROOT/f'figures/{name}.pdf',bbox_inches='tight');fig.savefig(ROOT/f'figures/{name}.png',dpi=300,bbox_inches='tight');plt.close(fig)
d=read('paper_final_review/data/camera_mechanism_results.csv');d=d[(d.model=='S5A_G3_structured')&(~d['mask'])]
order=['A-clean','D-clean','AD-A','AD-D','interaction','cam_s3-AD','absence-clean']
labels=['Appearance − clean','Depth − clean','Combined − appearance','Combined − depth','Appearance × depth','Add absence to combined','Absence − clean']
fig,axes=plt.subplots(1,2,figsize=(7.1,3.4),sharey=True,gridspec_kw={'wspace':.12})
for ax,metric,scale,title in zip(axes,['regret','unsafe'],[1,100],['Raw regret difference','Unsafe difference (pp)']):
 f=d[d.endpoint==metric].set_index('contrast').loc[order];y=np.arange(len(f));x=f.mean_diff.to_numpy()*scale
 ax.errorbar(x,y,xerr=np.vstack([x-f.ci95_lo.to_numpy()*scale,f.ci95_hi.to_numpy()*scale-x]),fmt='o',color=COLORS[0],capsize=3)
 ax.axvline(0,color='.4',lw=.8);ax.set_xlabel(title);ax.grid(axis='x',alpha=.2);ax.set_yticks(y,labels)
axes[0].invert_yaxis();save(fig,'mechanism')
d=read('strengthening_2026_10_18/phase_a/shapley/descriptive_intervals.csv')
d=d[(d.condition=='cam_s3')&(d.metric=='regret_loss_reduction')]
order=['height','material','hazard','terrain_semantics']
fig,axes=plt.subplots(1,2,figsize=(7.1,3.1),sharey=True,gridspec_kw={'wspace':.12})
for ax,arch,title in zip(axes,['KF-D','BF-D'],['KinAfford dynamic','BEVFusion-Exca']):
 for mask,color,offset,label,marker in [(0,COLORS[0],-.12,'Intact','o'),(1,COLORS[1],.12,'All cameras removed','s')]:
  f=d[(d.architecture==arch)&(d['mask']==mask)].set_index('factor').loc[order];x=f.estimate.to_numpy();y=np.arange(4)+offset
  ax.errorbar(x,y,xerr=np.vstack([x-f.lower95.to_numpy(),f.upper95.to_numpy()-x]),fmt=marker,color=color,capsize=3,label=label)
 ax.axvline(0,color='.4',lw=.8);ax.set_title(title);ax.set_yticks(range(4),['Height','Material','Hazard','Terrain semantics']);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Attributed raw-regret reduction')
axes[0].invert_yaxis();axes[0].legend(loc='lower right');save(fig,'attribution')
(ROOT/'data/existing_figure_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
