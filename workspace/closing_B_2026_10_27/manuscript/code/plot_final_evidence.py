"""Publication plots of recorded final outputs and validation histories only."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
PAPER=Path(__file__).resolve().parents[1];ROOT=PAPER.parent;PROJECT=ROOT.parent;OLD=PROJECT/'strengthening_2026_10_18'
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
CONDS=['clean','iid','cam_s1','cam_s2','cam_s3','single_severe','novel','all_s2','rough_iid']
NAMES=['Clean','Independent faults','Mild camera','Moderate camera','Severe camera','One-camera severe','Novel camera','Moderate all-sensor','Rough iid']
SOURCE={}
def read(p):SOURCE[str(p.relative_to(PROJECT))]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def save(fig,name):
    fig.savefig(PAPER/'figures'/f'{name}.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures'/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)
def main():
    frames=[read(OLD/'study3/analysis/complete_secondary_comparisons.csv'),read(ROOT/'study4/analysis/complete_secondary_comparisons.csv')]
    fig,axes=plt.subplots(2,2,figsize=(7.25,5.2),sharey=True,sharex='col')
    for i,d in enumerate(frames):
        for j,a in enumerate(['KF-D','BF-D']):
            arch=a if i==0 else a+'42';ax=axes[i,j]
            f=d[(d.architecture==arch)&(d.policy=='direct')&(d.contrast=='gate_minus_keep_all')&(d.metric=='regret')].set_index('pool').reindex(CONDS)
            assert f.estimate.notna().all();x=f.estimate.to_numpy();lo=f.lower95.to_numpy();hi=f.upper95.to_numpy();y=np.arange(9)
            ax.axvline(0,color='.55',lw=.7);ax.grid(axis='x',color='.92',lw=.7)
            color=['#0072B2','#D55E00'][j]
            ax.hlines(y,lo,hi,color=color,lw=1);ax.plot(x,y,'o',ms=3,color=color)
            ax.set_title(f"Study {i+3}: "+['KinAfford dynamic','BEVFusion-Exca'][j],fontsize=9)
            ax.set_yticks(y,NAMES);ax.invert_yaxis() if i==0 and j==0 else None
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            if i==1:ax.set_xlabel('Gate minus keep-all raw regret')
    fig.tight_layout();save(fig,'gate_condition_effects_final')
    curves=[];fig,axes=plt.subplots(3,2,figsize=(7.25,6.0),sharex=True)
    for j,a in enumerate(['KF-D','BF-D']):
        for seed,color in zip([10,11,12],['#0072B2','#D55E00','#009E73']):
            p=ROOT/f'checkpoints/{a}/seed{seed}/history.json';SOURCE[str(p.relative_to(PROJECT))]=hashlib.sha256(p.read_bytes()).hexdigest();h=json.loads(p.read_text());assert len(h)==42
            vals=[]
            for e in h:
                mat=np.asarray(e['material_confusion']);diag=np.diag(mat);union=mat.sum(0)+mat.sum(1)-diag
                record=dict(architecture=a,seed=seed,epoch=e['epoch'],validation_total=e['validation_losses']['total'],
                    training_total=e['train_losses']['total'],height_mae=e['height_absolute_sum']/e['height_cells'] if e['height_cells'] else np.nan,
                    material_accuracy=float(diag.sum()/mat.sum()) if mat.sum() else np.nan,lr=e['lr'])
                record.update({'validation_loss_'+k:v for k,v in e['validation_losses'].items()});curves.append(record);vals.append(record)
            f=pd.DataFrame(vals)
            for i,key in enumerate(['validation_total','height_mae','material_accuracy']):
                axes[i,j].plot(f.epoch,f[key],label=f'Seed {seed}',lw=.9,color=color)
                for ep in [14,28]:axes[i,j].axvline(ep,color='.8',lw=.5,ls=':')
            axes[0,j].set_title(['KinAfford dynamic','BEVFusion-Exca'][j]);axes[2,j].set_xlabel('Fixed training epoch')
    for i,label in enumerate(['Weighted validation loss','Reachable-terrain MAE (m)','Material accuracy']):axes[i,0].set_ylabel(label)
    axes[0,1].legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'validation_learning_curves')
    pd.DataFrame(curves).to_csv(PAPER/'data/validation_curve_metrics.csv',index=False)
    (PAPER/'data/figure_sources.json').write_text(json.dumps(dict(files=SOURCE,new_draws=0,neural_forwards=0,scope='plotting stored outcomes and prescribed validation curves; no checkpoint selection'),indent=2)+'\n')
if __name__=='__main__':main()
