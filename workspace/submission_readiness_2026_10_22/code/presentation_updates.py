"""Format stored results only; no estimates, scores, intervals or draws regenerated."""
from pathlib import Path
import csv,hashlib,json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT=Path(__file__).resolve().parents[1];PROJECT=OUT.parent
FINAL=PROJECT/'final_evidence_2026_10_20';OLD=PROJECT/'strengthening_2026_10_18';PAPER=OUT/'manuscript'
SOURCES={};REGISTRY=[]
def read(path):
    SOURCES[str(path.relative_to(PROJECT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)
def row(d,source,**kw):
    q=d
    for k,v in kw.items():q=q[q[k]==v]
    assert len(q)==1,(source,kw,len(q));r=q.iloc[0]
    for field in ['estimate','lower95','upper95','percent','lower95_percent','upper95_percent','removed','images']:
        if field in r:
            REGISTRY.append(dict(source=source,lookup=json.dumps(kw,sort_keys=True),field=field,value=r[field],scope='existing descriptive output or authorized C1 threshold application'))
    return r
def ci(r,scale=1,places=4,sign=True):
    form=('+' if sign else '')+'.'+str(places)+'f'
    return ' ['.join([format(r.estimate*scale,form),', '.join(format(r[k]*scale,form) for k in ['lower95','upper95'])+']'])
def write(name,s): (PAPER/name).write_text(s+'\n')
def table(name,caption,label,columns,header,rows,size='small'):
    write('tables/'+name+'.tex',f'\\begin{{table}}[tbp]\\centering\\{size}\\setlength{{\\tabcolsep}}{{3pt}}\n'
        f'\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{columns}}}\\toprule {header}\\\\\\midrule\n'+
        '\n'.join(' & '.join(r)+r'\\' for r in rows)+'\n\\bottomrule\\end{tabular}\\end{table}')
def main():
    src='final_evidence_2026_10_20/study4/analysis/complete_secondary_comparisons.csv';d=read(PROJECT/src)
    msrc='final_evidence_2026_10_20/study4/map_metrics/paired_comparisons.csv';m=read(PROJECT/msrc)
    names={'clean':'Clean','cam_s3':'Severe camera','rough_iid':'Rough iid'};rs=[]
    for c in names:
        r=row(d,src,architecture='new_BF_minus_new_KF',policy='direct',contrast='external_minus_lightweight_keep_all',pool=c,metric='regret')
        u=row(d,src,architecture='new_BF_minus_new_KF',policy='direct',contrast='external_minus_lightweight_keep_all',pool=c,metric='unsafe')
        h=row(m,msrc,left='BF-D42',right='KF-D42',condition=c,metric='height_frame_mae')
        rs.append([names[c],ci(r),ci(u,100,2),ci(h,1000,2)])
    table('architecture_replication','Strengthened 42-epoch recipe: keep-all BEVFusion-Exca minus KinAfford dynamic on the replication holdout. Direct decisions; ordinary 95\\% paired scene intervals. Unsafe differences are percentage points; height is reachable-terrain mean absolute error in millimetres.','tab:architecturereplication','lrrr','Condition & Regret [95\\%] & Unsafe pp [95\\%] & Height mm [95\\%]',rs)
    # Move the complete former supplementary table, without changing a value.
    t=(PAPER/'tables/material_and_rare_classes.tex').read_text()
    body=t.split('\\midrule\\endhead\n',1)[1].split('\\bottomrule',1)[0]
    material_rows=[x[:-2].split(' & ') for x in body.strip().splitlines()]
    table('material_main','Same-holdout keep-all map diagnostics for both recipes. Parentheses denote epochs; all values are percentages with ordinary 95\\% scene intervals. Material mIoU averages four classes; person and obstacle recalls are BEV-cell recalls. Lightweight denotes KinAfford dynamic and external denotes BEVFusion-Exca.','tab:materialrare','llrrrr','Model & Condition & Material acc. & Material mIoU & Person recall & Obstacle recall',material_rows,size='scriptsize')
    csrc='submission_readiness_2026_10_22/C1/deployed_threshold_real_transfer.csv';c=read(PROJECT/csrc)
    oldtable=(PAPER/'tables/real_gate_summary.tex').read_text()
    oldrows=oldtable.split('\\midrule\n',1)[1].split('\\bottomrule',1)[0]
    oldrows=oldrows.replace('Lightweight simulation threshold','First prospective gate study: KinAfford').replace('External simulation threshold','First prospective gate study: BEVFusion-Exca').replace('Structured-dropout threshold','Structured-dropout model').replace('Real training recalibration','Earlier training-only recalibration')
    newrows=[]
    for a,label in [('KF-D42','KinAfford'),('BF-D42','BEVFusion-Exca')]:
        clean=row(c,csrc,model=a,mechanism='clean',severity=0)
        mud=row(c,csrc,model=a,mechanism='mud_water',severity=3)
        glare=row(c,csrc,model=a,mechanism='sun_glare',severity=3)
        newrows.append(f"Replication: {label} ($\\tau_g={clean.threshold:.2f}$) & {int(clean.removed)}/192 & {clean.percent:.2f} [{clean.lower95_percent:.2f}, {clean.upper95_percent:.2f}] & {int(mud.removed)}/192 & {int(glare.removed)}/192\\\\")
    write('tables/real_gate_summary.tex',r'''\begin{table}[tbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Real-image RGB-gate transfer at the deployed thresholds, with earlier results retained. Original-image removal percentages have ordinary 95\% saved-sequence intervals; novel-fault columns give detection counts. Every denominator is 192 images. Replication rows apply the unchanged network at its selected thresholds without recalibration.}\label{tab:realgatesummary}
\begin{tabular}{lrrrr}\toprule Threshold source & Removed & Original \% [95\%] & Mud/water & Glare\\\midrule
'''+oldrows+'\n'+'\n'.join(newrows)+r'''
\bottomrule\end{tabular}\end{table}''')
    labels={'clean':'Original','occlusion':'Occlusion','haze':'Haze','blur_exposure':'Blur/exposure','noise':'Noise','mud_water':'Mud/water','sun_glare':'Glare'}
    # Full inventory is displayed beside both fixed thresholds, not selected rows.
    full=[]
    for mechanism,severity in c[['mechanism','severity']].drop_duplicates().itertuples(index=False,name=None):
        vals=[]
        for a in ['KF-D42','BF-D42']:
            r=row(c,csrc,model=a,mechanism=mechanism,severity=int(severity))
            vals.append(f'{int(r.removed)}/192; {r.percent:.2f} [{r.lower95_percent:.2f}, {r.upper95_percent:.2f}]')
        full.append([labels.get(mechanism,mechanism).replace('_',' '),str(severity),*vals])
    table('real_gate_replication_full','Replication-selected RGB thresholds on the same original and corrupted real images: counts and removal/detection percentages [ordinary 95\\% saved-sequence intervals]. KinAfford uses 0.75 and BEVFusion-Exca uses 0.25. No recalibration or new draws.','tab:realgatereplication','llrr','Mechanism & Severity & KinAfford & BEVFusion-Exca',full,size='scriptsize')
    # Stored interval figures: shared axes throughout each metric, no new bootstrap.
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    conds=['clean','iid','cam_s1','cam_s2','cam_s3','single_severe','novel','all_s2','rough_iid']
    cn=['Clean','Iid','Mild camera','Moderate camera','Severe camera','One-camera severe','Novel camera','Moderate all-sensor','Rough iid']
    first=read(OLD/'study3/analysis/complete_secondary_comparisons.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.25,5.3),sharex=True,sharey=True)
    limits=[]
    for i,df in enumerate([first,d]):
        for j,a in enumerate(['KF-D','BF-D']):
            q=df[(df.architecture==(a if i==0 else a+'42'))&(df.policy=='direct')&(df.contrast=='gate_minus_keep_all')&(df.metric=='regret')].set_index('pool').reindex(conds)
            ax=axes[i,j];ax.axvline(0,color='.5',lw=.7);ax.grid(axis='x',color='.92',lw=.6)
            ax.hlines(np.arange(9),q.lower95,q.upper95,color=['#0072B2','#D55E00'][j],lw=1)
            ax.plot(q.estimate,np.arange(9),'o',color=['#0072B2','#D55E00'][j],ms=3)
            ax.set_yticks(np.arange(9),cn)
            ax.set_title(('First prospective gate study' if i==0 else 'Replication')+'\n'+['KinAfford dynamic','BEVFusion-Exca'][j],fontsize=9)
            limits.extend([q.lower95.min(),q.upper95.max()]);ax.xaxis.set_major_locator(MaxNLocator(4))
            if i==1:ax.set_xlabel('Gate minus keep-all raw regret')
    axes[0,0].invert_yaxis();axes[0,0].set_xlim(min(limits)-.003,max(limits)+.003)
    fig.tight_layout();fig.savefig(PAPER/'figures/gate_condition_effects_final.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/gate_condition_effects_final.png',dpi=180,bbox_inches='tight');plt.close(fig)
    psrc='final_evidence_2026_10_20/study4/diagnostics/shapley/descriptive_intervals.csv';p=read(PROJECT/psrc)
    fig,axes=plt.subplots(2,2,figsize=(7.25,4.7),sharex='row',sharey=True)
    factors=['height','material','hazard','terrain_semantics'];fn=['Height','Material','Hazard','Terrain semantics']
    for j,a in enumerate(['KF-D42','BF-D42']):
        for i,metric in enumerate(['regret_loss_reduction','unsafe_loss_reduction']):
            ax=axes[i,j];scale=1 if i==0 else 100
            for arm,off,color in [('keep_all',-.13,'#0072B2'),('gate',.13,'#D55E00')]:
                rr=[row(p,psrc,architecture=a,arm=arm,condition='cam_s3',factor=f,metric=metric) for f in factors]
                x=np.array([r.estimate for r in rr])*scale;lo=np.array([r.lower95 for r in rr])*scale;hi=np.array([r.upper95 for r in rr])*scale;y=np.arange(4)+off
                ax.hlines(y,lo,hi,color=color,lw=1);ax.plot(x,y,'o',ms=3,label='Keep all' if arm=='keep_all' else 'Gate',color=color)
            ax.axvline(0,color='.5',lw=.7);ax.grid(axis='x',color='.92',lw=.6);ax.set_yticks(np.arange(4),fn)
            ax.set_xlabel('Regret reduction' if i==0 else 'Unsafe reduction (percentage points)')
            if i==0:ax.set_title(['KinAfford dynamic','BEVFusion-Exca'][j])
            ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0,0].invert_yaxis();axes[0,1].legend(frameon=False,fontsize=8,loc='lower right')
    fig.tight_layout();fig.savefig(PAPER/'figures/replication_attribution.pdf',bbox_inches='tight');fig.savefig(PAPER/'figures/replication_attribution.png',dpi=180,bbox_inches='tight');plt.close(fig)
    pd.DataFrame(REGISTRY).drop_duplicates().to_csv(PAPER/'data/readiness_number_registry.csv',index=False)
    (PAPER/'data/readiness_presentation_provenance.json').write_text(json.dumps(dict(files=SOURCES,new_bootstrap_draws=0,new_scoring=0,scope='formatting existing results and C1 fixed-threshold reporting'),indent=2)+'\n')
    print('Presentation tables and stored-interval figures formatted')

if __name__=='__main__':main()
