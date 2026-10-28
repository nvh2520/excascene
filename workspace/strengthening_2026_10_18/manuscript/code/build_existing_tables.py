"""Format immutable historical/Phase A exports; no scoring or resampling."""
from pathlib import Path
import csv,hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];NEW=ROOT.parent;PROJECT=NEW.parent
sources={}
def read(path):
 p=PROJECT/path;sources[str(path)]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def put(name,lines): (ROOT/'tables'/name).write_text('\n'.join(lines)+'\n')
def signed(x,n=4):return f'{x:+.{n}f}'
def interval(row,scale=1,n=4):return f"{signed(row.estimate*scale,n)} [{signed(row.lower95*scale,n)}, {signed(row.upper95*scale,n)}]"
primary=read('paper_final_review/data/primary_results.csv')
lines=[r'\begin{table}[tbp]\centering\small',r'\caption{Reference-study primary family. Differences are dynamic minus static, full minus ablated supervision, or supplied all-camera removal minus intact. Height is in millimetres; unsafe differences are percentage points (pp).}\label{tab:referenceprimary}',r'\begin{tabular}{lrr}\toprule Endpoint & Difference & Adjusted 98.75\% interval\\\midrule']
labels={'G':'Dynamic projection height (mm)','S':'Supervision ablation regret','C-regret':'Supplied-removal regret','C-unsafe':'Supplied-removal unsafe (pp)'}
for _,r in primary.iterrows():
 scale=1000 if r.question=='G' else 100 if r.question=='C-unsafe' else 1;n=3 if r.question=='G' else 4 if r.question!='C-unsafe' else 2
 lines.append(f"{labels[r.question]} & {signed(r.mean_diff*scale,n)} & [{signed(r.ci9875_lo*scale,n)}, {signed(r.ci9875_hi*scale,n)}] \\\\")
put('reference_primary.tex',lines+[r'\bottomrule\end{tabular}\end{table}'])
secondary=read('AFAST_reporting_closure/secondary_descriptive_intervals.csv')
print('comparisons',secondary.comparison.unique().tolist())
print('masks',secondary['mask'].unique().tolist())
absolute=read('AFAST_reporting_closure/condition_denominators.csv')
ref=read('submission_strengthening/references/task_value_compact_table.csv')
classical=read('strengthening_2026_10_18/phase_a/classical/afast_condition_point_estimates.csv')
print('reference methods',ref.method.unique().tolist())
lines=[r'\begin{table}[tbp]\centering\small',r'\caption{Descriptive task-value comparison on the external-study holdout. Each cell is raw regret / unsafe percentage per attempt, strict direct selection for the learned systems. The fixed-floor row gives uniform / training-prior policies separately; uniform unsafe values are analytical expectations. All displayed policies execute every attempt. Paired intervals and complete seven-condition inventories are in the supplement and repository.}\label{tab:taskvalue}',r'\begin{tabular}{lrrr}\toprule Policy & Clean & Severe camera & Rough iid\\\midrule']
conds=['clean','cam_s3','rough_iid']
def val(regret,unsafe):return f'{regret:.4f} / {100*unsafe:.2f}'
lines.append('Local-ground/hazard classical & '+' & '.join(val(*classical.set_index('condition').loc[c,['regret','unsafe']]) for c in conds)+r'\\')
for arch,label in [('KF-D','KinAfford dynamic'),('BF-D','BEVFusion-Exca')]:
 d=absolute[(absolute.architecture==arch)&(absolute['mask']==0)&(absolute.policy=='direct')].set_index('condition')
 lines.append(label+' & '+' & '.join(val(*d.loc[c,['regret_mean','unsafe_per_attempt']]) for c in conds)+r'\\')
for label in ref.method.unique():print(label)
# Original reference outputs use one fixed policy per frame, not three fits.
r=read('submission_strengthening/references/reference_per_frame.csv').groupby(['reference','condition'])[['regret','unsafe']].mean()
lines.append('Uniform; training-prior floors & '+' & '.join(r'\shortstack{'+val(*r.loc[('R0',c)])+r';\\'+val(*r.loc[('R1',c)])+'}' for c in conds)+r'\\')
put('task_value.tex',lines+[r'\bottomrule\end{tabular}\end{table}'])
classint=read('strengthening_2026_10_18/phase_a/classical/descriptive_intervals.csv')
labelscond={'clean':'Clean','iid':'Iid faults','cam_s1':'Mild camera','cam_s2':'Moderate camera','cam_s3':'Severe camera','all_s2':'Moderate all sensors','rough_iid':'Rough iid'}
lines=[r'\begin{longtable}{lrr}',r'\caption{Frozen classical baseline on the external-study holdout. Ordinary95\% scene-bootstrap intervals are post-hoc descriptive; every condition is retained.}\label{tab:classicalall}\\',r'\toprule Condition & Raw regret [95\%] & Unsafe percentage [95\%]\\\midrule\endfirsthead',r'\toprule Condition & Raw regret [95\%] & Unsafe percentage [95\%]\\\midrule\endhead']
for c in ['clean','iid','cam_s1','cam_s2','cam_s3','all_s2','rough_iid']:
 d=classint[classint.condition==c].set_index('metric');row=[]
 for metric,scale,n in [('regret',1,4),('unsafe',100,2)]:
  t=d.loc[metric];row.append(f'{t.estimate*scale:.{n}f} [{t.lower95*scale:.{n}f}, {t.upper95*scale:.{n}f}]')
 lines.append(labelscond[c]+' & '+' & '.join(row)+r'\\')
put('classical_all.tex',lines+[r'\bottomrule\end{longtable}'])
(ROOT/'data/existing_table_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
