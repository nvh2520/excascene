"""Format locked-study exports after completion; no scoring or resampling."""
from pathlib import Path
import hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];NEW=ROOT.parent;B=NEW/'study3';sources={}
def read(rel):
    p=B/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def table(name,caption,label,columns,header,rows,long=False):
    if long:
        lines=[r'\begingroup\small\setlength{\tabcolsep}{4pt}',r'\begin{longtable}{'+columns+'}',r'\caption{'+caption+'}\\label{'+label+r'}\\',r'\toprule '+header+r'\\\midrule\endfirsthead',r'\multicolumn{'+str(len(columns))+r'}{l}{\textit{Continued from previous page}}\\',r'\toprule '+header+r'\\\midrule\endhead'];end=[r'\bottomrule\end{longtable}\endgroup']
    else:
        lines=[r'\begin{table}[tbp]\centering\small\setlength{\tabcolsep}{4pt}',r'\caption{'+caption+'}\\label{'+label+'}',r'\begin{tabular}{'+columns+r'}\toprule '+header+r'\\\midrule'];end=[r'\bottomrule\end{tabular}\end{table}']
    (ROOT/'tables'/name).write_text('\n'.join(lines+[' & '.join(map(str,r))+r'\\' for r in rows]+end)+'\n')
def signed(x,n):
    # Do not round a strictly nonzero interval endpoint into an apparent zero.
    return f'{x:+.2e}' if x!=0 and round(x,n)==0 else f'{x:+.{n}f}'
def ci(r,scale=1,n=4,level='95'):
    return f"{signed(r.estimate*scale,n)} [{signed(r['lower'+level]*scale,n)}, {signed(r['upper'+level]*scale,n)}]"
def absolute_ci(r,scale=1,n=2):
    return f"{r.estimate*scale:.{n}f} [{r.lower95*scale:.{n}f}, {r.upper95*scale:.{n}f}]"
conds={'clean':'Clean','iid':'Iid','cam_s1':'Mild camera','cam_s2':'Moderate camera','cam_s3':'Severe camera','single_severe':'One-camera severe','novel':'One-camera novel','all_s2':'Moderate all','rough_iid':'Rough iid','equal9':'Equal nine','clean_heavy':'Half clean'}
arches={'KF-D':'KinAfford dynamic','BF-D':'BEVFusion-Exca','S5A':'Structured dropout','G6':'LiDAR specialist','KF-D_vs_BF-D':'Gated lightweight minus external'}
arms={'keep_all':'Keep all','gate':'Gate','always_mask':'Always mask','oracle_mask':'Oracle mask','lidar_specialist':'LiDAR only'}
assert json.loads((B/'campaign_completion.json').read_text())['status']=='ALL_LOCKED_JOBS_COMPLETE'
p=read('analysis/primary_results.csv');rows=[]
for _,r in p.iterrows():
    scale,n=(100,3) if r.metric=='unsafe' else (1,5)
    rows.append([arches[r.architecture],'Unsafe (pp)' if r.metric=='unsafe' else 'Raw regret',ci(r,scale,n,'98_75'),r.class98_75.lower()])
table('gate_primary.tex','Prospective gate-minus-keep-all primaries: strict direct selection, equal weight over nine conditions, three fixed seeds. Four-endpoint family; adjusted 98.75\\% marginal scene-bootstrap intervals. Unsafe contrasts are percentage points, not percentages of a pooled event count. Joint benefit requires both upper bounds below zero for a model.','tab:gateprimary','llrl','Model & Endpoint & Difference [98.75\\%] & Interval class',rows)
ps=read('analysis/primary_per_seed.csv');rows=[]
for (a,s),g in ps.groupby(['architecture','seed']):
    t=g.set_index('metric');rows.append([arches[a],s,f'{t.loc["regret","estimate"]:+.5f}',f'{100*t.loc["unsafe","estimate"]:+.3f}'])
table('gate_per_seed.tex','Every fitted-seed primary effect, with the same equal-nine-condition weights. Seeds are fixed fitted systems, not independent environmental bootstrap units.','tab:gateseeds','lrrr','Model & Seed & Regret & Unsafe pp',rows)
sec=read('analysis/complete_secondary_comparisons.csv');rows=[]
for a in ['KF-D','BF-D']:
    for c in conds:
        d=sec[(sec.architecture==a)&(sec.contrast=='gate_minus_keep_all')&(sec.pool==c)&(sec.policy=='direct')].set_index('metric')
        rows.append([arches[a],conds[c],ci(d.loc['regret']),ci(d.loc['unsafe'],100,2)])
table('gate_all_conditions.tex','Gate minus keep-all direct decisions: every condition and both prespecified pools. Ordinary 95\\% intervals are descriptive. Complete soft decisions, four within-model contrasts, cross-model contrasts and execution/abstention endpoints remain in the CSV inventory.','tab:gateconditions','llrr','Model & Condition/pool & Regret [95\\%] & Unsafe pp [95\\%]',rows,True)
absolute=read('analysis/absolute_condition_results.csv');rows=[]
for a in ['KF-D','BF-D']:
    for c in list(conds)[:9]:
        for arm in ['keep_all','gate']:
            r=absolute[(absolute.architecture==a)&(absolute.arm==arm)&(absolute.condition==c)&(absolute.policy=='direct')].iloc[0]
            rows.append([arches[a],conds[c],arms[arm],f'{r.regret:.4f}',f'{int(r.unsafe_events)}/{int(r.attempts)}',f'{int(r.executions)}/{int(r.attempts)}',int(r.abstentions)])
table('gate_absolute_behavior.tex','Absolute direct behavior for keep-all and gate. Fractions use actual attempted decisions, with every abstention shown. These condition counts are not substituted for equal-condition primary pooling. All four arms, both selectors and unsafe/execution denominators are supplied in the complete export.','tab:gateabsolute','lllrrrr','Model & Condition & Arm & Regret & Unsafe/attempt & Executed/attempt & Abstain',rows,True)
ref=read('diagnostics/references/descriptive_intervals.csv');rows=[]
for c in list(conds)[:9]:
    for a in ['S5A','G6']:
        r=absolute[(absolute.architecture==a)&(absolute.condition==c)&(absolute.policy=='direct')].iloc[0]
        rows.append([conds[c],arches[a],f'{r.regret:.4f}',f'{100*r.unsafe:.2f}',f'{int(r.executions)}/{int(r.attempts)}'])
    for a,label in [('classical_ground','Local-ground classical'),('R0','Uniform expectation'),('R1','Training prior')]:
        d=ref[(ref.architecture==a)&(ref.condition==c)].set_index('metric')
        rows.append([conds[c],label,f'{d.loc["regret","estimate"]:.4f}',f'{100*d.loc["unsafe","estimate"]:.2f}',f'{100*d.loc["executed","estimate"]:.2f}\\%'])
table('gate_secondary_systems.tex','Contextual systems and references, all nine conditions. Structured-dropout and LiDAR-specialist training histories differ from the main models. Uniform unsafe entries are expected percentages, not events. Complete paired model-minus-reference intervals and soft results accompany these point estimates.','tab:gatesecondary','llrrr','Condition & Policy & Regret & Unsafe \\% & Execution',rows,True)
sense=read('diagnostics/sensitivity/complete_comparison_intervals.csv');rows=[]
cn={'KF_gate_minus_keep':'Lightweight gate--keep','BF_gate_minus_keep':'External gate--keep','KF_gate_minus_always':'Lightweight gate--always','BF_gate_minus_always':'External gate--always','KF_gate_minus_BF_keep':'Lightweight gate--external keep'}
for (policy,contrast,metric),g in sense[(sense.setting!='nominal')&(sense.pool=='equal9')].groupby(['policy','contrast','metric']):
    rows.append([cn[contrast],policy,metric,len(g),int(g.retains_nominal_sign.sum()),int(g.retains_nominal_interval_class95.sum()),int(g.retains_nominal_interval_class98_75.sum())])
table('gate_sensitivity_summary.tex','Utility sensitivity, equal-nine pool: number of 24 nonbaseline settings retaining the nominal sign and interval classification. All settings, conditions and both pools remain in the complete CSV. Both interval-level sensitivity columns are descriptive; predicted aggregate hazard cannot be reparameterized by individual truth-buffer size.','tab:gatesensitivity','lllrrrr','Contrast & Selector & Metric & Settings & Sign & 95\\% class & 98.75\\% class',rows,True)
safe=read('diagnostics/safety/equal_condition_pool.csv');rows=[]
for (a,arm,policy),g in safe.groupby(['architecture','arm','policy']):
    if a in ['KF-D','BF-D','S5A','G6'] and policy!='direct':continue
    t=g.set_index('metric');label=arches.get(a,{'classical_ground':'Local-ground classical','R0':'Uniform expectation','R1':'Training prior'}.get(a,a))
    rows.append([label,arms.get(arm,arm.replace('_',' ')),absolute_ci(t.loc['multi_person_buffer'],1000,2),absolute_ci(t.loc['unsafe'],100,2)])
table('gate_safety_summary.tex','Equal-condition person-buffer selections per 1,000 attempts and total unsafe percentage, with ordinary 95\\% descriptive intervals. Full overlapping and priority-exclusive cause counts, all conditions/arms/selectors and exact execution denominators are in the source. Uniform entries are analytic expectations.','tab:gatesafety','llrr','Policy & Arm & Person/1,000 [95\\%] & Unsafe \\% [95\\%]',rows,True)
shares=read('analysis/oracle_benefit_recovery.csv');rows=[]
for _,r in shares[(shares.pool.isin(['equal9','clean','cam_s3','single_severe','novel','clean_heavy']))&(shares.policy=='direct')].iterrows():
    value='Undefined' if pd.isna(r.share_recovered) else f'{r.share_recovered:.3f}'
    interval='Undefined' if pd.isna(r.lower95) else f'[{r.lower95:.3f}, {r.upper95:.3f}]'
    rows.append([arches[r.architecture],conds[r.pool],r.metric,value,interval,int(r.undefined_or_nonpositive_oracle_draws)])
table('gate_oracle_recovery.tex','Share of oracle-removal benefit recovered, strict direct selection. A nonpositive oracle benefit makes the point undefined; any nonpositive resampled denominator makes the unconditional interval undefined. Ratios are not clipped. Full condition/selector inventory is supplied.','tab:gaterecovery','lllrrr','Model & Pool/condition & Metric & Share & 95\\% interval & Invalid draws',rows,True)
(ROOT/'data/study3_table_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
