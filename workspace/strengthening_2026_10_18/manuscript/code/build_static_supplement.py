"""Presentation only: format completed, versioned exports without new scoring."""
from pathlib import Path
import hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];NEW=ROOT.parent;PROJECT=NEW.parent
sources={}
def read(rel):
    p=PROJECT/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def obj(rel):
    p=PROJECT/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def put(name,lines): (ROOT/'tables'/name).write_text('\n'.join(lines)+'\n')
def table(name,caption,label,columns,header,rows,long=False):
    if long:
        lines=[r'\begingroup\small\setlength{\tabcolsep}{4pt}',r'\begin{longtable}{'+columns+'}',r'\caption{'+caption+'}\\label{'+label+r'}\\',r'\toprule '+header+r'\\\midrule\endfirsthead',r'\multicolumn{'+str(len(columns))+r'}{l}{\textit{Continued from previous page}}\\',r'\toprule '+header+r'\\\midrule\endhead']
        end=[r'\bottomrule\end{longtable}\endgroup']
    else:
        lines=[r'\begin{table}[tbp]\centering\small\setlength{\tabcolsep}{4pt}',r'\caption{'+caption+'}\\label{'+label+'}',r'\begin{tabular}{'+columns+r'}\toprule '+header+r'\\\midrule'];end=[r'\bottomrule\end{tabular}\end{table}']
    put(name,lines+[r' & '.join(map(str,r))+r'\\' for r in rows]+end)
def ci(r,scale=1,n=4):return f'{r.estimate*scale:+.{n}f} [{r.lower95*scale:+.{n}f}, {r.upper95*scale:+.{n}f}]'
conds={'clean':'Clean','iid':'Iid','cam_s1':'Mild camera','cam_s2':'Moderate camera','cam_s3':'Severe camera','all_s2':'Moderate all','rough_iid':'Rough iid'}
arches={'KF-D':'KinAfford dynamic','BF-D':'BEVFusion-Exca','S5A':'Structured dropout','G6':'LiDAR specialist'}
s=read('AFAST_reporting_closure/secondary_descriptive_intervals.csv')
rows=[]
for c,p in [('clean','direct'),('cam_s3','direct'),('rough_iid','direct'),('rough_iid','soft')]:
    d=s[(s.comparison=='BF-D minus KF-D')&(s['mask']==0)&(s.condition==c)&(s.policy==p)].set_index('metric')
    rows.append([conds[c]+' / '+p,ci(d.loc['regret']),ci(d.loc['unsafe'],100,2),ci(d.loc['H_reachable_terrain'],1000,2)])
table('architecture_tradeoffs.tex','Intact BEVFusion-Exca minus KinAfford dynamic. Ordinary 95\\% paired scene-bootstrap intervals are descriptive. Unsafe differences use percentage points; height is reachable-terrain mean absolute error in millimetres.','tab:architecture','lrrr','Condition / selector & Regret & Unsafe (pp) & Height (mm)',rows)
for name,comparison in [('external_all_removal.tex','camera masked minus intact'),('external_all_architecture.tex','BF-D minus KF-D')]:
    rows=[]
    if comparison.startswith('camera'):
        groups=[(a,None) for a in arches if a in ['KF-D','BF-D']]
    else:groups=[(None,m) for m in [0,1]]
    for a,m in groups:
        for c,label in conds.items():
            for p in ['direct','soft']:
                d=s[(s.comparison==comparison)&(s.condition==c)&(s.policy==p)]
                d=d[d.architecture==a] if a else d[d['mask']==m]
                d=d.set_index('metric');system=arches[a] if a else ('Intact' if m==0 else 'Masked')
                rows.append([system,label,p,ci(d.loc['regret']),ci(d.loc['unsafe'],100,2)])
    caption=('All-camera removal minus intact, both architectures' if comparison.startswith('camera') else 'External minus lightweight architecture, both camera states')+'. Complete condition/selector inventory; ordinary 95\\% descriptive intervals. Unsafe differences are percentage points. Height companions and all individual seeds are retained in the CSV inventory.'
    table(name,caption,'tab:'+name[:-4],'lllrr','System/state & Condition & Selector & Regret [95\\%] & Unsafe pp [95\\%]',rows,True)
i=read('AFAST_reporting_closure/posthoc_removal_interaction.csv')
rows=[]
for p in ['direct','soft']:
    d=i[(i.condition=='cam_s3')&(i.policy==p)].set_index('metric')
    rows.append([p,ci(d.loc['regret']),ci(d.loc['unsafe'],100,2),ci(d.loc['H_reachable_terrain'],1000,2)])
table('external_posthoc_interaction.tex','Labeled post-hoc severe-condition interaction: external removal effect minus lightweight removal effect. Ordinary 95\\% intervals; not a primary endpoint or multiplicity-controlled finding. The direct regret interval spans zero, while its unsafe interval is above zero.','tab:posthocinteraction','lrrr','Selector & Regret & Unsafe pp & Height mm',rows)
sh=read('strengthening_2026_10_18/phase_a/shapley/descriptive_intervals.csv');rows=[]
for arch in ['KF-D','BF-D']:
    for mask in [0,1]:
        for factor in ['height','material','hazard','terrain_semantics']:
            d=sh[(sh.architecture==arch)&(sh['mask']==mask)&(sh.condition=='cam_s3')&(sh.factor==factor)].set_index('metric')
            rows.append([arches[arch],['Intact','Masked'][mask],factor.replace('_',' '),ci(d.loc['regret_loss_reduction']),ci(d.loc['unsafe_loss_reduction'],100,2)])
table('phase_a_attribution.tex','Severe-camera soft-composition Shapley reductions. Ordinary 95\\% descriptive intervals; positive values mean lower loss after truth substitution. Full 16-subset and seven-condition inventories accompany the source.','tab:phasea-attribution','lllrr','Model & State & Truth factor & Regret reduction & Unsafe reduction pp',rows,True)
safe=read('strengthening_2026_10_18/phase_a/safety/complete_cause_inventory.csv');rows=[]
for _,r in safe[safe.condition=='cam_s3'].iterrows():
    name=arches.get(r.architecture,{'R0':'Uniform expected','R1':'Training prior','geometric_anchor':'Original geometric','classical_ground':'Local-ground classical'}.get(r.architecture,r.architecture))
    name+=(' masked' if r['mask']==1 else '')
    vals=[r['exclusive_'+c] for c in ['person_buffer','obstacle_buffer','truck_buffer','machine_footprint_buffer','non_terrain']]
    f=lambda x:f'{x:.2f}' if r.selection_kind=='analytic_expectation' else str(int(x))
    rows.append([name,r.policy if r.policy in ['direct','soft'] else '--',r.attempts,*[f(v) for v in vals],f'{r.person_per_1000_attempts:.2f}'])
table('phase_a_safety.tex','Severe-camera safety causes, priority-exclusive counts (person, obstacle, truck, machine, non-terrain). Uniform values are expected counts. The last column is person-buffer selections per 1,000 attempted decisions. Overlapping causes and every condition are retained in the machine-readable inventory.','tab:phasea-safety','llrrrrrrr','Policy & Selector & Attempts & Person & Obstacle & Truck & Machine & Other & /1,000',rows,True)
probe=obj('strengthening_2026_10_18/phase_a/material_probe/result.json')
rows=[[c,probe['train_counts'][j],probe['validation_support'][j],f"{100*probe['recall'][j]:.2f}"] for j,c in enumerate(probe['class_names'])]
table('material_probe.tex','Clean visible-terrain RGB-D patch probe. The fixed 20-epoch final model has balanced validation accuracy '+f"{100*probe['balanced_accuracy']:.2f}"+'\\%, above the locked 50\\% rule. No perception-model retraining followed. Patches are correlated within 400 training/100 validation scenes; this is a restricted learnability probe.','tab:materialprobe','lrrr','Class & Training patches & Validation patches & Recall (\\%)',rows)
th=obj('strengthening_2026_10_18/study3/thresholds.json');rows=[]
for a,r in th.items():rows.append([arches[a],r['tau'],r['delta'],f"{r['pooled_validation_regret']:.5f}"])
table('gate_thresholds.tex','Validation-frozen gate thresholds, shared across each architecture\'s seeds. Drop if severe probability exceeds or equals the RGB threshold or invalid-depth fraction exceeds or equals the depth threshold. An RGB threshold of 1.01 disables that branch.','tab:thresholds','lrrr','Downstream model & RGB threshold & Depth threshold & Validation regret',rows)
dep=read('strengthening_2026_10_18/study3/deployment/parameters_latency_and_flop_scope.csv');rows=[]
gate_cost=obj('strengthening_2026_10_18/manuscript/data/gate_analytical_dense_cost.json')
for _,r in dep.iterrows():
    rows.append([arches.get(r.architecture,'Three-camera RGB gate'),f'{r.parameters/1e6:.3f}',f'{1000*r.latency_median_seconds:.2f}',f'{1000*r.latency_p95_seconds:.2f}',f'{r.reported_dense_operator_flops_lower_bound/1e9:.2f}' if pd.notna(r.reported_dense_operator_flops_lower_bound) else f"{gate_cost['three_camera_conv_linear_flops']/1e9:.3f}"])
table('deployment.tex','Warm batch-one FP32 timing on the same A100 40 GB, 20 warmups/50 measurements. Model timing includes normalization, transforms, transfer, network and finite/dtype checks, but excludes disk, selector and gate. Gate time covers three cameras. Parameters include fixed parameters. The gate dense count is an analytic convolution/linear count (two FLOPs per multiply-accumulate); normalization, nonlinearities, pooling and the depth rule are excluded. Other FLOPs are dense-operator lower bounds; sparse/lifting/pooling operations are not fully counted and these are not comparable total-model FLOPs.','tab:deployment','lrrrr','System & Parameters (M) & Median ms & 95th ms & Dense GFLOPs lower bound',rows)
table('study_crosswalk.tex','Development crosswalk. Separate endpoint families and post-hoc additions are not merged into a chronological confirmatory sequence.','tab:crosswalk','lll','Evidence & Checkpoints / data & Statistical scope',[
['Reference study','Historical lightweight fits; 192 new scenes','Four primaries; 98.75\\%'],
['External study','Six random-init 14-epoch fits; 192 scenes','Two primaries; 97.5\\%'],
['Reference policies / diagnostics','Frozen outputs; original train/validation','Post-hoc descriptive'],
['Prospective gate study','Frozen perception; new gate; 767 retained scenes','Four primaries; 98.75\\%'],
['Image-level check','Two segmentation fits; real sequences','Descriptive; no action oracle']])
(ROOT/'data/static_supplement_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
