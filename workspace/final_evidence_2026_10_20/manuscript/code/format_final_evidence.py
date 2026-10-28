"""Presentation of completed records only; no predictions, resampling or tuning."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
import pandas as pd
PAPER=Path(__file__).resolve().parents[1];ROOT=PAPER.parent;PROJECT=ROOT.parent
OLD=PROJECT/'strengthening_2026_10_18';SOURCES={};REGISTRY=[]
LABEL={'KF-D':'KinAfford dynamic','BF-D':'BEVFusion-Exca','KF-D42':'KinAfford dynamic','BF-D42':'BEVFusion-Exca',
       'KF-D14':'KinAfford dynamic (14 epochs)','BF-D14':'BEVFusion-Exca (14 epochs)'}
COND={'clean':'Clean','iid':'Iid mixture','cam_s1':'Mild camera','cam_s2':'Moderate camera','cam_s3':'Severe camera',
      'single_severe':'One-camera severe','novel':'One-camera novel','all_s2':'Moderate all-sensor','rough_iid':'Rough iid',
      'equal9':'Equal nine-condition pool','clean_heavy':'Half-clean pool'}
CONDITIONS=list(COND)[:9]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);rel=str(p.relative_to(PROJECT));SOURCES[rel]=digest(p);df=pd.read_csv(p);df.attrs['source']=rel;return df
def load(p):
    p=Path(p);SOURCES[str(p.relative_to(PROJECT))]=digest(p);return json.loads(p.read_text())
def row(df,**query):
    take=np.ones(len(df),bool)
    for k,v in query.items():take&=df[k].eq(v).to_numpy()
    got=df[take];assert len(got)==1,(query,len(got));r=got.iloc[0]
    for k,v in r.items():
        if pd.notna(v) and (isinstance(v,(int,float,np.integer,np.floating)) or k.startswith('exact_')):
            REGISTRY.append(dict(family='presentation_lookup',architecture=r.get('architecture',''),metric=r.get('metric',''),pool=r.get('pool',r.get('condition','')),
                field=k,value=v,units='raw source units; table captions state display scaling',source=df.attrs.get('source',''),lookup=json.dumps(query,sort_keys=True)))
    return r
def signed(x,d=4):return f'{x:+.{d}f}'
def ci(r,scale=1,d=4,level='95'):
    lo=r['lower'+level];hi=r['upper'+level]
    if pd.isna(r['estimate']):return 'undefined'
    if pd.isna(lo) or pd.isna(hi):return f"{signed(scale*r['estimate'],d)} [undefined]"
    nonzero=[abs(scale*x) for x in [r['estimate'],lo,hi] if x!=0]
    while nonzero and min(nonzero)<.5*10**(-d) and d<8:d+=1
    return f"{signed(scale*r['estimate'],d)} [{signed(scale*lo,d)}, {signed(scale*hi,d)}]"
def text(name,value):(PAPER/name).write_text(value+'\n')
def table(name,caption,label,columns,head,rows,long=False,size='small'):
    if long:
        s='\\begingroup\\'+size+'\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{'+columns+'}\n'
        s+='\\caption{'+caption+'}\\label{'+label+'}\\\\\n\\toprule '+head+'\\\\\\midrule\\endfirsthead\n'
        s+='\\multicolumn{'+str(len(head.split('&')))+'}{l}{\\textit{Continued from previous page}}\\\\\n\\toprule '+head+'\\\\\\midrule\\endhead\n'
        s+='\n'.join(' & '.join(map(str,r))+'\\\\' for r in rows)+'\n\\bottomrule\\end{longtable}\\endgroup'
    else:
        s='\\begin{table}[tbp]\\centering\\'+size+'\\setlength{\\tabcolsep}{3pt}\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+columns+'}\\toprule '+head+'\\\\\\midrule\n'
        s+='\n'.join(' & '.join(map(str,r))+'\\\\' for r in rows)+'\n\\bottomrule\\end{tabular}\\end{table}'
    text('tables/'+name+'.tex',s)
def record(family,r,source,units):
    for k in ['estimate','lower95','upper95','lower98_75','upper98_75','exact_estimate','exact_lower98_75','exact_upper98_75']:
        if k in r and pd.notna(r[k]):REGISTRY.append(dict(family=family,architecture=r.get('architecture',''),metric=r.get('metric',''),pool=r.get('pool','equal9'),field=k,value=r[k],units=units,source=str(Path(source).relative_to(PROJECT))))

def gate_tables():
    frames={3:read(OLD/'study3/analysis/primary_results.csv'),4:read(ROOT/'study4/analysis/primary_results.csv')}
    absolute={3:read(OLD/'study3/analysis/absolute_condition_results.csv'),4:read(ROOT/'study4/analysis/absolute_condition_results.csv')}
    secondary={n:read((OLD/'study3' if n==3 else ROOT/'study4')/'analysis/complete_secondary_comparisons.csv') for n in [3,4]}
    shares={n:read((OLD/'study3' if n==3 else ROOT/'study4')/'analysis/oracle_benefit_recovery.csv') for n in [3,4]}
    operating={n:read((OLD/'study3' if n==3 else ROOT/'study4')/'analysis/gate_operating_inventory.csv') for n in [3,4]}
    results=[];allrows=[];seedrows=[];joint={};engineering=[]
    for n in [3,4]:
        for a in ['KF-D','BF-D']:
            aa=a if n==3 else a+'42';g=frames[n][frames[n].architecture==aa];assert len(g)==2
            joint[(n,a)]=bool((g.upper98_75<0).all())
            base=absolute[n][(absolute[n].architecture==aa)&(absolute[n].arm=='keep_all')&(absolute[n].policy=='direct')]
            basecol='regret' if n==3 else 'scene_equal_regret';baseval=float(base[base.condition.isin(CONDITIONS)][basecol].mean())
            for metric in ['regret','unsafe']:
                r=row(g,metric=metric);relative='---' if metric!='regret' or baseval<=0 else signed(100*r.estimate/baseval,1)+'\\%'
                results.append([str(n),LABEL[a],'Regret' if metric=='regret' else 'Unsafe (pp)',ci(r,100 if metric=='unsafe' else 1,3 if metric=='unsafe' else 5,'98_75'),relative,str(r.class98_75).lower()])
                record('Study'+str(n)+'_four_primary',r,(OLD/'study3' if n==3 else ROOT/'study4')/'analysis/primary_results.csv','probability' if metric=='unsafe' else 'raw utility regret')
            for c in CONDITIONS+['equal9','clean_heavy']:
                r=row(secondary[n],architecture=aa,policy='direct',contrast='gate_minus_keep_all',pool=c,metric='regret')
                u=row(secondary[n],architecture=aa,policy='direct',contrast='gate_minus_keep_all',pool=c,metric='unsafe')
                allrows.append([str(n),LABEL[a],COND[c],ci(r,d=4),ci(u,100,2)])
            for metric in ['regret','unsafe']:
                q=row(shares[n],architecture=aa,policy='direct',pool='cam_s3',metric=metric)
                share='undefined' if pd.isna(q.share_recovered) else f'{100*q.share_recovered:.1f}\\%'
                interval='undefined' if pd.isna(q.lower95) else f'[{100*q.lower95:.1f}, {100*q.upper95:.1f}]\\%'
                engineering.append([str(n),LABEL[a],metric,share,interval,str(int(q.undefined_or_nonpositive_oracle_draws))])
        perseed=read((OLD/'study3' if n==3 else ROOT/'study4')/'analysis/primary_per_seed.csv')
        for r in perseed.to_dict('records'):seedrows.append([n,LABEL[r['architecture']],r['seed'],r['metric'],signed(r['estimate']*(100 if r['metric']=='unsafe' else 1),5)])
    table('gate_primary_combined','Two distinct prospective four-primary families: gate minus keep-all, direct selection, equal nine-condition pool, three fixed seeds. Each family uses adjusted 98.75\\% marginal intervals with approximate coverage. Unsafe is in percentage points. Relative regret is the observed point contrast divided by keep-all regret, with no additional inferential claim.','tab:gateprimarycombined','lllrll','Study & Model & Endpoint & Difference [98.75\\%] & Relative & Class',results,size='footnotesize')
    table('gate_all_conditions_combined','Complete gate-minus-keep-all direct condition/pool inventory. Ordinary 95\\% intervals are descriptive. Both selectors, all other contrasts and denominators are retained in the CSV record.','tab:gateconditionscombined','lllrr','Study & Model & Condition/pool & Regret [95\\%] & Unsafe pp [95\\%]',allrows,long=True,size='scriptsize')
    table('gate_seed_effects_combined','Every fitted perception-seed effect for both prospective primary pools. Unsafe differences are percentage points; seeds are fixed and are not bootstrap units.','tab:gateseedscombined','llllr','Study & Model & Seed & Endpoint & Difference',seedrows,long=True,size='scriptsize')
    table('severe_oracle_recovery','Severe-camera share of generator-informed oracle-mask benefit recovered. Ordinary 95\\% paired scene intervals are descriptive; ratios are unclipped. Invalid denominator draws make the interval explicitly undefined.','tab:severeoracle','lllllr','Study & Model & Endpoint & Recovered & 95\\% interval & Undefined draws',engineering,size='scriptsize')
    # Narrative uses reported comparisons; it never searches conditions/thresholds.
    lines=[]
    for n in [3,4]:
        for a in ['KF-D','BF-D']:
            aa=a if n==3 else a+'42'
            op=operating[n];clean=op[(op.architecture==aa)&(op.condition=='clean')]
            lines.append(f'Study {n}, {LABEL[a]} removes {int(clean.removed.sum())}/{int(clean.present_camera_observations.sum())} clean camera observations.')
    text('generated/gate_engineering.tex','The engineering comparison is recovery of useful severe-fault removal while retaining clean views (Table~\\ref{tab:severeoracle}). '+ ' '.join(lines)+' The unchanged three-camera RGB classifier has a recorded warm batch-one median cost of 1.088 ms on the same A100, measured separately from perception and the depth rule; it has 0.408 million parameters and approximately 0.708 GFLOPs under the dense-operation convention.\n\\input{tables/severe_oracle_recovery.tex}')
    s3='The first prospective study meets the joint criterion for the lightweight model and not for the external model.'
    if joint[(4,'KF-D')]:rep='The fresh replication also meets the lightweight joint criterion.'
    else:rep='The fresh replication does not meet the lightweight joint criterion; the Study 3 joint result is therefore not replicated under the changed recipe.'
    bf='The external model '+('meets' if joint[(4,'BF-D')] else 'does not meet')+' the replication joint criterion.'
    text('generated/gate_primary_reading.tex',s3+' '+rep+' '+bf+' Table~\\ref{tab:gateprimarycombined} retains all adjusted intervals, including zero-crossing and unfavorable effects.\n\\input{tables/gate_primary_combined.tex}')
    later=[]
    for a in ['KF-D42','BF-D42']:
        for c in ['cam_s3','clean','single_severe','novel','rough_iid']:
            r=row(secondary[4],architecture=a,policy='direct',contrast='gate_minus_keep_all',pool=c,metric='regret')
            b=row(absolute[4],architecture=a,arm='keep_all',policy='direct',condition=c)
            relative='undefined' if b.scene_equal_regret<=0 else signed(100*r.estimate/b.scene_equal_regret,1)+'\\%'
            later.append([LABEL[a],COND[c],ci(r,d=5),relative])
    table('gate_selected_costs','Replication gate effects in the prespecified severe, clean, single-camera, novel and rough conditions. Ordinary 95\\% intervals are descriptive; relative percentages normalize the raw regret point contrast to keep-all. The full inventory is in the supplement.','tab:gateselected','llrl','Model & Condition & Regret [95\\%] & Relative',later,size='scriptsize')
    branch=[]
    for a in ['KF-D42','BF-D42']:
        for contrast,label in [('rgb_only_minus_keep_all','RGB-only'),('depth_only_minus_keep_all','Depth-only'),('gate_minus_rgb_only','OR minus RGB-only'),('gate_minus_depth_only','OR minus depth-only')]:
            r=row(secondary[4],architecture=a,policy='direct',contrast=contrast,pool='equal9',metric='regret');u=row(secondary[4],architecture=a,policy='direct',contrast=contrast,pool='equal9',metric='unsafe')
            branch.append([LABEL[a],label,ci(r,d=5),ci(u,100,3)])
    table('gate_branches','Replication branch and incremental OR-gate comparisons, equal-condition direct pool. Ordinary 95\\% descriptive intervals; no new primary endpoints or selected branches.','tab:gatebranches','llrr','Model & Contrast & Regret [95\\%] & Unsafe pp [95\\%]',branch,size='scriptsize')
    controls=[]
    for a in ['KF-D14','BF-D14']:
        for c in ['equal9','clean','cam_s3','rough_iid']:
            r=row(secondary[4],architecture=a,policy='direct',contrast='gate_minus_keep_all',pool=c,metric='regret')
            u=row(secondary[4],architecture=a,policy='direct',contrast='gate_minus_keep_all',pool=c,metric='unsafe')
            controls.append([LABEL[a],COND[c],ci(r,d=5),ci(u,100,3)])
    table('earlier_gate_fresh_controls','Unchanged earlier14-epoch models and Study3 thresholds on the fresh replication population. Gate minus keep-all; ordinary95\\% descriptive intervals, not an additional primary family. The full nine conditions and soft policy are retained in the CSV inventory.','tab:earliergatefresh','llrr','Control model & Pool / condition & Regret [95\\%] & Unsafe pp [95\\%]',controls,size='scriptsize')
    impact=read(ROOT/'study4/analysis/sensor_impact_comparisons.csv');rig=[]
    for a in ['KF-D14','BF-D14','KF-D42','BF-D42']:
        for c in ['single_severe','cam_s3']:
            r=row(impact,architecture=a,policy='direct',condition=c,metric='regret');u=row(impact,architecture=a,policy='direct',condition=c,metric='unsafe')
            rig.append([LABEL[a]+(' (42 epochs)' if a.endswith('42') else ''),COND[c],ci(r,d=5),ci(u,100,3)])
    table('single_vs_rig','Fresh-holdout keep-all condition minus clean, paired nominal scenes. One present-camera severe and compound rig-level severe costs are distinguished. Ordinary95\\% descriptive intervals; no equivalence test or threshold selection.','tab:singlevsrig','llrr','Model & Fault condition & Regret cost [95\\%] & Unsafe pp cost [95\\%]',rig,size='scriptsize')
    text('generated/recipe_discussion.tex','Training recipe and architecture choice interact with the gate. '+rep+' '+bf+' The paired recipe comparison holds scene, slot, seed and selector constant, but changes the three prescribed training components together. It supports choosing and validating the complete perception--fault-handling stack, rather than transferring an architecture ranking or a gate threshold independently of the recipe.')
    return dict(joint={str(n)+'_'+a:v for (n,a),v in joint.items()})

def recipe_tables():
    d=read(ROOT/'study4/analysis/complete_secondary_comparisons.csv');m=read(ROOT/'study4/map_metrics/paired_comparisons.csv');rows=[]
    for a in ['KF','BF']:
        for c in ['clean','cam_s3','rough_iid']:
            r=row(d,architecture=a+'_42_vs_14',policy='direct',pool=c,metric='regret',contrast='new_minus_old_keep_all')
            u=row(d,architecture=a+'_42_vs_14',policy='direct',pool=c,metric='unsafe',contrast='new_minus_old_keep_all')
            h=row(m,left=a+'-D42',right=a+'-D14',condition=c,metric='height_frame_mae')
            rows.append([LABEL[a+'-D42'],COND[c],ci(r,d=4),ci(u,100,2),ci(h,1000,2)])
    table('recipe_comparison','Paired strengthened-minus-14-epoch recipe comparison on the same fresh holdout, keep-all and direct decisions. Ordinary 95\\% intervals are descriptive. The recipe bundles horizon, class weighting and ImageNet initialization; height is reachable-terrain MAE in millimetres.','tab:recipecomparison','llrrr','Model & Condition & Regret [95\\%] & Unsafe pp [95\\%] & Height mm [95\\%]',rows,size='scriptsize')
    text('generated/recipe_map_reading.tex','Table~\\ref{tab:recipecomparison} compares the 42-epoch recipe with the earlier models on exactly the same fresh scene/slot/seed keys. Height and decision effects remain separate outcomes; neither the epoch count nor a favorable map contrast establishes convergence or better decisions. Full material accuracy/macro-IoU and person/obstacle cell recalls accompany the condition-specific comparisons in the supplement.')
    absmap=read(ROOT/'study4/map_metrics/absolute_metrics.csv');rows=[]
    for a in ['KF-D14','BF-D14','KF-D42','BF-D42']:
        for c in ['clean','cam_s3','rough_iid']:
            vals=[row(absmap,architecture=a,condition=c,metric=x) for x in ['material_accuracy','material_macro_iou','person_cell_recall','obstacle_cell_recall']]
            display=[]
            for v in vals:
                display.append('undefined' if pd.isna(v.estimate) else ci(v,100,1).replace('+',''))
            short=('Lightweight' if a.startswith('KF') else 'External')+(' (42)' if a.endswith('42') else ' (14)')
            rows.append([short,COND[c],*display])
    table('material_and_rare_classes','Same-holdout keep-all map diagnostics. Lightweight/external denote KinAfford dynamic/BEVFusion-Exca; parentheses give epochs. Material accuracy, four-class macro-IoU and person/obstacle BEV-cell recall are percentages with ordinary 95\\% scene intervals. These are cell recalls, not object detections. Full classes, support, seed effects and every condition are in the CSV record.','tab:materialrare','llrrrr','Model & Condition & Material acc. & Material mIoU & Person recall & Obstacle recall',rows,long=True,size='scriptsize')
    thresholds=load(ROOT/'study4/thresholds.json');rows=[]
    for a,t in thresholds.items():rows.append([LABEL[a]+(' (42 epochs)' if a.endswith('42') else ''),f"{t['tau']:.2f}",f"{t['delta']:.2f}",'disabled' if t['tau']>1 else 'enabled'])
    table('replication_thresholds','Validation-selected replication thresholds and unchanged earlier-model thresholds. A severe-probability threshold above one disables the RGB branch. All thresholds precede test generation.','tab:replicationthresholds','lrrl','Downstream model & RGB threshold & Depth threshold & RGB branch',rows,size='scriptsize')
    neural=read(ROOT/'study4/analysis/absolute_condition_results.csv');refs=read(ROOT/'study4/diagnostics/references/descriptive_intervals.csv');rows=[]
    def refcell(a,c):
        vals=[row(refs,architecture=a,condition=c,metric=k).estimate for k in ['regret','unsafe','executed']]
        return f'{vals[0]:.4f} / {100*vals[1]:.2f} / {100*vals[2]:.1f}'
    rows.append(['Classical local-ground',*[refcell('classical_ground',c) for c in ['clean','cam_s3','rough_iid']]])
    for a in ['KF-D14','BF-D14','KF-D42','BF-D42']:
        cells=[]
        for c in ['clean','cam_s3','rough_iid']:
            r=row(neural,architecture=a,arm='keep_all',policy='direct',condition=c)
            cells.append(f'{r.scene_equal_regret:.4f} / {100*r.scene_equal_unsafe:.2f} / {100*r.scene_equal_executed:.1f}')
        rows.append([LABEL[a]+(' (42 epochs)' if a.endswith('42') else ''),*cells])
    rows.append(['Uniform; training-prior',*['\\shortstack{'+refcell('R0',c)+';\\\\'+refcell('R1',c)+'}' for c in ['clean','cam_s3','rough_iid']]])
    table('task_value_replication','Fresh replication holdout: raw regret / unsafe percentage / execution percentage, equal-scene condition means. Learned rows keep all cameras and use direct selection. Uniform values are expectations. Paired descriptive intervals, actual counts and every condition/policy are in the complete reference and absolute-output tables.','tab:taskvaluereplication','lrrr','Policy & Clean & Severe camera & Rough iid',rows,size='scriptsize')

def image_tables():
    clean=read(ROOT/'image_check/metrics/clean_summary.csv');order=read(ROOT/'image_check/metrics/ordering_results.csv')
    old=load(OLD/'goose_ex/metrics/degradation_ordering.json');oldr=read(OLD/'goose_ex/metrics/degradation_results.csv');rows=[]
    sim=row(oldr,domain='sim',condition='clean');real=row(oldr,domain='real',condition='clean')
    rows.append(['Random / parity',f'{sim.miou:.4f}',f'{real.miou:.4f} [{real.miou_ci95_lo:.4f}, {real.miou_ci95_hi:.4f}]','20 / 42',f"{old['spearman_relative_drop_ordering']:.4f} [{old['ordinary_ci95'][0]:.4f}, {old['ordinary_ci95'][1]:.4f}]"])
    sim=row(clean,domain='sim',resolution='parity')
    for res,label in [('parity','ImageNet / parity'),('width512','ImageNet / real512')]:
        r=row(clean,domain='real',resolution=res);c=row(order,comparison='sim_parity_vs_real_'+res)
        rows.append([label,f'{sim.clean_miou:.4f}',f'{r.clean_miou:.4f} [{r.clean_lower95:.4f}, {r.clean_upper95:.4f}]',f'{int(r.supported_zero_tp_classes)} / {int(r.truth_supported_classes)}',f'{c.spearman:.4f} [{c.lower95:.4f}, {c.upper95:.4f}]'])
    table('image_upgrade','Descriptive image-check results. Parity is256$\\times$160; real512 is512$\\times$384 with the same final real checkpoint. Spearman compares14 nonclean degradation orderings with simulation parity. Real clean mIoU and zero-true-positive supported classes accompany ordering. Ordinary95\\% sequence-bootstrap intervals use saved draws. Native ontologies differ.','tab:imageupgrade','lllll','Initialization / evaluation & Sim mIoU & Real mIoU [95\\%] & Zero TP / support & Spearman [95\\%]',rows,size='scriptsize')
    cleanrows=[]
    for r in clean.to_dict('records'):
        cleanrows.append([r['domain'],r['resolution'],f"{r['clean_miou']:.4f} [{r['clean_lower95']:.4f}, {r['clean_upper95']:.4f}]",int(r['supported_zero_tp_classes']),int(r['truth_supported_classes'])])
    table('image_upgrade_clean','Clean performance for every pretrained domain/resolution, including simulation zero-true-positive classes. Counts use classes with true validation support. Ordinary95\\% descriptive sequence intervals.','tab:imageupgradeclean','llrrr','Domain & Resolution & Clean mIoU [95\\%] & Zero TP classes & Supported classes',cleanrows,size='scriptsize')
    allrows=[];f=read(ROOT/'image_check/metrics/degradation_results.csv')
    for r in f.to_dict('records'):
        allrows.append([r['domain']+' / '+r['resolution'],r['condition'].replace('_',' '),f"{r['miou']:.4f} [{r['lower95']:.4f}, {r['upper95']:.4f}]",f"{r['relative_miou']:.3f} [{r['relative_lower95']:.3f}, {r['relative_upper95']:.3f}]"])
    table('image_upgrade_full','Every condition in the pretrained image check. Relative mIoU is divided by clean mIoU for the same fitted model/resolution; ordinary95\\% sequence intervals are descriptive.','tab:imageupgradefull','llrr','Domain / resolution & Condition & mIoU [95\\%] & Relative mIoU [95\\%]',allrows,long=True,size='scriptsize')

def population():
    audit=load(ROOT/'study4/holdout/validity_audit.json');assert audit['status']=='PASS'
    excluded=audit['excluded_scenes'];desc='No scene was excluded whole.' if not excluded else 'Prespecified whole-scene exclusion removed '+', '.join(map(str,excluded))+' without replacement before inference.'
    receipts=[load(ROOT/f'study4/metrics/{a}_seed{s}.json') for a in ['KF-D42','BF-D42','KF-D14','BF-D14'] for s in [10,11,12]]
    logical=sum(r['logical_arm_forwards'] for r in receipts);actual=sum(r['actual_neural_forwards'] for r in receipts)
    assert logical==48*audit['records']
    text('generated/study4_population.tex',f"Study 4 retains {audit['nominal_scenes']} nominal and {audit['rough_scenes']} rough scene identities, {audit['records']:,} records and {audit['accepted_bundles']:,} accepted pose bundles, giving {logical:,} logical test forwards ({actual:,} actual forwards after identical-input reuse). "+desc+f" There are {audit['excluded_bundles']} excluded slots; all retained slots keep their complete paired condition bundles.")
    return audit

def diagnostics_tables():
    d=read(ROOT/'study4/diagnostics/safety/descriptive_intervals.csv');rows=[]
    for a,arm,policy,label in [('classical_ground','classical_ground','reference','Classical local-ground'),('R0','uniform_reachable','reference','Uniform (expected)'),
        ('R1','training_prior','reference','Training prior'),('KF-D42','keep_all','direct','KinAfford / keep-all'),('KF-D42','gate','direct','KinAfford / gate'),
        ('BF-D42','keep_all','direct','BEVFusion-Exca / keep-all'),('BF-D42','gate','direct','BEVFusion-Exca / gate')]:
        p=row(d,architecture=a,arm=arm,policy=policy,condition='cam_s3',metric='multi_person_buffer')
        u=row(d,architecture=a,arm=arm,policy=policy,condition='cam_s3',metric='unsafe')
        e=row(d,architecture=a,arm=arm,policy=policy,condition='cam_s3',metric='executed')
        rows.append([label,ci(p,1000,1),ci(u,100,2),f'{100*e.estimate:.2f}'])
    table('severe_person_safety','Replication severe-camera safety breakdown. Person-buffer selections are per1,000 attempts; unsafe and execution rates are percentages. Ordinary95\\% scene intervals are descriptive. Uniform rows are analytical probabilities, not integer events. Complete overlapping and exclusive cause inventories include every policy and condition.','tab:personsafety','lrrr','Policy & Person /1,000 [95\\%] & Unsafe \\% [95\\%] & Executes \\%',rows,size='scriptsize')
    curve=read(ROOT/'study4/diagnostics/cutoffs/equal_condition_pool.csv');rows=[]
    for a in ['KF-D42','BF-D42']:
        for arm in ['keep_all','gate']:
            for cutoff in [.5,.3,.2,.1,.05]:
                vals=[row(curve,architecture=a,arm=arm,policy='direct',cutoff=cutoff,metric=m) for m in ['regret','unsafe','person_buffer','abstained']]
                rows.append([LABEL[a]+' / '+arm.replace('_',' '),f'{cutoff:g}',*[x.replace('+','') for x in [ci(vals[0],d=4),ci(vals[1],100,2),ci(vals[2],1000,1),ci(vals[3],100,2)]]])
    table('fixed_cutoff_curve','Fixed direct hazard-cutoff curve, replication models, equal-condition pool. No cutoff is selected from these outcomes. Regret is raw; unsafe/abstention are percentages and person-buffer rate is per1,000 attempts. Ordinary95\\% descriptive intervals. Every learned arm and condition is retained in the full CSV inventory.','tab:cutoffcurve','lrrrrr','Model / arm & Cutoff & Regret [95\\%] & Unsafe \\% & Person /1,000 & Abstain \\%',rows,long=True,size='scriptsize')
    phi=read(ROOT/'study4/diagnostics/shapley/descriptive_intervals.csv');rows=[]
    for a in ['KF-D42','BF-D42']:
        for arm in ['keep_all','gate']:
            for factor in ['height','material','hazard','terrain_semantics']:
                r=row(phi,architecture=a,arm=arm,condition='cam_s3',factor=factor,metric='regret_loss_reduction')
                u=row(phi,architecture=a,arm=arm,condition='cam_s3',factor=factor,metric='unsafe_loss_reduction')
                rows.append([LABEL[a]+' / '+arm.replace('_',' '),factor.replace('_',' '),ci(r,d=4),ci(u,100,2)])
    table('replication_attribution','Severe-camera replication soft-composition Shapley loss reductions. Positive values favor replacing that output factor with truth; negative values are retained. Ordinary95\\% scene intervals are descriptive. All16 subsets, every learned arm, seed and condition remain in the CSV inventory.','tab:replicationattribution','llrr','Model / arm & Replaced factor & Regret reduction [95\\%] & Unsafe pp reduction [95\\%]',rows,long=True,size='scriptsize')

def main():
    assert load(ROOT/'study4/campaign_completion.json')['status']=='ALL_LOCKED_STUDY4_JOBS_COMPLETE'
    assert load(ROOT/'image_check/metrics/completion.json')['status']=='COMPLETE'
    (PAPER/'generated').mkdir(exist_ok=True)
    summary=gate_tables();recipe_tables();image_tables();diagnostics_tables();summary['population']=population()
    pd.DataFrame(REGISTRY).drop_duplicates().to_csv(PAPER/'data/number_registry.csv',index=False)
    (PAPER/'data/presentation_sources.json').write_text(json.dumps(dict(files=SOURCES,scope='formatting existing completed tables; no scientific computation or new draws'),indent=2)+'\n')
    (PAPER/'data/format_completion.json').write_text(json.dumps(dict(status='COMPLETE',**summary),indent=2)+'\n')
    print('Formatted completed final evidence',summary['joint'])
if __name__=='__main__':main()
