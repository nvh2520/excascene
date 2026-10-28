"""Format existing exports only; no statistical or model execution."""
from pathlib import Path
import csv
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper_submission_candidate'

def read(path):
    return list(csv.DictReader(path.open()))

def esc(value):
    return str(value).replace('_', r'\_').replace('%', r'\%')

def table_start(caption, label, cols, header):
    return [r'\begingroup\small', r'\setlength{\tabcolsep}{5pt}',
            r'\begin{longtable}{' + cols + '}',
            r'\caption{' + caption + r'}\label{' + label + r'}\\',
            r'\toprule', header + r'\\\midrule\endfirsthead',
            r'\toprule', header + r'\\\midrule\endhead',
            r'\midrule\endfoot', r'\bottomrule\endlastfoot']

def finish(lines, path):
    path.write_text('\n'.join(lines + [r'\end{longtable}', r'\endgroup']) + '\n')

def main():
    tables = PAPER / 'tables'
    rows = read(ROOT / 'AFAST_reporting_closure/secondary_descriptive_intervals.csv')
    inputs=[]
    specs = [('regret','Raw regret','raw utility units',4),
             ('unsafe','Unsafe frequency','percentage points',2),
             ('H_reachable_terrain','Reachable-terrain height MAE','mm',1)]
    for metric,title,unit,places in specs:
        name='afast_descriptive_' + metric
        lines=table_start(title + r': complete architectural and removal comparisons, with ordinary descriptive 95\% scene intervals added at reporting stage. No multiplicity control; units: ' + unit + '.',
            'tab:'+name, 'p{3.2cm}llrrr', r'Contrast/state & Condition & Policy & Effect & Lower & Upper')
        count=0
        for row in rows:
            if row['metric']!=metric: continue
            if row['comparison']=='BF-D minus KF-D':
                group='BF$-$KF ' + ('intact' if row['mask']=='0' else 'masked')
            else:
                group=row['architecture']+' removal'
            condition=esc(row['condition']).replace('pool','pool')
            if 'pool' in condition: condition='1/3 pool'
            values=[f"{float(row['display_'+k]):+.{places}f}" for k in ('estimate','lower95','upper95')]
            lines.append(' & '.join([group,condition,row['policy']]+values)+r'\\')
            count+=1
        assert count==60, (metric,count)
        finish(lines,tables/(name+'.tex')); inputs.append(r'\input{tables/'+name+'.tex}')
    (tables/'afast_all_descriptive.tex').write_text('\n'.join(inputs)+'\n')
    rows=read(ROOT/'AFAST_reporting_closure/posthoc_removal_interaction.csv')
    inputs=[]
    for metric,title,unit,places in specs:
        name='afast_interaction_'+metric
        lines=table_start('Post-hoc four-arm removal interaction: '+ title.lower()+r'. (BF masked $-$ BF intact) $-$ (KF masked $-$ KF intact); ordinary descriptive 95\% interval, '+unit+'.',
            'tab:'+name,'llrrr',r'Condition & Policy & Effect & Lower & Upper')
        for row in rows:
            if row['metric']==metric:
                values=[f"{float(row['display_'+k]):+.{places}f}" for k in ('estimate','lower95','upper95')]
                lines.append(' & '.join([esc(row['condition']),row['policy']]+values)+r'\\')
        finish(lines,tables/(name+'.tex'));inputs.append(r'\input{tables/'+name+'.tex}')
    (tables/'afast_posthoc.tex').write_text('\n'.join(inputs)+'\n')

    matdir=ROOT/'submission_strengthening/material/complete_v1'
    summaries=read(matdir/'summary_metrics.csv')
    # All pooled full-support rows, plus the deterministic training majority.
    conditions=['clean','iid','all_s2','cam_s1','cam_s2','cam_s3','rough_iid']
    def matrow(architecture,condition,mask,stratum):
        selected=[r for r in summaries if r['architecture']==architecture and r['condition']==condition and r['task']=='material' and r['stratum']==stratum and r['seed'] not in ('10','11','12') and (architecture=='TRAIN_MAJORITY' or r['mask']==mask)]
        assert len(selected)==1,selected
        return selected[0]
    for subset in ('all','coverage'):
        caption=(r'Material diagnostics across all conditions. Accuracy is percent; macro IoU is dimensionless. Values pool pixels over three separately fitted models; the training-selected soil reference is one deterministic classifier.' if subset=='all' else r'CLEAN material performance on approximate projected measured-depth coverage. Accuracy is percent; macro IoU is dimensionless. The support is defined by original CLEAN cameras for both model states, not by post-removal observation.')
        lines=table_start(caption,'tab:material_'+subset,'llrrrrrr',r'Condition/support & Cameras & \multicolumn{2}{c}{BF-D} & \multicolumn{2}{c}{KF-D} & \multicolumn{2}{c}{Train-soil}')
        lines.append(r' & & Acc. & mIoU & Acc. & mIoU & Acc. & mIoU\\\midrule')
        groups=[(c,'full_official_support',esc(c)) for c in conditions] if subset=='all' else [('clean','projected_depth_covered_terrain','Covered'),('clean','not_projected_depth_covered_terrain','Not covered')]
        for condition,stratum,label in groups:
            for mask in ('0','1'):
                values=[]
                for architecture in ('BF-D','KF-D','TRAIN_MAJORITY'):
                    r=matrow(architecture,condition,mask,stratum)
                    values += [f"{100*float(r['accuracy']):.2f}",f"{float(r['macro_iou']):.3f}"]
                lines.append(' & '.join([label,'intact' if mask=='0' else 'masked']+values)+r'\\')
        finish(lines,tables/('material_'+subset+'.tex'))

    data=PAPER/'data/strengthening'
    data.mkdir(exist_ok=True)
    shutil.copytree(matdir,data/'material',dirs_exist_ok=True)
    for f in ['training_material_prior.json','prior_lock.json','reference_comparison_intervals.csv','reference_denominators.csv','reference_support_summary.csv','task_value_compact_table.csv']:
        shutil.copy2(ROOT/'submission_strengthening/references'/f,data/f)
    shutil.copy2(ROOT/'submission_strengthening/material/convention_clarification_v1.json',data/'material/convention_clarification_v1.json')
    ref_inputs=[]
    for p in sorted((ROOT/'submission_strengthening/references/tables').glob('*.tex')):
        shutil.copy2(p,tables/p.name)
        if p.name.startswith('reference_absolute_'):
            metric=p.stem.rsplit('_',1)[-1]
            unit='raw utility regret' if metric=='regret' else 'unsafe probability per attempted decision (percent)'
            ref_inputs += [r'\begin{table}[tbp]\centering',r'\caption{Absolute reference '+unit+r', with ordinary descriptive 95\% scene intervals. R0 is analytically expected performance, not realized integer events. R1 is one fixed training-estimated policy. All conditions share their planned draws; the pool has fixed one-third weights.}',r'\label{tab:reference_absolute_'+metric+'}',r'\input{tables/'+p.name+'}',r'\end{table}']
        else:
            ref_inputs.append(r'\input{tables/'+p.name+'}')
    (tables/'reference_complete.tex').write_text('\n'.join(ref_inputs)+'\n')
    for name in ('geometric_descriptive_intervals.csv','geometric_denominators.csv','geometric_analysis_provenance.json','geometric_scoring_provenance.json','reference_inventory_summary.json'):
        shutil.copy2(ROOT/'submission_strengthening/references'/name,data/name)
    geometric=read(ROOT/'submission_strengthening/references/geometric_descriptive_intervals.csv')
    for metric,unit,places in [('regret','raw regret',4),('unsafe','percent for absolute values; percentage points for differences',2),('H_reachable_terrain','mm',1)]:
        selected=[r for r in geometric if r['metric']==metric]
        metric_title={'regret':'raw regret','unsafe':'unsafe probability per attempt','H_reachable_terrain':'reachable-terrain height MAE'}[metric]
        lines=table_start('Existing geometric anchor on AFAST: '+metric_title+'; '+unit+r'. Ordinary post-hoc 95\% scene intervals from saved draws. Model contrasts are intact model minus anchor; no anchor removal arm.',
            'tab:anchor_'+metric,'llrrrr',r'Condition & Arm/contrast & Policy & Effect & Lower & Upper')
        for r in selected:
            # CSV already declares and scales the display units.
            vals=[f"{float(r['display_'+k]):.{places}f}" for k in ('estimate','lower95','upper95')]
            condition='1/3 pool' if ('pool' in r['condition'] or r['condition'].startswith('fixed_one_third')) else esc(r['condition'])
            lines.append(' & '.join([condition,esc(r['arm']),esc(r['policy'])]+vals)+r'\\')
        finish(lines,tables/('anchor_'+metric+'.tex'))
    print('Formatted unchanged AFAST tables and copied new source data')

if __name__=='__main__': main()
