"""Render authorized C1 results into the closing manuscript, without new scoring."""
from pathlib import Path
import json,csv,re,shutil
import pandas as pd
O=Path(__file__).resolve().parents[1];P=O.parent;M=O/'manuscript'
def edit(rel,old,new):
    f=M/rel;s=f.read_text();assert old in s,(rel,old);f.write_text(s.replace(old,new))
def main():
    lock=json.loads((O/'C1/validation/selection_lock.json').read_text())
    a=pd.read_csv(O/'C1/holdout/condition_results.csv');p=pd.read_csv(O/'C1/holdout/paired_comparisons.csv')
    def row(cond,metric):return a[(a.condition==cond)&(a.metric==metric)].iloc[0]
    def ratio(arch):return p[(p.architecture==arch)&(p.contrast=='learned_divided_by_posthoc_classical')].iloc[0]
    kr=ratio('KF-D42');br=ratio('BF-D42');clean=row('clean','regret');unsafe=row('clean','unsafe')
    taus=lock['tau'];buf=lock['buffer'];rlo=min(kr.estimate,br.estimate);rhi=max(kr.estimate,br.estimate)
    # Table 2: preserve all historical rows byte-for-byte and add one policy.
    f=M/'tables/task_value_replication.tex';s=f.read_text();needle='Classical local-ground & 0.4700 & 20.21 & 0.4726 & 18.60 & 0.5849 & 42.19\\\\'
    vals=[]
    for cond in ['clean','cam_s3','rough_iid']:vals += [f'{row(cond,"regret").estimate:.4f}',f'{100*row(cond,"unsafe").estimate:.2f}']
    new='Classical (post hoc, validation-tuned) & '+' & '.join(vals)+r'\\'
    assert needle in s;s=s.replace(needle,needle+'\n'+new)
    s=s.replace('Complete paired intervals and condition inventories remain',f'The original classical row is frozen; the added row selects a {buf:g} m buffer and {taus:g} m height threshold from validation alone after extending the grid. Complete paired intervals and condition inventories remain')
    if any(row(c,'abstained').estimate for c in ['clean','cam_s3','rough_iid']):
        rates=', '.join(f'{100*row(c,"abstained").estimate:.2f}\\%' for c in ['clean','cam_s3','rough_iid'])
        s=s.replace('All displayed policies execute on 100\\% of attempts.',f'The post-hoc policy abstains on {rates} of clean, severe-camera and rough attempts, respectively; all other rows execute on every attempt.')
    f.write_text(s)
    # Table 11: new severe cause row with the same denominator and 95% intervals.
    f=M/'tables/severe_person_safety.tex';s=f.read_text();needle='Classical local-ground & 93.8 [76.7, 111.3] & 18.60 [16.41, 20.85]\\\\'
    person=row('cam_s3','multi_person_buffer');u=row('cam_s3','unsafe')
    new=f'Classical (post hoc, validation-tuned) & {1000*person.estimate:.1f} [{1000*person.lower95:.1f}, {1000*person.upper95:.1f}] & {100*u.estimate:.2f} [{100*u.lower95:.2f}, {100*u.upper95:.2f}]'+r'\\'
    assert needle in s;s=s.replace(needle,needle+'\n'+new)
    if row('cam_s3','abstained').estimate:
        replacement=f'The post-hoc classical row abstains on {100*row("cam_s3","abstained").estimate:.2f}\\% of attempts; all other rows execute throughout.'
        s=s.replace('Every policy executes on every attempt.',replacement)
        edit('sections/diagnostics.tex','Every policy executes on every attempt in this table.',replacement)
    f.write_text(s)
    # Complete 60-pair validation inventory; no results selected for presentation.
    grid=pd.read_csv(O/'C1/validation/extended_grid.csv').sort_values(['grid_part','tau','buffer'],key=lambda c:c.map({'original':0,'added':1}) if c.name=='grid_part' else c)
    cap=(r'Classical-policy validation for the original 30 pairs and 30 added pairs, on the same 1,200 frames. Threshold $\tau$ and dilation $b$ are metres. Unsafe and abstention use attempts; excluded is the percentage of reachable cells removed by the policy hazard mask, pooled across frames. F marks the frozen pair and P the post-hoc validation-selected pair. Selection minimizes mean validation regret, with exact ties broken by smaller $b$ and then smaller $\tau$. ')
    cap+=f'The selected pair is $\\tau={taus:g}$ m, $b={buf:g}$ m; '+('it remains at the largest buffer.' if lock['largest_buffer_selected'] else 'it is inside the extended buffer range.')
    text=r'\begingroup\scriptsize\setlength{\tabcolsep}{5pt}'+'\n'+r'\begin{longtable}{rrlrrrr}'+'\n'+r'\caption{'+cap+r'}\label{tab:classicalvalidation}\\\toprule'+'\n'
    head=r'$\tau$ & $b$ & Grid & Regret & Unsafe (\%) & Abstain (\%) & Excluded (\%)\\\midrule'
    text+=head+r'\endfirsthead'+'\n'+r'\multicolumn{7}{c}{Table \thetable\ continued}\\\toprule'+head+r'\endhead'+'\n'
    for _,r in grid.iterrows():
        mark=' F' if (r.tau,r.buffer)==(1.,4.) else ' P' if (r.tau,r.buffer)==(taus,buf) else ''
        text+=f'{r.tau:g} & {r.buffer:g} & {r.grid_part.capitalize()}{mark} & {r.regret:.5f} & {100*r.unsafe:.2f} & {100*r.abstained:.2f} & {100*r.excluded_reachable_share:.2f}'+r'\\'+'\n'
    text+=r'\bottomrule\end{longtable}\endgroup'+'\n';(M/'tables/classical_validation.tex').write_text(text)
    # Method, scope and adverse validation trade-off.
    old='The selected values are 1.0 m and 4.0 m; regret was still falling at that buffer boundary for every tested height threshold (Table~\\ref{S-tab:classicalvalidation}).'
    new=f'The original frozen values are 1.0 m and 4.0 m. A post-hoc extension adds six larger buffers at each of the five original height thresholds. The same minimum-regret rule selects {taus:g} m and {buf:g} m from the combined 60-pair validation grid; this one setting is then evaluated on the replication holdout (Table~\\ref{{S-tab:classicalvalidation}}).'
    edit('sections/design.tex',old,new)
    f=M/'sections/maps_results.tex';s=f.read_text();begin=s.index('Table~\\ref{tab:taskvaluereplication}');end=s.index('\\input{tables/task_value_replication.tex}')
    paragraph=(f'Table~\\ref{{tab:taskvaluereplication}} compares the classical baseline, both training recipes and the fixed references on the replication holdout. Extending the validation buffer grid selects {buf:g} m rather than the frozen 4 m. The post-hoc, validation-tuned policy has clean regret {clean.estimate:.4f} and unsafe frequency {100*unsafe.estimate:.2f}\\%, compared with 0.4700 and 20.21\\% for the frozen row. The 42-epoch models retain clean regret 0.1171 for KinAfford and 0.1119 for BEVFusion-Exca. KinAfford incurs {100*kr.estimate:.1f}\\% of the extended baseline\'s clean regret [95\\%: ${100*kr.lower95:.1f},{100*kr.upper95:.1f}$]. BEVFusion-Exca incurs {100*br.estimate:.1f}\\% [${100*br.lower95:.1f},{100*br.upper95:.1f}$]. These ratios are descriptive paired scene comparisons. Uniform selection and the fixed training prior remain separate references.\n')
    f.write_text(s[:begin]+paragraph+s[end:])
    old='The classical baseline has one fixed ground-estimation procedure and a bounded validation grid; validation regret was still decreasing at the largest tested buffer, 4.0 m (Table~\\ref{S-tab:classicalvalidation}).'
    new=(f'The classical baseline has one fixed ground-estimation procedure. Extending its buffer grid post hoc selects an interior {buf:g} m setting by validation regret, while retaining the frozen 4 m control. Validation regret falls from 0.4357 to {lock["selected_validation_regret"]:.4f}, but unsafe frequency rises from 20.75\\% to {100*lock["selected_validation_unsafe"]:.2f}\\%; the selected setting excludes {100*lock["selected_excluded_reachable_share"]:.2f}\\% of reachable cells (Table~\\ref{{S-tab:classicalvalidation}}). The extension follows inspection of the research holdout and is descriptive, although its parameter choice uses validation alone.')
    edit('sections/limitations.tex',old,new)
    # K1: report all retained and newly added comparator signs, including unfavorable ones.
    comp=pd.read_csv(O/'C1/sensitivity/learned_comparisons.csv');direct=comp[comp.policy=='direct'];neg=int((direct.sign=='negative').sum());total=len(direct)
    if neg==total:
        edit('sections/diagnostics.tex','and both models keep lower clean regret than the classical policy.','and both models keep lower clean regret than both the frozen and post-hoc validation-tuned classical policies.')
        edit('tables/replication_sensitivity.tex',"both models' clean-regret advantages over the classical policy stay negative throughout.","both models' clean-regret contrasts against both the frozen and post-hoc validation-tuned classical policies stay negative throughout.")
    else:
        edit('sections/diagnostics.tex','and both models keep lower clean regret than the classical policy.',f'and both models keep lower clean regret than the frozen classical policy. Against the post-hoc validation-tuned policy, {neg} of {total} model-by-setting direct regret contrasts are negative; every changed sign is reported in the supplementary inventory.')
    f=M/'sections/sensitivity_replication.tex';s=f.read_text();s+=f'\nThe post-hoc validation-tuned classical policy is added by rescoring its fixed clean choices under the same 27 settings. Learned-policy scene aggregates and all primary classifications are unchanged. Its learned-minus-classical direct comparisons are negative in {neg} of {total} model-by-setting rows; the complete inventory includes both selectors and ordinary 95\\% paired intervals.\n';f.write_text(s)
    # Abstract and highlights: ratios against extended setting, never the frozen one.
    f=M/'generated/abstract.tex';s=f.read_text();s=s.replace("to about one quarter of a classical baseline's",f"to {100*rlo:.0f}--{100*rhi:.0f}\\% of the extended classical baseline's")
    s=s.replace('About 5\\% of decisions entered person buffers.','About 5\\% of decisions entered person buffers, although a reachable safe cell always existed.')
    f.write_text(s)
    highlights=[
      'Paired sensor faults connect excavator perception errors to point-selection loss.',
      f'Clean learned regret is {100*rlo:.0f}–{100*rhi:.0f}% of a post-hoc, validation-tuned classical policy\'s.',
      'Camera fallback helps KinAfford under familiar severe faults in two studies.',
      'A simple depth-validity rule matches the tested RGB monitor in simulation.',
      'About 5% of picks enter person buffers although a reachable safe cell always exists.'
    ];assert all(len(x)<=85 for x in highlights),[(len(x),x) for x in highlights]
    (M/'highlights.txt').write_text('\n'.join(highlights)+'\n')
    f=O/'cover_letter_draft.md';s=f.read_text();s=s.replace("to roughly one quarter of a classical policy's, and removed the earlier resolved clean architecture ranking",f"to {100*rlo:.0f}–{100*rhi:.0f}% of the extended, validation-selected classical policy's, while the earlier clean-regret architecture difference became inconclusive")
    s=s.replace('Residual person-buffer choices and weak real-image transfer delimit the engineering claim.','About 5% of decisions still entered person buffers despite reachable safe alternatives; weak real-image transfer further delimits the engineering claim.')
    f.write_text(s);(M/'cover_letter_draft.md').write_text(s)
    header=r'\documentclass[10pt]{article}'+'\n'+r'\usepackage[a4paper,margin=2cm]{geometry}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{lmodern,microtype}\setlength{\parindent}{0pt}\setlength{\parskip}{7pt}\pagestyle{empty}'+'\n'+r'\begin{document}'+'\n'
    body=s.replace('%',r'\%').replace('&',r'\&').replace('_',r'\_').replace('Sincerely,\nHung Viet Nguyen and Jinhong Yang\n','Sincerely,\\\\{}\nHung Viet Nguyen and Jinhong Yang\\\\{}\n')
    (M/'cover_letter.tex').write_text(header+body+'\n'+r'\end{document}'+'\n')
    for name in ['condition_results.csv','paired_comparisons.csv']:
        shutil.copy2(O/'C1/holdout'/name,M/'data'/('C1_'+name))
    shutil.copy2(O/'C1/validation/extended_grid.csv',M/'data/C1_validation_grid.csv')
    shutil.copy2(O/'C1/sensitivity/learned_comparisons.csv',M/'data/C1_sensitivity_comparisons.csv')
    shutil.copy2(O/'C1/sensitivity/compact_sensitivity_table_extended.csv',M/'data/C1_sensitivity_extended.csv')
    (O/'report/C1_presentation_receipt.json').write_text(json.dumps({'ratio_low':rlo,'ratio_high':rhi,'highlight_lengths':[len(x) for x in highlights],
        'new_direct_sensitivity_negative':neg,'new_direct_sensitivity_total':total,'frozen_rows_retained':True},indent=2)+'\n')
if __name__=='__main__':main()
