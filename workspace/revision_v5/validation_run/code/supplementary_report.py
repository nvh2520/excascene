"""Report the locked comparisons and all decision accounting; no new contrasts."""
from common import *
import pandas as pd

def md(df, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(f'{row[c]:.7g}' if isinstance(row[c], float) else str(row[c]) for c in columns) + ' |')
    return '\n'.join(lines)

def main():
    locked()
    a = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in sorted((RUN/'metrics_corrected').glob('*.csv'))], ignore_index=True)
    group = ['model','checkpoint','seed','mask','condition','policy']
    accounting = a.groupby(group).agg(
        attempted=('executed','size'), executed=('executed','sum'), abstained=('abstained','sum'),
        unsafe_events=('unsafe','sum'), empty_reach=('empty_reach','sum'), empty_candidates=('empty_candidates','sum'),
        exact_tie_frames=('exact_tie','sum'), reach_boundary_cells=('reach_boundary','sum'), hazard_boundary_cells=('hazard_boundary','sum'),
        nonfinite_score_cells=('nonfinite_scores','sum'), nonfinite_reach_cells=('nonfinite_reach','sum'),
        nonfinite_hazard_cells=('nonfinite_hazard','sum'), nonfinite_height_cells=('nonfinite_height','sum'),
        maximum_global_feasible_oracle_gap=('global_feasible_gap','max'))
    accounting['unsafe_per_attempt'] = accounting.unsafe_events/accounting.attempted
    accounting['unsafe_per_executed'] = accounting.unsafe_events/accounting.executed.replace(0,np.nan)
    accounting.to_csv(RUN/'decision_accounting.csv')
    # These are arm means for the four existing endpoints, not additional tests.
    p = pd.read_csv(RUN/'primary_results.csv')
    specs = [('G','H_reachable_terrain','G3_dyn','G2_static','last','direct',['clean','wider_articulation']),
      ('S','regret','R2_full','R2_noaff','last','soft',['iid','all_s2','rough_iid']),
      ('C-regret','regret','S5A_G3_structured','S5A_G3_structured','best','direct',['cam_s3']),
      ('C-unsafe','unsafe','S5A_G3_structured','S5A_G3_structured','best','direct',['cam_s3'])]
    arms=[]; readings=[]; quantized=[]
    for question, endpoint, ma, mb, checkpoint, policy, conditions in specs:
        z=p[p.question.eq(question)].iloc[0]
        for arm, model, mask in [('A',ma,question.startswith('C-')),('B',mb,False)]:
            t=a[a.model.eq(model)&a.checkpoint.eq(checkpoint)&a.policy.eq(policy)&a.condition.isin(conditions)&a['mask'].eq(mask)]
            per=t.groupby(['condition','seed'])[endpoint].mean()
            arms.append(dict(question=question,arm=arm,model=model,checkpoint=checkpoint,mask=mask,policy=policy,
                mean=float(per.groupby('condition').mean().mean()), per_seed=json.dumps(per.groupby('seed').mean().to_dict()),
                attempted_seed_decisions=len(t),executed=int(t.executed.sum()),unsafe_events=int(t.unsafe.sum())))
            if endpoint=='regret':
                quantized.append(dict(question=question,arm=arm,
                    fp32_utility_mean=float(per.groupby('condition').mean().mean()),
                    fp16_utility_mean=float(t.groupby(['condition','seed']).regret_fp16_target.mean().groupby('condition').mean().mean()),
                    picks_and_reach_unchanged=True))
        direction='lower loss supported' if z.ci9875_hi<0 else 'higher loss supported' if z.ci9875_lo>0 else 'direction unresolved'
        readings.append(dict(question=question,adjusted_interval_interpretation=direction,
            scope='Locked synthetic comparison, environmental uncertainty conditional on three fitted seeds; no equivalence claim.'))
    csvwrite(RUN/'primary_arm_means.csv',arms);csvwrite(RUN/'primary_interpretations.csv',readings)
    quantized_effects=[]
    for question in ['S','C-regret']:
        qa=next(x for x in quantized if x['question']==question and x['arm']=='A')
        qb=next(x for x in quantized if x['question']==question and x['arm']=='B')
        fp32=qa['fp32_utility_mean']-qb['fp32_utility_mean'];fp16=qa['fp16_utility_mean']-qb['fp16_utility_mean']
        quantized_effects.append(dict(question=question,fp32_utility_effect=fp32,fp16_utility_effect=fp16,
            fp16_minus_fp32_effect=fp16-fp32,scope='Descriptive utility-storage-only sensitivity; same picks, reachable candidates, sensor inputs and forward outputs; no new test.'))
    csvwrite(RUN/'primary_target_quantization_sensitivity.csv',quantized_effects)
    seconds=pd.read_csv(RUN/'secondary_results.csv');mechanisms=pd.read_csv(RUN/'camera_mechanism_results.csv')
    cells=[]
    for _,r in mechanisms.drop_duplicates(['model','mask','endpoint']).iterrows():
        cells.append(dict(model=r.model,mask=r['mask'],endpoint=r.endpoint,**json.loads(r.cell_means)))
    csvwrite(RUN/'camera_mechanism_cell_means.csv',cells)
    # Unique geometry/support rows, avoiding repeated policies and seed-varying predictions in counts.
    frame=a.drop_duplicates(['model','checkpoint','seed','mask','condition','scene_id','frame_id'])
    support=frame.groupby(['model','checkpoint','seed','mask','condition']).agg(
        frames=('frame_id','size'),reachable=('n_reachable_terrain','sum'),visible=('n_suite_visible','sum'),clean=('n_clean_fixed','sum'),
        intersection=('n_intersection','sum'),lost=('visibility_lost','sum'),gained=('visibility_gained','sum'),
        reachable_MAE=('H_reachable_terrain','mean'),visible_MAE=('H_suite_visible','mean'),clean_MAE=('H_clean_fixed','mean'),intersection_MAE=('H_intersection','mean'))
    support['visibility_jaccard']=support.intersection/(support.visible+support.clean-support.intersection).replace(0,np.nan)
    support.to_csv(RUN/'support_diagnostics.csv')
    # Do not convert a zero-event sample into a zero population probability.
    zero_cells=int((accounting.unsafe_events==0).sum())
    text=['# Fixed secondary results and decision accounting','',
      'All contrasts below were specified before outcomes. Their intervals are descriptive environmental intervals, conditional on the fitted seeds. Specialist contrasts estimate differences; they do not test equivalence. All complete numeric rows and per-seed sensitivities remain in the associated CSV files.','',
      '## Primary arm means','',md(pd.DataFrame(arms),['question','arm','model','mask','mean','attempted_seed_decisions','executed','unsafe_events']),'',
      'Means use the locked condition weights. Pooled event totals in the S arm rows are accounting totals, not the one-third-weighted S estimand.','',
      '## Utility ablation and checkpoint sensitivity','',md(seconds[seconds.question.str.startswith('S_')],['question','mean_diff','ci95_lo','ci95_hi']),'',
      'The primary uses soft without extra hazard exclusion and final epoch 14. Best-checkpoint, hard, and hazard-filtered soft contrasts remain separate. Equal epoch does not remove original training initialization/data-order asymmetries; both soft arms use privileged grade and reach. Contextual selectors remain on their originally supervised backbones.','',
      '## Camera controls','',md(seconds[seconds.question.str.startswith('C_')&seconds.question.str.endswith(('_regret','_unsafe'))],['question','mean_diff','ci95_lo','ci95_hi']),'',
      'The 28-scene combined-present subset has a different sampling denominator from full-condition results. Compare mechanism terms only within its complete paired block. Camera-only interventions preserve LiDAR and joints; all-camera masking preserves calibration and kinematic conditioning as specified.','',
      '## Paired mechanism cell means','',md(pd.DataFrame(cells),['model','mask','endpoint','clean','appearance_only','depth_only','combined_present','cam_s3','absence_only']),'',
      'All seven prespecified mechanism contrasts per model/endpoint, including interaction and cam_s3 minus combined-present, are in camera_mechanism_results.csv. These are descriptive subset analyses; neither their favorable nor unfavorable findings add a primary test.','',
      '## Numerical and support denominators','',
      f'decision_accounting.csv records every model/checkpoint/seed/mask/condition/policy count, including threshold boundaries, exact ties, nonfinite cells, abstention, empty reach and candidate sets, and unsafe rates per attempted and executed decision. There are {zero_cells} zero-event cells; a degenerate bootstrap interval cannot prove zero population risk. Utility strata and undefined low-oracle normalized regrets remain explicit in utility_strata.csv and per-frame exports. No fallback relaxes a restriction.','',
      'support_diagnostics.csv reports per-seed reachable, suite-visible, clean-fixed and intersection MAE with cell denominators and visibility losses/gains. Visibility is rendered terrain-hit support before measured-depth corruption. Camera absence/mount drift can alter that support. Rough clean references use the exact accepted poses in a versioned CPU-only correction; primary reachable support is unchanged. No calibrated masked-model coverage claim is made.','',
      'primary_target_quantization_sensitivity.csv reports the two primary regret point effects under FP32 versus FP16 utility storage at the same picks and candidate support. It is a descriptive scoring sensitivity, separate from sensor-input precision and actual BF16/FP32 inference. No input, reach, policy or model output is changed.','',
      'The three fixed seeds are not an independent sample of all possible retraining outcomes. Per-seed and leave-one-seed-out effects are exported separately from shared-scene intervals. No dropout-necessity contrast, additional architecture, or post-outcome model selection was added.']
    (RUN/'reports/secondary_results_report.md').write_text('\n'.join(text)+'\n')
    event('fixed_secondary_report_complete',primary_arms=len(arms),secondary_rows=len(seconds),mechanism_rows=len(mechanisms),accounting_rows=len(accounting))
    print('Fixed secondary report complete',len(accounting))

if __name__=='__main__':main()
