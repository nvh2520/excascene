"""Readable task-value tables/support inventory, using completed saved scores."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from reference_policies import write_json,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--afast',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output
    r=pd.read_csv(out/'reference_per_frame.csv');ci=pd.read_csv(out/'reference_comparison_intervals.csv')
    m=pd.read_csv(a.afast/'analysis/condition_results.csv')
    assert m.abstained_count.sum()==0
    support=r[['reference','condition','scene_id','frame_id','oracle','reachable_cells','eligible_cells','regret']].copy()
    support['utility_stratum']=np.where(support.oracle==0,'zero',np.where(support.oracle<=.05,'low','positive'))
    support['normalized_regret']=np.where(support.oracle>.05,support.regret/np.where(support.oracle>0,support.oracle,1),np.nan)
    support['normalized_regret_undefined']=support.oracle<=.05
    support.to_csv(out/'reference_frame_support.csv',index=False)
    rows=[]
    for key,g in support.groupby(['reference','condition']):
        rows.append(dict(reference=key[0],condition=key[1],records=len(g),oracle_zero=int((g.utility_stratum=='zero').sum()),
            oracle_low=int((g.utility_stratum=='low').sum()),oracle_positive=int((g.utility_stratum=='positive').sum()),
            normalized_regret_undefined=int(g.normalized_regret_undefined.sum()),normalized_regret_defined=int(g.normalized_regret.notna().sum()),
            source='reference_per_frame.csv; oracle>0.05 required for normalized regret'))
    pd.DataFrame(rows).to_csv(out/'reference_support_summary.csv',index=False)
    m[['architecture','mask','condition','policy','attempted_count','unsafe_count','executed_count','abstained_count',
       'unsafe_per_execution_pooled_diagnostic','unsafe_per_execution_undefined','oracle_zero_frame_seed_count',
       'oracle_low_frame_seed_count','oracle_positive_frame_seed_count','undefined_normalized_regret_frame_seed_count']].to_csv(out/'saved_model_denominators.csv',index=False)
    table=[]
    for cond in ['clean','cam_s3','rough_iid']:
        for method in ['R0','R1','BF-D','KF-D']:
            if method.startswith('R'):
                g=r[(r.reference==method)&(r.condition==cond)]
                table.append(dict(condition=cond,method=method,regret=float(g.regret.mean()),unsafe_percent=100*float(g.unsafe.mean()),
                    unsafe_event_count='' if method=='R0' else int(g.unsafe.sum()),attempted_decisions=len(g),execution_percent=100*float(g.executed.mean()),
                    inference='analytic/deterministic reference evaluated once',unsafe_kind='expected_probability' if method=='R0' else 'deterministic_events',source='reference_per_frame.csv'))
            else:
                row=m[(m.architecture==method)&(m['mask']==0)&(m.policy=='direct')&(m.condition==cond)].iloc[0]
                table.append(dict(condition=cond,method=method,regret=float(row.regret),unsafe_percent=100*float(row.unsafe),
                    unsafe_event_count=int(row.unsafe_count),attempted_decisions=int(row.attempted_count),execution_percent=100*float(row.executed),
                    inference='mean of three separately scored fitted seeds; intact direct',unsafe_kind='observed_events',source='original analysis/condition_results.csv'))
    pd.DataFrame(table).to_csv(out/'task_value_compact_table.csv',index=False)
    quality=[]
    for ref in ['R0','R1']:
        g=ci[(ci.reference==ref)&ci.model.isin(['BF-D','KF-D'])&ci.metric.isin(['regret','unsafe'])]
        quality.append(dict(reference=ref,loss_contrasts=len(g),positive_point_effects=int((g.estimate>0).sum()),intervals_including_zero=int(((g.lower95<=0)&(g.upper95>=0)).sum()),
            inference='inventory descriptive count, not multiple-testing evidence or a selection rule'))
    write_json(out/'reference_inventory_summary.json',dict(status='COMPLETE',reference_records=len(r),interval_rows=len(ci),model_denominator_rows=len(m),
        all_reference_executions=int(r.executed.sum()),all_reference_abstentions=int(r.abstained.sum()),all_model_abstentions=int(m.abstained_count.sum()),
        priors_selected_cells=r[r.reference=='R1'][['pick_row','pick_col']].drop_duplicates().astype(int).to_dict('records'),
        direction_inventory=quality,files={p.name:digest(p) for p in [out/'reference_support_summary.csv',out/'saved_model_denominators.csv',out/'task_value_compact_table.csv']}))
    print(pd.DataFrame(table).to_string(index=False))

if __name__=='__main__':main()
