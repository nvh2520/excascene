"""Generate readable per-metric reference tables, never recalculating evidence."""
import argparse
from pathlib import Path
import pandas as pd

CONDITIONS=['clean','iid','all_s2','cam_s1','cam_s2','cam_s3','rough_iid','fixed_one_third_iid_all_s2_rough']
LABELS=['Clean','IID','All S2','Camera S1','Camera S2','Camera S3','Rough IID','Fixed pool']

def cell(row,digits):
    vals=[row.display_estimate,row.display_lower95,row.display_upper95]
    est,lo,hi=[f'{v:.{digits}f}' for v in vals]
    return r'\shortstack{$'+est+r'$\\$['+lo+','+hi+r']$}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    t=out/'tables';t.mkdir(exist_ok=True)
    data=pd.read_csv(out/'reference_comparison_intervals.csv')
    for metric,digits,unit in [('regret',4,'raw regret'),('unsafe',2,'percentage points')]:
        for policy in ['direct','soft']:
            lines=['% Generated from reference_comparison_intervals.csv. Full machine precision is retained there.',
                r'\begingroup\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{longtable}{lllcc}',
                r'\caption{Post-hoc '+policy+' policy differences from the two references ('+unit+r'). Each cell gives model minus reference, then its ordinary unadjusted 95\% paired scene interval. Camera state 0 is intact; 1 is all cameras removed. The fixed pool equally weights IID, All S2 and Rough IID. These intervals use the existing saved draws and are descriptive, with approximate coverage.}',
                r'\label{tab:reference_'+policy+'_'+metric+r'}\\',r'\toprule',
                r'Condition & Model & Camera state & Versus R0 & Versus R1\\',r'\midrule\endfirsthead',
                r'\toprule Condition & Model & Camera state & Versus R0 & Versus R1\\\midrule\endhead',
                r'\midrule\endfoot\bottomrule\endlastfoot']
            for cond,label in zip(CONDITIONS,LABELS):
                for arch in ['BF-D','KF-D']:
                    for mask in [0,1]:
                        g=data[(data.metric==metric)&(data.model_policy==policy)&(data.condition==cond)&(data.model==arch)&(data.camera_mask==mask)]
                        assert len(g)==2
                        lines.append(' & '.join([label,arch,str(mask),cell(g[g.reference=='R0'].iloc[0],digits),cell(g[g.reference=='R1'].iloc[0],digits)])+r'\\')
            lines.extend([r'\end{longtable}',r'\endgroup'])
            (t/f'reference_{policy}_{metric}.tex').write_text('\n'.join(lines)+'\n')
    for metric,digits,unit in [('regret',4,'raw regret'),('unsafe',2,'percent expected/observed unsafe per attempt')]:
        lines=['% Generated from reference_comparison_intervals.csv; R0 probabilities are expectations, never events.',
            r'\begingroup\small',r'\begin{tabular}{lcc}',r'\toprule Condition & R0 & R1\\\midrule']
        for cond,label in zip(CONDITIONS,LABELS):
            g=data[(data.model=='reference')&(data.condition==cond)&(data.metric==metric)]
            lines.append(' & '.join([label,cell(g[g.reference=='R0'].iloc[0],digits),cell(g[g.reference=='R1'].iloc[0],digits)])+r'\\')
        lines.extend([r'\bottomrule\end{tabular}',r'\endgroup'])
        (t/f'reference_absolute_{metric}.tex').write_text('\n'.join(lines)+'\n')
    print('Six reference TeX table fragments written, with complete saved-draw comparison inventory.')

if __name__=='__main__':main()
