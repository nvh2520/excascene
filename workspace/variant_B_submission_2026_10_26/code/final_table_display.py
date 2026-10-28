"""Presentation precision/spacing repairs only; input estimates stay unchanged."""
from pathlib import Path
import csv
O=Path(__file__).resolve().parents[1];P=O/'manuscript/tables'
f=P/'replication_sensitivity.tex';f.write_text(f.read_text().replace(r'{\tabcolsep}{2pt}',r'{\tabcolsep}{3pt}'))
f=P/'prevalence.tex';s=f.read_text();a=s.index(r'\multicolumn{4}{l}{\textbf{KinAfford');b=s.index(r'\bottomrule',a)
data=list(csv.DictReader((O/'K3/prevalence_mixtures.csv').open()));rows=[]
for arch,name in [('KF-D42','KinAfford dynamic'),('BF-D42','BEVFusion-Exca')]:
    rows.append(r'\multicolumn{4}{l}{\textbf{'+name+r' (42)}}\\')
    for p in [1,5,10,25,50]:
        for m in ['regret','unsafe']:
            r=next(r for r in data if r['architecture']==arch and int(r['severe_prevalence_percent'])==p and r['metric']==m)
            scale=1 if m=='regret' else 100;digits=5 if m=='regret' else 4
            x=[scale*float(r[k]) for k in ['estimate','lower95','upper95']]
            y=[float(r[k]) for k in ['percent_of_clean_regret','percent_clean_lower95','percent_clean_upper95']]
            raw=f'${x[0]:+.{digits}f}$ [$ {x[1]:+.{digits}f}, {x[2]:+.{digits}f} $]'
            relative=f'${y[0]:+.3f}$ [$ {y[1]:+.3f}, {y[2]:+.3f} $]'
            rows.append(f'{p} & '+('Regret' if m=='regret' else 'Unsafe (pp)')+' & '+raw+' & '+relative+r'\\')
f.write_text(s[:a]+'\n'.join(rows)+'\n'+s[b:])
