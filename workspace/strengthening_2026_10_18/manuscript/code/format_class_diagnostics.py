"""Format every saved confusion cell/class; no inference, scoring or resampling."""
from pathlib import Path
import ast,csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];NEW=ROOT.parent;OUT=ROOT/'data/class_diagnostics';OUT.mkdir(exist_ok=False)
names={'semantic':['terrain','rock','truck','person','obstacle','machine'],'material':['sand','soil','gravel','rocky']}
rows=[];cells=[];totals={};sources={}
for p in sorted((NEW/'study3/metrics').glob('*_confusions.json')):
    sources[str(p.relative_to(NEW))]=hashlib.sha256(p.read_bytes()).hexdigest()
    a,seed=p.stem.removesuffix('_confusions').rsplit('_seed',1)
    for key,heads in json.loads(p.read_text()).items():
        condition,arm=ast.literal_eval(key)
        for head,raw in heads.items():
            c=np.array(raw,dtype=np.int64);assert c.shape==(len(names[head]),len(names[head])) and (c>=0).all()
            totals.setdefault((a,'all_three',condition,arm,head),np.zeros_like(c));totals[(a,'all_three',condition,arm,head)]+=c
            totals[(a,seed,condition,arm,head)]=c
assert len(sources)==12
for (a,seed,condition,arm,head),c in sorted(totals.items()):
    ident=dict(architecture=a,seed=seed,condition=condition,arm=arm,head=head)
    for i,name in enumerate(names[head]):
        tp=int(c[i,i]);truth=int(c[i].sum());predicted=int(c[:,i].sum());union=truth+predicted-tp
        rows.append(dict(**ident,class_id=i,class_name=name,tp=tp,truth_support=truth,predicted_support=predicted,
            recall=None if not truth else tp/truth,precision=None if not predicted else tp/predicted,iou=None if not union else tp/union,
            support_rule='all BEV cells' if head=='semantic' else 'truth-terrain BEV cells'))
        for j,pred_name in enumerate(names[head]):cells.append(dict(**ident,truth_class=i,predicted_class=j,count=int(c[i,j])))
for name,data in [('all_class_metrics.csv',rows),('all_confusion_cells.csv',cells)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(OUT/'provenance.json').write_text(json.dumps(dict(status='COMPLETE',sources=sources,class_metric_rows=len(rows),confusion_cell_rows=len(cells),new_forwards=0,new_bootstrap_draws=0,new_decision_scoring=False,note='Undefined ratios stay empty; no selection by favorable result. Semantic mapping is the original six BEV classes (sky is not a BEV label).'),indent=2)+'\n')
print(len(rows),'class rows',len(cells),'confusion cells')
