"""New pretrained-only descriptive results; prior image/gate results untouched."""
from pathlib import Path
import csv,hashlib,importlib.util,json
import numpy as np
OUT=Path(__file__).resolve().parents[1];NEW=OUT.parent;PROJECT=NEW.parent
spec=importlib.util.spec_from_file_location('fixed_image_statistics',PROJECT/'strengthening_2026_10_18/goose_ex/code/analyze_segmentation.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
def write(name,rows):
    with (OUT/'metrics'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
draws=np.load(OUT/'protocol/shared_sequence_draws.npz',allow_pickle=False)
domains={};rows=[];classrows=[];summary=[]
for domain,res in [('sim','parity'),('real','parity'),('real','width512')]:
    with np.load(OUT/'metrics'/f'{domain}_{res}_segmentation_confusions.npz',allow_pickle=False) as z:
        cms=z['sequence_confusions'];seq=z['sequences'];conditions=z['conditions'].tolist()
    assert np.array_equal(seq,draws[domain+'_sequences'])
    means,iou,gt,pr,tp=core.domain_stats(cms.sum(1));boots,relboots=core.bootstrap_domain(cms,draws[domain+'_counts'])
    rel=means/means[0] if means[0]>0 else np.full(len(means),np.nan)
    key=domain+'_'+res;domains[key]=dict(relative=rel,replicates=relboots,conditions=conditions)
    for i,c in enumerate(conditions):
        ci,undefined=core.ci(boots[:,i]);rc,ru=core.ci(relboots[:,i]);dc,du=core.ci(boots[:,0]-boots[:,i])
        rows.append(dict(domain=domain,resolution=res,condition=c,images=300 if domain=='sim' else 192,sequences=len(seq),miou=float(means[i]),lower95=ci[0],upper95=ci[1],relative_miou=float(rel[i]),relative_lower95=rc[0],relative_upper95=rc[1],drop=float(means[0]-means[i]),drop_lower95=dc[0],drop_upper95=dc[1],undefined_miou_draws=undefined,undefined_relative_draws=ru,scope='descriptive, ordinary95 sequence bootstrap; same fitted checkpoints; not multiplicity controlled'))
        for k in range(len(iou[i])):
            classrows.append(dict(domain=domain,resolution=res,condition=c,class_id=k,truth_pixels=int(gt[i,k]),predicted_pixels=int(pr[i,k]),true_positive_pixels=int(tp[i,k]),included_in_miou=bool(gt[i,k]>0),iou=float(iou[i,k]),clean_iou=float(iou[0,k]),absolute_drop=float(iou[0,k]-iou[i,k]),relative_iou=float(iou[i,k]/iou[0,k]) if iou[0,k]>0 else None))
    ci,_=core.ci(boots[:,0]);summary.append(dict(domain=domain,resolution=res,clean_miou=float(means[0]),clean_lower95=ci[0],clean_upper95=ci[1],truth_supported_classes=int((gt[0]>0).sum()),supported_zero_tp_classes=int(((gt[0]>0)&(tp[0]==0)).sum())))
corr=[];rep={}
for realkey in ['real_parity','real_width512']:
    sim=domains['sim_parity'];real=domains[realkey];assert sim['conditions']==real['conditions']
    point=float(core.corr_rows((1-sim['relative'][1:])[None],(1-real['relative'][1:])[None])[0]);b=core.corr_rows(1-sim['replicates'][:,1:],1-real['replicates'][:,1:]);ci,u=core.ci(b)
    corr.append(dict(comparison='sim_parity_vs_'+realkey,spearman=point,lower95=ci[0],upper95=ci[1],undefined_draws=u,scope='descriptive sequence-bootstrap ordering over14non-clean conditions'))
    rep[realkey]=b
write('degradation_results.csv',rows);write('per_class_results.csv',classrows);write('clean_summary.csv',summary);write('ordering_results.csv',corr)
np.savez_compressed(OUT/'metrics/ordering_replicates.npz',**rep)
(OUT/'metrics/completion.json').write_text(json.dumps(dict(status='COMPLETE',summary=summary,orderings=corr,new_bootstrap_draws=0,saved_draws_sha256=hashlib.sha256((OUT/'protocol/shared_sequence_draws.npz').read_bytes()).hexdigest(),earlier_image_and_gate_results='unchanged',scope='new pretrained image-check outputs only; no decision/safety claim'),indent=2)+'\n')
print(json.dumps(dict(summary=summary,orderings=corr),indent=2))
