"""Descriptive paired sequence-bootstrap degradation ordering, no tuning."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
OUT=Path(__file__).resolve().parents[1]

def domain_stats(cm):
    # [..., true class, predicted class]; ignored truth was excluded upstream.
    gt=cm.sum(-1);pred=cm.sum(-2);tp=np.diagonal(cm,axis1=-2,axis2=-1)
    union=gt+pred-tp
    iou=np.divide(tp,union,out=np.full(tp.shape,np.nan),where=union>0)
    # Same fixed semantic evaluation rule under clean and all faults.
    present=gt>0
    miou=np.divide(np.where(present,iou,0).sum(-1),present.sum(-1),
                   out=np.full(present.shape[:-1],np.nan),where=present.sum(-1)>0)
    return miou,iou,gt,pred,tp

def bootstrap_domain(cms,draws):
    # Preserve each sequence as a whole and share counts across all conditions.
    gt=cms.sum(-1);pred=cms.sum(-2);tp=np.diagonal(cms,axis1=-2,axis2=-1)
    g=np.einsum('bs,csk->bck',draws,gt,optimize=True)
    p=np.einsum('bs,csk->bck',draws,pred,optimize=True)
    t=np.einsum('bs,csk->bck',draws,tp,optimize=True)
    iou=np.divide(t,g+p-t,out=np.full(t.shape,np.nan),where=(g+p-t)>0)
    present=g>0
    means=np.divide(np.where(present,iou,0).sum(-1),present.sum(-1),
                    out=np.full(present.shape[:-1],np.nan),where=present.sum(-1)>0)
    rel=np.divide(means,means[:,0,None],out=np.full(means.shape,np.nan),where=means[:,0,None]>0)
    return means,rel

def corr_rows(x,y):
    a=rankdata(x,axis=-1);b=rankdata(y,axis=-1);a-=a.mean(-1,keepdims=True);b-=b.mean(-1,keepdims=True)
    den=np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    return np.divide((a*b).sum(-1),den,out=np.full(den.shape,np.nan),where=den>0)

def ci(x):
    good=np.isfinite(x)
    return (np.quantile(x[good],[.025,.975]).tolist() if good.any() else [None,None]),int((~good).sum())

def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--draws',type=int,default=20000);ap.add_argument('--seed',type=int,default=20261019)
    a=ap.parse_args()
    assert not (OUT/'metrics/sequence_bootstrap_draws.npz').exists(), 'Reuse the saved draws for any reporting correction'
    rng=np.random.default_rng(a.seed);domains={};rows=[];classrows=[];all_draws={}
    for domain in ('sim','real'):
        with np.load(OUT/'metrics'/f'{domain}_segmentation_confusions.npz',allow_pickle=False) as z:
            cms=z['sequence_confusions'];conditions=z['conditions'].tolist();seq=z['sequences'].tolist()
        n=len(seq);counts=rng.multinomial(n,np.ones(n)/n,size=a.draws);all_draws[domain+'_counts']=counts;all_draws[domain+'_sequences']=np.array(seq)
        means,iou,gt,pr,tp=domain_stats(cms.sum(1));boots,relboots=bootstrap_domain(cms,counts)
        rel=np.divide(means,means[0],out=np.full(means.shape,np.nan),where=means[0]>0)
        domains[domain]=dict(relative=rel,relative_boot=relboots,conditions=conditions)
        for c,name in enumerate(conditions):
            ic,iu=ci(boots[:,c]);rc,ru=ci(relboots[:,c]);dc,du=ci(boots[:,0]-boots[:,c])
            rows.append(dict(domain=domain,condition=name,sequences=n,miou=means[c],miou_ci95_lo=ic[0],miou_ci95_hi=ic[1],
                relative_miou=rel[c],relative_ci95_lo=rc[0],relative_ci95_hi=rc[1],miou_drop=means[0]-means[c],
                drop_ci95_lo=dc[0],drop_ci95_hi=dc[1],bootstrap_undefined_miou=iu,bootstrap_undefined_relative=ru,
                interval_scope='descriptive sequence bootstrap; not multiplicity controlled'))
            for k in range(len(iou[c])):
                classrows.append(dict(domain=domain,condition=name,class_id=k,gt_pixels=int(gt[c,k]),
                    predicted_pixels=int(pr[c,k]),true_positive_pixels=int(tp[c,k]),iou=iou[c,k],
                    clean_iou=iou[0,k],absolute_iou_drop=iou[0,k]-iou[c,k],
                    relative_iou=(iou[c,k]/iou[0,k] if iou[0,k]>0 else float('nan')),
                    included_in_miou=bool(gt[c,k]>0)))
    assert domains['sim']['conditions']==domains['real']['conditions']
    corr=float(corr_rows((1-domains['sim']['relative'][1:])[None],(1-domains['real']['relative'][1:])[None])[0])
    cb=corr_rows(1-domains['sim']['relative_boot'][:,1:],1-domains['real']['relative_boot'][:,1:])
    cc,undef=ci(cb)
    np.savez_compressed(OUT/'metrics/sequence_bootstrap_draws.npz',**all_draws)
    np.savez_compressed(OUT/'metrics/sequence_bootstrap_results.npz',correlation=cb,
                        sim_relative=domains['sim']['relative_boot'],real_relative=domains['real']['relative_boot'])
    write_csv(OUT/'metrics/degradation_results.csv',rows);write_csv(OUT/'metrics/degradation_per_class.csv',classrows)
    result=dict(spearman_relative_drop_ordering=corr,ordinary_ci95=cc,undefined_bootstrap_draws=undef,
        bootstrap_draws=a.draws,bootstrap_seed=a.seed,conditions=domains['sim']['conditions'][1:],
        sampling_unit='ALICE recording sequence; simulator scene, including all three cameras',
        paired_within_domain=True,independent_across_domains=True,
        scope='descriptive image-level degradation ordering; native domain ontologies differ; no decision or safety inference')
    (OUT/'metrics/degradation_ordering.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
