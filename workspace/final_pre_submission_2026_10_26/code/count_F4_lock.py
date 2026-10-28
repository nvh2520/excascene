"""Saved observations only; fix examples before rendering."""
from pathlib import Path
import csv,json,hashlib,datetime
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1];P=OUT.parent;F=P/'final_evidence_2026_10_20'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sources={};pairs=[]
    for arch in ['KF-D42','BF-D42']:
        for seed in [10,11,12]:
            src=F/f'study4/metrics/{arch}_seed{seed}.csv';sources[str(src.relative_to(P))]=sha(src)
            f=pd.read_csv(src);f=f[(f.condition=='cam_s3')&f.arm.isin(['gate','keep_all'])]
            a=f[f.arm=='gate'].set_index(['scene_id','frame_id','policy']);b=f[f.arm=='keep_all'].set_index(['scene_id','frame_id','policy']).reindex(a.index)
            assert a.index.is_unique and not b.regret.isna().any()
            r=a[['architecture','seed']].copy();r['changed']=(a.pick_row!=b.pick_row)|(a.pick_col!=b.pick_col)
            r['delta']=a.regret-b.regret;r['unsafe_to_safe']=(b.unsafe==1)&(a.unsafe==0);r['safe_to_unsafe']=(b.unsafe==0)&(a.unsafe==1)
            for arm,d in [('gate',a),('keep_all',b)]:
                for k in ['prediction_path','pick_row','pick_col','regret','unsafe']:r[arm+'_'+k]=d[k]
            pairs.append(r.reset_index())
    allf=pd.concat(pairs,ignore_index=True);allf.to_csv(OUT/'F4/paired_saved_frames.csv',index=False)
    rows=[]
    for scope,keys in [('all_three_seeds',['architecture','policy']),('per_seed',['architecture','policy','seed'])]:
        for key,g in allf.groupby(keys):
            c=g[g.changed];n=len(g);nc=len(c);row=dict(zip(keys,key));row.update(scope=scope,attempts=n,changed_frames=nc,changed_share=nc/n,
                changed_lower=int((c.delta<0).sum()),changed_equal=int((c.delta==0).sum()),changed_higher=int((c.delta>0).sum()),
                lower_share_changed=float((c.delta<0).mean()) if nc else None,equal_share_changed=float((c.delta==0).mean()) if nc else None,
                higher_share_changed=float((c.delta>0).mean()) if nc else None,mean_regret_change_changed=float(c.delta.mean()) if nc else None,
                unsafe_to_safe=int(g.unsafe_to_safe.sum()),safe_to_unsafe=int(g.safe_to_unsafe.sum()))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'F4/decision_change_counts.csv',index=False)
    f=allf[(allf.architecture=='KF-D42')&(allf.seed==10)&(allf.policy=='direct')]
    chosen=[]
    for label,g in [('improved',f[f.delta<0]),('worsened',f[f.delta>0]),('unchanged',f[~f.changed])]:
        g=g.sort_values(['scene_id','frame_id'] if label=='unchanged' else ['delta','scene_id','frame_id'])
        if not len(g):chosen.append(dict(category=label,status='NONE_EXIST'));continue
        i=0 if label=='unchanged' else (len(g)-1)//2;r=g.iloc[i];sid=int(r.scene_id);slot=int(r.frame_id);data=F/f'study4/holdout/data/cam_s3/f_{sid}_{slot}.npz'
        chosen.append(dict(category=label,status='SELECTED',rank_index=i,category_size=len(g),scene_id=sid,frame_id=slot,delta=float(r.delta),data=str(data.relative_to(P)),data_sha256=sha(data),
            **{arm:{k:(int(r[arm+'_'+k]) if k.startswith('pick_') else r[arm+'_'+k]) for k in ['prediction_path','pick_row','pick_col']} for arm in ['keep_all','gate']}))
    prev=json.loads((P/'pre_submission_2026_10_24/visualizations/selection_lock.json').read_text())
    lock=dict(status='LOCKED_BEFORE_RENDERING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rule='KinAfford 42-epoch seed10, Severe camera, direct policy. Improved and worsened groups defined by strictly negative and positive saved gate-minus-keep-all regret. Order each by delta ascending, scene_id, frame_id; select lower median floor((n-1)/2). Unchanged means identical selected cell; choose lowest scene_id then frame_id. No display was inspected to choose frames.',
        F3_frames=chosen,F3_source=prev['F3_source'],F2_source=prev['F2_source'],F2_source_sha256=prev['F2_source_sha256'],source_sha256=sources,
        old_figure='pre_submission_2026_10_24/manuscript/figures/qualitative_decisions.pdf',new_thresholds=0,new_metrics='Only user-authorized descriptive saved-frame counts; no selector or inference',gpu_hours=0)
    dest=OUT/'visualizations/selection_lock.json';assert not dest.exists();dest.write_text(json.dumps(lock,indent=2)+'\n')
    (OUT/'F4/receipt.json').write_text(json.dumps(dict(status='COMPLETE',source_sha256=sources,rows=len(allf),counts_rows=len(rows),rule_lock_sha256=sha(dest),new_bootstrap_draws=0,gpu_hours=0),indent=2)+'\n')
    print(pd.DataFrame(rows).query("scope=='all_three_seeds' and policy=='direct'").to_string(index=False));print([(r['category'],r.get('scene_id'),r.get('frame_id'),r.get('delta')) for r in chosen])
if __name__=='__main__':main()
