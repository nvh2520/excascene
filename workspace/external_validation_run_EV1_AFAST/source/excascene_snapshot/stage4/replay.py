"""Compare stored frozen endpoints; explicitly cannot establish absent map identity."""
import hashlib,json
import numpy as np
import pandas as pd
from stage4.common import ROOT,RESULTS,sha256,write_json,utc
SUITES=['cam_s3','all_s3','test_iid']
def compare(a,b,keys):
 a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
 if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index):raise ValueError('Replay pairing differs')
 cols=[c for c in a if c in b and pd.api.types.is_numeric_dtype(a[c]) and c not in ['seed']];rows=[]
 for c in cols:
  x,y=a[c],b[c];eq=(x==y)|(x.isna()&y.isna());finite=np.isfinite(x)&np.isfinite(y)
  rows.append(dict(column=c,frames=len(a),max_abs_deviation=float((x[finite]-y[finite]).abs().max()) if finite.any() else None,unequal_values=int((~eq).sum()),missingness_mismatches=int((x.isna()!=y.isna()).sum()),nonfinite_mismatches=int(((~finite)&(~eq)).sum())))
 canonical=lambda f:hashlib.sha256(f[cols].to_csv(index=True).encode()).hexdigest()
 return rows,dict(keys_identical=True,numeric_columns=cols,original_common_sha256=canonical(a),replayed_common_sha256=canonical(b),stored_numeric_columns_identical=all(r['unequal_values']==0 for r in rows))
def run():
 records=[];differences=[];cache_hashes={}
 for original,tag in [('R2_full','S4B_R2_cache'),('G3_dyn','S4B_G3_cache')]:
  for seed in range(3):
   source=ROOT/'results'/f'{original}_s{seed}.csv';new=RESULTS/'B'/f'{tag}_s{seed}.csv';a=pd.read_csv(source);b=pd.read_csv(new)
   assert set(b.suite)==set(SUITES)
   for suite in SUITES:
    rows,receipt=compare(a[a.suite==suite],b[b.suite==suite],['suite','scene_id','frame_id']);differences.extend(dict(original=original,tag=tag,seed=seed,suite=suite,**r) for r in rows)
    records.append(dict(original=original,tag=tag,seed=seed,suite=suite,original_csv_sha256=sha256(source),replayed_csv_sha256=sha256(new),checkpoint_sha256=sha256(ROOT/'runs'/original/f'seed{seed}/best.pt'),**receipt))
   for p in sorted((RESULTS/'B/cache'/f'{tag}_s{seed}').rglob('*.npz')):cache_hashes[str(p.relative_to(RESULTS))]=sha256(p)
 pd.DataFrame(differences).to_csv(RESULTS/'audit/replay_column_differences.csv',index=False)
 write_json(RESULTS/'audit/replay_hashes.json',dict(time=utc(),records=records,cache_files=len(cache_hashes),cellwise_argmax_class_identity='UNVERIFIABLE: original CSVs do not store class maps',cellwise_height_identity='UNVERIFIABLE: original CSVs do not store height maps',requested_stored_proxies=['sem_acc_vis','cm_0 through cm_35','mat_acc','H_mae_vis','H_mae_all','H_mae_reach','regret_direct'],frozen_csv_changed=False))
 write_json(RESULTS/'audit/probability_cache_hashes.json',cache_hashes)
 print('REPLAY: 18 suite/seed comparisons; all stored numeric columns checked. Cellwise class/height identity unavailable in frozen CSVs.',flush=True)
if __name__=='__main__':run()
