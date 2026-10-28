"""Audit frozen inference on matching common rows; no replacement comparator outputs."""
import json
import numpy as np
import pandas as pd
from stage3.common import ROOT,RESULTS,ANALYSIS,write_json,sha256
import hashlib

def run():
 reports=[];differences=[]
 for tag,original in [('S3A_R2','R2_full'),('S3A_G3','G3_dyn')]:
  for seed in range(3):
   path=RESULTS/'A'/f'{tag}_s{seed}.csv';old=ROOT/'results'/f'{original}_s{seed}.csv'
   if not path.exists():continue
   a=pd.read_csv(old);b=pd.read_csv(path);a=a[a.suite.isin(['iid','all_s2','ood_terrain'])];keys=['suite','scene_id','frame_id'];a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index();assert a.index.equals(b.index)
   cols=[c for c in a if c in b and c not in ['tag','variant','seed']];same=True
   for c in cols:
    x,y=a[c],b[c];numeric=pd.api.types.is_numeric_dtype(x);missing=int((x.isna()!=y.isna()).sum());equal=(x==y)|(x.isna()&y.isna());diff=float((x-y).abs().max()) if numeric and x.notna().any() else None
    differences.append(dict(tag=tag,seed=seed,column=c,numeric=numeric,max_abs_deviation=diff,unequal_values=int((~equal).sum()),missingness_mismatches=missing));same &= bool(equal.all())
   canonical=lambda f:hashlib.sha256(f[cols].to_csv(index=True).encode()).hexdigest()
   reports.append(dict(tag=tag,seed=seed,original_csv_sha256=sha256(old),new_csv_sha256=sha256(path),checkpoint_sha256=sha256(ROOT/'runs'/original/f'seed{seed}/best.pt'),rows=len(a),keys_identical=True,common_columns_identical=same,original_common_sha256=canonical(a),replayed_common_sha256=canonical(b),full_csv_byte_identity_claimed=False))
 pd.DataFrame(differences).to_csv(RESULTS/'audit/replay_column_differences.csv',index=False);write_json(RESULTS/'audit/replay_hashes.json',reports)
 cache_hashes={str(p.relative_to(RESULTS)):sha256(p) for p in sorted((RESULTS/'A/cache').rglob('*.npz'))}
 write_json(RESULTS/'audit/probability_cache_hashes.json',cache_hashes)
 # Resolve the editorial rounding query from raw per-seed measurements, not the display alone.
 raw=[];sources={}
 for seed in range(3):
  p=ROOT/'results_stage2'/f'S2D_R2_oracle_s{seed}.csv';d=pd.read_csv(p);sources[str(p.relative_to(ROOT))]=sha256(p)
  for suite in ['cam_s3','all_s2']:
   g=d[d.suite==suite];raw.append(dict(seed=seed,suite=suite,rows=len(g),mean_H_mae_vis=g.groupby('scene_id').H_mae_vis.mean().mean()))
 d=pd.DataFrame(raw);pooled=d.groupby('suite').mean_H_mae_vis.mean();export=pd.read_csv(ROOT/'results_stage2/analysis/requested_six_variant_endpoints.csv');values={}
 for suite,v in pooled.items():
  saved=export[(export.tag=='S2D_R2_oracle')&(export.suite==suite)].H_mae_vis.iloc[0];assert np.isclose(v,saved,rtol=0,atol=1e-14);values[suite]=dict(raw_mean=v,exported_mean=saved,display=f'{v:.4f}')
 difference=float(pooled['cam_s3']-pooled['all_s2']);report=dict(sources=sources,values=values,difference=difference,exactly_equal=difference==0,identical_four_decimal_display=values['cam_s3']['display']==values['all_s2']['display'],conclusion='Rounding coincidence; independently aggregated raw values differ and match the export.')
 d.to_csv(RESULTS/'audit/oracle_rounding_per_seed.csv',index=False);write_json(RESULTS/'audit/oracle_rounding_check.json',report);(RESULTS/'audit/oracle_rounding_check.md').write_text('# D-oracle Table-8 rounding check\n\n'+json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':run()
