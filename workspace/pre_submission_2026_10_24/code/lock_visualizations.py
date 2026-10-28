"""Fix examples before rendering, using a declared saved-output rank rule."""
from pathlib import Path
import csv,hashlib,json,datetime
OUT=Path(__file__).resolve().parents[1];PROJECT=OUT.parent;F=PROJECT/'final_evidence_2026_10_20'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 src=F/'study4/metrics/KF-D42_seed10.csv';rows=[r for r in csv.DictReader(src.open()) if r['condition']=='cam_s3' and r['policy']=='direct' and r['arm'] in ['keep_all','gate']]
 pairs={}
 for r in rows:pairs.setdefault((int(r['scene_id']),int(r['frame_id'])),{})[r['arm']]=r
 ordered=sorted(pairs,key=lambda k:(float(pairs[k]['gate']['regret'])-float(pairs[k]['keep_all']['regret']),*k))
 picked=[]
 for p in [.25,.50,.75]:
  index=int((len(ordered)-1)*p+.5);sid,slot=ordered[index];pair=pairs[(sid,slot)]
  data=F/f'study4/holdout/data/cam_s3/f_{sid}_{slot}.npz'
  picked.append(dict(percentile_rank=int(100*p),rank_index=index,scene_id=sid,frame_id=slot,data=str(data.relative_to(PROJECT)),data_sha256=sha(data),
    keep_all={k:pair['keep_all'][k] for k in ['prediction_path','pick_row','pick_col']},gate={k:pair['gate'][k] for k in ['prediction_path','pick_row','pick_col']}))
 clean=F/'study4/holdout/data/clean/f_1200000_0.npz'
 lock=dict(status='LOCKED_BEFORE_RENDERING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 F2_rule='First accepted clean fixture record, scene1200000 slot0; all three camera views; fixed image-check fault implementations, severe level3 and both novel mechanisms.',
 F2_source=str(clean.relative_to(PROJECT)),F2_source_sha256=sha(clean),
 F3_rule='KinAfford final42 seed10, severe-camera direct policy; sort every accepted frame by saved gate-minus-keep-all regret, breaking ties by scene_id then frame_id; choose nearest integer rank at25/50/75percent. No image inspection precedes selection.',
 F3_source=str(src.relative_to(PROJECT)),F3_source_sha256=sha(src),F3_frames=picked,
 reported_new_metrics=False,neural_forwards_planned=0,gpu_hours_forecast=0,gpu_cap_hours=1,
 inference='All required input and prediction maps already exist; no GPU or checkpoint replay is necessary.')
 (OUT/'visualizations/selection_lock.json').write_text(json.dumps(lock,indent=2)+'\n');print([(r['scene_id'],r['frame_id'],r['percentile_rank']) for r in picked])
if __name__=='__main__':main()
