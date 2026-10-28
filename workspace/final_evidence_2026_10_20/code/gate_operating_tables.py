"""Descriptive operating counts from frozen gate outputs; no choices or new draws."""
from common import *
import pandas as pd
def main():
    from infer_study4 import verify_lock
    verify_lock();source=ROOT/'study4/holdout';scores=pd.read_csv(source/'gate_scores.csv').set_index(['condition','scene_id','frame_id'])
    thresholds=json.loads((ROOT/'study4/thresholds.json').read_text());rows=[]
    for r in csv.DictReader((source/'files.csv').open()):
        sid=int(r['scene_id']);slot=int(r['frame_id']);condition=r['condition'];p=scores.loc[(condition,sid,slot)]
        d=load_npz(source/r['path'],['camera_fault_severity','camera_novel_fault','cam_present'])
        for arch,t in thresholds.items():
            for j in range(3):
                present=int(d['cam_present'][j]);rgb=int(p[f'p{j}']>=t['tau']);depth=int(p[f'invalid{j}']>=t['delta'])
                rows.append(dict(architecture=arch,condition=condition,scene_id=sid,frame_id=slot,camera=j,present=present,
                    injected_severity=int(d['camera_fault_severity'][j]),novel=int(d['camera_novel_fault'][j]),
                    supplied_oracle_drop=int(d['camera_fault_severity'][j]>=3 or d['camera_novel_fault'][j]),
                    rgb_trigger=rgb,depth_trigger=depth,both_triggers=rgb*depth,removed=int(rgb or depth),
                    probability_severe=float(p[f'p{j}']),invalid_depth_fraction=float(p[f'invalid{j}'])))
    out=ROOT/'study4/analysis';f=pd.DataFrame(rows);f.to_csv(out/'gate_per_camera.csv',index=False)
    present=f[f.present==1];keys=['architecture','condition','injected_severity','novel']
    agg=present.groupby(keys).agg(present_camera_observations=('present','size'),removed=('removed','sum'),rgb_trigger=('rgb_trigger','sum'),
        depth_trigger=('depth_trigger','sum'),both_triggers=('both_triggers','sum'),oracle_drop=('supplied_oracle_drop','sum')).reset_index()
    agg['removal_fraction']=agg.removed/agg.present_camera_observations
    agg.to_csv(out/'gate_operating_inventory.csv',index=False)
    js(out/'gate_operating_receipt.json',dict(status='COMPLETE_DESCRIPTIVE_COUNTS',new_forwards=0,new_bootstrap_draws=0,
        severity_scope='known injected appearance/depth mechanisms; novel flag separately; legacy rel_gt proxy is not a novel-fault severity label',
        absent_cameras_excluded_from_operating_denominators=int((f.present==0).sum()),count_scope='camera observations; architectures share one frozen classifier and have distinct validation thresholds; no independent repetitions implied'))
if __name__=='__main__':main()
