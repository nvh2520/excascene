"""Input-only audit of all original validation poses/targets and fault replay."""
from common import *
import pandas as pd
def main():
    source=ROOT/'study4/validation';complete=json.loads((source/'completion.json').read_text())
    assert complete['nominal_source_frames']==1200
    files=pd.read_csv(source/'files.csv');original=pd.read_csv(source/'original.csv');slots=pd.read_csv(source/'slots.csv')
    assert len(original)==1200 and original[['scene_id','frame_id']].drop_duplicates().shape[0]==1200
    assert set(original.scene_id)==set(range(3000,3300))
    assert not (ROOT/'study4/holdout').exists()
    nominal=files[files.population=='nominal'];rough=files[files.population=='rough']
    assert len(nominal)==8400 and set(rough.condition)=={'rough_iid'}
    assert not files.duplicated(['condition','scene_id','frame_id']).any()
    checks=0;unknown_component=0
    for (sid,frame),g in nominal.groupby(['scene_id','frame_id']):
        assert len(g)==7 and set(g.condition)=={'clean','iid','cam_s1','cam_s2','cam_s3','single_severe','all_s2'}
        orig=original[(original.scene_id==sid)&(original.frame_id==frame)].iloc[0]
        oldpath=PROJECT/orig.source_path;assert sha(oldpath)==orig.source_sha256
        old=load_npz(oldpath);ds={}
        for r in g.to_dict('records'):
            path=source/r['path'];assert sha(path)==r['sha256'];d=load_npz(path);ds[r['condition']]=d
            for k in ['q_true','gt_H','gt_sem','gt_mat','gt_haz','gt_aff','gt_reach','z_target']:assert np.array_equal(d[k],old[k]),(sid,frame,k)
            unknown_component+=int(str(d['component'])=='UNRECORDED_ORIGINAL_VALIDATION_COMPONENT')
            assert np.isfinite(d['rgb']).all() and np.isfinite(d['depth']).all()
            checks+=1
        clean=ds['clean']
        for c in ['cam_s1','cam_s2','cam_s3','single_severe']:
            for k in ['lidar_range','lidar_valid','q_rep','q_true','drift']:assert np.array_equal(ds[c][k],clean[k])
        assert ds['single_severe']['cam_present'].all()
    for r in rough.to_dict('records'):
        path=source/r['path'];assert sha(path)==r['sha256'];d=load_npz(path)
        for k in ['rgb','depth','gt_H','gt_aff','gt_reach','q_true']:assert np.isfinite(d[k]).all()
        assert int(d['scene_id'])==int(r['scene_id']) and int(d['frame_id'])==int(r['frame_id']);checks+=1
    assert len(nominal)==8400 and len(rough)>0
    js(source/'validation_audit.json',dict(status='PASS_INPUTS_ONLY',nominal_source_frames=1200,nominal_records=len(nominal),rough_records=len(rough),
        rough_excluded_slots=int((~slots[slots.population=='rough'].accepted).sum()),checked_records=checks,
        exact_original_nominal_pose_and_targets=True,unknown_original_component_records=unknown_component,
        manifest_sha256=sha(source/'files.csv'),no_holdout_generated=True,neural_forwards=0,
        interpretation='Full original nominal validation pose/target bank; standardized known-fault input replay; separate rough development replay. No claim of byte-identical original RGB.'))
    print('PASS full validation inputs',checks,flush=True)
if __name__=='__main__':main()
