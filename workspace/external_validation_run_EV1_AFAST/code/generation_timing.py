"""Timing only: re-render existing train scene0's original frame0 pose."""
import pathlib,sys,json,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/excascene_snapshot'))
from kinafford.sim.scene import generate_scene,bev_ground_truth
from kinafford.sim.render import Renderer
from kinafford.sim.kinematics import CAMERAS
p=ROOT.parent/'data/train/f_000000_0.npz'
with np.load(p) as z:q=z['q_true'].astype(np.float64)
s=generate_scene(0);r=Renderer(s);box=r._machine_boxes(q);static=r._static_objects();times={}
start=time.monotonic();gt=bev_ground_truth(s,q);times['oracle_seconds']=time.monotonic()-start
for c in CAMERAS:
    start=time.monotonic();r.render_camera(c,q,np.random.default_rng(40),machine_boxes=box,static=static);times['camera_'+c+'_seconds']=time.monotonic()-start
start=time.monotonic();r.render_lidar(q,None,machine_boxes=box,static=static);times['lidar_seconds']=time.monotonic()-start
# Upper work estimate: 3 distinct extrinsic renders/view and3 LiDAR renders,
# doubling measured CPU work for rough variation, degradation/compression/validity,
# with two single-thread workers. Only training pose is used for this forecast.
bundle=sum(v for k,v in times.items() if k.startswith('camera'))*3+times['lidar_seconds']*3+times['oracle_seconds']
forecast=768*bundle*2/2
receipt=dict(status='MEASURED_ORIGINAL_TRAIN_INPUT',input=str(p.relative_to(ROOT.parent)),scene_id=0,frame_id=0,
    timings=times,forecast_generation_seconds=forecast,workers=2,threads_per_worker=1,
    forecast_multiplier_before_general_margin=2.,new_holdout_inputs_generated=0,utc_unix=time.time())
(ROOT/'reports/generation_timing.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
