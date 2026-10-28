"""Standalone scalar reconstruction: NumPy + standard library; no Torch/checkpoints."""
import argparse,csv,json,pathlib,collections
import numpy as np
from statistical_core import continuous_bootstrap,unsafe_bootstrap,direction
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);a=p.parse_args()
directory=a.root/'analysis'
rows=list(csv.DictReader((directory/'primary_scene_inputs.csv').open()))
frames=list(csv.DictReader((directory/'primary_paired_frames.csv').open()))
grouped=collections.defaultdict(list)
for frame in frames:grouped[int(frame['scene_id'])].append(frame)
for row in rows:
    group=grouped[int(row['scene_id'])]
    assert len(group)==int(row['attempted_paired_seed_slots'])
    difference=sum(int(g['unsafe_masked'])-int(g['unsafe_intact']) for g in group)
    assert difference==int(row['unsafe_difference_count'])
    assert sum(int(g['unsafe_intact']) for g in group)==int(row['unsafe_intact_count'])
    assert sum(int(g['unsafe_masked']) for g in group)==int(row['unsafe_masked_count'])
    frame_mean=np.mean([float(g['regret_effect']) for g in group],dtype=np.float64)
    # A fixed float64 serialization/reduction check, never effect snapping.
    assert abs(frame_mean-float(row['raw_regret_effect']))<=32*np.finfo(np.float64).eps
with np.load(directory/'bootstrap_multiplicities.npz') as b:
    assert list(map(int,b['nominal_scene_ids']))==[int(r['scene_id']) for r in rows]
    matrix=b['nominal']
r,ri,_=continuous_bootstrap([float(x['raw_regret_effect']) for x in rows],matrix)
u,ui,_,denom=unsafe_bootstrap([int(x['unsafe_difference_count']) for x in rows],[int(x['attempted_paired_seed_slots']) for x in rows],matrix)
saved=json.loads((directory/'primary_results.json').read_text())['endpoints']
assert r==saved[0]['estimate'] and list(ri)==saved[0]['ordinary95']+saved[0]['adjusted97_5']
assert str(u)==saved[1]['estimate_exact'] and list(map(str,ui))==saved[1]['ordinary95_exact']+saved[1]['adjusted97_5_exact']
receipt=dict(status='SCALAR_RECONSTRUCTION_MATCHED',raw_regret_estimate=r,raw_regret_bounds=ri.tolist(),
    unsafe_estimate_exact=str(u),unsafe_bounds_exact=list(map(str,ui)),unsafe_bootstrap_denominator=denom,
    raw_regret_interpretation=direction(ri[2],ri[3]),unsafe_interpretation=direction(ui[2],ui[3]),
    paired_frame_event_counts_verified=True,continuous_frame_scene_reduction_tolerance=32*np.finfo(np.float64).eps)
(a.root/'records/scalar_reconstruction.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
