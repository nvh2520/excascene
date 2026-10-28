"""Reporting-only dense operation count from the frozen gate's declared shapes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];NEW=ROOT.parent
source=NEW/'code/gate.py';body=source.read_text()
assert 'for out in [32,64,96,128,192]' in body and 'nn.Conv2d(ch,out,3,stride=2,padding=1)' in body and 'nn.Linear(192,4)' in body
h,w,ch=160,256,3;rows=[];macs=0;params=0
for index,out in enumerate([32,64,96,128,192]):
    h=(h+2-3)//2+1;w=(w+2-3)//2+1
    n=h*w*out*ch*3*3;macs+=n;params+=out*ch*3*3+out
    rows.append(dict(layer=index+1,output_h=h,output_w=w,in_channels=ch,out_channels=out,multiply_accumulate_pairs=n));ch=out
macs+=192*4;params+=192*4+4;assert params==407652
p=ROOT/'data/gate_analytical_dense_cost.json';assert not p.exists()
p.write_text(json.dumps(dict(status='REPORTING_DERIVATION_FROM_FROZEN_SOURCE',parameters=params,per_camera_conv_linear_macs=macs,
    three_camera_conv_linear_flops=2*3*macs,flop_convention='one multiplication plus one addition per MAC; three160x256RGB cameras per frame',
    scope='Analytic convolution/linear dense lower bound only; excludes bias additions, ReLU, pooling, softmax, input normalization, depth-validity calculation and memory traffic. Distinct from measured latency and operator-profiler model lower bounds.',
    new_neural_forwards=0,new_training=False,new_scientific_outcomes=False,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),layers=rows),indent=2)+'\n')
print(2*3*macs,'dense convolution/linear FLOPs for three cameras; no inference')
