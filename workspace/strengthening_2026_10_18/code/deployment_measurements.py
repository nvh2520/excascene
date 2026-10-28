"""Warm batch1 FP32 timing and explicitly partial operator FLOP accounting."""
from common import *
import time,torch
from neural import backend,frozen,batch,ensure_fp32
from gate import load_gate,severe_probability

def measure():
    backend();records=[load_npz(PROJECT/f'data/val/f_{sid:06d}_0.npz') for sid in range(3000,3032)];rows=[]
    for architecture,seed in [('KF-D',10),('BF-D',10),('S5A',0),('G6',0)]:
        model,cp=frozen(architecture,seed);times=[]
        with torch.inference_mode():
            for i in range(70):
                d=records[i%32];drop=np.ones(3,bool) if architecture=='G6' else d['cam_present']<=.5
                torch.cuda.synchronize();start=time.perf_counter();b=batch([d],[drop]);out=model(b);ensure_fp32(out);torch.cuda.synchronize()
                if i>=20:times.append(time.perf_counter()-start)
            # PyTorch reports FLOPs for selected dense conv/matmul operators only.
            # Sparse custom kernels, BEV pooling and several elementwise ops do
            # not have complete FLOP metadata. Never label this a full total.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True,record_shapes=True) as prof:
                out=model(b);torch.cuda.synchronize()
        entries=[dict(operator=e.key,calls=e.count,reported_flops=e.flops) for e in prof.key_averages()]
        flops=sum(e['reported_flops'] for e in entries if e['reported_flops'])
        csvwrite(ROOT/f'study3/deployment/{architecture}_operator_flops.csv',entries)
        rows.append(dict(architecture=architecture,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
            reported_dense_operator_flops_lower_bound=int(flops),full_architecture_flops='NOT_FULLY_MEASURABLE_WITH_AVAILABLE_OPERATOR_COUNTERS',
            batch=1,warmup_frames=20,measured_frames=50,latency_median_seconds=float(np.median(times)),latency_p95_seconds=float(np.quantile(times,.95)),
            precision='actual FP32;TF32 off',includes='input normalization/extrinsics,host-device transfer,network,finite/dtype check; already loaded original validation frame',
            excludes='disk read,decision selection,gate',device=torch.cuda.get_device_name(0),checkpoint_sha256=cp['sha256']))
        js(ROOT/f'study3/deployment/{architecture}_latency_samples.json',times)
        del model;torch.cuda.empty_cache()
    gate=load_gate(ROOT/'study3/gate/final.pt');times=[]
    for i in range(70):
        torch.cuda.synchronize();start=time.perf_counter();severe_probability(gate,records[i%32]['rgb']);torch.cuda.synchronize()
        if i>=20:times.append(time.perf_counter()-start)
    rows.append(dict(architecture='RGB_gate_three_cameras_per_frame',parameters=sum(p.numel() for p in gate.parameters()),batch=1,warmup_frames=20,measured_frames=50,
        latency_median_seconds=float(np.median(times)),latency_p95_seconds=float(np.quantile(times,.95)),precision='actual FP32;TF32 off',device=torch.cuda.get_device_name(0)))
    csvwrite(ROOT/'study3/deployment/parameters_latency_and_flop_scope.csv',rows)
    js(ROOT/'study3/deployment/receipt.json',dict(status='COMPLETE_WITH_EXPLICIT_PARTIAL_FLOP_COVERAGE',neural_frames=4*71,gate_frame_calls=70,
        inputs='32originalvalidationframes;noholdoutinput',code_sha256=sha(__file__),comparison_limit='Unsupported sparse/lifting/pooling operations prevent an honest full FLOP total; dense-operator counts are lower bounds, not comparable full model costs.'))
if __name__=='__main__':measure()
