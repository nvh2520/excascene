"""Prospective seed-10 deployment timing on original validation inputs only."""
import io,json,pathlib,time,zipfile
import numpy as np
import torch
from models import build,neural_inputs
from data import SharedDataset
from scoring import neutral
ROOT=pathlib.Path(__file__).resolve().parents[1]
release=json.loads((ROOT/'records/holdout_prediction_release.json').read_text());assert release['status']=='RELEASED_AFTER_SIX_FINALS'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False
ds=SharedDataset('val',10,False);rows=[];total=0
for arch in ['BF-D','KF-D']:
    checkpoint=torch.load(ROOT/'checkpoints'/arch/'seed10/final.pt',map_location='cpu',weights_only=False)
    m=build(arch).cuda().float();m.load_state_dict(checkpoint['model'],strict=True);m.eval()
    def run(indices):
        global total
        cpu=[ds[i] for i in indices]
        batch={k:torch.stack([c[k] for c in cpu]).cuda() for k in neural_inputs(cpu[0])}
        total+=len(indices)
        with torch.no_grad():out=m(batch)
        # Include synchronized output transfer and neutral FP32 serialization.
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for i in range(len(indices)):
                for k,v in neutral(out,i).items():
                    encoded=io.BytesIO();np.save(encoded,v,allow_pickle=False);archive.writestr(f'{i}_{k}.npy',encoded.getvalue())
        torch.cuda.synchronize();return len(buffer.getvalue())
    for i in range(8):run([i])
    single=[];bytecounts=[]
    for i in range(32):
        torch.cuda.synchronize();begin=time.perf_counter();bytecounts.append(run([i]));single.append(time.perf_counter()-begin)
    for i in range(4):run(list(range(i*4,i*4+4)))
    batched=[]
    for i in range(8):
        torch.cuda.synchronize();begin=time.perf_counter();run(list(range(i*4,i*4+4)));batched.append(time.perf_counter()-begin)
    rows.append(dict(architecture=arch,seed=10,inputs='first 32 original validation frames in stored order; no holdout',
        includes='NPZ deserialize, input normalization and dynamic transforms, host/device transfer, network, neutral FP32 output transfer and in-memory ZIP serialization',
        excludes='model/checkpoint setup, disk output writes, oracle/selector scoring',precision='actual FP32; TF32 off',
        single_frame_latency_seconds_mean=float(np.mean(single)),single_frame_latency_seconds_p50=float(np.quantile(single,.5)),
        single_frame_latency_seconds_p95=float(np.quantile(single,.95)),single_frame_seconds=single,
        batch4_seconds=batched,batch4_throughput_frames_per_second=32/sum(batched),
        measured_single_frames=32,single_warmup_frames=8,batch4_warmup_frames=16,batch4_measured_frames=32,
        parameters_total=sum(p.numel() for p in m.parameters()),parameters_trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),
        serialized_output_bytes=bytecounts,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    del m;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
assert total==176
receipt=dict(status='COMPLETE',logical_forwards=total,rows=rows)
(ROOT/'reports/deployment_timing.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
