"""Replace the historical proxy with recorded SegFormer pilot measurements."""
import json,math
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
p=json.loads((OUT/'resources/segformer_pilot.json').read_text())
assert p['status']=='COMPLETE'
train=p['train_seconds_per_batch_p95'];ev=p['eval_seconds_per_batch_p95'];margin=1.25
jobs=[]
for domain,images in [('sim',1200),('real',2164)]:
    steps=24*math.ceil(images/16)
    # Pilot includes dataloader wait, transfers/forward/backward/step; add startup/checkpoint allowance.
    seconds=margin*(steps*train+60)
    jobs.append(dict(name=f'C_{domain}_fit',kind='fit',domain=domain,epochs=24,updates=steps,
                     forecast_seconds_with_margin=seconds,rate_source='recorded end-to-end pilot p95 plus60s setup/checkpoint'))
for domain,images in [('sim',300),('real',192)]:
    batches=15*math.ceil(images/16)
    corruption=json.loads((OUT/'resources/corruption_cpu_timing.json').read_text())
    novel=json.loads((OUT/'resources/novel_cpu_timing.json').read_text())
    cpu_batch=16*max(r['p95_seconds'] for r in corruption['rows']+novel['rows'])
    seconds=margin*(batches*(ev+cpu_batch)+60)
    jobs.append(dict(name=f'C_{domain}_evaluate',kind='evaluate',domain=domain,forwards=images*15,
                     forecast_seconds_with_margin=seconds,rate_source='recorded end-to-end pilot p95 plus worst recorded CPU corruption p95 times16 (conservative no CPU parallelism credit), plus60s setup'))
r=dict(status='MEASURED_FORECAST',margin=margin,pilot_source='resources/segformer_pilot.json',jobs=jobs,
       training_and_segmentation_evaluation_forecast_hours=sum(x['forecast_seconds_with_margin'] for x in jobs)/3600,
       pilot_actual_gpu_hours=p['occupied_gpu_hours'],phase_cap_gpu_hours=6,
       gate_forecast='Pending recorded gate timing; gate probabilities evaluated once, threshold variants reuse arrays')
r['fits_before_gate']=r['training_and_segmentation_evaluation_forecast_hours']+r['pilot_actual_gpu_hours']<6
dst=OUT/'resources/segmentation_measured_forecast.json'
assert not dst.exists(),'Never overwrite an executed forecast record'
dst.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
