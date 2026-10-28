"""Summarize actual metered jobs and conservative allocated-cap accounting."""
from pathlib import Path
import argparse,datetime,json,time
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'submission_strengthening'
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
cfg=json.loads((S/'resources/start_clock.json').read_text())
jobs=[json.loads(p.read_text()) for p in sorted((S/'resources').glob('job_*.json'))]
events=[]
for j in jobs:
    events.extend([(j['start_unix'],j['worker_cores']),(j['end_unix'],-j['worker_cores'])])
active=peak=0
for _,change in sorted(events,key=lambda x:(x[0],x[1])):active+=change;peak=max(peak,active)
sizes={}
for name in ('submission_strengthening','release_candidate','paper_submission_candidate'):
    sizes[name]=sum(p.stat().st_size for p in (ROOT/name).rglob('*') if p.is_file() and not p.is_symlink())
measured=sum(j['cpu_core_hours'] for j in jobs)
allocated=sum(j['wall_seconds']*j['worker_cores']/3600 for j in jobs)
allowance=cfg['unmetered_shell_and_agent_overhead_allowance_cpu_core_hours']
gpu=sum(j['gpu_device_hours'] for j in jobs)
checkpoint=json.loads((S/'reproduction/checkpoint_smoke64_recovery1/checkpoint_reproduction.json').read_text())
generator=json.loads((S/'reproduction/generator_smoke7/generator_reproduction.json').read_text())
now=time.time()
out=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),start_utc=cfg['first_action_utc'],
    elapsed_hours=(now-cfg['start_unix'])/3600,completed_metered_jobs=len(jobs),
    measured_process_cpu_core_hours=measured,launched_allocated_core_hours=allocated,
    unmetered_shell_agent_allowance_core_hours=allowance,
    conservatively_charged_cpu_core_hours=max(measured,allocated)+allowance,
    charged_gpu_device_hours=gpu,peak_aggregate_declared_worker_cores=peak,
    neural_sample_forwards=checkpoint['completed_sample_forwards'],reproduced_existing_sensor_records=generator['reproduced_existing_sensor_records'],
    new_scene_identities=0,new_bootstrap_draws=0,gradient_updates=0,dependency_acquisition_bytes=0,
    output_bytes_by_root=sizes,total_new_output_bytes=sum(sizes.values()),
    failures=[dict(name=j['name'],returncode=j['returncode'],error=j['error'],cpu_core_hours=j['cpu_core_hours'],gpu_device_hours=j['gpu_device_hours']) for j in jobs if j['returncode']!=0 or j['error']],
    caps=cfg['caps'],scope='All completed metered jobs, including failed jobs and recovery; allocation charges wall time times declared cores. Shell/agent reporting overhead has a separate conservative allowance. Symlinked external assets are not new output bytes. This receipt is a current snapshot, not historical backdating.')
out['within_caps']=out['elapsed_hours']<36 and out['conservatively_charged_cpu_core_hours']<64 and gpu<2 and peak<=8 and out['total_new_output_bytes']<10_000_000_000 and out['neural_sample_forwards']<=128 and out['reproduced_existing_sensor_records']<=8
target=ROOT/a.output;assert not target.exists(),target;target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('elapsed_hours','conservatively_charged_cpu_core_hours','charged_gpu_device_hours','peak_aggregate_declared_worker_cores','total_new_output_bytes','within_caps')},indent=2))
assert out['within_caps']
