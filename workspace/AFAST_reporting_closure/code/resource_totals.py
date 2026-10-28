"""Snapshot this reporting closure's costs, distinct from historical AFAST costs."""
import datetime
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'AFAST_reporting_closure'
config=json.loads((C/'resources/closure_start.json').read_text())
jobs=[json.loads(p.read_text()) for p in sorted((C/'resources').glob('job_*.json'))]
now=time.time()
cpu=sum(j['cpu_core_hours'] for j in jobs)
size=sum(p.stat().st_size for base in [C,ROOT/'paper_v5_2'] for p in base.rglob('*') if p.is_file())
result=dict(stage=sys.argv[1] if len(sys.argv)>1 else 'snapshot',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            recorded_start_utc=config['recorded_start_utc'],elapsed_hours_since_captured_start=(now-config['recorded_start_unix'])/3600,
            conservative_preflight_allowance_hours=config['unmetered_read_only_preflight_allowance_seconds']/3600,
            charged_elapsed_hours=(now-config['budget_start_unix'])/3600,
            metered_completed_job_cpu_core_hours=cpu,
            unmetered_read_only_shell_reporting_cpu_allowance_core_hours=config['unmetered_command_cpu_allowance_core_hours'],
            charged_cpu_core_hours=cpu+config['unmetered_command_cpu_allowance_core_hours'],
            launched_completed_job_count=len(jobs),failed_job_names=[j['name'] for j in jobs if j['returncode']!=0],
            new_output_bytes=size,output_roots=['AFAST_reporting_closure','paper_v5_2'],
            max_requested_worker_cores=max([j['cores'] for j in jobs] or [0]),aggregate_affinity_ceiling=8,
            gpu_device_hours=0,new_training_updates=0,new_neural_forwards=0,new_bootstrap_draws=0,new_synthetic_records=0,
            network_acquisition_bytes=0,purchases=0,local_electricity_and_amortization='not priced',
            caps=dict(elapsed_hours=8,cpu_core_hours=24,worker_cores=8,new_output_bytes=5000000000),
            metering_scope='Completed CPU-runner child process user+system time including descendants. Small shell/edit/inspection overhead and this final snapshot use an explicit conservative allowance; no exact OS-wide process census is claimed.',
            historical_costs='Actual historical AFAST final totals are in delivery_sidecar and are not added to this new instruction budget.')
result['within_caps']=result['charged_elapsed_hours']<8 and result['charged_cpu_core_hours']<24 and size<5000000000 and result['max_requested_worker_cores']<=8
assert result['within_caps'],result
name='final_resource_totals.json' if result['stage']=='final' else 'resources_at_package_freeze.json'
(C/'resources'/name).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
