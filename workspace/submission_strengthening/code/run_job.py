"""Meter bounded strengthening jobs. GPU permitted only for declared replay jobs."""
import argparse, datetime, json, os, resource, subprocess, time
from pathlib import Path

S=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--cores',type=int,default=1)
p.add_argument('--cpu-offset',type=int,default=0)
p.add_argument('--gpu-replay',action='store_true')
p.add_argument('--reporting',action='store_true')
p.add_argument('--timeout-seconds',type=float,default=7200)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
cfg=json.loads((S/'resources/start_clock.json').read_text());start=time.time()
deadline=cfg['hard_deadline_unix'] if a.reporting else cfg['scientific_work_cutoff_unix']
remaining=min(deadline-start,a.timeout_seconds)
assert remaining>0 and 1<=a.cores<=8 and 0<=a.cpu_offset and a.cpu_offset+a.cores<=8
finished=[json.loads(f.read_text()) for f in (S/'resources').glob('job_*.json')]
cpu=sum(j.get('cpu_core_hours',0) for j in finished)
gpu=sum(j.get('gpu_device_hours',0) for j in finished)
assert cpu+cfg['unmetered_shell_and_agent_overhead_allowance_cpu_core_hours']<64
if a.gpu_replay:
    assert not list((S/'resources').glob('gpu_job_running.json')),'Only one GPU replay job may run'
    remaining=min(remaining,(2-gpu)*3600);assert remaining>0
    (S/'resources/gpu_job_running.json').write_text(json.dumps({'job':a.name,'started':start}))
path=S/'resources'/f'job_{a.name}.json'
assert not path.exists(),'Use a fresh job name; never overwrite failed jobs'
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
env['CUDA_VISIBLE_DEVICES']='0' if a.gpu_replay else ''
if not a.gpu_replay:env['NVIDIA_VISIBLE_DEVICES']='none'
cpus=cfg['cpu_affinity_ceiling'][a.cpu_offset:a.cpu_offset+a.cores]
def constrain():
    os.sched_setaffinity(0,cpus)
    cap=max(1,int(min(remaining*a.cores,(62-cpu)*3600)))
    resource.setrlimit(resource.RLIMIT_CPU,(cap,cap))
before=resource.getrusage(resource.RUSAGE_CHILDREN);code=None;error=None
try:
    code=subprocess.run(command,env=env,preexec_fn=constrain,timeout=remaining).returncode
except BaseException as exc:error=repr(exc)
finally:
    after=resource.getrusage(resource.RUSAGE_CHILDREN);end=time.time()
    record=dict(name=a.name,command=command,start_unix=start,end_unix=end,wall_seconds=end-start,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_core_hours=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)/3600,
        gpu_device_hours=(end-start)/3600 if a.gpu_replay else 0,declared_gpu_reproduction=a.gpu_replay,
        worker_cores=a.cores,cpu_affinity=cpus,max_rss_kib=after.ru_maxrss,returncode=code,error=error)
    path.write_text(json.dumps(record,indent=2)+'\n')
    if a.gpu_replay:(S/'resources/gpu_job_running.json').unlink(missing_ok=True)
if error:raise RuntimeError(error)
raise SystemExit(code)
