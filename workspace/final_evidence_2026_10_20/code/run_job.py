"""Single-A100 metering with complete-scope forecast reservations and75h cap."""
from pathlib import Path
import argparse,datetime,fcntl,json,os,resource,signal,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--gpu',action='store_true');p.add_argument('--cores',type=int,default=4)
p.add_argument('--forecast-seconds',type=float);p.add_argument('--forecast-source');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
assert 1<=a.cores<=8;cmd=a.command[1:] if a.command[:1]==['--'] else a.command
path=ROOT/f'resources/job_{a.name}.json';assert not path.exists(),'Preserve every executed receipt'
plan=json.loads((ROOT/'resources/whole_round_forecast.json').read_text());planned={r['name']:r for r in plan['jobs']}
done=[json.loads(x.read_text()) for x in (ROOT/'resources').glob('job_*.json')];used=sum(r.get('gpu_hours',0) for r in done)
forecast=a.forecast_seconds if a.forecast_seconds is not None else planned[a.name]['forecast_seconds']
basis=a.forecast_source or planned[a.name]['basis'];assert forecast>0
future=sum(r['forecast_seconds'] for n,r in planned.items() if n!=a.name and not (ROOT/f'resources/job_{n}.json').exists())
lock=None
if a.gpu:
    lock=(ROOT/'resources/gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert used*3600+forecast+future<=75*3600,('STOP_TOTAL_FORECAST_EXCEEDS_CAP',used,forecast/3600,future/3600)
    assert a.name in planned,'Every GPU job needs a recorded whole-scope reservation'
    device=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,compute_cap','--format=csv,noheader'],text=True).strip()
    assert device.startswith('NVIDIA A100-PCIE-40GB, GPU-a3d97219-0fd5-b540-0fa1-46b9eb1dcf2c'),device
    timeout=75*3600-used*3600-future
else:device=None;timeout=forecast
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0' if a.gpu else '')
available=sorted(os.sched_getaffinity(0));cpus=available[:a.cores] if a.gpu else available[4:4+a.cores]
if len(cpus)<a.cores:cpus=available[:a.cores]
def constrain():os.sched_setaffinity(0,cpus)
start=time.time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
r=dict(name=a.name,command=cmd,gpu=a.gpu,device=device,cores=a.cores,cpu_affinity=cpus,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),start_unix=start,forecast_seconds_with25percent_margin=forecast,forecast_basis=basis,gpu_hours_used_before=used,future_reserved_gpu_hours=future/3600,cap_gpu_hours=75)
running=ROOT/f'resources/running_{a.name}.json';running.write_text(json.dumps(r,indent=2)+'\n')
with (ROOT/'resources/jobs.jsonl').open('a') as f:f.write(json.dumps(dict(event='launch',**r))+'\n')
proc=None;error=None;code=1
try:
    with (ROOT/f'logs/{a.name}.log').open('w') as log:
        proc=subprocess.Popen(cmd,env=env,preexec_fn=constrain,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
        code=proc.wait(timeout=timeout)
except BaseException as exc:
    error=repr(exc)
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    code=124 if isinstance(exc,subprocess.TimeoutExpired) else 1
finally:
    after=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.time()-start
    r.update(returncode=code,error=error,wall_seconds=wall,gpu_hours=wall/3600 if a.gpu else 0,cpu_core_hours=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)/3600,max_rss_kib=after.ru_maxrss,finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(r,indent=2)+'\n');running.unlink(missing_ok=True)
    with (ROOT/'resources/jobs.jsonl').open('a') as f:f.write(json.dumps(dict(event='finish',**r))+'\n')
print(json.dumps(r,indent=2),flush=True);raise SystemExit(code)
