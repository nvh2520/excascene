"""Meter isolated jobs, enforce the single GPU and forecasted phase budgets."""
import argparse, datetime, fcntl, json, os, resource, signal, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--phase',choices=['A','probe','gate','study3','goose'],required=True)
p.add_argument('--gpu',action='store_true')
p.add_argument('--forecast-seconds',type=float,required=True,help='Conservative prediction INCLUDING 25% margin')
p.add_argument('--forecast-source',required=True)
p.add_argument('--cores',type=int,default=4)
p.add_argument('--recovery-of')
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
assert a.forecast_seconds>0 and 1<=a.cores<=8
records=ROOT/'resources';path=records/f'job_{a.name}.json'
assert not path.exists(), 'Never overwrite an executed job receipt'
caps={'probe':2,'gate':4,'study3':12,'goose':6}
gpu_lock=None
if a.gpu:
    assert a.phase in caps
    gpu_lock=(records/'gpu.lock').open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
done=[json.loads(f.read_text()) for f in records.glob('job_*.json')]
used=sum(j.get('gpu_device_hours',0) for j in done)
phase_used=sum(j.get('gpu_device_hours',0) for j in done if j['phase']==a.phase)
limit=caps.get(a.phase,0)*3600-phase_used*3600
if a.gpu:
    if a.recovery_of:
        old=json.loads((records/f'job_{a.recovery_of}.json').read_text())
        assert old['returncode']!=0 and old['phase']==a.phase
        limit=min(36*3600-used*3600,12*3600-sum(j.get('gpu_device_hours',0)*3600 for j in done if j.get('recovery_of')))
    else:
        planned_used=sum(j.get('gpu_device_hours',0) for j in done if not j.get('recovery_of'))
        limit=min(limit,24*3600-planned_used*3600)
    assert a.forecast_seconds<=limit, ('FORECAST_EXCEEDS_BUDGET',a.forecast_seconds,limit)
start=time.time();state=json.loads((ROOT/'research_state.json').read_text())
# Conservative deadline: five days after midnight of the start date, earlier than five elapsed days.
deadline=datetime.datetime(2026,10,23,tzinfo=datetime.timezone.utc).timestamp()
timeout=min(deadline-start,a.forecast_seconds if not a.gpu else limit)
assert timeout>0
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0' if a.gpu else '')
cpus=sorted(os.sched_getaffinity(0))[:a.cores]
def constrain(): os.sched_setaffinity(0,cpus)
before=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt=dict(name=a.name,phase=a.phase,command=cmd,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),start_unix=start,
    forecast_seconds_with_margin=a.forecast_seconds,forecast_margin=1.25,forecast_source=a.forecast_source,cores=a.cores,cpu_affinity=cpus,gpu=a.gpu,recovery_of=a.recovery_of)
(records/f'running_{a.name}.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (records/'jobs.jsonl').open('a') as f:f.write(json.dumps(dict(event='launch',**receipt))+'\n')
code=None;error=None;proc=None
try:
    with (ROOT/'logs'/f'{a.name}.log').open('w') as log:
        proc=subprocess.Popen(cmd,env=env,preexec_fn=constrain,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
        code=proc.wait(timeout=timeout)
except BaseException as exc:
    error=repr(exc)
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    code=124 if isinstance(exc,subprocess.TimeoutExpired) else 1
finally:
    after=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.time()-start
    receipt.update(returncode=code,error=error,wall_seconds=wall,gpu_device_hours=wall/3600 if a.gpu else 0,
        cpu_core_hours=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)/3600,max_rss_kib=after.ru_maxrss,
        finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    with (records/'jobs.jsonl').open('a') as f:f.write(json.dumps(dict(event='finish',**receipt))+'\n')
    (records/f'running_{a.name}.json').unlink(missing_ok=True)
print(json.dumps(receipt,indent=2),flush=True)
raise SystemExit(code)
