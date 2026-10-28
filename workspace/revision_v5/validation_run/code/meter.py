"""Resource-limited job launcher; wall reservation is a conservative GPU-hour charge."""
import os,sys,time,resource,subprocess,json,signal
from common import *
import psutil
name=sys.argv[1];gpu=sys.argv[2]=='gpu';cmd=sys.argv[3:]
assert cmd
start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
done=[]
if (RUN/'ledger.jsonl').exists():
    done=[json.loads(l) for l in (RUN/'ledger.jsonl').read_text().splitlines() if json.loads(l)['kind']=='job_end']
spent_cpu=sum(x['cpu_seconds'] for x in done);spent_gpu=sum(x['gpu_reserved_seconds'] for x in done)
spent_wall=sum(x['wall_seconds'] for x in done)
assert spent_cpu<CAPS['cpu_seconds'] and spent_gpu<CAPS['gpu_seconds'] and spent_wall<CAPS['active_wall_seconds']
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
         NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0' if gpu else '')
event('job_start',job=name,command=cmd,gpu=gpu,pid=os.getpid())
f=(RUN/'logs'/f'{name}.log').open('a')
p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
peak_rss=0;reason=None;last=0
while p.poll() is None:
    time.sleep(1);elapsed=time.monotonic()-start
    try:
        procs=[psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)
        cpu=sum(q.cpu_times().user+q.cpu_times().system for q in procs)
        peak_rss=max(peak_rss,sum(q.memory_info().rss for q in procs))
    except psutil.Error:cpu=0
    if spent_cpu+cpu>CAPS['cpu_seconds']:reason='CPU cap'
    if gpu and spent_gpu+elapsed>CAPS['gpu_seconds']:reason='GPU reservation cap'
    if spent_wall+elapsed>CAPS['active_wall_seconds']:reason='active wall cap'
    lock=RUN/'manifests/artifact_lock.json'
    if lock.exists() and time.time()-json.loads(lock.read_text())['unix_time']>CAPS['elapsed_seconds']:reason='elapsed cap'
    if elapsed-last>60:
        last=elapsed;used=storage()
        js(RUN/'live_usage.json',dict(job=name,elapsed_seconds=elapsed,live_cpu_seconds=cpu,storage_bytes=used,peak_rss=peak_rss))
        if used>CAPS['storage_bytes']:reason='storage cap'
    if reason:
        os.killpg(p.pid,signal.SIGTERM);p.wait();break
rc=p.wait();f.close();after=resource.getrusage(resource.RUSAGE_CHILDREN)
cpu=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime);elapsed=time.monotonic()-start
event('job_end',job=name,returncode=rc,cpu_seconds=cpu,wall_seconds=elapsed,
      gpu_reserved_seconds=elapsed if gpu else 0,peak_rss=peak_rss,storage_bytes=storage(),stop_reason=reason)
print(json.dumps(dict(job=name,returncode=rc,cpu_seconds=cpu,wall_seconds=elapsed,reason=reason)),flush=True)
sys.exit(rc if rc else (1 if reason else 0))
