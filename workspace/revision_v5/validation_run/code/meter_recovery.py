"""Attach resource monitoring to the unchanged, briefly paused inference process.

The original locked wrapper is preserved. It failed while statting a renamed
temporary file; the scientific child survived and is not relaunched.
"""
from common import *
import psutil,signal,datetime

def stable_storage():
    size=0;vanished=0
    for p in RUN.rglob('*'):
        try:
            if p.is_file():size+=p.stat().st_size
        except FileNotFoundError:vanished+=1
    return size,vanished

def main():
    locked();receipt=json.loads((RUN/'monitor_recovery_process.json').read_text())
    proc=psutil.Process(receipt['pid']);assert proc.create_time()==receipt['create_time']
    assert proc.cmdline()==receipt['cmdline'] and proc.status()==psutil.STATUS_STOPPED
    rows=[json.loads(x) for x in (RUN/'ledger.jsonl').read_text().splitlines()]
    start=next(x for x in rows if x['kind']=='job_start' and x.get('job')=='fresh')
    start_epoch=datetime.datetime.fromisoformat(start['time']).timestamp()
    done=[x for x in rows if x['kind']=='job_end']
    spent_cpu=sum(x['cpu_seconds'] for x in done);spent_gpu=sum(x['gpu_reserved_seconds'] for x in done);spent_wall=sum(x['wall_seconds'] for x in done)
    cpu_times=proc.cpu_times();last_cpu=cpu_times.user+cpu_times.system;peak=proc.memory_info().rss
    used,vanished=stable_storage();elapsed=time.time()-start_epoch
    assert spent_cpu+last_cpu<CAPS['cpu_seconds'] and spent_gpu+elapsed<CAPS['gpu_seconds'] and used+1_000_000_000<CAPS['storage_bytes']
    event('resource_monitor_recovery',original_wrapper_returncode=1,reason='FileNotFoundError between is_file and stat during atomic .tmp rename',
       child_pid=proc.pid,child_cpu_seconds_observed=last_cpu,original_job_elapsed_seconds=elapsed,storage_bytes=used,
       resumed_same_process=True,new_forwards=0,inference_code_unchanged=True,
       accounting='Full child cumulative CPU and entire original GPU reservation retained, including the monitoring gap and pause. Child OS exit code may be unavailable after reparenting; scientific completion is separately verified.')
    os.kill(proc.pid,signal.SIGCONT)
    last_sample=time.monotonic();last_scan=0;reason=None;last_seen=last_sample
    while True:
        try:
            if not proc.is_running() or proc.status()==psutil.STATUS_ZOMBIE:break
            t=proc.cpu_times();last_cpu=t.user+t.system;last_seen=time.monotonic();peak=max(peak,proc.memory_info().rss)
        except psutil.NoSuchProcess:break
        elapsed=time.time()-start_epoch
        if spent_cpu+last_cpu>CAPS['cpu_seconds']:reason='CPU cap'
        if spent_gpu+elapsed>CAPS['gpu_seconds']:reason='GPU reservation cap'
        if spent_wall+elapsed>CAPS['active_wall_seconds']:reason='active wall cap'
        if time.time()-json.loads((RUN/'manifests/artifact_lock.json').read_text())['unix_time']>CAPS['elapsed_seconds']:reason='elapsed cap'
        if time.monotonic()-last_scan>30:
            last_scan=time.monotonic();used,n=stable_storage();vanished+=n
            js(RUN/'live_usage_recovered.json',dict(job='fresh',elapsed_seconds=elapsed,live_cpu_seconds=last_cpu,storage_bytes=used,peak_rss=peak,vanished_paths_during_snapshot=vanished))
            # 1 GB reserve bounds live-directory snapshot drift far above observed throughput.
            if used+1_000_000_000>CAPS['storage_bytes']:reason='storage reserve'
        if reason:
            os.kill(proc.pid,signal.SIGTERM);break
        time.sleep(.5)
    tail=time.monotonic()-last_seen
    # Conservative tail CPU charge using all host logical CPUs, because a reparented
    # process may be reaped before its exact final cumulative CPU is sampled.
    tail_bound=tail*(psutil.cpu_count() or 1)
    proof=RUN/'fresh_status.json';complete=proof.exists() and json.loads(proof.read_text()).get('complete') is True
    count=json.loads((RUN/'forward_counter.json').read_text())
    complete=bool(complete and count['by_phase']['fresh']==58608 and not reason)
    used,n=stable_storage();elapsed=time.time()-start_epoch
    event('job_end',job='fresh',returncode=None,original_wrapper_returncode=1,
       scientific_completion_verified=complete,completion_basis='fresh_status.json and exact forward count; separate result completeness gate follows',
       cpu_seconds=last_cpu+tail_bound,cpu_seconds_observed=last_cpu,cpu_unsampled_tail_upper_bound_seconds=tail_bound,
       cpu_accounting='Observed process lifetime CPU plus conservative final polling-tail bound; OS exit status unavailable for reparented child',
       wall_seconds=elapsed,gpu_reserved_seconds=elapsed,peak_rss=peak,storage_bytes=used,stop_reason=reason,
       vanished_paths_during_snapshot=vanished+n,recovery='Same child; no repeat or substitute forwards')
    print(json.dumps(dict(scientific_completion_verified=complete,cpu_seconds_observed=last_cpu,cpu_tail_bound=tail_bound,wall_seconds=elapsed,stop_reason=reason)),flush=True)
    assert complete,'fresh inference did not furnish completion proof'

if __name__=='__main__':main()
