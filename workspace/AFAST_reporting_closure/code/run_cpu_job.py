"""Bounded reporting-only command runner. No GPU access is requested."""
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--cores', type=int, default=1)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
assert 1 <= a.cores <= 8
cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
config = json.loads((ROOT/'resources/closure_start.json').read_text())
start = time.time()
remaining = config['deadline_unix'] - start
assert remaining > 0, 'Reporting elapsed-time cap reached'
prior = list((ROOT/'resources').glob('job_*.json'))
cpu_used = sum(json.loads(f.read_text()).get('cpu_core_hours', 0) for f in prior)
assert cpu_used + config['unmetered_command_cpu_allowance_core_hours'] < 24
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
# All reporting workers share the same eight-core affinity ceiling.
cpus = sorted(os.sched_getaffinity(0))[:8]
def constrain():
    os.sched_setaffinity(0, cpus[:a.cores])
    resource.setrlimit(resource.RLIMIT_CPU, (int(min(remaining*a.cores, (23-cpu_used)*3600)),)*2)
before = resource.getrusage(resource.RUSAGE_CHILDREN)
code, error = None, None
try:
    code = subprocess.run(cmd, env=env, preexec_fn=constrain, timeout=remaining).returncode
except BaseException as exc:
    error = repr(exc)
finally:
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    end = time.time()
    record = dict(name=a.name, command=cmd, start_unix=start, end_unix=end,
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  wall_seconds=end-start, cpu_core_hours=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)/3600,
                  max_rss_kib=after.ru_maxrss, cores=a.cores, affinity=cpus[:a.cores],
                  gpu_requested=False, returncode=code, error=error)
    (ROOT/'resources'/f'job_{a.name}.json').write_text(json.dumps(record, indent=2)+'\n')
if error:
    raise RuntimeError(error)
raise SystemExit(code)
