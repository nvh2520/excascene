#!/usr/bin/env python3
"""Bound and record each launched job; no shell interpolation or remote reporting."""
import argparse, datetime, json, os, pathlib, resource, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--gpu', type=int, default=0, help='number of occupied GPU devices')
p.add_argument('--hours', type=float, default=2)
p.add_argument('--threads', type=int, default=4)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
clock = json.loads((ROOT/'records/start_clock.json').read_text())
start = time.time(); mono = time.monotonic()
elapsed_t0=max(start-clock['t0_unix'],mono-clock['t0_monotonic'])
stage=26 if 'formal_seed' in a.name else 32 if 'holdout_inference' in a.name else 36
if elapsed_t0 >= stage*3600:
    raise SystemExit('Hard deadline reached; refusing launch')
allowed = sorted(os.sched_getaffinity(0))[:8]
os.sched_setaffinity(0, allowed)
env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads),
           MAX_JOBS=str(a.threads), NUMEXPR_NUM_THREADS=str(a.threads), CUBLAS_WORKSPACE_CONFIG=':4096:8', NVIDIA_TF32_OVERRIDE='0')
env['PATH'] = str(ROOT/'environment/venv/bin') + os.pathsep + env['PATH']
log = ROOT/'logs'/f'{a.name}.log'
pre = resource.getrusage(resource.RUSAGE_CHILDREN)
launch = dict(name=a.name, command=cmd, start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              start_unix=start, elapsed_from_t0_hours=(start-clock['t0_unix'])/3600,
              cpu_affinity=allowed, occupied_gpu_devices=a.gpu, log=str(log.relative_to(ROOT)))
with (ROOT/'resources/jobs.jsonl').open('a') as f: f.write(json.dumps(dict(event='launch',**launch))+'\n')
with log.open('w') as f:
    proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = proc.wait(timeout=min(a.hours*3600, stage*3600-elapsed_t0))
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(proc.pid, signal.SIGTERM)
        try: code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); code = proc.wait()
elapsed = time.monotonic()-mono
post = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = dict(event='finish', **launch, returncode=code, elapsed_seconds=elapsed,
               cpu_user_seconds=post.ru_utime-pre.ru_utime,
               cpu_system_seconds=post.ru_stime-pre.ru_stime,
               cpu_core_hours=((post.ru_utime-pre.ru_utime)+(post.ru_stime-pre.ru_stime))/3600,
               occupied_gpu_device_hours=a.gpu*elapsed/3600,
               finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (ROOT/'resources/jobs.jsonl').open('a') as f: f.write(json.dumps(receipt)+'\n')
print(json.dumps(receipt), flush=True)
raise SystemExit(code)
