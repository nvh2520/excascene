"""Isolated, persistent wall/job-hour accounting and same-seed single-retry queue."""
from __future__ import annotations
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import statistics
import shutil
import fcntl
from stage4.common import ROOT, ANALYSIS, RESULTS, CAPS, utc, write_json, deviation, sha256, commit


@dataclass
class Job:
    key: str
    command: list[str]
    cwd: Path = ROOT
    sentinel: str | None = None
    archive: list[str] = field(default_factory=list)
    train_dir: str | None = None


def environment(cwd=ROOT):
    return dict(os.environ, PATH=str(ROOT / '.venv/bin') + os.pathsep + os.environ.get('PATH', ''),
                PYTHONPATH=str(cwd), PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='4')


class Runner:
    def __init__(self):
        lock = ROOT / 'logs_stage4/runner.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        self.lock = lock.open('a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = ANALYSIS / 'compute_usage.json'
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = dict(window_start=None, deadline=None, experiment_stop=None, wall_window_hours=None,
                              analysis_reserve_hours=1.75, total_job_hour_cap=8, block_caps=CAPS, parallel=3,
                              jobs=[], actual_order=[], events=[], code_commit=commit())
        self.running = {}
        if any(j['status'] == 'running' for j in self.state['jobs']):
            raise RuntimeError('Ledger has an unfinished job; reconcile its PID and artefacts before resuming')
        self.save()

    def save(self):
        now = datetime.now(timezone.utc)
        sums = {b:0.0 for b in CAPS}
        for j in self.state["jobs"]:
            end=datetime.fromisoformat(j["end"]) if j.get("end") else now
            j["seconds"]=max(0,(end-datetime.fromisoformat(j["start"])).total_seconds())
            sums[j["block"]]+=j["seconds"]/3600
        self.state.update(generated_at=now.isoformat(), job_hours_by_block=sums, total_job_hours=sum(sums.values()))
        experimental=[j['start'] for j in self.state['jobs'] if j['key'].startswith(('train/','eval/'))]
        self.state['first_experiment']=min(experimental) if experimental else None
        if self.state['window_start']:
            self.state['elapsed_wall_hours'] = (now-datetime.fromisoformat(self.state['window_start'])).total_seconds()/3600
            self.state['remaining_wall_hours'] = None
        write_json(self.path, self.state)

    def start_window(self):
        if self.state['window_start'] is None:
            now = datetime.now(timezone.utc)
            self.state.update(window_start=now.isoformat(), deadline=None, experiment_stop=None)
            self.save()

    def available(self, block):
        self.save()
        if self.state['total_job_hours'] >= 8:
            return False
        if self.state['job_hours_by_block'][block] >= CAPS[block]:
            return False
        stop = self.state['deadline'] if block == 'analysis' else self.state['experiment_stop']
        return stop is None or datetime.now(timezone.utc) < datetime.fromisoformat(stop)

    def event(self, kind, **kwargs):
        self.state['events'].append(dict(time=utc(), kind=kind, **kwargs))
        self.save()

    def begin_block(self, block):
        if block not in self.state['actual_order']:
            self.state['actual_order'].append(block)
        self.event('block_start', block=block)

    def validation(self, path):
        records = {}
        if Path(path).exists():
            for line in Path(path).read_text().splitlines():
                try:
                    row = json.loads(line)
                    if 'val' in row:
                        records[row['val']['epoch']] = row['val']
                except json.JSONDecodeError:
                    continue
        return records

    def divergence(self, job):
        if not job.train_dir:
            return None
        path = Path(job.train_dir)
        current = self.validation(path / 'log.jsonl')
        peer_paths=[p for p in path.parent.glob('seed*') if p != path]
        peers = [self.validation(p/'log.jsonl') for p in peer_paths]
        for ep, row in current.items():
            other = [p[ep]['H_mae_vis'] for p in peers if ep in p]
            if len(other) >= 2 and row['H_mae_vis'] > 2*statistics.median(other):
                return dict(epoch=ep, height=row['H_mae_vis'], other_seeds=other)
        return None

    def stop(self, proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def run(self, jobs, block, parallel=None):
        parallel = min(parallel or self.state['parallel'], self.state['parallel'])
        finished = {j['key'] for j in self.state['jobs'] if j['status'] == 'complete'}
        pending = [j for j in jobs if j.key not in finished]
        failures = []
        while pending or self.running:
            while pending and len(self.running) < parallel and self.available(block):
                job = pending.pop(0)
                previous = [j for j in self.state['jobs'] if j['key'] == job.key]
                attempt = len(previous)+1
                if attempt > 2:
                    failures.append(job.key)
                    continue
                if job.sentinel and Path(job.sentinel).exists():
                    raise RuntimeError(f'Unaccounted existing output, refusing reevaluation: {job.sentinel}')
                self.start_window()
                log = ROOT/'logs_stage4'/block/(job.key.replace('/','_')+f'.attempt{attempt}.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open('x')
                record = dict(key=job.key, block=block, command=job.command, cwd=str(job.cwd), start=utc(),
                              end=None, status='running', attempt=attempt, log=str(log.relative_to(ROOT)), code_commit=commit())
                process = subprocess.Popen(job.command, cwd=job.cwd, env=environment(job.cwd), stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                record['pid'] = process.pid
                record['process_start_ticks'] = Path(f'/proc/{process.pid}/stat').read_text().split()[21]
                self.state['jobs'].append(record)
                self.running[process.pid] = (process, job, stream, record)
                self.save()
                print(f'[stage4] START {block} {job.key} attempt={attempt} pid={process.pid}', flush=True)
                time.sleep(2)
            if not self.available(block):
                reason = f'{block} stopped by wall/job-hour cap; remaining jobs: {[j.key for j in pending]}'
                deviation(reason, 'BUDGET')
                failures.extend(j.key for j in pending)
                pending = []
                for proc, job, _, record in self.running.values():
                    record['stop_reason'] = 'budget cap'
                    self.stop(proc)
            for pid, (proc, job, stream, record) in list(self.running.items()):
                reason = self.divergence(job) if proc.poll() is None else None
                if reason:
                    record['stop_reason'] = f'validation divergence: {reason}'
                    self.stop(proc)
                if proc.poll() is None:
                    continue
                stream.close()
                rc = proc.returncode
                success = rc == 0 and not record.get('stop_reason') and (not job.sentinel or Path(job.sentinel).exists())
                record.update(end=utc(), returncode=rc, status='complete' if success else 'failed')
                if success and job.sentinel:
                    record.update(output=job.sentinel, output_sha256=sha256(job.sentinel))
                self.running.pop(pid)
                if not success:
                    log_text = (ROOT / record['log']).read_text()
                    if 'out of memory' in log_text.lower():
                        self.state['parallel'] = parallel = min(parallel, 2)
                        deviation(f'OOM in {job.key}; Stage-4 concurrency reduced to {parallel}; configuration unchanged.', 'RETRY')
                    archive = RESULTS/'_invalid'/datetime.now(timezone.utc).strftime('%Y-%m-%d')/(job.key.replace('/','_')+f'_attempt{record["attempt"]}')
                    archive.mkdir(parents=True, exist_ok=True)
                    for i, p in enumerate(job.archive):
                        p = Path(p)
                        if p.exists():
                            shutil.move(str(p), str(archive/f'{i}_{p.name}'))
                    write_json(archive/'note.json', record)
                    deviation(f'{job.key} attempt {record["attempt"]} failed: rc={rc}, {record.get("stop_reason", "see log")}; archive {archive.relative_to(ROOT)}.', 'RETRY')
                    if record['attempt'] < 2 and self.available(block):
                        pending.append(job)
                    else:
                        failures.append(job.key)
                self.save()
                print(f'[stage4] {record["status"].upper()} {job.key} rc={rc}', flush=True)
            if self.running:
                self.save()
                time.sleep(10)
        self.save()
        return failures
