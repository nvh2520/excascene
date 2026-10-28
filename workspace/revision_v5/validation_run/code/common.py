from pathlib import Path
import csv,datetime,hashlib,json,os,sys,time,fcntl
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'revision_v5/validation_run'
sys.path.insert(0,str(ROOT))
for name in ['manifests','logs','legacy','reports','data','predictions','metrics','precision','conditional','original_prompt1']:
    (RUN/name).mkdir(parents=True,exist_ok=True)
CAPS=dict(forwards=120000,precision_forwards=2048,gpu_seconds=24*3600,cpu_seconds=96*3600,
          active_wall_seconds=72*3600,elapsed_seconds=7*86400,storage_bytes=50_000_000_000,records=5500)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()
def js(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(p)
def csvwrite(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with p.open('w',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def event(kind,**kw):
    with (RUN/'ledger.jsonl').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(dict(time=utc(),kind=kind,**kw),default=str)+'\n');f.flush()
def storage():return sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
def reserve_forwards(n,phase):
    with (RUN/'forward_counter.json').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);raw=f.read();d=json.loads(raw) if raw else dict(total=0,precision=0,by_phase={})
        assert d['total']+n<=CAPS['forwards'],'total forward cap'
        if phase=='precision':assert d['precision']+n<=CAPS['precision_forwards'],'precision cap';d['precision']+=n
        d['total']+=n;d['by_phase'][phase]=d['by_phase'].get(phase,0)+n
        f.seek(0);f.truncate();json.dump(d,f);f.flush()
def locked():
    m=json.loads((RUN/'manifests/artifact_lock.json').read_text())
    for rel,digest in m['files'].items():assert sha(RUN/rel)==digest,('lock changed',rel)
    return m
def cp_path(tag,seed,ckpt):
    folder='runs_stage5' if tag=='S5A_G3_structured' else 'runs_stage2' if tag=='S2E_G6_lidar' else 'runs'
    return ROOT/f'{folder}/{tag}/seed{seed}/{ckpt}.pt'
def cache_path(tag,seed,suite,scene,frame):
    code='R2' if tag=='R2_full' else 'G3'
    if suite in ['iid','all_s2','ood_terrain']:return ROOT/f'results_stage3/A/cache/S3A_{code}_s{seed}/{suite}/{scene}_{frame}.npz'
    return ROOT/f'results_stage4/B/cache/S4B_{code}_cache_s{seed}/{suite}/{scene}_{frame}.npz'
def npzwrite(path,**arrays):
    # Lossless low-level deflate, preserving float32 outputs rather than quantizing prediction caches.
    import zipfile,io
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path.with_suffix('.tmp'), 'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for k,a in arrays.items():
            b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(k+'.npy',b.getvalue())
    path.with_suffix('.tmp').replace(path)
