"""Isolated generation support; never writes any historical validation directory."""
from pathlib import Path
import csv,datetime,hashlib,json,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT
sys.path.insert(0,str(ROOT/'source/excascene_snapshot'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def array_sha(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()
def js(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(p)
def csvwrite(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with p.open('w',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def npzwrite(path,**arrays):
    import zipfile,io
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path.with_suffix('.tmp'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for k,a in arrays.items():
            b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(k+'.npy',b.getvalue())
    path.with_suffix('.tmp').replace(path)
def locked():
    m=json.loads((ROOT/'manifests/execution_lock.json').read_text())
    assert m['status']=='AUTHORIZED_GATES_PASSED'
    for rel,digest in m['locked_files'].items():assert sha(ROOT/rel)==digest,('lock changed',rel)
    return m
