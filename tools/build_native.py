#!/usr/bin/env python3
"""Compile the unchanged original three-operator build entry point and record hashes."""
from pathlib import Path
import os,sys,subprocess,hashlib,json,time,datetime,importlib.metadata as m
REPO=Path(__file__).resolve().parents[1];A=REPO/'workspace/external_validation_run_EV1_AFAST'
out=REPO/'native_build_receipt.json';assert not out.exists()
ops=A/'source/bevfusion_adapted/mmdet3d/ops'
assert not list(ops.rglob('*.so')),'Fresh compilation requires no pre-existing operators'
start=time.time();env=os.environ.copy();env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','');env.update(CUDA_HOME='/usr/local/cuda-12.9',TORCH_CUDA_ARCH_LIST='8.0',MAX_JOBS='4',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run([sys.executable,'-B',str(REPO/'tools/native_compile_entry.py')],env=env)
bins=[dict(path=p.relative_to(REPO).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(ops.rglob('*.so'))]
receipt=dict(status='PASS' if r.returncode==0 and len(bins)==3 else 'FAILED',fresh_compilation=True,exit_code=r.returncode,binaries=bins,wall_seconds=time.time()-start,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,packages={n:m.version(n) for n in ['torch','torchvision','numpy','mmcv','ninja']},cuda_toolkit=subprocess.check_output(['/usr/local/cuda-12.9/bin/nvcc','--version'],text=True),compiler=subprocess.check_output(['g++','--version'],text=True))
out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));sys.exit(0 if receipt['status']=='PASS' else 1)
