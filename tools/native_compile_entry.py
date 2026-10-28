#!/usr/bin/env python3
"""Quote linker search paths when the checkout contains spaces.

PyTorch 2.8's JIT builder quotes include/source paths but not generated -L paths.
Only shell quoting is adapted; compiler options, source and numerical code stay
unchanged. The original build entry point is executed verbatim.
"""
from pathlib import Path
import runpy,shlex,sys
import torch.utils.cpp_extension as ce
original=ce._prepare_ldflags

def quote_library_paths(*args,**kwargs):
 return ['-L'+shlex.quote(x[2:]) if x.startswith('-L') else x for x in original(*args,**kwargs)]
ce._prepare_ldflags=quote_library_paths
source=Path(__file__).resolve().parents[1]/'workspace/external_validation_run_EV1_AFAST/code/build_native.py'
sys.argv=[str(source)]+sys.argv[1:]
runpy.run_path(str(source),run_name='__main__')
