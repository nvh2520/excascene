"""Compile the pinned native operators, preserving their arithmetic and topology."""
import argparse, json, os, pathlib, shutil, time
import torch
from torch.utils.cpp_extension import load

ROOT = pathlib.Path(__file__).resolve().parents[1]
original = ROOT/'source/bevfusion-db75150717a9462cb60241e36ba28d65f6908607'
adapted = ROOT/'source/bevfusion_adapted'
if not adapted.exists(): shutil.copytree(original, adapted)
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
os.environ['MAX_JOBS'] = '4'
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.9'
ops = adapted/'mmdet3d/ops'
inc = ops/'spconv/include'
parser=argparse.ArgumentParser();parser.add_argument('--only');args=parser.parse_args()
# C++17 is required by the installed PyTorch headers. No algorithm changes.
f = inc/'torch_utils.h'
f.write_text(f.read_text().replace('tensor.type().scalarType()', 'tensor.scalar_type()'))
for name, folder, sources, includes in [
    ('bev_pool_ext', 'bev_pool', ['bev_pool.cpp','bev_pool_cuda.cu'], []),
    ('sparse_conv_ext', 'spconv', ['all.cc','reordering.cc','reordering_cuda.cu',
                                  'indice.cc','indice_cuda.cu','maxpool.cc','maxpool_cuda.cu'], [str(inc)]),
    ('voxel_layer', 'voxel', ['voxelization.cpp','scatter_points_cpu.cpp','scatter_points_cuda.cu',
                            'voxelization_cpu.cpp','voxelization_cuda.cu'], [])]:
    if args.only and name!=args.only:continue
    dest=ops/folder; build=ROOT/'environment/build'/name; build.mkdir(parents=True,exist_ok=True)
    print('BUILD',name,flush=True); start=time.monotonic()
    module=load(name=name, sources=[str(dest/'src'/s) for s in sources],
                extra_include_paths=includes, extra_cflags=['-O2','-std=c++17','-w','-DWITH_CUDA'],
                extra_cuda_cflags=['-O2','-std=c++17','-w','-DWITH_CUDA','-D__CUDA_NO_HALF_OPERATORS__',
                                  '-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__'],
                build_directory=str(build), verbose=True)
    shutil.copy2(module.__file__, dest/pathlib.Path(module.__file__).name)
    print(json.dumps({'operator':name,'build_s':time.monotonic()-start,'binary':module.__file__}),flush=True)
