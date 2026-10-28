"""Import only the pinned segmentation graph, without unrelated detector extensions.

Network modules and native operators remain the original implementations. Registries
are MMCV registries; package shims avoid eager imports of unused detection models.
PatchEmbed/PatchMerging and helpers are compiled from unchanged upstream AST nodes.
"""
import ast, importlib.util, logging, pathlib, sys, types
from mmcv.utils import Registry, get_logger
ROOT=pathlib.Path(__file__).resolve().parents[1]
SRC=ROOT/'source/bevfusion_adapted'
DET=ROOT/'environment/venv/lib/python3.11/site-packages/mmdet'

def package(name,path=None):
    if name in sys.modules: return sys.modules[name]
    m=types.ModuleType(name); m.__path__=[] if path is None else [str(path)]
    sys.modules[name]=m
    if '.' in name:
        parent,leaf=name.rsplit('.',1)
        if parent in sys.modules: setattr(sys.modules[parent],leaf,m)
    return m

def load(name,path):
    if name in sys.modules: return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path,
        submodule_search_locations=[str(path.parent)] if path.name=='__init__.py' else None)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m)
    if '.' in name:
        parent,leaf=name.rsplit('.',1); setattr(sys.modules[parent],leaf,m)
    return m

def initialize():
    package('mmdet',DET)
    models=package('mmdet.models',DET/'models')
    builder=package('mmdet.models.builder')
    for n in ['BACKBONES','NECKS']:
        reg=Registry('afast_'+n.lower()); setattr(builder,n,reg); setattr(models,n,reg)
    package('mmdet.utils').get_root_logger=lambda **kw:get_logger('mmdet',**kw)
    utils=package('mmdet.models.utils',DET/'models/utils')
    tr=package('mmdet.models.utils.transformer')
    path=DET/'models/utils/transformer.py'; tree=ast.parse(path.read_text())
    keep={'AdaptivePadding','PatchEmbed','PatchMerging','nlc_to_nchw','nchw_to_nlc'}
    # Imports needed by these exact original nodes, excluding unrelated Deformable DETR.
    imports=ast.parse('import math\nimport warnings\nfrom typing import Sequence\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom mmcv.cnn import build_conv_layer, build_norm_layer\nfrom mmcv.runner import BaseModule\nfrom mmcv.utils import to_2tuple\n').body
    selected=ast.Module(body=imports+[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in keep],type_ignores=[])
    exec(compile(ast.fix_missing_locations(selected),str(path),'exec'),tr.__dict__)
    load('mmdet.models.utils.ckpt_convert',DET/'models/utils/ckpt_convert.py')
    utils.ResLayer=load('mmdet.models.utils.res_layer',DET/'models/utils/res_layer.py').ResLayer
    package('mmdet.models.backbones',DET/'models/backbones')
    load('mmdet.models.backbones.resnet',DET/'models/backbones/resnet.py')
    swin=load('mmdet.models.backbones.swin',DET/'models/backbones/swin.py').SwinTransformer
    package('mmdet3d',SRC/'mmdet3d')
    ops=package('mmdet3d.ops',SRC/'mmdet3d/ops')
    ops.spconv=load('mmdet3d.ops.spconv',SRC/'mmdet3d/ops/spconv/__init__.py')
    ops.bev_pool=load('mmdet3d.ops.bev_pool',SRC/'mmdet3d/ops/bev_pool/__init__.py').bev_pool
    sparseblock=load('mmdet3d.ops.sparse_block',SRC/'mmdet3d/ops/sparse_block.py')
    ops.SparseBasicBlock=sparseblock.SparseBasicBlock
    ops.make_sparse_convmodule=sparseblock.make_sparse_convmodule
    package('mmdet3d.ops.voxel',SRC/'mmdet3d/ops/voxel')
    voxel=load('mmdet3d.ops.voxel.voxelize',SRC/'mmdet3d/ops/voxel/voxelize.py').Voxelization
    package('mmdet3d.models',SRC/'mmdet3d/models')
    b=package('mmdet3d.models.builder')
    for n in ['VTRANSFORMS','FUSERS']:setattr(b,n,Registry('afast_'+n.lower()))
    for p in ['backbones','necks','vtransforms','fusers']:
        package('mmdet3d.models.'+p,SRC/'mmdet3d/models'/p)
    sparse=load('mmdet3d.models.backbones.sparse_encoder',SRC/'mmdet3d/models/backbones/sparse_encoder.py').SparseEncoder
    second=load('mmdet3d.models.backbones.second',SRC/'mmdet3d/models/backbones/second.py').SECOND
    fpn=load('mmdet3d.models.necks.second',SRC/'mmdet3d/models/necks/second.py').SECONDFPN
    imagefpn=load('mmdet3d.models.necks.generalized_lss',SRC/'mmdet3d/models/necks/generalized_lss.py').GeneralizedLSSFPN
    load('mmdet3d.models.vtransforms.base',SRC/'mmdet3d/models/vtransforms/base.py')
    lss=load('mmdet3d.models.vtransforms.lss',SRC/'mmdet3d/models/vtransforms/lss.py').LSSTransform
    fuser=load('mmdet3d.models.fusers.conv',SRC/'mmdet3d/models/fusers/conv.py').ConvFuser
    return swin,voxel,sparse,second,fpn,imagefpn,lss,fuser,ops.bev_pool
