"""Independent arithmetic checks of real pinned CUDA forward/backward operators."""
import importlib.util, json, pathlib, sys, time, types
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
SRC=ROOT/'source/bevfusion_adapted'
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
torch.manual_seed(40)
for name,path in [('mmdet3d',SRC/'mmdet3d'),('mmdet3d.ops',SRC/'mmdet3d/ops')]:
    mod=types.ModuleType(name); mod.__path__=[str(path)]; sys.modules[name]=mod
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path,
        submodule_search_locations=[str(path.parent)] if path.name=='__init__.py' else None)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
pool=load('mmdet3d.ops.bev_pool',SRC/'mmdet3d/ops/bev_pool/__init__.py').bev_pool
sp=load('mmdet3d.ops.spconv',SRC/'mmdet3d/ops/spconv/__init__.py')
records=[]
x=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],device='cuda',requires_grad=True)
c=torch.tensor([[0,1,0,0],[0,1,0,0],[1,2,0,1]],device='cuda',dtype=torch.int32)
y=pool(x,c,2,1,3,4)
expected=torch.zeros_like(y); expected[0,:,0,0,1]=torch.tensor([4.,6.],device='cuda'); expected[1,:,0,1,2]=torch.tensor([5.,6.],device='cuda')
torch.testing.assert_close(y,expected,atol=0,rtol=0)
y.square().sum().backward()
torch.testing.assert_close(x.grad,torch.tensor([[8.,12.],[8.,12.],[10.,12.]],device='cuda'),atol=0,rtol=0)
records.append({'test':'native_bev_pool_forward_backward','passed':True,'dtype':str(y.dtype),'sample_forwards':2})
for kind in ['subm','stride']:
    coords=torch.tensor([[0,1,2,3],[0,2,2,3],[1,3,4,5]],dtype=torch.int32,device='cuda')
    feats=torch.randn(3,2,device='cuda',requires_grad=True)
    conv=(sp.SubMConv3d(2,3,3,padding=1,bias=False) if kind=='subm' else
          sp.SparseConv3d(2,3,3,padding=1,stride=2,bias=False)).cuda().float()
    sparse=conv(sp.SparseConvTensor(feats,coords,[5,6,7],2))
    actual=sparse.dense()
    f2=feats.detach().clone().requires_grad_(); w2=conv.weight.detach().clone().requires_grad_()
    dense=torch.zeros(2,2,5,6,7,device='cuda')
    dense[coords[:,0],:,coords[:,1],coords[:,2],coords[:,3]]=f2
    expected=torch.nn.functional.conv3d(dense,w2.permute(4,3,0,1,2),padding=1,stride=1 if kind=='subm' else 2)
    if kind=='subm':
        mask=torch.zeros_like(expected); mask[coords[:,0],:,coords[:,1],coords[:,2],coords[:,3]]=1
        expected=expected*mask
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    actual.square().sum().backward(); expected.square().sum().backward()
    torch.testing.assert_close(feats.grad,f2.grad,atol=3e-5,rtol=3e-5)
    torch.testing.assert_close(conv.weight.grad,w2.grad,atol=3e-5,rtol=3e-5)
    records.append({'test':'native_sparse_'+kind+'_forward_backward_vs_dense','passed':True,
                    'shape':list(actual.shape),'sample_forwards':2,'max_error':float((actual-expected).abs().max())})
torch.cuda.synchronize()
receipt={'status':'PASSED','torch':torch.__version__,'cuda':torch.version.cuda,
         'device':torch.cuda.get_device_name(),'fp32':True,'tf32':False,'tests':records,
         'non_training_forwards':sum(r['sample_forwards'] for r in records), 'utc_unix':time.time()}
(ROOT/'tests/native_operator_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
