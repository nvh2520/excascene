"""Declared ExcaScene task adapter around the original BEVFusion segmentation graph."""
import pathlib, sys
import torch
from torch import nn
from torch.nn import functional as F
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/excascene_snapshot'))
from kinafford.models.blocks import Heads
from kinafford.models.variants import GeometricFusionNet
from kinafford.sim.kinematics import lidar_rays,SENSORS
from upstream_loader import initialize

Swin,Voxel,Sparse,SECOND,SECONDFPN,ImageFPN,LSS,Fuser,native_pool=initialize()
NEURAL_FIELDS=('img','depth','cam_present','lidar_range','lidar_valid','q','T_dyn','K','meta')
CFG=dict(sensors=['cab','boom','stick','lidar'],extrinsics='dynamic',fusion='geometric',
         gating=False,hetero=True,aff_head=True,pretrained=False,feat_ch=64)
def neural_inputs(batch):
    return {k:batch[k] for k in NEURAL_FIELDS}

class ExcaLSS(LSS):
    def create_frustum(self):
        h,w=self.feature_size; depths=torch.arange(*self.dbound,dtype=torch.float32)
        # Pixel centres: 4x4 patch embedding followed by 2x2 merge => stride 8,
        # centres 4+8j. H/W are divisible by 8, so no padding at this level.
        yy,xx=torch.meshgrid(4+8*torch.arange(h),4+8*torch.arange(w),indexing='ij')
        return nn.Parameter(torch.stack([xx.expand(len(depths),-1,-1),yy.expand(len(depths),-1,-1),
                         depths[:,None,None].expand(-1,h,w)],-1).float(),requires_grad=False)

    def forward(self,img,points,camera2ego,lidar2ego,lidar2camera,lidar2image,
                camera_intrinsics,camera2lidar,img_aug_matrix,lidar_aug_matrix,img_metas=None,
                *,cam_present):
        self._cam_present=cam_present
        try:
            result=super().forward(img,points,camera2ego,lidar2ego,lidar2camera,lidar2image,
                            camera_intrinsics,camera2lidar,img_aug_matrix,lidar_aug_matrix)
        finally: self._cam_present=None
        # Learned BN offsets in native downsample must not create an absent branch.
        return result*(cam_present.sum(1)>0)[:,None,None,None]

    def get_cam_feats(self,x):
        lifted=super().get_cam_feats(x)
        return lifted*self._cam_present[:,:,None,None,None,None]

    def bev_pool(self,geometry,features):
        B,N,D,H,W,C=features.shape
        coords=torch.floor((geometry-(self.bx-self.dx/2))/self.dx).long().reshape(-1,3)
        values=features.reshape(-1,C)
        batch=torch.arange(B,device=values.device).repeat_interleave(N*D*H*W)
        keep=((coords>=0)&(coords<self.nx)).all(1)
        # Presence filtering before pooling eliminates all removed-view contributions.
        keep=keep&(self._cam_present>0.5)[:,:,None,None,None].expand(B,N,D,H,W).reshape(-1)
        coords=torch.cat([coords,batch[:,None]],1)[keep]
        if len(coords)==0:
            return values.sum()*0+values.new_zeros(B,C,int(self.nx[0]),int(self.nx[1]))
        result=native_pool(values[keep],coords,B,int(self.nx[2]),int(self.nx[0]),int(self.nx[1]))
        return torch.cat(result.unbind(2),1)

class BEVFusionExca(nn.Module):
    def __init__(self):
        super().__init__()
        self.camera_backbone=Swin(in_channels=4,embed_dims=96,depths=[2,2,6,2],
            num_heads=[3,6,12,24],window_size=7,mlp_ratio=4,qkv_bias=True,qk_scale=None,
            drop_rate=0.,attn_drop_rate=0.,drop_path_rate=.3,patch_norm=True,
            out_indices=[1,2,3],with_cp=False,convert_weights=True,pretrained=None,init_cfg=None)
        self.camera_backbone.init_weights()
        self.camera_neck=ImageFPN(in_channels=[192,384,768],out_channels=256,start_level=0,
            num_outs=3,norm_cfg=dict(type='BN2d',requires_grad=True),
            act_cfg=dict(type='ReLU',inplace=True),upsample_cfg=dict(mode='bilinear',align_corners=False))
        self.lift=ExcaLSS(256,80,(160,256),(20,32),(-8.,16.,.125),(-12.,12.,.125),
                          (-8.,8.,16.),(.05,40.05,.5),downsample=2)
        self.voxelize=Voxel(voxel_size=[.03125,.03125,.4],point_cloud_range=[-8,-12,-8,16,12,8],
                           max_num_points=10,max_voxels=(90000,120000),deterministic=True)
        self.sparse_encoder=Sparse(in_channels=3,sparse_shape=[768,768,41],output_channels=128,
            order=('conv','norm','act'),encoder_channels=((16,16,32),(32,32,64),(64,64,128),(128,128)),
            encoder_paddings=((0,0,1),(0,0,1),(0,0,(1,1,0)),(0,0)),block_type='basicblock')
        self.fuser=Fuser([80,256],256)
        norm=dict(type='BN',eps=1e-3,momentum=.01)
        self.decoder_backbone=SECOND(in_channels=256,out_channels=[128,256],layer_nums=[5,5],
            layer_strides=[1,2],norm_cfg=norm,conv_cfg=dict(type='Conv2d',bias=False),pretrained=None,init_cfg=None)
        self.decoder_neck=SECONDFPN(in_channels=[128,256],out_channels=[256,256],upsample_strides=[1,2],
            norm_cfg=norm,upsample_cfg=dict(type='deconv',bias=False),use_conv_for_no_stride=True,init_cfg=None)
        self.condition=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,1024))
        self.heads=Heads(512,True,True)
        self.image_head=nn.Conv2d(256,7,1)
        self.register_buffer('rays',torch.tensor(lidar_rays(SENSORS['lidar']),dtype=torch.float32))
        # Auxiliary targets retain the reference's measured-pixel support 2::4,
        # pixel centres 2.5+4j; sample from the native stride-8 lattice explicitly.
        yy,xx=torch.meshgrid(2.5+4*torch.arange(40),2.5+4*torch.arange(64),indexing='ij')
        grid=torch.stack([2*((xx-4)/8+.5)/32-1,2*((yy-4)/8+.5)/20-1],-1)
        self.register_buffer('aux_grid',grid[None])

    def lidar_points(self,b):
        T=b['T_dyn'][:,3]
        raw=b['lidar_range'][...,None]*self.rays[None]
        upper=torch.einsum('bij,braj->brai',T[:,:3,:3],raw)+T[:,None,None,:3,3]
        valid=(b['lidar_valid']>.5)&(b['lidar_range']>.3)&torch.isfinite(upper).all(-1)
        return [upper[i][valid[i]].contiguous() for i in range(len(T))]

    def lidar_features(self,points):
        B=len(points); feats=[]; coords=[]
        for i,p in enumerate(points):
            if len(p)==0: continue
            v,c,n=self.voxelize(p)
            if len(c)==0:continue
            feats.append(v.sum(1)/n.to(v.dtype)[:,None]);coords.append(F.pad(c,(1,0),value=i))
        if not feats:
            # Real B/order, zero branch, no invented points and no sparse BN on zero rows.
            return points[0].new_zeros(B,256,96,96)
        result=self.sparse_encoder(torch.cat(feats),torch.cat(coords),B)
        assert result.shape==(B,256,96,96),('native_sparse_shape',result.shape)
        # Upstream voxel CUDA stores XYZ, despite the stale zyx docstring.
        return result.transpose(-1,-2).contiguous()

    def forward(self,b):
        assert set(b)==set(NEURAL_FIELDS),('neural_field_allowlist',sorted(set(b)-set(NEURAL_FIELDS)))
        B=b['img'].shape[0]; present=b['cam_present']
        images=(b['img']*present[:,:,None,None,None]).reshape(B*3,4,160,256)
        pyramid=self.camera_backbone(images); feature=self.camera_neck(pyramid)[0]
        assert feature.shape==(B*3,256,20,32),feature.shape
        aux=F.grid_sample(self.image_head(feature),self.aux_grid.expand(B*3,-1,-1,-1),
                          mode='bilinear',padding_mode='border',align_corners=False).view(B,3,7,40,64)
        points=self.lidar_points(b); TU=b['T_dyn'][:,:3]
        eye=torch.eye(4,device=feature.device).expand(B,4,4)
        aug=eye[:,None].expand(B,3,4,4)
        cam=self.lift(feature.view(B,3,256,20,32),points,TU,eye,None,None,b['K'],TU,
                      aug,eye,[{} for _ in range(B)],cam_present=present).transpose(-1,-2).contiguous()
        assert cam.shape==(B,80,96,96),cam.shape
        lid=self.lidar_features(points)
        f=self.decoder_neck(self.decoder_backbone(self.fuser([cam,lid])))[0]
        assert f.shape==(B,512,96,96),f.shape
        gamma,beta=self.condition(b['q']).chunk(2,1)
        f=f*(1+gamma[:,:,None,None])+beta[:,:,None,None]
        out=self.heads(f);out['img_seg']=[(k,aux[:,k]) for k in range(3)]
        return out

def build(architecture):
    def forbidden_download(*args,**kwargs):
        raise RuntimeError('EV1-AFAST forbids network weight retrieval')
    import urllib.request
    urllib.request.urlopen=forbidden_download
    torch.hub.load_state_dict_from_url=forbidden_download
    torch.hub.download_url_to_file=forbidden_download
    import torchvision.models._api
    torchvision.models._api.load_state_dict_from_url=forbidden_download
    class StrictKinAfford(GeometricFusionNet):
        def forward(self,batch):
            assert set(batch)==set(NEURAL_FIELDS),('neural_field_allowlist',sorted(set(batch)-set(NEURAL_FIELDS)))
            # Canonicalize absent content at the interface before the original
            # encoder, as well as its existing pre-pooling presence filter.
            # Already canonical training/validation inputs remain identical.
            masked=dict(batch);presence=batch['cam_present']
            masked['img']=batch['img']*presence[:,:,None,None,None]
            masked['depth']=batch['depth']*presence[:,:,None,None]
            return super().forward(masked)
    model=BEVFusionExca() if architecture=='BF-D' else StrictKinAfford(CFG)
    # Some MMCV constructors assign local Kaiming init_cfg defaults even when
    # passed None. Clear this metadata: initialization already occurred through
    # explicit random PyTorch initializers; no later runner init/load is used.
    for module in model.modules():
        if hasattr(module,'init_cfg'):module.init_cfg=None
    assert not any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters())
    return model
