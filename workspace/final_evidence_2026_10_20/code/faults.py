"""Fixed novel image-fault mechanisms; simulation designs, not physical calibration."""
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates

def novel_latents(shape,kind,rng):
    h,w=shape
    if kind=='mud_water':
        # Opaque irregular spots plus translucent refracting droplets, all geometry
        # drawn before touching the image. Same arrays can be reused across arms.
        return dict(kind=kind,spots=np.column_stack([rng.uniform(0,w,12),rng.uniform(0,h,12),rng.uniform(.035,.10,12)*min(h,w),rng.uniform(.5,1.,12)]),
            droplets=np.column_stack([rng.uniform(0,w,20),rng.uniform(0,h,20),rng.uniform(.018,.055,20)*min(h,w)]),
            texture=rng.normal(0,8,(h,w,3)).astype(np.float32))
    if kind=='sun_glare':
        return dict(kind=kind,center=rng.uniform([.15*w,.05*h],[.85*w,.45*h]),sigma=float(rng.uniform(.12,.22)*w),
            gain=float(rng.uniform(1.4,1.8)),strength=float(rng.uniform(130,210)))
    raise ValueError(kind)

def apply_novel(rgb,depth,latent):
    h,w=depth.shape;yy,xx=np.mgrid[:h,:w];img=rgb.astype(np.float32);d=depth.astype(np.float32).copy()
    if latent['kind']=='mud_water':
        mud=np.zeros((h,w),bool)
        for x,y,r,aspect in latent['spots']:mud|=((xx-x)/r)**2+((yy-y)/(r*aspect))**2<=1
        blob=gaussian_filter(latent['texture'],(2,2,0))+np.array([63,48,29],np.float32)
        img[mud]=blob[mud];d[mud]=0
        sx=xx.astype(float);sy=yy.astype(float);droplet=np.zeros((h,w),bool)
        for x,y,r in latent['droplets']:
            rr=((xx-x)**2+(yy-y)**2)/r**2;inside=rr<1
            factor=.18*(1-rr[inside]);sx[inside]+=(xx[inside]-x)*factor;sy[inside]+=(yy[inside]-y)*factor;droplet|=inside
        warped=np.stack([map_coordinates(img[...,c],[sy,sx],order=1,mode='nearest') for c in range(3)],-1)
        img[droplet]=.85*warped[droplet]+.15*255
        d[droplet]=0
    elif latent['kind']=='sun_glare':
        x,y=latent['center'];s=latent['sigma'];bloom=np.exp(-((xx-x)**2+(yy-y)**2)/(2*s*s))
        img=img*latent['gain']+bloom[...,None]*latent['strength']
        saturated=(img>=250).all(-1);d[saturated]=0
    else:raise ValueError(latent['kind'])
    return np.clip(img,0,255).astype(np.uint8),d.astype(np.float32)

def novel_camera(rgb,depth,kind,rng):
    return apply_novel(rgb,depth,novel_latents(depth.shape,kind,rng))
