"""Local-only ALICE official-label inventory and reduced-resolution cache.

No downloads of images; never read an annotation source outside the official
GOOSE-Ex labels tree. Cached images are private derived data, not release assets.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, shutil, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
GOOSE = ROOT / 'data_stage2/external/goose_ex'
SIM = ROOT / 'strengthening_2026_10_18/phase_a/material_replay'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def inventory():
    entries=[]
    for split in ('train','val'):
        for p in sorted((GOOSE/'images'/split).rglob('alice*.png')):
            prefix=p.name.rsplit('_camera_',1)[0]
            q=GOOSE/'labels'/split/p.parent.name/(prefix+'_labelids.png')
            if not q.is_file(): raise FileNotFoundError(q)
            seq=re.fullmatch(r'(alice_scenario\d+_sequence\d+)_.*',p.name).group(1)
            entries.append(dict(dataset='real',split=split,id=p.stem,sequence=seq,
                rgb=str(p.relative_to(ROOT)),label=str(q.relative_to(ROOT)),
                rgb_bytes=p.stat().st_size,label_bytes=q.stat().st_size))
    train={e['sequence'] for e in entries if e['split']=='train'}
    val={e['sequence'] for e in entries if e['split']=='val'}
    assert not train & val, 'Sequence collision across official splits'
    assert len(entries)==2356, len(entries)
    (OUT/'manifests').mkdir(exist_ok=True,parents=True)
    p=OUT/'manifests/real_images.json'; p.write_text(json.dumps(entries,indent=2)+'\n')
    meta=GOOSE/'_archive_metadata/gooseEx_2d_train'
    (OUT/'evidence').mkdir(exist_ok=True,parents=True)
    for name in ('LICENSE','goose_label_mapping.csv','CHANGELOG'):
        shutil.copyfile(meta/name, OUT/'evidence'/('GOOSE-Ex_'+name))
    counts={s:dict(images=sum(e['split']==s for e in entries),
                  sequences=len({e['sequence'] for e in entries if e['split']==s})) for s in ('train','val')}
    rec=dict(status='AVAILABLE_LOCAL',platform='ALICE only',counts=counts,
        official_labels_only=True,sequence_overlap=[],manifest_sha256=sha(p),
        license_source=str((meta/'LICENSE').relative_to(ROOT)),license_sha256=sha(meta/'LICENSE'),
        mapping_sha256=sha(meta/'goose_label_mapping.csv'),images_redistributable_in_this_package=False,
        no_asset_acquisition=True)
    (OUT/'manifests/local_asset_inventory.json').write_text(json.dumps(rec,indent=2)+'\n')
    return rec

def cache_real(e):
    target=OUT/'cache/real'/e['split']/(e['id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(ROOT/e['rgb']) as im:
        source_size=im.size
        rgb=np.array(im.convert('RGB').resize((256,160),Image.Resampling.BILINEAR))
    with Image.open(ROOT/e['label']) as im:
        assert im.size==source_size
        label=np.array(im.resize((256,160),Image.Resampling.NEAREST),dtype=np.uint8)
    assert label.ndim==2 and label.max()<64
    np.savez_compressed(target,rgb=rgb,label=label)
    return {**e,'cache':str(target.relative_to(ROOT)),'source_size':source_size,
            'source_rgb_sha256':sha(ROOT/e['rgb']),'source_label_sha256':sha(ROOT/e['label']),
            'cache_sha256':sha(target)}

def cache_sim():
    rows=[]
    for split,start,n in [('train',0,400),('val',3000,100)]:
        for scene in range(start,start+n):
            src=SIM/split/f'f_{scene:06d}_0.npz'
            if not src.is_file(): raise FileNotFoundError(src)
            with np.load(src,allow_pickle=False) as d:
                rgb=d['rgb'];lab=d['img_sem']
                assert rgb.shape==(3,160,256,3) and lab.shape==(3,160,256)
                assert lab.max()<7
                for camera in range(3):
                    id_=f'f_{scene:06d}_0_c{camera}'
                    dst=OUT/'cache/sim'/split/(id_+'.npz');dst.parent.mkdir(exist_ok=True,parents=True)
                    np.savez_compressed(dst,rgb=rgb[camera],label=lab[camera])
                    rows.append(dict(dataset='sim',split=split,id=id_,sequence=f'scene_{scene:06d}',
                        source=str(src.relative_to(ROOT)),camera=camera,cache=str(dst.relative_to(ROOT)),
                        cache_sha256=sha(dst)))
    (OUT/'manifests/sim_cache.json').write_text(json.dumps(rows,indent=2)+'\n')
    return dict(images=len(rows),train_images=1200,val_images=300)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inventory','real-cache','sim-cache'])
    ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();t=time.monotonic()
    if a.action=='inventory': r=inventory()
    elif a.action=='real-cache':
        es=json.loads((OUT/'manifests/real_images.json').read_text())
        with ThreadPoolExecutor(max_workers=a.workers) as ex: rows=list(ex.map(cache_real,es))
        (OUT/'manifests/real_cache.json').write_text(json.dumps(rows,indent=2)+'\n')
        r=dict(images=len(rows),train_images=2164,val_images=192)
    else:r=cache_sim()
    r.update(action=a.action,elapsed_seconds=time.monotonic()-t,utc_unix=time.time())
    (OUT/'resources'/f'prepare_{a.action}.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
