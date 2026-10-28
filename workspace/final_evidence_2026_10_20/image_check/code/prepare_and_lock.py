"""Declared image-check setup: immutable caches and aspect-preserving real inputs."""
from pathlib import Path
import csv,datetime,hashlib,json,sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
OUT=Path(__file__).resolve().parents[1];NEW=OUT.parent;PROJECT=NEW.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def one(e):
    source=PROJECT/e['rgb'];label=PROJECT/e['label']
    with Image.open(source) as im:
        assert im.size==(2048,1536);rgb=np.array(im.convert('RGB').resize((512,384),Image.Resampling.BILINEAR))
    with Image.open(label) as im:y=np.array(im.resize((512,384),Image.Resampling.NEAREST),dtype=np.uint8)
    target=OUT/'cache/real_width512'/e['split']/(e['id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
    np.savez_compressed(target,rgb=rgb,label=y)
    return dict(id=e['id'],sequence=e['sequence'],source=e['rgb'],source_rgb_sha256=e['source_rgb_sha256'],source_label_sha256=e['source_label_sha256'],shape=[384,512],cache=str(target.relative_to(NEW)),sha256=sha(target))
if __name__=='__main__':
    assert not (OUT/'protocol/image_check_lock.json').exists()
    rows=json.loads((OUT/'manifests/real_cache.json').read_text());selected=[r for r in rows if r['split']=='val'];assert len(selected)==192
    with ThreadPoolExecutor(max_workers=4) as pool:made=list(pool.map(one,selected))
    (OUT/'manifests/real_width512_cache.json').write_text(json.dumps(made,indent=2)+'\n')
    paths=list((OUT/'code').glob('*.py'))+list((OUT/'manifests').glob('*.json'))+[OUT/'protocol/shared_sequence_draws.npz',NEW/'pretrained/mit_b0_imagenet1k.bin',NEW/'pretrained/mit_b0_config.json',NEW/'code/faults.py']
    record=dict(status='LOCKED_BEFORE_IMAGE_CHECK_FITTING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),epochs=24,seed=2026102003,training='clean1200sim images and2164official real training images;256x160; exact existing split/caches; new ImageNetMiTB0encoder, random decoder and task head',evaluation='15fixed conditions; sim300images parity;real192images parity and512x384aspect-preserving; same real checkpoint for both resolutions',precision='FP32;TF32/autocast off',bootstrap='reuse20,000 saved scene/sequence multiplicities; same independent sim/real draws; real draws shared across resolutions',frozen_original_gate_transfer='unchanged; no gate evaluation or recalibration in this upgrade',novel_module=str(NEW/'code/faults.py'),real_images_redistributed=False,license='NVIDIA pretraining source terms and official GOOSEEx terms retained; local research use only; no new license grant',files={str(p.relative_to(NEW)):sha(p) for p in paths})
    (OUT/'protocol/image_check_lock.json').write_text(json.dumps(record,indent=2)+'\n');print('Locked image check',len(paths),'artifacts')
