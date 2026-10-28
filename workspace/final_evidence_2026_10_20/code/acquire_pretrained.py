"""Acquire only explicitly requested encoder weights from official sources."""
from pathlib import Path
import datetime,hashlib,json,shutil,urllib.request
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'pretrained'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def fetch(url,path):
    assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(url,headers={'User-Agent':'ExcaScene-local-research/1.0'})
    with urllib.request.urlopen(req,timeout=120) as r, path.with_suffix(path.suffix+'.part').open('wb') as f:
        status=r.status;final=r.url
        while b:=r.read(2**20):f.write(b)
    path.with_suffix(path.suffix+'.part').replace(path)
    return dict(url=url,http_status=status,final_url=final,bytes=path.stat().st_size,sha256=sha(path),path=str(path.relative_to(ROOT)))
receipt=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Only ResNet18/SwinT/ImageNetMiTB0 weights for authorized final round; no task-trained weight acquisition or distribution',assets={},errors={})
cache=Path('/home/choilee-1/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth');assert sha(cache).startswith('f37072fd')
shutil.copy2(cache,OUT/'resnet18_imagenet1k_v1.pth')
receipt['assets']['resnet18']=dict(path='pretrained/resnet18_imagenet1k_v1.pth',sha256=sha(cache),bytes=cache.stat().st_size,source='https://download.pytorch.org/models/resnet18-f37072fd.pth',availability='Existing local cache copied; official filename SHA256 prefix verified now; historical download event not asserted',official_mapping='https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html')
try:receipt['assets']['swin_t']=fetch('https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',OUT/'swin_t_imagenet1k.pth')
except Exception as e:receipt['errors']['swin_t']=repr(e)
try:
    info=fetch('https://huggingface.co/api/models/nvidia/mit-b0',OUT/'mit_b0_repository.json')
    metadata=json.loads((OUT/'mit_b0_repository.json').read_text());rev=metadata['sha']
    receipt['mit_b0_repository_revision']=rev
    base=f'https://huggingface.co/nvidia/mit-b0/resolve/{rev}/'
    receipt['assets']['mit_b0']=fetch(base+'pytorch_model.bin',OUT/'mit_b0_imagenet1k.bin')
    receipt['assets']['mit_b0_config']=fetch(base+'config.json',OUT/'mit_b0_config.json')
    receipt['assets']['mit_b0_card']=fetch(base+'README.md',OUT/'mit_b0_README.md')
except Exception as e:receipt['errors']['mit_b0']=repr(e)
for key,url in [('swin_notice','https://raw.githubusercontent.com/microsoft/Swin-Transformer/main/LICENSE'),('mit_notice','https://raw.githubusercontent.com/NVlabs/SegFormer/master/LICENSE')]:
    try:receipt['assets'][key]=fetch(url,OUT/(key+'.txt'))
    except Exception as e:receipt['errors'][key]=repr(e)
(OUT/'acquisition_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(assets={k:{q:v[q] for q in ['path','bytes','sha256'] if q in v} for k,v in receipt['assets'].items()},errors=receipt['errors']),indent=2))
