"""Render the two completed PDFs for the requested page inspection."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'AFAST_reporting_closure/renders'
OUT.mkdir(exist_ok=True)
names=sys.argv[1:] or ['main_v5_2_review','appendix_v5_2_review']
manifest=ROOT/'AFAST_reporting_closure/records/render_manifest.json'
records=[r for r in json.loads(manifest.read_text()) if Path(r['pdf']).stem not in names] if manifest.exists() else []
for name in names:
    pdf=ROOT/'paper_v5_2'/f'{name}.pdf'
    directory=OUT/name
    directory.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm','-r','105','-png',str(pdf),str(directory/'page')],check=True)
    subprocess.run(['pdftotext','-layout',str(pdf),str(directory/'text.txt')],check=True)
    pages=sorted(directory.glob('page-*.png'))
    sheets=[]
    for i in range(0,len(pages),2):
        pair=[Image.open(p).convert('RGB') for p in pages[i:i+2]]
        width=max(im.width for im in pair);height=max(im.height for im in pair)
        sheet=Image.new('RGB',(width*2,height+35),'#d4d4d4')
        d=ImageDraw.Draw(sheet)
        for j,im in enumerate(pair):
            sheet.paste(im,(j*width,35))
            d.text((j*width+10,10),f'{name} page {i+j+1}',fill='black')
        target=directory/f'sheet_{i+1:02d}_{min(i+2,len(pages)):02d}.png'
        sheet.save(target)
        sheets.append(str(target.relative_to(ROOT)))
    records.append(dict(pdf=str(pdf.relative_to(ROOT)),sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                        pages=len(pages),dpi=105,sheets=sheets,inspection_scope='All pages, covering all changed and repaginated pages'))
manifest.write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='sheets'} for r in records],indent=2))
