"""Render every candidate PDF page; no scientific computation."""
from pathlib import Path
import hashlib,json,subprocess
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'submission_strengthening/renders'
OUT.mkdir(exist_ok=True)
records=[]
for name in ('main_submission_candidate','appendix_submission_candidate'):
    pdf=ROOT/'paper_submission_candidate'/(name+'.pdf')
    folder=OUT/name;folder.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm','-r','105','-png',str(pdf),str(folder/'page')],check=True)
    subprocess.run(['pdftotext','-layout',str(pdf),str(folder/'text.txt')],check=True)
    pages=sorted(folder.glob('page-*.png'));sheets=[]
    for i in range(0,len(pages),2):
        images=[Image.open(p).convert('RGB') for p in pages[i:i+2]]
        width=max(im.width for im in images);height=max(im.height for im in images)
        sheet=Image.new('RGB',(2*width,height+35),'#dddddd');draw=ImageDraw.Draw(sheet)
        for j,im in enumerate(images):
            sheet.paste(im,(width*j,35));draw.text((width*j+10,8),f'{name}: page {i+j+1}',fill='black')
        path=folder/f'sheet_{i+1:02d}_{min(i+2,len(pages)):02d}.png';sheet.save(path)
        sheets.append(str(path.relative_to(ROOT)))
    records.append({'pdf':str(pdf.relative_to(ROOT)),'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':len(pages),'dpi':105,'sheets':sheets,'scope':'all pages including every changed or repaginated page'})
(ROOT/'submission_strengthening/records/render_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='sheets'} for r in records],indent=2))
