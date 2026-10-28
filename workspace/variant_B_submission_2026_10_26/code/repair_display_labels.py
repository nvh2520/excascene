"""Display-only vector-label repairs; no observation or numerical changes."""
from pathlib import Path
import hashlib,json,shutil
import fitz
O=Path(__file__).resolve().parents[1]
P=O/'manuscript'
old=O.parent/'final_pre_submission_2026_10_26/manuscript/figures'
rows=[]
for name,a,b in [('framework','at the saved fixture pose','at the example pose'),
                 ('example_inputs','Clean saved','Clean')]:
    source=old/(name+'.svg'); s=source.read_text(); assert s.count(a)==1
    target=P/'figures'/(name+'.svg'); target.write_text(s.replace(a,b))
    with fitz.open(target) as doc:
        pdf=doc.convert_to_pdf()
    (P/'figures'/(name+'.pdf')).write_bytes(pdf)
    rows.append(dict(figure=name,source=str(source.relative_to(O.parent)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),old_label=a,new_label=b,
        output_sha256=hashlib.sha256(pdf).hexdigest()))
for ext in ['pdf','svg']:
    shutil.copy2(P/'figures'/('framework.'+ext),P/('graphical_abstract.'+ext))
for name in ['generated/gate_engineering.tex','tables/real_gate_summary.tex']:
    f=P/name;s=f.read_text().replace('a recorded warm','a measured warm').replace('the recorded dense-operation convention','the dense-operation convention').replace('saved-sequence intervals','sequence-bootstrap intervals');f.write_text(s)
(O/'report/display_label_receipt.json').write_text(json.dumps(dict(scope='vector text only; original SVG shapes and raster pixels retained',figures=rows),indent=2)+'\n')
