"""Find extracted-text number/letter joins, separating identifiers from prose defects."""
from pathlib import Path
import re,csv,subprocess,json
O=Path(__file__).resolve().parents[1];P=O/'manuscript'
ALLOW=re.compile(r'^(?:[A-Za-z]*RGB[A-Za-z]*|BEVFusion|Robo3D|RELLIS|ResNet|Swin|MiT|SegFormer|PointNet|NeuroNCAP|WoodScape|ImageNet|3D|2D|BF16|FP32|TF32|RGB|A100|ISO|S\d+|G\d+|S5A|G6|EC3|IAARC|CVPR|ICCV|ECCV|ICRA|FPN|B0|DOI|ISBN|ISSN|BK21|S\-tab)',re.I)
def main():
 out=[]
 for name in ['main_submission','supplement_submission']:
  dst=O/'report'/f'{name}_extracted.txt';subprocess.run(['pdftotext','-layout',str(P/(name+'.pdf')),str(dst)],check=True)
  text=dst.read_text();reference=False
  for page,line in [(i+1,l) for i,s in enumerate(text.split('\f')) for l in s.splitlines()]:
   if re.match(r'^\s*(?:\d+\s+)?References\s*$',line):reference=True
   if re.search(r'https?://|doi:|@|arxiv.org|\bURL\b',line):continue
   for token in re.findall(r'\S*(?:[A-Za-z]\d|\d[A-Za-z])\S*',line):
    # Bib entries, class identifiers and conventional sensor/model names remain explicit.
    clean=token.strip('.,;:()[]')
    allowed_ordinal=bool(re.match(r'^\d+(?:st|nd|rd|th)(?:\b|[-,])',clean))
    allowed_path=bool(re.search(r'[_/]',clean))
    allowed_native=clean in ['sm80','lo2','hi2']
    allowed_code=bool(re.search(r'np\.|rng\.|nn\.|torch\.|self\.|r\.(?:layer|bn)|conv[0-9]|lat[0-9]|z0=|[0-9]e[+-]',clean))
    status='LEGITIMATE_IDENTIFIER_ORDINAL_PATH_OR_REFERENCE' if ALLOW.search(clean) or reference or allowed_ordinal or allowed_path or allowed_native or allowed_code else 'REVIEW'
    out.append(dict(document=name,page=page,token=token,line=line.strip(),status=status))
 with (O/'report/pdf_digit_letter_join_inventory.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print('Candidate joins:')
 seen=set()
 for r in out:
  if r['status']=='REVIEW' and r['token'] not in seen:print(r['document'],r['page'],r['token'],'|',r['line']);seen.add(r['token'])
 print('All candidates',len(out),'review tokens',len(seen))
 (O/'report/pdf_join_check.json').write_text(json.dumps(dict(status='PASS' if not seen else 'REVIEW',candidate_occurrences=len(out),unresolved_prose_tokens=sorted(seen),recognized_exceptions='Conventional identifiers, ordinals, exact artifact paths, native sm80 target, and bibliography DOI/URL fragments.',known_repairs=['comma spacing in material and buffer lists','0.25 m','and b','learning rate 10^-4','Prescribed 42-epoch','epochs 14 and 28','only epoch 42','earlier 14-epoch','recorded 200-row','real width 512']),indent=2)+'\n')
if __name__=='__main__':main()
