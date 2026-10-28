"""Targeted preservation/new-claim/build check, not a historical audit."""
from pathlib import Path
import csv,hashlib,json,re,datetime
ROOT=Path(__file__).resolve().parents[2]
S=ROOT/'submission_strengthening';P=ROOT/'paper_submission_candidate'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,condition,detail=''):
    checks.append({'check':name,'pass':bool(condition),'detail':detail})
for r in json.loads((S/'records/preserved_targeted_baseline.json').read_text())['records']:
    check('preserved:'+r['path'],sha(ROOT/r['path'])==r['sha256'])
for rel in ('numbers.tex','tables/primary.tex','tables/historical_49.tex','data/historical_49_complete_key.csv','figures/stored_example.pdf'):
    check('unchanged_candidate_copy:'+rel,sha(P/rel)==sha(ROOT/'paper_v5_2'/rel))
old=json.loads((ROOT/'AFAST_reporting_closure/publication_numbers_v5_2.json').read_text())
new=json.loads((S/'publication_numbers_submission_candidate.json').read_text())
check('registry_old565_prefix_exact',new['records'][:len(old['records'])]==old['records'],len(old['records']))
sources={r['source_file']:r['source_sha256'] for r in new['records'][len(old['records']):]}
for f,digest in sources.items():check('new_registry_source:'+f,sha(ROOT/f)==digest)
main=(P/'main_submission_candidate.tex').read_text()
abstract=main.split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0]
words=len(abstract.split());check('editorial_abstract_200_230_words',200<=words<=230,words)
sections='\n'.join(p.read_text() for p in (P/'sections').glob('*.tex'))
check('no_seed_prediction_ensemble_wording','small frozen ensemble' not in sections)
check('figure_grammar_fixed','Figure 5 show ' not in sections)
for document in ('main_submission_candidate','appendix_submission_candidate'):
    log=(P/(document+'.log')).read_text(errors='replace')
    failures=[line for line in log.splitlines() if ('undefined' in line.lower() and ('reference' in line.lower() or 'citation' in line.lower())) or line.startswith('!')]
    check('complete_build:'+document,(P/(document+'.pdf')).exists() and not failures,failures)
    # Overfull warnings are recorded for targeted visual inspection, not auto-hidden.
    over=[line for line in log.splitlines() if 'Overfull' in line]
    check('overfull_inspection_inventory:'+document,True,over)
check('independent_new_reference_check',json.loads((S/'records/targeted_reference_and_preservation_check.json').read_text()).get('status','').startswith('PASS'))
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'preserved primary/table digests, new registry source identities, targeted prose and final builds; no general audit','checks':checks,'status':'PASS' if all(c['pass'] for c in checks) else 'BLOCKED'}
(S/'records/targeted_candidate_check.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'checks':len(checks),'failed':[c for c in checks if not c['pass']]},indent=2))
if out['status']!='PASS':raise SystemExit(1)
