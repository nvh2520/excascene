"""Targeted display/source check; no scoring, inference, resampling or re-estimation."""
from pathlib import Path
from functools import lru_cache
import ast,csv,json,re,hashlib,math,subprocess
O=Path(__file__).resolve().parents[1];R=O.parent;P=O/'manuscript';OLD=R/'pre_submission_2026_10_24/manuscript'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@lru_cache(None)
def rows(p):return list(csv.DictReader(Path(p).open()))
def same(a,b):
 try:return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=2e-13)
 except (ValueError,TypeError):return str(a)==str(b)
def get(path,lookup):
 found=[r for r in rows(str(path)) if all(k in r and same(r[k],v) for k,v in lookup.items())];assert len(found)==1,(path,lookup,len(found));return found[0]
def numeric_table(path):return [re.findall(r'[-+]?\d+(?:\.\d+)?',l) for l in path.read_text().splitlines() if '&' in l and re.search(r'\d',l) and not any(x in l for x in ['caption','multicolumn','toprule','cmidrule'])]
def writecsv(p,rs):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
 checks=[];source_hashes={}
 assert sha(P/'numbers.tex')==sha(OLD/'numbers.tex')
 for name in ['reference_primary','external_primary','gate_primary_combined','architecture_tradeoffs','architecture_replication','material_main','task_value','task_value_replication']:
  assert numeric_table(P/'tables'/f'{name}.tex')==numeric_table(OLD/'tables'/f'{name}.tex'),name
  checks.append({'check':name,'status':'NUMERIC_ROWS_IDENTICAL'})
 verified=[]
 for q in rows(str(P/'data/context_checked_claims.csv')):
  src=R/q['source'];lookup=json.loads(q['lookup']);fields=q['field'].split('/');expected=json.loads(q['raw_value'])
  if '/' in str(lookup.get('architecture','')):
   root=lookup['architecture'].split('-D')[0];values=[float(get(src,dict(lookup,architecture=root+'-D'+suffix))[q['field']]) for suffix in ['14','42']]
  else:r=get(src,lookup);values=[float(r[k]) for k in fields]
  exp=expected if isinstance(expected,list) else [expected];assert len(values)==len(exp) and all(same(a,b) for a,b in zip(values,exp)),q
  verified.append(q);source_hashes[q['source']]=sha(src)
 for q in rows(str(P/'data/readiness_number_registry.csv')):
  src=R/q['source'];r=get(src,json.loads(q['lookup']));assert same(r[q['field']],q['value']);verified.append(q);source_hashes[q['source']]=sha(src)
 # Newly displayed counts and means are copied from the authorized saved-frame report.
 f4=[]
 for r in rows(str(O/'F4/decision_change_counts.csv')):
  if r['scope']!='all_three_seeds' or r['policy']!='direct':continue
  expected=[f"{100*float(r[k]):.1f}" for k in ['changed_share','lower_share_changed','equal_share_changed','higher_share_changed']]+[f"{float(r['mean_regret_change_changed']):+.5f}",r['unsafe_to_safe'],r['safe_to_unsafe']]
  line=next(l for l in (P/'tables/decision_changes.tex').read_text().splitlines() if l.startswith('KinAfford dynamic' if r['architecture']=='KF-D42' else 'BEVFusion-Exca'))
  assert all(v in line for v in expected),(r,expected,line);f4.append(r)
 assert len(f4)==2
 rr=json.loads((O/'R1/analysis_receipt.json').read_text());assert rr['settings']==25 and rr['new_draws']==0
 changes=rows(str(O/'R1/sign_class_changes.csv'))
 counts={k:sum(r['kind']=='primary' and r['field']==k for r in changes) for k in ['class98_75','sign']};assert counts=={'class98_75':2,'sign':1}
 direct=[r for r in changes if r['kind']=='descriptive' and r['identity'].startswith('direct')]
 assert sum(r['identity'].endswith('clean|regret') for r in direct)==13
 assert sum(r['identity'].endswith('clean|unsafe') for r in direct)==9
 # Every specification entry's line interval must reproduce the frozen excerpt.
 spec=rows(str(O/'specification/source_parameters.csv'))
 for q in spec:
  src=R/q['source_file'];assert sha(src)==q['source_sha256'],q['source_file']
  lines=src.read_text().splitlines();excerpt='\n'.join(lines[int(q['line_start'])-1:int(q['line_end'])])
  assert q['source_excerpt'].strip() in excerpt.strip(),(q['id'],q['parameter'])
 cov=json.loads((O/'specification/coverage_scope.json').read_text());assert 'all truth terrain' in cov['denominator'] and 'not restricted to reachable' in cov['denominator']
 assert f"{cov['saved_camera_coverage_percent']:.2f}" in (P/'generated/coverage.tex').read_text()
 assert 'Hazard cutoff & 0.5' in (P/'tables/decision_grounding.tex').read_text()
 # Record every changed main-text numeric occurrence, including source-derived specification constants.
 active=[]
 def visit(f):
  if f in active or f.name in ['numbers.tex','preamble.tex']:return
  active.append(f)
  for name in re.findall(r'\\input\{([^}]+)\}',f.read_text()):visit(P/(name if name.endswith('.tex') else name+'.tex'))
 visit(P/'main_submission.tex');registry=[]
 mapping={'tables/decision_changes.tex':'F4/decision_change_counts.csv','tables/decision_grounding.tex':'specification/decision_constants.csv; specification/source_parameters.csv; specification/hazard_cutoff_provenance.json','generated/coverage.tex':'specification/coverage_scope.json','generated/abstract.tex':'data/context_checked_claims.csv; data/readiness_number_registry.csv; data/number_registry.csv; tables/reference_primary.tex; tables/task_value_replication.tex','sections/diagnostics.tex':'R1/analysis_receipt.json; R1/sign_class_changes.csv; data/context_checked_claims.csv','sections/gate_results.tex':'F4/decision_change_counts.csv; visualizations/selection_lock.json; data/context_checked_claims.csv; ../pre_submission_2026_10_24/V1/' }
 for f in active:
  rel=str(f.relative_to(P));old=OLD/rel;prior=set(old.read_text().splitlines()) if old.exists() else set()
  for i,line in enumerate(f.read_text().splitlines(),1):
   if line.startswith('%'):continue
   for tok in re.findall(r'(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?',line):
    registry.append({'file':rel,'line':i,'number_token':tok,'changed_line':line not in prior,'source':mapping.get(rel,'Retained accepted number registry and frozen source/configuration specification'),'display_context':line})
 writecsv(P/'data/final_number_occurrence_registry.csv',registry)
 # Text and build-level checks are on the actual PDF, not just source.
 for name in ['main_submission','supplement_submission']:
  subprocess.run(['pdftotext','-layout',str(P/(name+'.pdf')),str(O/'report'/(name+'_extracted.txt'))],check=True)
  log=(P/(name+'.log')).read_text();assert not re.search(r'Overfull|undefined references|undefined on input|multiply defined',log),name
  text=(O/'report'/(name+'_extracted.txt')).read_text();assert '??' not in text,name
 abstract=(P/'generated/abstract.tex').read_text();wc=len(abstract.split());assert wc<=150
 highlights=(P/'highlights.txt').read_text().splitlines();assert 3<=len(highlights)<=5 and max(map(len,highlights))<=85
 for rel in ['R1/primary_sensitivity.csv','R1/descriptive_signs.csv','R1/sign_class_changes.csv','F4/decision_change_counts.csv','visualizations/selection_lock.json','specification/source_parameters.csv']:source_hashes[str(O.relative_to(R)/rel)]=sha(O/rel)
 out={'status':'PASS','scope':'Saved-source fields, authorized R1/F4 display, source-line specification and compiled references; no scientific recomputation','retained_source_fields':len(verified),'retained_table_checks':checks,'specification_entries':len(spec),'main_numeric_occurrences':len(registry),'changed_line_occurrences':sum(q['changed_line'] for q in registry),'abstract_words':wc,'highlight_characters':list(map(len,highlights)),'R1_grid_changes':counts,'source_sha256':source_hashes,'gpu_hours':0,'new_neural_forwards':0,'new_bootstrap_draws':0}
 (O/'report/targeted_number_claim_check.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],wc,len(spec),len(registry))
if __name__=='__main__':main()
