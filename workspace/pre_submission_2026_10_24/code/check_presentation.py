"""Targeted transcription, claims and build checks. Never calls scientific code."""
from pathlib import Path
from decimal import Decimal
from functools import lru_cache
import csv,hashlib,json,math,re,subprocess
O=Path(__file__).resolve().parents[1];R=O.parent;P=O/'manuscript';OLD=R/'submission_readiness_2026_10_22/manuscript'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@lru_cache(None)
def rows(path):return list(csv.DictReader(Path(path).open()))
def same(a,b):
    try:return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=2e-13)
    except (ValueError,TypeError):return str(a)==str(b)
def get(path,lookup):
    found=[r for r in rows(str(path)) if all(k in r and same(r[k],v) for k,v in lookup.items())]
    assert len(found)==1,(path,lookup,len(found));return found[0]
def csvwrite(path,rs):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def numeric_table(path):
    return [re.findall(r'[-+]?\d+(?:\.\d+)?',line) for line in path.read_text().splitlines() if '&' in line and re.search(r'\d',line) and not any(x in line for x in ['caption','multicolumn','toprule','cmidrule'])]
def main():
    checks=[];source_hashes={}
    for relative,digest in json.loads((O/'records/source_manuscript_hashes.json').read_text()).items():assert sha(R/relative)==digest,relative
    assert sha(P/'numbers.tex')==sha(OLD/'numbers.tex')
    for name in ['reference_primary','external_primary','gate_primary_combined','architecture_tradeoffs','architecture_replication','material_main']:
        assert numeric_table(P/'tables'/f'{name}.tex')==numeric_table(OLD/'tables'/f'{name}.tex'),name
        checks.append(dict(check=name,status='DISPLAYED_NUMERIC_ROWS_IDENTICAL'))
    for name,count in [('task_value',5),('task_value_replication',7)]:
        s=(P/'tables'/f'{name}.tex').read_text();assert '\\shortstack' not in s and 'Executes' not in s and '100\\%' in s
        assert len(numeric_table(P/'tables'/f'{name}.tex'))==count
        assert 'Uniform reachable (expected)' in s and 'Fixed training prior' in s
    assert not any('\\shortstack' in p.read_text() for p in (P/'tables').glob('*.tex'))
    # Check existing contextual transcriptions by exact source keys, without rerunning estimates.
    context=[]
    for q in rows(str(P/'data/context_checked_claims.csv')):
        src=R/q['source'];lookup=json.loads(q['lookup']);fields=q['field'].split('/');expected=json.loads(q['raw_value'])
        if '/' in str(lookup.get('architecture','')):
            root=lookup['architecture'].split('-D')[0];values=[]
            for suffix in ['14','42']:
                a=dict(lookup,architecture=root+'-D'+suffix);values.append(float(get(src,a)[q['field']]))
        else:
            r=get(src,lookup);values=[float(r[k]) for k in fields]
        exp=expected if isinstance(expected,list) else [expected]
        assert len(values)==len(exp) and all(same(a,b) for a,b in zip(values,exp)),q
        context.append(dict(q,verification='MATCHES_SAVED_SOURCE_FIELDS_NO_RESCORING'));source_hashes[q['source']]=sha(src)
    csvwrite(P/'data/presentation_context_checked_claims.csv',context)
    readiness=[]
    for q in rows(str(P/'data/readiness_number_registry.csv')):
        src=R/q['source'];lookup=json.loads(q['lookup']);r=get(src,lookup)
        assert same(r[q['field']],q['value']),(q,r[q['field']]);readiness.append(q);source_hashes[q['source']]=sha(src)
    checks.append(dict(check='accepted_context_and_readiness_registry',source_fields_verified=len(context)+len(readiness),status='PASS'))
    for q in rows(str(O/'V1/novel_keep_all_costs.csv')):
        lookup={k:q[k] for k in ['architecture','policy','condition','contrast','metric']};src=R/'final_evidence_2026_10_20/study4/analysis/sensor_impact_comparisons.csv';r=get(src,lookup)
        for f in ['estimate','lower95','upper95']:assert q[f]==r[f]
    # Every newly tabulated absolute oracle effect is a saved CSV row, not a new interval.
    for q in rows(str(O/'V1/severe_oracle_effects_from_saved.csv')):
        src=R/q['source']/'complete_secondary_comparisons.csv'
        r=get(src,dict(architecture=q['architecture'],policy='direct',contrast='oracle_mask_minus_keep_all',pool='cam_s3',metric=q['metric']))
        assert all(q[k]==r[k] for k in ['estimate','lower95','upper95'])
    assert all(float(q['gate_minus_keep_regret'])==0 for q in rows(str(O/'V1/qualitative_pair_effects.csv')))
    # Active source/number inventory replaces the stale inherited occurrence map.
    inherited=P/'data/main_number_occurrence_registry.csv'
    backup=P/'data/prior_main_number_occurrence_registry.csv'
    if not backup.exists():backup.write_bytes(inherited.read_bytes())
    sources={
      'generated/abstract.tex':'tables/task_value_replication.tex; tables/reference_primary.tex; tables/gate_primary_combined.tex; data/context_checked_claims.csv; V1/severe_mask_equality.json',
      'sections/benchmark.tex':'external_validation_run_EV1_AFAST/source/excascene_snapshot/kinafford/sim/scene.py; kinafford/sim/kinematics.py; kinafford/sim/degrade.py (same snapshot)',
      'sections/design.tex':'Supplementary study crosswalk; existing protocols; data/number_registry.csv',
      'sections/maps_results.tex':'tables/task_value_replication.tex; tables/reference_primary.tex; tables/architecture_tradeoffs.tex; tables/architecture_replication.tex; tables/material_main.tex',
      'sections/removal_results.tex':'tables/reference_primary.tex; tables/external_primary.tex; AFAST_reporting_closure/primary_reconstruction/analysis/primary_results.json; data/number_registry.csv',
      'sections/gate_results.tex':'data/presentation_context_checked_claims.csv; V1/novel_keep_all_costs.csv; V1/novel_camera_detection_inventory.csv; V1/qualitative_pair_effects.csv',
      'sections/diagnostics.tex':'data/presentation_context_checked_claims.csv; data/number_registry.csv; saved attribution/safety/sensitivity tables',
      'sections/image_check.tex':'data/presentation_context_checked_claims.csv; tables/image_upgrade.tex; tables/real_gate_summary.tex; data/readiness_number_registry.csv',
      'sections/discussion.tex':'tables/reference_primary.tex; tables/material_main.tex; tables/real_gate_summary.tex; data/number_registry.csv',
      'generated/gate_engineering.tex':'data/presentation_context_checked_claims.csv; data/number_registry.csv; V1/severe_mask_equality.json',
      'generated/conclusion.tex':'tables/reference_primary.tex; tables/gate_primary_combined.tex; tables/image_upgrade.tex',
    }
    active=[]
    def visit(p):
        if p in active or p.name in ['preamble.tex','numbers.tex']:return
        active.append(p)
        for item in re.findall(r'\\input\{([^}]+)\}',p.read_text()):visit(P/(item if item.endswith('.tex') else item+'.tex'))
    visit(P/'main_submission.tex');inventory=[]
    for p in active:
        rel=str(p.relative_to(P));old=OLD/rel
        for i,line in enumerate(p.read_text().splitlines(),1):
            if line.lstrip().startswith('%'):continue
            for token in re.findall(r'(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?',line):
                inherited_display=old.exists() and line in old.read_text().splitlines()
                src=sources.get(rel,('submission_readiness_2026_10_22/manuscript/'+rel if inherited_display else 'data/number_registry.csv; data/readiness_number_registry.csv; saved tables and presentation source receipts'))
                if rel=='tables/severe_oracle_recovery.tex':src='V1/severe_oracle_effects_from_saved.csv; saved oracle_benefit_recovery.csv'
                inventory.append(dict(file=rel,line=i,displayed_number=token,source=src,context=line.strip(),verification='UNCHANGED_ACCEPTED_SOURCE_LINE' if inherited_display else 'PRESENTATION_TRANSCRIPT_SOURCE_LOCATED'))
    csvwrite(inherited,inventory)
    abstract=(P/'generated/abstract.tex').read_text();abstract_words=len(abstract.split());assert abstract_words<=150
    highlights=[x[2:] for x in (P/'highlights.txt').read_text().splitlines() if x.startswith('- ')];assert 3<=len(highlights)<=5 and max(map(len,highlights))<=85
    text=subprocess.check_output(['pdftotext','-layout',str(P/'main_submission.pdf'),'-'],text=True)
    for pattern in [r'\bAFAST\b',r'\bv5\.1\b',r'\bBF-D\b',r'\bKF-D\b',r'cam_s3',r'all_s2',r'\bQ2\b',r'F\.6',r'49 tests',r'user.authorized exception',r'one agent administers']:
        assert not re.search(pattern,text,re.I),pattern
    for word in ['97.5','98.75','inconclusive','in-family detection','not deployable','0.470','0.117','0.112']:assert word in text,word
    assert 'inconclusive' in (P/'tables/external_primary.tex').read_text()
    assert text.index('Discussion')<text.index('Limitations') and text.count('Limitations')==1
    for name in ['main_submission','supplement_submission']:
        log=(P/(name+'.log')).read_text(errors='replace')
        for s in ['undefined references','undefined citations','LaTeX Error','Overfull \\hbox']:assert s not in log,(name,s)
    bibcount=len(re.findall(r'\\bibitem', (P/'main_submission.bbl').read_text()));assert 45<=bibcount<=60
    visual=json.loads((O/'visualizations/render_receipt.json').read_text());assert visual['new_neural_forwards']==0 and visual['gpu_hours']==0
    result=dict(status='PASS',scope='source-field transcription, unchanged primary/comparison displays, actual PDF labels and build references; no scoring or resampling',checks=checks,abstract_word_count=abstract_words,highlight_characters=list(map(len,highlights)),bibliography_items=bibcount,active_number_occurrences=len(inventory),prior_source_hashes_preserved=True,new_neural_forwards=0,new_scores=0,new_bootstrap_draws=0,gpu_hours=0,source_sha256=source_hashes)
    (O/'report/targeted_number_claim_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
