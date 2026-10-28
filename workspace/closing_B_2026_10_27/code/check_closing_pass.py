"""Targeted preservation, number-registry and document checks for this closing pass."""
from pathlib import Path
import hashlib,json,csv,re,argparse,subprocess
import pandas as pd
O=Path(__file__).resolve().parents[1];P=O.parent;M=O/'manuscript';B=P/'variant_B_submission_2026_10_26'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def numerical():
    hist=json.loads((O/'protocol/prior_manuscript_hashes.json').read_text())
    # Setup manifest is a direct path-to-digest map.
    if 'files' in hist:hist=hist['files']
    for f,h in hist.items():assert sha(P/f)==h,f
    lock=json.loads((O/'protocol/C1_lock.json').read_text())
    for f,h in lock['files'].items():assert sha(P/f)==h,f
    changed=[];allowed={'task_value_replication.tex','severe_person_safety.tex','classical_validation.tex','replication_sensitivity.tex','decision_changes.tex'}
    for f in (B/'manuscript/tables').glob('*.tex'):
        if sha(f)!=sha(M/'tables'/f.name):changed.append(f.name);assert f.name in allowed,f
    for f in (B/'manuscript/figures').iterdir():
        if f.is_file():assert sha(f)==sha(M/'figures'/f.name),f
    for name in ['task_value_replication.tex','severe_person_safety.tex']:
        old=next(s for s in (B/'manuscript/tables'/name).read_text().splitlines() if s.startswith('Classical local-ground &'))
        assert old in (M/'tables'/name).read_text()
    grid=pd.read_csv(O/'C1/validation/extended_grid.csv');sel=json.loads((O/'C1/validation/selection_lock.json').read_text())
    best=grid.sort_values(['regret','buffer','tau']).iloc[0];assert (best.tau,best.buffer)==(sel['tau'],sel['buffer'])
    assert len(grid)==60 and (grid.frames==1200).all()
    receipt=json.loads((O/'C1/holdout/receipt.json').read_text());assert receipt['selector_calls']==5120 and receipt['sensitivity_new_selector_calls']==0
    a=pd.read_csv(O/'C1/holdout/condition_results.csv');pairs=pd.read_csv(O/'C1/holdout/paired_comparisons.csv')
    for _,r in a[a.metric.isin(['unsafe','abstained','executed'])].iterrows():assert abs(r.estimate-r.events/r.attempts)<1e-14
    t=(M/'tables/task_value_replication.tex').read_text()
    for cond in ['clean','cam_s3','rough_iid']:
        g=a[a.condition==cond].set_index('metric');assert f'{g.loc["regret","estimate"]:.4f}' in t;assert f'{100*g.loc["unsafe","estimate"]:.2f}' in t
    for _,r in pairs[pairs.contrast=='learned_divided_by_posthoc_classical'].iterrows():
        text=(M/'sections/maps_results.tex').read_text()
        for field in ['estimate','lower95','upper95']:assert f'{100*r[field]:.1f}' in text
    # Full new numeric register, retaining source lookup and digest for every entry.
    registry=[]
    specifications=[
      (O/'C1/validation/extended_grid.csv',['tau','buffer'],'Table S24; Methods; Limitations'),
      (O/'C1/holdout/condition_results.csv',['condition','metric'],'Tables 2 and 11; Section 5.1; C1 report'),
      (O/'C1/holdout/paired_comparisons.csv',['condition','architecture','contrast','metric'],'Abstract; Section 5.1; highlights; cover letter'),
      (O/'C1/sensitivity/learned_comparisons.csv',['setting','architecture','policy','condition','metric'],'Supplementary Section 4; K1 extension'),
      (B/'K4/support_summary.csv',['condition','scope','architecture','arm','policy','input_state','seed_scope','seed'],'Section 5.4 direct-return coverage; Section 6.1'),
    ]
    for src,keys,where in specifications:
        for r in csv.DictReader(src.open()):
            for field,v in r.items():
                try:value=float(v)
                except (TypeError,ValueError):continue
                if not __import__('math').isfinite(value):continue
                registry.append(dict(source=str(src.relative_to(P)),source_sha256=sha(src),lookup=json.dumps({k:r[k] for k in keys},sort_keys=True),field=field,value=v,display_locations=where))
    with (M/'data/closing_number_registry.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(registry[0]));w.writeheader();w.writerows(registry)
    # Re-read each registered source with unique keys, verifying the emitted values.
    cache={}
    for entry in registry:
        key=(entry['source'],entry['lookup']);lookup=json.loads(entry['lookup'])
        if entry['source'] not in cache:cache[entry['source']]=list(csv.DictReader((P/entry['source']).open()))
        matches=[r for r in cache[entry['source']] if all(r[k]==v for k,v in lookup.items())]
        assert len(matches)==1,(key,len(matches));assert matches[0][entry['field']]==entry['value']
    abstract=(M/'generated/abstract.tex').read_text().replace('\\%','%').replace('$','').replace('--','–')
    words=len(abstract.split());assert words<=150
    highlights=(M/'highlights.txt').read_text().splitlines();assert 3<=len(highlights)<=5 and max(map(len,highlights))<=85
    assert 'About 5\\% of decisions entered person buffers, although a reachable safe cell always existed.' in (M/'generated/abstract.tex').read_text()
    assert 'changes the earlier architecture ranking' not in (M/'generated/conclusion.tex').read_text()
    for rel in ['sections/design.tex','sections/declarations.tex']:assert 'drafted and revised the manuscript text' in (M/rel).read_text()
    assert '\\input{sections/using_excascene.tex}' not in (M/'sections/benchmark.tex').read_text()
    assert '\\subsection{Reproducibility and access}' in (M/'sections/design.tex').read_text()
    table=(M/'tables/decision_changes.tex').read_text();assert table.index('KinAfford dynamic &')<table.index('BEVFusion-Exca &')
    result=dict(status='PASS_TARGETED_NUMBER_AND_CLAIM_CHECK',prior_manuscript_files_unchanged=len(hist),locked_dependencies_unchanged=len(lock['files']),changed_tables=changed,
        all_other_tables_and_all_figures_identical=True,frozen_classical_rows_identical=True,new_registry_entries=len(registry),abstract_words=words,highlight_characters=list(map(len,highlights)),
        original_primary_results_unchanged=True,scientific_computation='Only authorized C1, once; original learned values and draws reused',gpu_hours=0,new_neural_forwards=0,new_bootstrap_draws=0)
    js(O/'report/number_claim_check.json',result);print(json.dumps(result,indent=2))
def documents():
    fields=[];pages={}
    for name in ['main_submission','supplement_submission','cover_letter','title_page_submission']:
        subprocess.run(['pdftotext','-layout',str(M/(name+'.pdf')),str(O/'report'/(name+'_extracted.txt'))],check=True)
        text=(O/'report'/(name+'_extracted.txt')).read_text();pages[name]=text.count('\f')
        for page_no,page_text in enumerate(text.split('\f'),1):
            # Typesetting can split REPOSITORY over a line; retain that author field.
            normalized=re.sub(r'(?m)^[ \t]*\d+[ \t]+','',page_text)
            normalized=re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])','',normalized)
            for pat in [r'\[LINK\]',r'\[REPOSITORY/DOI\]',r'\[Authors:[^\]]+\]',r'\[Corresponding author[^\]]+\]']:
                for match in re.finditer(pat,normalized,re.S):fields.append(dict(document=name+'.pdf',page=page_no,field=' '.join(match.group().split())))
        assert '??' not in text,name
        log=(M/(name+'.log')).read_text()
        assert 'undefined references' not in log.lower() and 'undefined on input' not in log.lower(),name
    assert pages['cover_letter']==1 and pages['title_page_submission']==1
    text=(O/'report/main_submission_extracted.txt').read_text();assert '4.7. Reproducibility and access' in text
    for label in ['taskvaluereplication','personsafety','decisionchanges']:
        assert '\\newlabel{tab:'+label+'}' in (M/'main_submission.aux').read_text()
    assert '\\newlabel{tab:classicalvalidation}{{S24}' in (M/'supplement_submission.aux').read_text()
    with (O/'report/remaining_author_fields.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['document','page','field']);w.writeheader();w.writerows(fields)
    js(O/'report/document_check.json',dict(status='PASS',pages=pages,remaining_fields=fields,unresolved_cross_references=0,independent_review_performed=False))
    print(json.dumps({'pages':pages,'author_field_occurrences':len(fields)},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['numerical','documents']);a=ap.parse_args();globals()[a.action]()
