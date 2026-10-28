"""Index displayed numerical exports; do not rescore, fit or resample."""
from pathlib import Path
import csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1];NEW=ROOT.parent;PROJECT=NEW.parent
rows=[];sources={}
def load(path):
    p=PROJECT/path;sources[path]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open() as f:return list(csv.DictReader(f))
def add(study,path,keys,point,lo,hi,level,default_units='raw regret',scope='descriptive'):
    for i,r in enumerate(load(path)):
        rows.append(dict(study=study,source=path,source_sha256=sources[path],row_index=i,selector=json.dumps({k:r[k] for k in keys},sort_keys=True),
            estimate=r[point],lower=r[lo],upper=r[hi],interval_level=level,units=default_units,status=scope))
add('reference','paper_final_review/data/primary_results.csv',['question','endpoint','conditions'],'mean_diff','ci9875_lo','ci9875_hi','98.75% adjusted marginal','source units; height metres, regret raw utility, unsafe probability difference','original four-primary family preserved')
external_primary='AFAST_reporting_closure/primary_reconstruction/analysis/primary_results.json'
p=PROJECT/external_primary;sources[external_primary]=hashlib.sha256(p.read_bytes()).hexdigest()
for i,r in enumerate(json.loads(p.read_text())['endpoints']):
    rows.append(dict(study='external_primary',source=external_primary,source_sha256=sources[external_primary],row_index=i,selector=json.dumps(dict(name=r['name'])),estimate=r['estimate'],lower=r['adjusted97_5'][0],upper=r['adjusted97_5'][1],interval_level='97.5% adjusted marginal',units='raw regret' if i==0 else 'unsafe probability difference',status='original two-primary family; inconclusive; joint criterion not met'))
add('external','AFAST_reporting_closure/secondary_descriptive_intervals.csv',['comparison','mask','condition','policy','metric'],'estimate','lower95','upper95','95% ordinary','source units','reporting-stage descriptive comparisons, preserved')
add('classical','strengthening_2026_10_18/phase_a/classical/descriptive_intervals.csv',['condition','metric'],'estimate','lower95','upper95','95% ordinary','source units','post-hoc descriptive')
add('gate','strengthening_2026_10_18/study3/analysis/primary_results.csv',['architecture','contrast','policy','metric'],'estimate','lower98_75','upper98_75','98.75% adjusted marginal','raw regret / unsafe probability difference','prospectively locked four-primary family; authorized population exclusion')
add('gate_secondary','strengthening_2026_10_18/study3/analysis/complete_secondary_comparisons.csv',['architecture','contrast','policy','pool','metric'],'estimate','lower95','upper95','95% ordinary','source units','prespecified descriptive')
for p in sorted((ROOT/'data').glob('*_sources.json')):
    for key,value in json.loads(p.read_text()).items():sources[str(p.name)+':'+key]=value
with (ROOT/'data/number_registry.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'data/registry_provenance.json').write_text(json.dumps(dict(rows=len(rows),sources=sources,source_values_unscaled=True,display_scaling='height metres x1000 when labelled mm; unsafe probability contrasts x100 for percentage points; absolute probabilities x100 for percentages',old_primaries='Four reference and two external primary records remain in their immutable original/corrected exports; original tests/families retain their memberships',new_scoring=False,new_bootstrap_draws=0),indent=2)+'\n')
print(len(rows),'registry rows')
