"""Targeted read-only source/number check. No scientific pipeline or new draws."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,hashlib,re
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parents[1];P=O.parent;M=O/'manuscript';F=P/'final_evidence_2026_10_20'
old=P/'final_pre_submission_2026_10_26/variants/B/manuscript'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
checks=[]
def check(name,ok,detail=''):
    assert bool(ok),(name,detail)
    checks.append(dict(check=name,status='PASS',detail=detail))
def read(f):return pd.read_csv(O/f)
def texrows(n):return [line for line in (M/'tables'/n).read_text().splitlines() if ' & ' in line and line.endswith('\\') and not line.startswith('\\')]
num=r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?'
def values(s):return [float(x) for x in re.findall(num,s)]
boot=F/'study4/analysis/bootstrap_multiplicities.npz'
check('Existing multiplicities digest',sha(boot)=='6701bccbac734469666abf921af07034ec17418e28e6004592fbf093de234296')
check('K1 pre-computation lock digest',sha(O/'protocol/K1_lock.json')=='58406a5b7a4142d57a97b77cc0dae09c0b4da97dbbfbc98e73c40631bc9ec4f9')
k1=read('K1/primary_sensitivity.csv');d1=read('K1/descriptive_signs.csv')
check('K1 complete grid',len(k1)==108 and k1.setting.nunique()==27 and len(d1)==540)
check('KinAfford both adjusted classes at all settings',(k1[k1.architecture=='KF-D42'].class98_75=='LOWER').all())
b=k1[(k1.architecture=='BF-D42')&(k1.metric=='regret')]
check('BEVFusion regret lower in two settings',set(b[b.class98_75=='LOWER'].setting)=={'grid_04','grid_06'})
check('BEVFusion regret positive in four settings',(b.sign=='positive').sum()==4)
check('BEVFusion unsafe remains inconclusive',(k1[(k1.architecture=='BF-D42')&(k1.metric=='unsafe')].class98_75=='INCONCLUSIVE').all())
rec=json.loads((O/'K1/analysis_receipt.json').read_text())
check('Nominal matches frozen primaries',all(r['max_absolute_difference']<2e-11 and r['class_preserved'] for r in rec['nominal_reconstruction']))
changed=json.loads((O/'K1/display_change_counts.json').read_text())
check('Clean architecture sign counts',changed['descriptive|direct|BF42_minus_KF42|clean|regret|sign']==11 and changed['descriptive|direct|BF42_minus_KF42|clean|unsafe|sign']==9)
for cond in ['cam_s3','rough_iid']:
    for metric in ['regret','unsafe']:
        q=d1[(d1.policy=='direct')&(d1.contrast=='BF42_minus_KF42')&(d1.condition==cond)&(d1.metric==metric)]
        check('Direct architecture signs persist '+cond+' '+metric,q.sign.nunique()==1)
check('Severe KinAfford effect retains sign',(d1[d1.contrast=='KF_severe_gate_minus_keep'].sign=='negative').all())
check('Learned clean regret below classical',(d1[d1.contrast.str.contains('minus_classical')].sign=='negative').all())
# Display table retains each primary class and sign in setting order.
compact=read('K1/compact_sensitivity_table.csv')
rows=texrows('replication_sensitivity.tex')
check('K1 displayed settings',len(rows)==27)
for row,(_,r) in zip(rows,compact.iterrows()):
    cells=row.split(' & ')
    for j,(a,m) in enumerate([(a,m) for a in ['KF-D42','BF-D42'] for m in ['regret','unsafe']]):
        sign,cl=r[a+'_'+m].split('; ');expected={'negative':'$-$','positive':'$+$','zero':'0'}[sign]+{'LOWER':'L','INCONCLUSIVE':'I','HIGHER':'H'}[cl]
        assert cells[5+j]==expected
check('K1 all displayed primary signs/classes',True)

k2=read('K2/stratified_safety.csv');pop=read('K2/population_support.csv')
check('K2 complete direct/reference inventory',len(k2)==57 and set(k2.pool)=={'clean','cam_s3','equal9'})
check('K2 safe/positive population',(pop[pop.flag=='safe_available'].attempt_share==1).all() and (pop[pop.flag=='zero_oracle'].attempt_share==0).all())
check('K2 undefined empty strata',(k2.no_safe_status=='UNDEFINED_NO_ATTEMPTS').all() and (k2.oracle_zero_status=='UNDEFINED_NO_ATTEMPTS').all() and k2.no_safe_unsafe_rate.isna().all() and k2.oracle_zero_unsafe_rate.isna().all())
check('K2 unsafe alternatives',np.allclose(k2.unsafe_share_with_safe_available,1,atol=1e-14))
orig=pd.read_csv(F/'study4/diagnostics/safety/complete_cause_inventory.csv')
pool=pd.read_csv(F/'study4/diagnostics/safety/equal_condition_pool.csv')
errors=[]
for _,r in k2.iterrows():
    keys=dict(architecture=r.architecture,arm=r.arm,policy=r.policy)
    q=pool if r.pool=='equal9' else orig
    for key,v in keys.items():q=q[q[key]==v]
    if r.pool=='equal9':
        expected=[q[q.metric==m].estimate.iloc[0] for m in ['unsafe','multi_person_buffer']]
    else:
        q=q[q.condition==r.pool].iloc[0];expected=[q.unsafe/q.attempts,q.multi_person_buffer/q.attempts]
    errors.extend(abs(np.asarray(expected)-[r.unsafe_rate,r.person_rate]))
check('K2 rates reconstruct original cause inventory',max(errors)<1e-12,f'max_abs={max(errors):.3g}')
for row,(_,g) in zip(texrows('always_act_strata.tex'),k2.groupby(['architecture','arm'],sort=True)):
    expected=[]
    for pool_name in ['clean','cam_s3','equal9']:
        r=g[g.pool==pool_name].iloc[0];expected.extend([round(100*r.unsafe_rate,2),round(100*r.person_rate,2)])
    assert np.allclose(values(' & '.join(row.split(' & ')[2:])),expected,atol=1e-12)
check('K2 complete display rates',len(texrows('always_act_strata.tex'))==19)
pose_rows=[r for f in (O/'K2/scenes').glob('*_population.csv') for r in csv.DictReader(f.open())]
check('K2 strictly positive oracle, every pose',len(pose_rows)==3072 and all(float(r['oracle_utility'])>0 and int(r['safe_reachable_cells'])>0 for r in pose_rows))

k3=read('K3/prevalence_mixtures.csv');original=pd.read_csv(F/'study4/analysis/all_per_scene_results.csv')
for _,r in k3.iterrows():
    q=original[(original.architecture==r.architecture)&(original.policy=='direct')]
    means=q.groupby(['arm','condition'])[r.metric].mean()
    delta=lambda c:means.loc[('gate',c)]-means.loc[('keep_all',c)]
    p=r.severe_prevalence_percent/100
    assert abs(r.estimate-((1-p)*delta('clean')+p*delta('cam_s3')))<1e-12
    assert abs(r.percent_of_clean_regret-100*r.estimate/r.clean_keep_regret)<1e-11
check('K3 all twenty mixture points/normalizations',True)
check('K3 BEVFusion intervals include zero',((k3[k3.architecture=='BF-D42'].lower95<0)&(k3[k3.architecture=='BF-D42'].upper95>0)).all())
rows=texrows('prevalence.tex');check('K3 full display inventory',len(rows)==20)
for row,(_,r) in zip(rows,k3.sort_values(['architecture','severe_prevalence_percent','metric'],key=lambda x:x.map({'KF-D42':0,'BF-D42':1}) if x.name=='architecture' else x).iterrows()):
    x=values(row);scale=1 if r.metric=='regret' else 100;dp=5 if r.metric=='regret' else 4
    expected=[r.severe_prevalence_percent]+[round(scale*r[c],dp) for c in ['estimate','lower95','upper95']]+[round(r[c],3) for c in ['percent_of_clean_regret','percent_clean_lower95','percent_clean_upper95']]
    assert np.allclose(x,expected,atol=1e-12),(row,x,expected)
check('K3 display units/precision',True)

k4=read('K4/support_summary.csv')
for m in ['camera','lidar','either']:
    check('K4 '+m+' counts/fractions',np.allclose(k4[m+'_share'],k4[m+'_supported']/k4.denominator,atol=1e-14))
check('K4 union bounds',((k4.either_supported>=k4[['camera_supported','lidar_supported']].max(axis=1))&(k4.either_supported<=k4.camera_supported+k4.lidar_supported)&(k4.either_supported<=k4.denominator)).all())
masked=k4[(k4.arm=='always_mask')|(k4.input_state=='all_cameras_masked_support_only')]
check('K4 all-camera removal',not masked.camera_supported.any() and np.array_equal(masked.lidar_supported,masked.either_supported))
actual=k4[(k4.seed_scope=='all_seeds')&(k4.input_state=='actual')]
for scope in ['reachable_candidates','oracle_cell']:
    q=actual[(actual.scope==scope)&(actual.architecture=='input')&(actual.arm=='keep_all')]
    check('K4 camera-only intervention preserves LiDAR '+scope,q.lidar_supported.nunique()==1 and q.denominator.nunique()==1)
rows=texrows('selected_support.tex');i=0
for policy in ['direct','soft']:
 for condition in ['clean','cam_s3']:
  for architecture in ['KF-D42','BF-D42']:
   for arm in ['keep_all','gate','always_mask']:
    q=actual[(actual.scope=='selected_cell')&(actual.policy==policy)&(actual.condition==condition)&(actual.architecture==architecture)&(actual.arm==arm)].iloc[0]
    got=values(' & '.join(rows[i].split(' & ')[3:]));expected=[round(100*q[m+'_share'],2) for m in ['camera','lidar','either']]
    assert np.allclose(got,expected,atol=1e-12);i+=1
check('K4 all selected-cell display rows',i==24)
rows=texrows('region_support.tex');i=0
for condition in ['clean','cam_s3']:
 for architecture,arm in [('input','keep_all'),('KF-D42','gate'),('BF-D42','gate'),('input','always_mask')]:
    if condition=='cam_s3' and arm=='always_mask':continue
    q=actual[(actual.condition==condition)&(actual.architecture==architecture)&(actual.arm==arm)]
    expected=[]
    for scope in ['reachable_candidates','oracle_cell']:
        r=q[q.scope==scope].iloc[0];expected.extend([round(100*r[m+'_share'],2) for m in ['camera','lidar','either']])
    assert np.allclose(values(' & '.join(rows[i].split(' & ')[2:])),expected,atol=1e-12);i+=1
check('K4 all candidate/oracle display rows',i==7)

k5=read('K5/validation_grid.csv');lock=json.loads((O/'K5/validation_lock_original.json').read_text())
check('K5 original grid digest',sha(O/'K5/validation_grid_original.csv')==lock['grid_sha256'])
check('K5 all thirty points and boundary trend',len(k5)==30 and all((np.diff(g.sort_values('buffer').regret)<0).all() for _,g in k5.groupby('tau')))
for row,(_,g) in zip(texrows('classical_validation.tex'),k5.groupby('tau')):
    got=values(row);expected=[float(g.tau.iloc[0])]+[round(x,5) for x in g.sort_values('buffer').regret]
    assert np.allclose(got,expected,atol=1e-12)
check('K5 displayed original values',True)

changed_tables={'decision_grounding.tex','replication_sensitivity.tex','always_act_strata.tex','prevalence.tex','region_support.tex','selected_support.tex','classical_validation.tex'}
unchanged=[]
for f in (M/'tables').glob('*.tex'):
    prior=old/'tables'/f.name
    if prior.exists() and f.name not in changed_tables:
        assert re.findall(num,f.read_text())==re.findall(num,prior.read_text()),f.name
        unchanged.append(f.name)
check('Prior table numerical token sequences unchanged',True,', '.join(unchanged))
check('Abstract unchanged',sha(M/'generated/abstract.tex')==sha(old/'generated/abstract.tex'))
check('Highlights unchanged',sha(M/'highlights.txt')==sha(old/'highlights.txt'))
abstract=re.sub(r'\\[A-Za-z]+(?:\{([^}]*)\})?',lambda m:m.group(1) or '',(M/'generated/abstract.tex').read_text()).replace('$','')
check('Abstract <=150 words',len(abstract.split())<=150,str(len(abstract.split())))
highlights=[line.lstrip('- ').strip() for line in (M/'highlights.txt').read_text().splitlines() if line.strip()]
check('3-5 highlights <=85 characters',3<=len(highlights)<=5 and all(len(x)<=85 for x in highlights),str(list(map(len,highlights))))
check('Generator summary source digest',sha(P/json.loads((O/'specification/plain_summary_constants.json').read_text())['scene_source'])==json.loads((O/'specification/plain_summary_constants.json').read_text())['scene_source_sha256'])
for r in csv.DictReader((O/'specification/plain_summary_sources.csv').open()):assert sha(P/r['source_file'])==r['source_sha256']
check('Generator summary all AST extraction sources',True)
for r in csv.DictReader((O/'report/new_number_sources.csv').open()):
    d=read(r['source'])
    for key,val in json.loads(r['lookup']).items():d=d[d[key]==val]
    assert len(d)==1 and abs(float(d.iloc[0][r['field']])-float(r['value']))<1e-12
check('Changed prose source lookups',True,'29 source-indexed values plus sensitivity sign/class counts and original F4 denominator conversion')

# Every numeric output cell retains exact row/field provenance; no computations here.
registry=[]
sources=['K1/settings.csv','K1/primary_sensitivity.csv','K1/descriptive_signs.csv','K1/compact_sensitivity_table.csv','K2/stratified_safety.csv','K2/population_support.csv','K3/prevalence_mixtures.csv','K4/support_summary.csv','K5/validation_grid.csv']
for source in sources:
    digest=sha(O/source)
    for i,r in enumerate(csv.DictReader((O/source).open()),2):
        for field,value in r.items():
            try:float(value)
            except (ValueError,TypeError):continue
            registry.append(dict(source=source,csv_line=i,field=field,value=value,source_sha256=digest))
with (O/'report/numerical_output_registry.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(registry[0]));w.writeheader();w.writerows(registry)
(O/'report/number_check_receipt.json').write_text(json.dumps(dict(status='PASS',checks=checks,registered_numeric_cells=len(registry),scope='Changed K1-K5 outputs, source/display values, original primary/table numerical preservation, abstract/highlights and generator-summary extraction; no scientific rerun',new_draws=0,neural_forwards=0,gpu_hours=0),indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=len(checks),registered_numeric_cells=len(registry))))
