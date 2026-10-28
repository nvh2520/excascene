"""One targeted numerical/unit/sign/count and manuscript-source check.

Reads existing observations and completed reporting exports only. No resampling,
model imports, training, inference, or historical experimental replay.
"""
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'AFAST_reporting_closure'
R=ROOT/'external_validation_run_EV1_AFAST'
P=ROOT/'paper_v5_2'

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def near(a,b,tol=1e-12):assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=tol),(a,b)
def main():
    checks=[]
    s=json.loads((C/'scalar_reconstruction_summary.json').read_text())
    old=json.loads((R/'analysis/primary_results.json').read_text())
    assert s['primary']==old['endpoints']
    assert s['primary_checker']['status']=='SCALAR_RECONSTRUCTION_MATCHED'
    assert s['joint_lower_loss'] is False
    assert s['new_neural_forwards']==s['new_bootstrap_draws']==0
    near(s['primary'][0]['estimate'],-0.00683635302705928)
    assert s['primary'][0]['adjusted97_5']==[-0.017448174364284343,0.003264970965968443]
    assert s['primary'][1]['estimate_exact']=='-1/1536'
    assert s['primary'][1]['adjusted97_5_exact']==['-17/1536','5/512']
    checks.append('Frozen primary values, exact event fractions, adjusted levels and inconclusive status preserved')
    transitions=Counter(); seed_changes={}; mean_effects=[]
    for seed in [10,11,12]:
        data=[r for r in rows(R/f'metrics/BF-D_seed{seed}.csv') if r['condition']=='cam_s3' and r['policy']=='direct']
        a={(r['scene_id'],r['frame_id']):r for r in data if r['mask']=='0'}
        b={(r['scene_id'],r['frame_id']):r for r in data if r['mask']=='1'}
        assert len(a)==len(b)==512 and a.keys()==b.keys()
        seed_changes[seed]=sum(int(b[k]['unsafe'])-int(a[k]['unsafe']) for k in a)
        mean_effects.append(sum(float(b[k]['regret'])-float(a[k]['regret']) for k in a)/512)
        for k in a:
            x,y=a[k],b[k]
            assert x['attempted']==y['attempted']==x['executed']==y['executed']=='1'
            assert x['abstained']==y['abstained']=='0'
            transitions[(int(x['unsafe']),int(y['unsafe']))]+=1
            transitions['picks_changed']+=(x['pick_row'],x['pick_col'])!=(y['pick_row'],y['pick_col'])
    assert seed_changes=={10:2,11:-8,12:5}
    assert all(x<0 for x in mean_effects)
    near(sum(mean_effects)/3,s['primary'][0]['estimate'])
    assert transitions==Counter({(1,0):22,(0,1):21,(1,1):97,(0,0):1396,'picks_changed':576})
    checks.append('Original BF severe observations independently recover transitions, changed picks, attempts and all seed signs')
    secondary=rows(C/'secondary_descriptive_intervals.csv'); post=rows(C/'posthoc_removal_interaction.csv')
    assert len(secondary)==180 and len(post)==42
    condition_names={'clean','iid','all_s2','cam_s1','cam_s2','cam_s3','rough_iid'}
    assert {r['condition'] for r in secondary if 'fixed' not in r['condition']}==condition_names
    assert {r['policy'] for r in secondary}=={'direct','soft'}
    for r in secondary+post:
        assert r['interval_level']=='0.95' and r['adjusted']=='False'
        assert r['preregistered']==r['multiplicity_controlled']=='False'
        assert r['draws']=='20000' and r['quantiles']=='1/40;39/40'
        scale={'regret':1,'unsafe':100,'H_reachable_terrain':1000}[r['metric']]
        near(r['display_scale'],scale)
        for key in ['estimate','lower95','upper95']:near(r['display_'+key],float(r[key])*scale)
        if r['metric']=='unsafe':
            assert r['display_unit']=='percentage points'
            for key in ['estimate','lower95','upper95']:near(r[key],Fraction(r[key+'_exact']))
        assert int(r['undefined_frames'])==int(r['excluded_bundles'])==0
    key=lambda r:(r['comparison'],r['architecture'],r['mask'],r['condition'],r['policy'],r['metric'])
    assert len({key(r) for r in secondary})==180
    for cond,metric,target in [('clean','regret',(0.022404747593100186,0.004614738780583138,0.041191679489080944)),('clean','unsafe',(2.799479166666667,0.6510416666666667,5.078125)),('cam_s3','regret',(-0.05608708,-0.07912239,-0.03495099)),('cam_s3','unsafe',(-7.486979,-10.221354,-5.013021)),('cam_s3','H_reachable_terrain',(30.96270,21.60110,40.41894))]:
        r=next(r for r in secondary if r['comparison']=='BF-D minus KF-D' and r['mask']=='0' and r['policy']=='direct' and r['condition']==cond and r['metric']==metric)
        for name,v in zip(['estimate','lower95','upper95'],target):near(r['display_'+name],v,1e-5)
    pr=next(r for r in post if r['condition']=='cam_s3' and r['policy']=='direct' and r['metric']=='regret')
    pu=next(r for r in post if r['condition']=='cam_s3' and r['policy']=='direct' and r['metric']=='unsafe')
    assert float(pr['lower95'])<0<float(pr['upper95']) and Fraction(pu['estimate_exact'])==Fraction(55,1536)
    assert float(pu['lower95'])>0
    for r in post:assert r['evidence_class']=='POST_HOC_DESCRIPTIVE_INTERACTION'
    checks.append('Complete descriptive inventory, post-hoc classification, units and paired interval signs checked')
    for r in rows(C/'pixel_class_diagnostics.csv'):
        if int(r['union'])==0:assert r['iou'] in ['', 'None', 'nan']
        else:
            near(r['iou'],int(r['tp'])/int(r['union']))
            if int(r['truth_support'])==0:assert float(r['iou'])==0
    checks.append('All class IoUs obey true support/false-positive/zero-union rules')
    assert sha(P/'numbers.tex')==sha(ROOT/'paper_v5_1/numbers.tex')
    for f in (ROOT/'paper_v5_1/tables').glob('*.tex'):assert sha(f)==sha(P/'tables'/f.name),f.name
    for f in (C/'tables').glob('*.tex'):assert sha(f)==sha(P/'tables'/f.name),f.name
    text='\n'.join(p.read_text() for p in list(P.glob('*.tex'))+list((P/'sections').glob('*.tex')))
    for word in ['BEVFusion-Exca','INCONCLUSIVE','97.5','FP32','BF16','CVAT','MacBook Pro M1','NOT_TESTED']:
        assert word.casefold().replace('_',' ') in text.casefold().replace('_',' '),word
    assert 'task-matched external fusion NOT_RUN' not in text
    checks.append('Original numerical macros/tables preserved; AFAST table copies, study labels, precision distinction and author correction present')
    registry=json.loads((C/'publication_numbers_v5_2.json').read_text())
    original_registry=json.loads((ROOT/'revision_v5/readiness_v5_1/publication_numbers.json').read_text())
    assert registry['records'][:314]==original_registry['records']
    assert rows(C/'claim_evidence_v5_2.csv')[:52]==rows(ROOT/'revision_v5/manuscript_closure/claim_evidence_publication.csv')
    checks.append('314 historical publication records and 52 original claim rows remain unchanged prefixes')
    receipt={'status':'TARGETED_CHECK_PASSED','checks':checks,'historical_experimental_replay':False,
             'new_draws':0,'new_neural_forwards':0,'scope':'Only requested reporting changes; visual/build checks are separate.'}
    (C/'records/targeted_check.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
