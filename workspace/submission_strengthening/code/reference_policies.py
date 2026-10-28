"""Post-hoc R0/R1 construction/scoring; CPU NumPy, no draws or model imports.

The original training inventory and strict selector are explicit inputs. The
selector's function AST is loaded unchanged, without importing its ML helpers.
All files are written outside the immutable source root. No threshold fitting.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, io, json, time
from pathlib import Path
import numpy as np

MATERIAL_NAMES = ['sand', 'soil', 'gravel', 'rocky']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_selector(path):
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='choose']
    assert len(nodes)==1
    namespace={'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['choose']

def check_lock(plan,receipt):
    record=json.loads(Path(receipt).read_text())
    actual=digest(plan)
    assert actual in json.dumps(record), 'Plan digest does not match lock receipt'
    return actual

def make_priors(project,inventory,out,plan,lock):
    assert not (out/'prior_lock.json').exists(), 'Never overwrite a pinned prior'
    plan_hash=check_lock(plan,lock)
    with inventory.open() as f: rows=[r for r in csv.DictReader(f) if r['split']=='train']
    assert len(rows)==12000
    assert {(int(r['scene_id']),int(r['frame_id'])) for r in rows}=={(s,t) for s in range(3000) for t in range(4)}
    # Preserve original inventory ordering and inclusion, including invalid cases.
    write_csv(out/'training_inventory.csv',rows)
    utility_sum=np.zeros((96,96),np.float64);hazard_sum=np.zeros((96,96),np.int64)
    material_counts=np.zeros(4,np.int64);semantic_counts=np.zeros(6,np.int64)
    terrain_pixels=0;invalid_terrain=0;invalid_other=0;nonfinite_utility=0;nonfinite_hazard=0
    nontarget_hazard=0;shape_counts={};start=time.time()
    for i,row in enumerate(rows):
        path=project/row['path'];raw=path.read_bytes()
        assert len(raw)==int(row['bytes']) and hashlib.sha256(raw).hexdigest()==row['sha256'],str(path)
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            assert (int(z['scene_id']),int(z['frame_id']))==(int(row['scene_id']),int(row['frame_id']))
            u=z['gt_aff'];h=z['gt_haz'];sem=z['gt_sem'];mat=z['gt_mat']
            assert u.shape==h.shape==sem.shape==mat.shape==(96,96)
            shape_counts[str(u.dtype)]=shape_counts.get(str(u.dtype),0)+1
            nonfinite_utility+=int((~np.isfinite(u)).sum());nonfinite_hazard+=int((~np.isfinite(h)).sum())
            nontarget_hazard+=int((~np.isin(h,[0,1])).sum())
            utility_sum+=u.astype(np.float64);hazard_sum+=(h>0.5)
            terrain=sem==0;terrain_pixels+=int(terrain.sum())
            valid_material=(mat>=0)&(mat<4)
            invalid_terrain+=int((terrain&~valid_material).sum())
            invalid_other+=int((~terrain&~valid_material).sum())
            material_counts+=np.bincount(mat[terrain&valid_material].astype(np.int64),minlength=4)
            assert np.isin(sem,np.arange(6)).all()
            semantic_counts+=np.bincount(sem.ravel().astype(np.int64),minlength=6)
        if (i+1)%1000==0: print('training labels',i+1,flush=True)
    assert nonfinite_utility==nonfinite_hazard==nontarget_hazard==invalid_terrain==0
    prior=out/'training_prior_maps.npz'
    np.savez_compressed(prior,mean_A_train=utility_sum/12000,mean_h_train=hazard_sum.astype(np.float64)/12000,
                        training_frames=np.array(12000),material_counts=material_counts,semantic_counts=semantic_counts)
    majority=int(material_counts.argmax())
    material=dict(status='PINNED_TRAINING_ONLY',frames=12000,scenes=3000,source_population='exact original training inventory; no case filtering',
        true_terrain_rule='gt_sem == 0',material_names=MATERIAL_NAMES,counts=material_counts.tolist(),
        total_true_terrain_pixels=terrain_pixels,probabilities=(material_counts/terrain_pixels).tolist(),
        smoothing='none; no probabilistic log-loss reference requested',
        majority_class_id=majority,majority_class_name=MATERIAL_NAMES[majority],tie_rule='lowest class ID (np.argmax)',
        semantic_counts=semantic_counts.tolist(),invalid_material_on_true_terrain=invalid_terrain,
        invalid_material_elsewhere=invalid_other,nonfinite_utility_cells=nonfinite_utility,
        nonfinite_hazard_cells=nonfinite_hazard,nonbinary_hazard_cells=nontarget_hazard,
        utility_storage_dtypes=shape_counts,map_shape=[96,96],prior_map_sha256=digest(prior),
        training_inventory_sha256=digest(out/'training_inventory.csv'),plan_sha256=plan_hash)
    write_json(out/'training_material_prior.json',material)
    write_json(out/'prior_lock.json',dict(status='PINNED_BEFORE_REFERENCE_TEST_SCORING',unix_time=time.time(),
        elapsed_seconds=time.time()-start,plan_sha256=plan_hash,source_inventory=str(inventory),
        source_inventory_sha256=digest(inventory),files={p.name:digest(p) for p in [prior,out/'training_inventory.csv',out/'training_material_prior.json']},
        estimator='equal-frame per-cell float64 sum of original stored utility; binary hazard > 0.5; all 12000 original train frames',
        hash_verification='every streamed training archive verified against frozen inventory during same read'))
    print(json.dumps(material,indent=2))

def score_one(data,reference,choose,prior):
    u=data['gt_aff'].astype(np.float64);reach=data['gt_reach'];haz=data['gt_haz'];sem=data['gt_sem']
    feasible=np.isfinite(reach)&(reach>=0.05);n=int(feasible.sum())
    assert np.isfinite(u[feasible]).all()
    optimum=float(u[feasible].max()) if n else 0.0
    unsafe_truth=(haz>0.5)|(sem!=0)
    if optimum>0:
        oracle=choose(u,reach)[0];assert not unsafe_truth[oracle]
    common=dict(reference=reference,attempted=1,oracle=optimum,reachable_cells=n,
        nonfinite_reach=int((~np.isfinite(reach)).sum()),empty_reach=int(n==0))
    if reference=='R0':
        utility=float(np.sum(u[feasible],dtype=np.float64)/n) if n else 0.
        numerator=int(unsafe_truth[feasible].sum()) if n else 0
        denominator=n if n else 1
        common.update(regret=optimum-utility,unsafe=numerator/denominator,executed=int(n>0),abstained=int(n==0),
            unsafe_event='',expected_unsafe_numerator=numerator,expected_unsafe_denominator=denominator,
            selected_utility=utility,pick_row='',pick_col='',eligible_cells=n,tie_count='',empty_candidates=int(n==0),
            nonfinite_scores=0,nonfinite_hazard='',selection_kind='analytic_uniform_expectation',
            unsafe_measure_kind='expected_probability_no_realized_event')
    elif reference=='R1':
        score=prior['mean_A_train'];predhaz=prior['mean_h_train']
        pick,ties,eligible=choose(score,reach,predhaz)
        executed=int(pick is not None);utility=float(u[pick]) if executed else 0.
        unsafe=int(executed and unsafe_truth[pick])
        common.update(regret=optimum-utility,unsafe=unsafe,executed=executed,abstained=1-executed,
            unsafe_event=unsafe,expected_unsafe_numerator='',expected_unsafe_denominator='',
            selected_utility=utility,pick_row=-1 if pick is None else int(pick[0]),pick_col=-1 if pick is None else int(pick[1]),
            eligible_cells=eligible,tie_count=ties,empty_candidates=int(eligible==0),
            nonfinite_scores=int((~np.isfinite(score)).sum()),nonfinite_hazard=int((~np.isfinite(predhaz)).sum()),
            selection_kind='strict_direct_fixed_training_prior',unsafe_measure_kind='observed_deterministic_event')
    else: raise ValueError(reference)
    return common

def score_all(afast,out,selector):
    lock=json.loads((out/'prior_lock.json').read_text())
    for name,sha in lock['files'].items():assert digest(out/name)==sha
    choose=load_selector(selector)
    with np.load(out/'training_prior_maps.npz',allow_pickle=False) as z:prior={k:z[k] for k in ['mean_A_train','mean_h_train']}
    with (afast/'manifests/generated_files.csv').open() as f:inventory=list(csv.DictReader(f))
    assert len(inventory)==3328
    rows=[];source_dtype={}
    for i,r in enumerate(inventory):
        path=afast/r['path'];raw=path.read_bytes()
        assert len(raw)==int(r['bytes']) and hashlib.sha256(raw).hexdigest()==r['sha256'],str(path)
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            data={k:z[k] for k in ['gt_aff','gt_reach','gt_haz','gt_sem']}
            assert int(z['scene_id'])==int(r['scene_id']) and int(z['frame_id'])==int(r['frame_id'])
        assert data['gt_aff'].dtype==np.float32
        for reference in ['R0','R1']:
            row=score_one(data,reference,choose,prior)
            row.update(condition=r['condition'],scene_id=int(r['scene_id']),frame_id=int(r['frame_id']),
                       source_path=r['path'],source_sha256=r['sha256'])
            rows.append(row)
        if (i+1)%512==0:print('holdout scored',i+1,flush=True)
    write_csv(out/'reference_per_frame.csv',rows)
    write_json(out/'scoring_provenance.json',dict(status='POSTHOC_DESCRIPTIVE_REFERENCE_SCORING_COMPLETE',
        records=3328,reference_rows=len(rows),selector_source=str(selector),selector_sha256=digest(selector),
        prior_lock_sha256=digest(out/'prior_lock.json'),generated_inventory_sha256=digest(afast/'manifests/generated_files.csv'),
        output_sha256=digest(out/'reference_per_frame.csv'),new_model_forwards=0,new_random_draws=0,
        reference_camera_states='same deterministic policy in both camera states; stored once, not refitted or replicated as independent fits'))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['construct','score'])
    p.add_argument('--project',type=Path,required=True);p.add_argument('--afast',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path);p.add_argument('--lock',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    assert a.output.resolve()!=a.afast.resolve() and a.afast.resolve() not in a.output.resolve().parents
    if a.command=='construct':make_priors(a.project,a.afast/'manifests/train_validation_inventory.csv',a.output,a.plan,a.lock)
    else:score_all(a.afast,a.output,a.afast/'source/selectors_v5_original.py')

if __name__=='__main__':main()
