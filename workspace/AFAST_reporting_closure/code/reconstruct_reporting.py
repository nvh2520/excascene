"""CPU-only AFAST reporting closure from frozen scalar observations and saved draws.

No model imports, random generator, inference, training, or acquisition. Requires
NumPy only. The original primary checker is copied before its receipt is written.
Run: python reconstruct_reporting.py --source EXTRACTED_AFAST --output NEW_OUTPUT
"""
import argparse
import ast
import collections
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

SEEDS = (10, 11, 12)
CONDITIONS = ('clean', 'iid', 'all_s2', 'cam_s1', 'cam_s2', 'cam_s3', 'rough_iid')
METRICS = ('regret', 'unsafe', 'H_reachable_terrain')
POOL = 'fixed_one_third_iid_all_s2_rough'
WEIGHTING = 'metric per frame; equal fitted seeds; equal four slots within scene; equal scenes; no prediction ensemble'
Q = (Fraction(1, 40), Fraction(39, 40))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    assert rows
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def exact_quantile(numerators, denominator, q):
    ordered = np.sort(np.asarray(numerators, dtype=np.int64))
    position = q * (len(ordered) - 1)
    left = position.numerator // position.denominator
    weight = position - left
    right = min(left + 1, len(ordered) - 1)
    return ((1-weight)*int(ordered[left]) + weight*int(ordered[right])) / denominator


def esc(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def signed(value, digits=6):
    return f'{value:+.{digits}f}'


def ci(row, digits=6):
    return f"[{signed(row['display_lower95'], digits)}, {signed(row['display_upper95'], digits)}]"


def tabular(path, columns, headers, rows, size='small'):
    text = ['% Generated from scalar observations by reconstruct_reporting.py.', '\\'+size,
            r'\begin{tabular}{' + columns + '}', r'\toprule', ' & '.join(headers) + r' \\', r'\midrule']
    text += [' & '.join(map(str, row)) + r' \\' for row in rows]
    text += [r'\bottomrule', r'\end{tabular}', r'\normalsize']
    path.write_text('\n'.join(text)+'\n')


def longtable(path, columns, headers, rows, caption):
    text = ['% Generated; ordinary intervals are descriptive reporting-stage additions.', r'\begingroup\footnotesize',
            r'\setlength{\tabcolsep}{3pt}', r'\begin{longtable}{'+columns+'}',
            r'\caption{'+caption+r'}\\', r'\toprule', ' & '.join(headers)+r'\\', r'\midrule\endfirsthead',
            r'\toprule', ' & '.join(headers)+r'\\', r'\midrule\endhead', r'\midrule\endfoot', r'\bottomrule\endlastfoot']
    text += [' & '.join(map(str, row))+r'\\' for row in rows]
    text += [r'\end{longtable}', r'\endgroup']
    path.write_text('\n'.join(text)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', '--source-root', type=Path, required=True)
    p.add_argument('--output', '--output-root', type=Path, required=True)
    a = p.parse_args()
    source, out = a.source.resolve(), a.output.resolve()
    assert source != out and source not in out.parents, 'Outputs must be outside the frozen source tree'
    out.mkdir(parents=True, exist_ok=True)
    (out/'tables').mkdir(exist_ok=True)
    source_files = set()

    def read_source(rel):
        f = source/rel
        source_files.add(f)
        return f

    # Keep an exact input/output copy for the historical standalone checker.
    primary_copy = out/'primary_reconstruction'
    for rel in ('code/reconstruct_primary.py', 'code/statistical_core.py',
                'analysis/primary_scene_inputs.csv', 'analysis/primary_paired_frames.csv',
                'analysis/bootstrap_multiplicities.npz', 'analysis/primary_results.json'):
        dest = primary_copy/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(read_source(rel), dest)
    (primary_copy/'records').mkdir(exist_ok=True)
    checked = subprocess.run([sys.executable, str(primary_copy/'code/reconstruct_primary.py'),
                              '--root', str(primary_copy)], capture_output=True, text=True)
    (primary_copy/'checker_stdout.txt').write_text(checked.stdout)
    (primary_copy/'checker_stderr.txt').write_text(checked.stderr)
    assert checked.returncode == 0, checked.stderr
    primary = json.loads(read_source('analysis/primary_results.json').read_text())

    with np.load(read_source('analysis/bootstrap_multiplicities.npz')) as z:
        ids = {s: np.array(z[s+'_scene_ids'], dtype=np.int64) for s in ('nominal', 'rough')}
        matrices = {s: np.array(z[s], dtype=np.int64) for s in ids}
    for s in ids:
        assert matrices[s].shape == (20000, len(ids[s]))
        assert np.all(matrices[s] >= 0) and np.all(matrices[s].sum(1) == len(ids[s]))
    float_matrices = {s: m.astype(np.float64) for s, m in matrices.items()}

    # Arrays retain scene x seed x slot cardinality, and every paired source key.
    grouped = collections.defaultdict(dict)
    total_rows = 0
    for arch in ('BF-D', 'KF-D'):
        for seed in SEEDS:
            with read_source(f'metrics/{arch}_seed{seed}.csv').open() as f:
                for row in csv.DictReader(f):
                    key = (arch, int(row['mask']), row['condition'], row['policy'])
                    subkey = (int(row['scene_id']), seed, int(row['frame_id']))
                    assert subkey not in grouped[key]
                    assert row['architecture'] == arch and int(row['seed']) == seed
                    assert int(row['attempted']) == 1
                    assert int(row['unsafe']) <= int(row['executed'])
                    assert int(row['executed']) + int(row['abstained']) == 1
                    grouped[key][subkey] = row
                    total_rows += 1
    assert total_rows == 79872 and len(grouped) == 56
    arms, denominators, seed_condition_rows = {}, [], []
    tracked = ['attempted', 'unsafe', 'executed', 'abstained', 'empty_candidates', 'empty_reach',
               'exact_tie', 'reach_boundary', 'hazard_boundary', 'nonfinite_scores', 'nonfinite_reach',
               'nonfinite_hazard', 'nonfinite_height', 'n_reachable_terrain']
    for key, observations in grouped.items():
        arch, mask, condition, policy = key
        stratum = 'rough' if condition == 'rough_iid' else 'nominal'
        order = [(int(s), seed, slot) for s in ids[stratum] for seed in SEEDS for slot in range(4)]
        assert set(observations) == set(order)
        raw = [observations[k] for k in order]
        arm = {m: np.array([float(r[m]) for r in raw], dtype=np.float64).reshape(len(ids[stratum]), 3, 4)
               for m in METRICS+('oracle', 'global_oracle', 'reachable_cells', 'n_reachable_terrain')}
        assert all(np.isfinite(arm[m]).all() for m in METRICS)
        assert (arm['n_reachable_terrain'] > 0).all()
        for m in ('unsafe', 'attempted', 'executed', 'abstained', 'pick_row', 'pick_col'):
            arm[m] = np.array([int(r[m]) for r in raw], dtype=np.int64).reshape(len(ids[stratum]), 3, 4)
        arm['raw'], arm['stratum'] = raw, stratum
        arms[key] = arm
        counts = {f'{field}_count': sum(int(r[field]) for r in raw) for field in tracked}
        denom = dict(architecture=arch, mask=mask, condition=condition, policy=policy,
                     scene_count=len(ids[stratum]), distinct_scene_slots=4*len(ids[stratum]), fitted_seeds=3,
                     **counts, unsafe_per_attempt=counts['unsafe_count']/counts['attempted_count'],
                     execution_rate=counts['executed_count']/counts['attempted_count'],
                     unsafe_per_execution=None if counts['executed_count']==0 else counts['unsafe_count']/counts['executed_count'],
                     unsafe_per_execution_undefined=int(counts['executed_count']==0),
                     height_undefined_frames=sum(not r['H_reachable_terrain'] for r in raw),
                     regret_undefined_frames=sum(not r['regret'] for r in raw),
                     normalized_regret_undefined_frames=sum(not r['normalized_regret'] for r in raw),
                     oracle_zero_frames=sum(r['utility_stratum']=='zero' for r in raw),
                     oracle_low_frames=sum(r['utility_stratum']=='low' for r in raw),
                     oracle_positive_frames=sum(r['utility_stratum']=='positive' for r in raw),
                     height_support='true reachable terrain; per-frame MAE before seed/slot/scene averaging',
                     weighting=WEIGHTING,
                     source_key=f'{arch}/mask{mask}/{condition}/{policy}/scene_id/seed/frame_id')
        for m in METRICS:
            denom[m+'_mean'] = float(arm[m].mean(axis=1).mean(axis=1).mean())
        denominators.append(denom)
        for seed_index,seed in enumerate(SEEDS):
            subset=[r for r in raw if int(r['seed'])==seed]
            sr=dict(architecture=arch,seed=seed,mask=mask,condition=condition,policy=policy,
                scene_count=len(ids[stratum]),slots_per_scene=4,
                **{f'{field}_count':sum(int(r[field]) for r in subset) for field in tracked},
                normalized_regret_undefined_frames=sum(not r['normalized_regret'] for r in subset),
                height_undefined_frames=sum(not r['H_reachable_terrain'] for r in subset),
                evidence_class='DESCRIPTIVE_SINGLE_FIT_CONDITION_SUMMARY',weighting='equal slots then equal scenes; one fitted seed')
            sr.update({m+'_mean':float(arm[m][:,seed_index,:].mean(axis=1).mean()) for m in METRICS})
            sr['unsafe_per_attempt']=sr['unsafe_count']/sr['attempted_count']
            sr['execution_rate']=sr['executed_count']/sr['attempted_count']
            seed_condition_rows.append(sr)
    write_csv(out/'condition_denominators.csv', denominators)
    write_csv(out/'per_seed_condition_results_reconstructed.csv',seed_condition_rows)

    def contrast(group, metric, condition, policy):
        """group is (integer coefficient, architecture, mask) for paired arms."""
        stratum = 'rough' if condition == 'rough_iid' else 'nominal'
        n = len(ids[stratum])
        paired = [(coefficient, arms[(arch, mask, condition, policy)]) for coefficient, arch, mask in group]
        for _, arm in paired[1:]:
            for field in ('oracle', 'global_oracle', 'reachable_cells', 'n_reachable_terrain'):
                assert np.array_equal(arm[field], paired[0][1][field]), (condition, field)
        if metric == 'unsafe':
            difference = sum(coefficient*arm[metric] for coefficient, arm in paired)
            per_scene = difference.sum(axis=(1, 2), dtype=np.int64)
            numerator = matrices[stratum] @ per_scene
            denominator = n*12
            estimate = Fraction(int(per_scene.sum()), denominator)
            return dict(estimate=estimate, numerators=numerator, denominator=denominator,
                        draws=numerator/denominator, scene_values=per_scene/12,
                        unsafe_difference_events=int(per_scene.sum()))
        difference = sum(coefficient*arm[metric] for coefficient, arm in paired)
        scene_values = difference.mean(axis=1).mean(axis=1)
        return dict(estimate=float(scene_values.mean()),
                    draws=float_matrices[stratum]@scene_values/n, scene_values=scene_values)

    def record(kind, group, metric, condition, policy, architecture='', mask='', posthoc=False):
        conditions = ('iid', 'all_s2', 'rough_iid') if condition == POOL else (condition,)
        vals = [contrast(group, metric, cond, policy) for cond in conditions]
        if metric == 'unsafe':
            d = math.lcm(*(v['denominator'] for v in vals))
            nums = sum(v['numerators']*(d//v['denominator']) for v in vals)
            den = d*len(vals)
            estimate = sum((v['estimate'] for v in vals), Fraction())/len(vals)
            bounds = [exact_quantile(nums, den, q) for q in Q]
            exact = dict(estimate_exact=str(estimate), lower95_exact=str(bounds[0]), upper95_exact=str(bounds[1]),
                         bootstrap_integer_denominator=den)
        else:
            estimate = sum(v['estimate'] for v in vals)/len(vals)
            draws = sum(v['draws'] for v in vals)/len(vals)
            bounds = np.quantile(draws, [float(q) for q in Q], method='linear')
            exact = dict(estimate_exact='', lower95_exact='', upper95_exact='', bootstrap_integer_denominator='')
        scale = 100 if metric == 'unsafe' else 1000 if metric == 'H_reachable_terrain' else 1
        support = []
        for cond in conditions:
            for coefficient, arch, state in group:
                arm = arms[(arch, state, cond, policy)]
                support.append(dict(coefficient=coefficient, architecture=arch, mask=state, condition=cond,
                    attempts=int(arm['attempted'].sum()), executions=int(arm['executed'].sum()),
                    abstentions=int(arm['abstained'].sum()), unsafe_events=int(arm['unsafe'].sum()),
                    height_support_cells=int(arm['n_reachable_terrain'].sum()),
                    scenes=int(arm[metric].shape[0]), slots_per_scene=4, fitted_seeds=3))
        return dict(version='reporting_closure_v1', comparison=kind, architecture=architecture, mask=mask,
                    condition=condition, policy=policy, metric=metric, estimate=float(estimate),
                    lower95=float(bounds[0]), upper95=float(bounds[1]), **exact,
                    native_unit='probability difference' if metric=='unsafe' else 'm' if metric=='H_reachable_terrain' else 'raw utility regret',
                    display_unit='percentage points' if metric=='unsafe' else 'mm' if metric=='H_reachable_terrain' else 'raw utility regret',
                    display_scale=scale, display_estimate=float(estimate)*scale,
                    display_lower95=float(bounds[0])*scale, display_upper95=float(bounds[1])*scale,
                    interval_level=0.95, adjusted=False, quantiles='1/40;39/40', draws=20000,
                    evidence_class='POST_HOC_DESCRIPTIVE_INTERACTION' if posthoc else 'REPORTING_STAGE_DESCRIPTIVE_INTERVAL',
                    preregistered=False, multiplicity_controlled=False,
                    point_estimate_status='new post-hoc four-arm diagnostic' if posthoc else 'existing descriptive contrast; interval added at reporting stage',
                    coverage='approximate percentile coverage; environmental scenes conditional on three fitted seeds',
                    weighting=WEIGHTING, pool_weights='iid=1/3;all_s2=1/3;rough_iid=1/3' if condition==POOL else '1',
                    independent_scene_identities=192 if condition==POOL else len(vals[0]['scene_values']),
                    pairing_key='condition;policy;scene_id;seed;frame_id',
                    draw_source='analysis/bootstrap_multiplicities.npz:'+('nominal+rough; shared replicate index' if condition==POOL else 'rough' if condition=='rough_iid' else 'nominal'),
                    source_keys=json.dumps([dict(coefficient=c, architecture=b, mask=m, condition=list(conditions), policy=policy, seeds=list(SEEDS), frame_ids=[0,1,2,3]) for c,b,m in group],separators=(',',':')),
                    source_files=';'.join(f'metrics/{b}_seed{s}.csv' for b in sorted({b for _,b,_ in group}) for s in SEEDS),
                    denominators_json=json.dumps(support,separators=(',',':')),
                    undefined_frames=0, excluded_bundles=0,
                    support='paired common oracle/reach/true-terrain support verified; all four slots, all three seeds retained')

    secondary = []
    for mask in (0, 1):
        for condition in CONDITIONS+(POOL,):
            for policy in ('direct', 'soft'):
                for metric in METRICS:
                    secondary.append(record('BF-D minus KF-D', [(1,'BF-D',mask),(-1,'KF-D',mask)], metric, condition, policy, mask=mask))
    for architecture in ('BF-D','KF-D'):
        for condition in CONDITIONS:
            for policy in ('direct','soft'):
                for metric in METRICS:
                    secondary.append(record('camera masked minus intact', [(1,architecture,1),(-1,architecture,0)], metric, condition, policy, architecture=architecture))
    assert len(secondary)==180
    write_csv(out/'secondary_descriptive_intervals.csv',secondary)
    interactions = [record('(BF masked - BF intact) - (KF masked - KF intact)',
                    [(1,'BF-D',1),(-1,'BF-D',0),(-1,'KF-D',1),(1,'KF-D',0)], metric, condition, policy, posthoc=True)
                    for condition in CONDITIONS for policy in ('direct','soft') for metric in METRICS]
    write_csv(out/'posthoc_removal_interaction.csv',interactions)

    # Inventory agreement against the existing descriptive exports, without overwriting them.
    existing_arch = {(int(r['mask']),r['condition'],r['policy']):r for r in csv.DictReader(read_source('analysis/architecture_comparisons.csv').open())}
    existing_remove = {(r['architecture'],r['condition'],r['policy']):r for r in csv.DictReader(read_source('analysis/camera_removal_costs.csv').open())}
    errors = []
    for r in secondary:
        if r['comparison']=='BF-D minus KF-D':
            old = existing_arch[(r['mask'],r['condition'],r['policy'])][r['metric']]
        else:
            old = existing_remove[(r['architecture'],r['condition'],r['policy'])][r['metric']+'_effect']
        errors.append(abs(float(old)-r['estimate']))
    assert max(errors)<1e-14

    seed_rows=[]
    b0,b1=arms[('BF-D',0,'cam_s3','direct')],arms[('BF-D',1,'cam_s3','direct')]
    selections=[('all_three',SEEDS)]+[(f'seed_{s}',(s,)) for s in SEEDS]+[(f'leave_out_{s}',tuple(t for t in SEEDS if t!=s)) for s in SEEDS]
    for name,seeds in selections:
        ix=[SEEDS.index(s) for s in seeds]
        rd=b1['regret'][:,ix,:]-b0['regret'][:,ix,:]
        ud=b1['unsafe'][:,ix,:]-b0['unsafe'][:,ix,:]
        n=int(ud.size); u=Fraction(int(ud.sum()),n)
        seed_rows.append(dict(summary=name,seeds=';'.join(map(str,seeds)),scenes=128,slots_per_scene=4,
            raw_regret_effect=float(rd.mean(axis=1).mean(axis=1).mean()),unsafe_effect=float(u),unsafe_effect_exact=str(u),
            unsafe_difference_integer=int(ud.sum()),attempts_per_state=n,
            unsafe_intact=int(b0['unsafe'][:,ix,:].sum()),unsafe_masked=int(b1['unsafe'][:,ix,:].sum()),
            executions_intact=int(b0['executed'][:,ix,:].sum()),executions_masked=int(b1['executed'][:,ix,:].sum()),
            abstentions_intact=int(b0['abstained'][:,ix,:].sum()),abstentions_masked=int(b1['abstained'][:,ix,:].sum()),
            evidence_class='FROZEN_PRIMARY_SEED_SENSITIVITY; descriptive across fitted seeds',weighting=WEIGHTING))
    write_csv(out/'primary_per_seed_leave_one_out_reconstructed.csv',seed_rows)
    transitions=[]
    changed=(b0['pick_row']!=b1['pick_row']) | (b0['pick_col']!=b1['pick_col'])
    for name,seeds in [('all_three',SEEDS)]+[(f'seed_{s}',(s,)) for s in SEEDS]:
        ix=[SEEDS.index(s) for s in seeds];u0=b0['unsafe'][:,ix,:];u1=b1['unsafe'][:,ix,:]
        transitions.append(dict(summary=name,seeds=';'.join(map(str,seeds)),condition='cam_s3',policy='direct',architecture='BF-D',
            attempted_pairs=int(u0.size),unsafe_to_safe=int(((u0==1)&(u1==0)).sum()),safe_to_unsafe=int(((u0==0)&(u1==1)).sum()),
            unsafe_in_both=int(((u0==1)&(u1==1)).sum()),safe_in_both=int(((u0==0)&(u1==0)).sum()),
            selected_cells_changed=int(changed[:,ix,:].sum()),selected_cells_changed_fraction=float(changed[:,ix,:].mean()),
            abstentions_intact=int(b0['abstained'][:,ix,:].sum()),abstentions_masked=int(b1['abstained'][:,ix,:].sum()),
            source='metrics/BF-D_seed10.csv;metrics/BF-D_seed11.csv;metrics/BF-D_seed12.csv; saved pick_row/pick_col/unsafe',
            evidence_class='REPORTING_STAGE_DESCRIPTIVE_TRANSITIONS'))
    write_csv(out/'severe_BF_D_decision_transitions.csv',transitions)
    mapping=[r for r in secondary if r['comparison']=='camera masked minus intact' and r['architecture']=='BF-D' and r['policy']=='direct' and r['condition'] in ('clean','cam_s3')]
    write_csv(out/'mapping_decision_companions.csv',mapping)

    # Name classes by parsing the frozen simulator, never importing its model stack.
    scene_path=read_source('source/excascene_snapshot/kinafford/sim/scene.py')
    # Custody for the mixed likelihood + absolute-error objective labels below.
    read_source('source/excascene_snapshot/kinafford/losses.py')
    constants={}
    for node in ast.parse(scene_path.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('SEM_CLASSES','N_BEV_SEM','MAT_CLASSES'):
                    constants[target.id]=ast.literal_eval(node.value)
    semnames=constants['SEM_CLASSES'][:constants['N_BEV_SEM']];matnames=constants['MAT_CLASSES']
    write_json(out/'class_mapping.json',dict(semantic_BEV=semnames,material=matnames,
        image_only_excluded_from_BEV=constants['SEM_CLASSES'][constants['N_BEV_SEM']:],source=str(scene_path.relative_to(source)),
        source_sha256=digest(scene_path),confusion_orientation='rows truth, columns prediction',
        undefined_rule='IoU undefined only when union=0; truth-absent with false positives has IoU=0'))
    diagnostics=json.loads(read_source('analysis/pixel_diagnostics.json').read_text())
    class_rows=[];matrix_rows=[];hazard_rows=[]
    for d in diagnostics:
        key={k:d[k] for k in ('architecture','seed','mask','condition')}
        for task,names,matrixfield,classfield in [('semantic',semnames,'semantic_confusion','semantic_classes'),('material_true_terrain',matnames,'material_true_terrain_confusion','material_classes')]:
            c=np.asarray(d[matrixfield],dtype=np.int64)
            assert c.shape==(len(names),len(names))
            for i,name in enumerate(names):
                tp=int(c[i,i]);fp=int(c[:,i].sum())-tp;fn=int(c[i].sum())-tp;union=tp+fp+fn
                saved=d[classfield][i]
                assert (tp,fp,fn,union)==tuple(saved[k] for k in ('tp','fp','fn','union'))
                iou=tp/union if union else None
                assert iou==saved['iou']
                saved['class_name']=name
                class_rows.append(dict(**key,task=task,class_id=i,class_name=name,tp=tp,fp=fp,fn=fn,
                    truth_support=int(c[i].sum()),predicted_support=int(c[:,i].sum()),union=union,iou=iou,
                    undefined=int(union==0),weighting=d['pixel_weighting'],evidence_class='EXISTING_PIXEL_DIAGNOSTIC_WITH_CLASS_NAMES'))
                for j,predname in enumerate(names):
                    matrix_rows.append(dict(**key,task=task,truth_class_id=i,truth_class_name=name,
                        predicted_class_id=j,predicted_class_name=predname,count=int(c[i,j])))
        tp,fp,fn,tn=(d['hazard_'+k] for k in ('tp','fp','fn','tn'))
        assert tp+fp==d['hazard_precision_denominator'] and tp+fn==d['hazard_recall_denominator']
        hazard_rows.append(dict(**key,**{k:v for k,v in d.items() if k.startswith('hazard_') or k in ('material_terrain_cells','utility_absolute_sum','utility_cells','utility_mae')},
            hazard_precision_undefined=int(tp+fp==0),hazard_recall_undefined=int(tp+fn==0),
            total_hazard_cells=tp+fp+fn+tn,evidence_class='EXISTING_POOLED_PIXEL_DIAGNOSTIC'))
    write_json(out/'pixel_diagnostics_named.json',diagnostics)
    write_csv(out/'pixel_class_diagnostics.csv',class_rows)
    write_csv(out/'pixel_confusion_matrices_long.csv',matrix_rows)
    write_csv(out/'hazard_and_material_support.csv',hazard_rows)

    trajectory=[]
    for architecture in ('BF-D','KF-D'):
        for seed in SEEDS:
            rel=f'checkpoints/{architecture}/seed{seed}/history.json'
            history=json.loads(read_source(rel).read_text())
            assert [e['epoch'] for e in history]==list(range(1,15))
            for e in history:
                hc=e['hazard_counts']
                row=dict(architecture=architecture,seed=seed,epoch=e['epoch'],source=rel,
                    validation_height_absolute_sum=e['height_absolute_sum'],validation_height_cells=e['height_cells'],
                    validation_height_mae_m=e['height_absolute_sum']/e['height_cells'],
                    validation_utility_absolute_sum=e['utility_absolute_sum'],validation_utility_cells=e['utility_cells'],
                    validation_utility_mae=e['utility_absolute_sum']/e['utility_cells'],
                    validation_height_support='ground-truth reach >=0.05 and semantic terrain; pooled cells',
                    validation_utility_support='all BEV cells; pooled cells',
                    validation_hazard_precision=hc['tp']/(hc['tp']+hc['fp']) if hc['tp']+hc['fp'] else None,
                    validation_hazard_recall=hc['tp']/(hc['tp']+hc['fn']) if hc['tp']+hc['fn'] else None,
                    validation_hazard_precision_denominator=hc['tp']+hc['fp'],validation_hazard_recall_denominator=hc['tp']+hc['fn'],
                    training_seconds=e['training_seconds'],validation_seconds=e['validation_seconds'],learning_rate=e['lr'],
                    semantic_confusion_json=json.dumps(e['semantic_confusion'],separators=(',',':')),
                    evidence_class='SAVED_DEVELOPMENT_OPTIMIZATION_DIAGNOSTIC; no convergence inference')
                for phase in ('train','validation'):
                    row.update({f'{phase}_{k}':v for k,v in e[phase+'_losses'].items()})
                row.update(e['counts'])
                trajectory.append(row)
    write_csv(out/'optimization_epoch_trajectories.csv',trajectory)

    # Paper fragments: all numbers below come from reconstructed arrays/tables.
    tables=out/'tables'
    pr=primary['endpoints'][0];pu=primary['endpoints'][1]
    tabular(tables/'afast_primary.tex','lll',
        ['Severe BF-D masked $-$ intact','Estimate',r'Adjusted 97.5\% marginal interval'],
        [['Raw regret',f"{pr['estimate']:+.6f}",f"[{pr['adjusted97_5'][0]:+.6f}, {pr['adjusted97_5'][1]:+.6f}]"],
         ['Unsafe (percentage points)',f"{100*pu['estimate']:+.6f}",f"[{100*pu['adjusted97_5'][0]:+.6f}, {100*pu['adjusted97_5'][1]:+.6f}]"],
         ['Unsafe count: intact / masked',f"{pu['unsafe_intact_count']} / {pu['unsafe_masked_count']}",f"{pu['attempts_per_state']:,} attempts per state"],
         ['Executions; abstentions per state',f"{pu['executions_intact']:,}; 0",'Both endpoints INCONCLUSIVE']])
    tradeoff=[r for r in secondary if r['comparison']=='BF-D minus KF-D' and r['mask']==0 and r['policy']=='direct' and r['condition'] in ('clean','cam_s3')]
    metriclabel={'regret':'Regret','unsafe':'Unsafe (pp)','H_reachable_terrain':'Height (mm)'}
    tabular(tables/'afast_architecture_tradeoff.tex','llrrl',
        ['Condition / endpoint','BF-D absolute','KF-D absolute','Difference',r'Ordinary 95\% interval'],
        [[('Clean' if r['condition']=='clean' else 'Severe')+' / '+(r'Unsafe (\%; pp)' if r['metric']=='unsafe' else metriclabel[r['metric']]),
          f"{float(arms[('BF-D',0,r['condition'],'direct')][r['metric']].mean())*r['display_scale']:.6f}",
          f"{float(arms[('KF-D',0,r['condition'],'direct')][r['metric']].mean())*r['display_scale']:.6f}",
          signed(r['display_estimate']),ci(r)] for r in tradeoff],size='footnotesize')
    # Percentages in absolute columns, percentage points in contrast/CI columns.
    tabular(tables/'afast_primary_seed_sensitivity.tex','lrrrl',
        ['Fitted seeds','Regret change','Unsafe events','Attempts/state','Unsafe change (pp)'],
        [[esc(r['seeds']),signed(r['raw_regret_effect']),str(r['unsafe_difference_integer']),str(r['attempts_per_state']),signed(100*r['unsafe_effect'])] for r in seed_rows],size='footnotesize')
    # Complete 180-row inventory, grouped compactly as 60 rows x 3 metrics.
    wide=[]
    for index in range(0,len(secondary),3):
        rr=secondary[index:index+3];r=rr[0]
        name=('BF$-$KF '+('intact' if r['mask']==0 else 'masked')) if r['comparison']=='BF-D minus KF-D' else r['architecture']+' removal'
        cond='1/3 pool' if r['condition']==POOL else esc(r['condition'])
        wide.append([name,cond,r['policy']]+[signed(t['display_estimate'],4)+' '+ci(t,4) for t in rr])
    longtable(tables/'afast_all_descriptive.tex','lllp{3.1cm}p{3.1cm}p{3.1cm}',
        ['Contrast','Condition','Policy',r'Regret [95\% CI]',r'Unsafe pp [95\% CI]',r'Height mm [95\% CI]'],wide,
        r'Complete AFAST secondary inventory: ordinary 95\% descriptive reporting-stage intervals; no multiplicity control. The fixed pool assigns one third to iid, all\_s2 and rough\_iid in every saved draw.')
    postrows=[]
    for index in range(0,len(interactions),3):
        rr=interactions[index:index+3];r=rr[0]
        postrows.append([esc(r['condition']),r['policy']]+[signed(t['display_estimate'],4)+' '+ci(t,4) for t in rr])
    longtable(tables/'afast_posthoc.tex','llp{3.6cm}p{3.6cm}p{3.6cm}',
        ['Condition','Policy',r'Regret [95\% CI]',r'Unsafe pp [95\% CI]',r'Height mm [95\% CI]'],postrows,
        r'Post-hoc four-arm removal interaction: (BF masked $-$ BF intact) $-$ (KF masked $-$ KF intact). Ordinary descriptive 95\% intervals; neither endpoint is a primary interaction test.')
    tabular(tables/'afast_mapping_transitions.tex','llrl',
        ['Condition','BF-D removal endpoint','Effect',r'Ordinary 95\% interval'],
        [[esc(r['condition']),metriclabel[r['metric']],signed(r['display_estimate']),ci(r)] for r in mapping],size='footnotesize')
    tr=transitions[0]
    tabular(tables/'afast_decision_transitions.tex','lr', ['Severe BF-D direct transition','Seed--slot decisions'],
        [['Unsafe to safe',tr['unsafe_to_safe']],['Safe to unsafe',tr['safe_to_unsafe']],['Unsafe in both',tr['unsafe_in_both']],['Safe in both',tr['safe_in_both']],['Selected cell changed',f"{tr['selected_cells_changed']} / {tr['attempted_pairs']}"]])
    selected=[d for d in diagnostics if d['seed']=='all_three' and d['mask']==0 and d['condition'] in ('clean','cam_s3')]
    pixrows=[]
    for d in selected:
        for task,field in [('semantic','semantic_classes'),('material','material_classes')]:
            for c in d[field]:
                pixrows.append([d['architecture'],esc(d['condition']),task,c['class_name'],c['truth_support'],c['predicted_support'],c['union'],'undefined' if c['iou'] is None else f"{c['iou']:.5f}"])
    longtable(tables/'afast_pixel_diagnostics.tex','llllrrrr',
        ['Model','Condition','Task','Class','Truth','Predicted','Union','IoU'],pixrows,
        'Intact clean/severe pooled-three-seed class diagnostics. All 112 condition/mask/seed summaries and confusion matrices are supplied as machine-readable supplemental data; truth-absent with false positives gives zero IoU, not undefined.')
    matrixtex=[]
    for d in selected:
        matrixtex += [r'\noindent\textbf{'+d['architecture']+' '+esc(d['condition'])+r', intact; three fitted seeds pooled.}\par',r'\smallskip']
        for names,field,caption in [(semnames,'semantic_confusion','Semantic'),(matnames,'material_true_terrain_confusion','Material on true terrain')]:
            matrixtex += [r'\noindent '+caption+r' confusion (rows truth, columns prediction).\par',r'\begingroup\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{l'+'r'*len(names)+'}',r'\toprule',' & '+' & '.join(names)+r'\\',r'\midrule']
            matrixtex += [names[i]+' & '+' & '.join(map(str,row))+r'\\' for i,row in enumerate(d[field])]
            matrixtex += [r'\bottomrule\end{tabular}\endgroup\par\medskip']
    (tables/'afast_confusion_matrices.tex').write_text('\n'.join(matrixtex)+'\n')
    tabular(tables/'afast_hazard_support.tex','llrrrrrr',
        ['Model','Condition','TP','FP','FN','TN','Precision','Recall'],
        [[d['architecture'],esc(d['condition']),d['hazard_tp'],d['hazard_fp'],d['hazard_fn'],d['hazard_tn'],
          f"{d['hazard_precision']:.5f}",f"{d['hazard_recall']:.5f}"] for d in selected],size='scriptsize')
    # Mixed likelihood + L1 objectives are not the absolute-error metrics.
    optrows=[]
    for r in trajectory:
        if r['epoch'] in (1,14):
            optrows.append([r['architecture'],r['seed'],r['epoch'],f"{1000*r['validation_height_mae_m']:.2f}",f"{r['validation_utility_mae']:.5f}",
                f"{r['validation_H']:.3f}",f"{r['validation_aff']:.3f}",f"{r['validation_sem']:.3f}",f"{r['validation_mat']:.3f}",f"{r['validation_haz']:.3f}"])
    longtable(tables/'afast_optimization.tex','lrrrrrrrrr',
        ['Model','Seed','Epoch','Height mm','Utility MAE','H loss','Aff loss','Sem CE','Mat CE','Haz BCE'],optrows,
        'Saved development-validation absolute errors and objective components at epochs 1 and 14. H and Aff losses combine Gaussian likelihood with weighted absolute-error terms; they are not pure NLL or MAE. All 84 epoch records, training/validation components and denominators are supplied separately; reaching epoch 14 does not establish convergence.')

    provenance=[]
    for f in sorted(source_files):
        provenance.append(dict(path=str(f.relative_to(source)),bytes=f.stat().st_size,sha256=digest(f)))
    write_json(out/'scalar_source_provenance.json',dict(source_root_name=source.name,files=provenance,
        source_hash_scope='Only scalar reporting inputs and copied checker; no checkpoints or model binaries rehashed',
        output_version='reporting_closure_v1',new_random_draws=0,model_imports=0,inference_forwards=0,training_updates=0,
        draw_identity='Saved nominal and rough multiplicities from analysis/bootstrap_multiplicities.npz',
        source_key_schema='architecture;seed;mask;condition;policy;scene_id;frame_id',
        weighting=WEIGHTING,pool='fixed one third iid/all_s2/rough_iid; nominal conditions share draws',
        code_sha256=digest(Path(__file__).resolve())))
    owned_csv_names=('condition_denominators.csv','per_seed_condition_results_reconstructed.csv',
        'secondary_descriptive_intervals.csv','posthoc_removal_interaction.csv',
        'primary_per_seed_leave_one_out_reconstructed.csv','severe_BF_D_decision_transitions.csv',
        'mapping_decision_companions.csv','pixel_class_diagnostics.csv','pixel_confusion_matrices_long.csv',
        'hazard_and_material_support.csv','optimization_epoch_trajectories.csv')
    outputs={name:digest(out/name) for name in owned_csv_names}
    summary=dict(status='SCALAR_REPORTING_RECONSTRUCTED',metric_rows=total_rows,logical_neural_forwards_reused=39936,new_neural_forwards=0,
        new_bootstrap_draws=0,saved_draws_per_stratum=20000,secondary_rows=len(secondary),posthoc_rows=len(interactions),
        diagnostic_groups=len(diagnostics),class_rows=len(class_rows),matrix_cells=len(matrix_rows),optimization_epochs=len(trajectory),
        existing_descriptive_max_abs_difference=max(errors),primary=primary['endpoints'],joint_lower_loss=primary['joint_lower_loss'],
        primary_checker=json.loads((primary_copy/'records/scalar_reconstruction.json').read_text()),
        severe_BF_D_transitions=transitions[0],output_csv_sha256=outputs,
        original_reviewer_script_available=False,reviewer_note_use='Cross-check targets only; actual observations and saved draws are calculation inputs',
        limitations='Intervals conditional on fitted seeds; approximate scene-percentile coverage. No equivalence, retraining uncertainty or physical-safety inference.')
    write_json(out/'scalar_reconstruction_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','metric_rows','secondary_rows','posthoc_rows','diagnostic_groups','optimization_epochs','existing_descriptive_max_abs_difference')},indent=2))


if __name__=='__main__':
    main()
