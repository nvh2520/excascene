"""Render the bounded material report from completed saved-data diagnostics."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S = ROOT/'submission_strengthening'
D = S/'material/complete_v1'
rows = json.loads((D/'full_confusions_and_support.json').read_text())
prior = json.loads((S/'references/training_material_prior.json').read_text())
receipt = json.loads((D/'diagnostic_receipt.json').read_text())
classes = list(csv.DictReader((D/'class_metrics.csv').open()))
hazards = list(csv.DictReader((D/'hazard_head_metrics.csv').open()))
conditions = ['clean','iid','all_s2','cam_s1','cam_s2','cam_s3','rough_iid']


def get(architecture, condition, mask=0, stratum='full_official_support'):
    seed = 'not_applicable' if architecture == 'TRAIN_MAJORITY' else ('all_three' if stratum == 'full_official_support' else 'pooled_three_fitted_seeds')
    mask = 'not_applicable' if architecture == 'TRAIN_MAJORITY' else mask
    return next(r for r in rows if r['architecture']==architecture and r['condition']==condition and r['mask']==mask and r['seed']==seed and r['stratum']==stratum and r['task']=='material')


def table(headers, data):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in data])


train_table = table(['Material ID/name','Training terrain pixels','Empirical training prior'],
    [[f'{i} / {name}',f'{count:,}',f'{100*prob:.4f}%'] for i,(name,count,prob) in enumerate(zip(prior['material_names'],prior['counts'],prior['probabilities']))])
all_table = []
for condition in conditions:
    for mask in [0,1]:
        bf,kf,base=(get(a,condition,mask) for a in ['BF-D','KF-D','TRAIN_MAJORITY'])
        all_table.append([condition, 'intact' if not mask else 'camera-masked',
            f"{100*bf['accuracy']:.3f}% / {bf['macro_iou']:.5f}",
            f"{100*kf['accuracy']:.3f}% / {kf['macro_iou']:.5f}",
            f"{100*base['accuracy']:.3f}% / {base['macro_iou']:.5f}"])
all_table = table(['Condition','Model camera state','BF-D accuracy / macro IoU','KF-D accuracy / macro IoU','Train-soil accuracy / macro IoU'], all_table)
support_table = []
for stratum in ['projected_depth_covered_terrain','not_projected_depth_covered_terrain']:
    for mask in [0,1]:
        bf,kf,base=(get(a,'clean',mask,stratum) for a in ['BF-D','KF-D','TRAIN_MAJORITY'])
        support_table.append(['covered' if stratum.startswith('projected') else 'not covered', f"{base['cells']:,}",
            'intact' if not mask else 'camera-masked',f"{100*bf['accuracy']:.3f}% / {bf['macro_iou']:.5f}",
            f"{100*kf['accuracy']:.3f}% / {kf['macro_iou']:.5f}",f"{100*base['accuracy']:.3f}% / {base['macro_iou']:.5f}"])
support_table = table(['Approximate depth coverage','Distinct terrain cells','Model state','BF-D accuracy / macro IoU','KF-D accuracy / macro IoU','Train-soil accuracy / macro IoU'], support_table)
rare_table = []
for condition in ['clean','cam_s3','rough_iid']:
    for architecture in ['BF-D','KF-D']:
        for name in ['person','obstacle']:
            r=next(r for r in classes if r['architecture']==architecture and r['seed']=='all_three' and r['mask']=='0' and r['condition']==condition and r['task']=='semantic' and r['class_name']==name)
            rare_table.append([condition,architecture,name,r['truth_support'],r['predicted_support'],r['tp'],
                'undefined (0 predictions)' if not r['precision'] else f"{float(r['precision']):.4f}",f"{float(r['recall']):.4f}",f"{float(r['iou']):.5f}"])
rare_table=table(['Condition','Model','Class','Truth pixels','Predicted pixels','TP','Precision','Recall','IoU'],rare_table)
haz_table=[]
for r in hazards:
    if r['seed']=='all_three' and r['mask']=='0' and r['condition'] in ['clean','cam_s3','rough_iid']:
        haz_table.append([r['condition'],r['architecture'],f"{r['tp']}/{r['precision_denominator']}",f"{float(r['precision']):.4f}",f"{r['tp']}/{r['recall_denominator']}",f"{float(r['recall']):.4f}"])
haz_table=table(['Condition','Model','Hazard precision counts','Precision','Hazard recall counts','Recall'],haz_table)
cov=receipt['coverage']
report=f'''# Material and camera-support diagnostics

The existing six 14-epoch fits show weak material and rare-class predictions. A training-selected soil-only classifier has slightly higher full-map material accuracy than either intact model on CLEAN and rough conditions, while both models have higher material macro IoU. This is a limitation of these saved checkpoints and their task-specific evidence, not a diagnosis of class unlearnability, architecture failure, undertraining or sensor dominance.

The analysis is **post-hoc and descriptive**, following the already-inspected AFAST holdout and the locked descriptive plan. It adds no primary endpoint, neural execution, data generation, annotation or bootstrap draw. Material/semantic diagnostics pool pixels; they are not the frame/scene estimands of the primary families. All conditions, camera states and individual fitted seeds are retained in the machine-readable outputs.

## Training-only reference and label checks

The exact original 12,000 retained training frames (3,000 scenes, including historically retained invalid cases) contain **{prior['total_true_terrain_pixels']:,} true-terrain pixels**. The streaming construction verified each file against the original training inventory and pinned the prior before new material-reference test calculations. Material ID 1, **soil**, is the training majority; there was no test-selected class or support. The pinned prior contains unsmoothed empirical probabilities; no probability smoothing is used.

{train_table}

There were zero invalid material IDs on terrain or elsewhere, zero nonfinite utility/hazard labels and zero nonbinary hazard labels in that training stream. These checks do not filter any historical training frame. The material reference uses the unweighted count of true-terrain pixels, not the material-loss visibility weights.

The [narrow source review](material/source_review.md) found a consistent material ID order and canonical U-frame target convention, direct integer label loading without spatial relabelling, four material logits, a material cross-entropy term (coefficient 0.5) on true terrain, and a separate hazard head/loss. Saved material losses are present and finite for all 84 existing epoch records. The renderer uses material-specific albedo/texture through the same `scene.material_at` as the oracle. Material-dependent camera information exists in the implementation; complete target-cell observation does not follow. No concrete label, transform, omitted-head or omitted-loss defect was identified by this bounded check.

## Complete condition/camera inventory

Each nominal condition has {get('TRAIN_MAJORITY','clean')['cells']:,} distinct terrain cells across 512 frames; rough has {get('TRAIN_MAJORITY','rough_iid')['cells']:,} across 256 frames. Model pooled counts include the same support three times for the three separately fitted seeds. The train-majority reference is one deterministic policy; its repeated displayed value across camera states does not create independent replicates. Macro IoU includes every class with nonzero union; only zero union is undefined. A truth-absent class with false positives has IoU zero.

{all_table}

On intact CLEAN, BF-D/KF-D predict soil on {100*get('BF-D','clean')['prediction_counts'][1]/get('BF-D','clean')['cells']:.3f}% / {100*get('KF-D','clean')['prediction_counts'][1]/get('KF-D','clean')['cells']:.3f}% of terrain cells. Their accuracies are {100*get('BF-D','clean')['accuracy']:.4f}% / {100*get('KF-D','clean')['accuracy']:.4f}%, versus {100*get('TRAIN_MAJORITY','clean')['accuracy']:.4f}% for the **training-selected** soil class. This happens to equal the reviewer’s earlier test-soil arithmetic; the new reference’s class was selected exclusively from training. Macro IoU gives a different ranking and is retained rather than suppressed.

## CLEAN camera-support diagnostic and its limit

Exact raw `hits_B` separated by camera/pixel and image-material identities were not serialized. Exact camera-material observation is therefore **NOT RECOVERABLE from the retained arrays** without additional generation, which was not performed for this diagnostic. The implemented diagnostic is **approximate projected measured-depth coverage**, not exact visibility: original CLEAN camera-presence flags, saved terrain image pixels with finite optical depth >0.05 m, original +0.5 pixel centres/intrinsics, q_true and stored mount drift project the saved FP16 depth into canonical U cells. XY flooring uses the original 0.25 m grid and the mask intersects true BEV terrain. There is no dilation, post-hoc tolerance, LiDAR contribution or `gt_vis` use.

The result covers **{cov['projected_depth_covered']:,}/{cov['terrain_cells']:,} ({100*cov['projected_depth_covered']/cov['terrain_cells']:.3f}%)** distinct true-terrain cells; **{cov['not_projected_depth_covered']:,}** are outside this approximate measure. Quantized depth/joints and unknown within-cell material-boundary correspondence prevent interpreting the latter as proven invisible or the former as a certified match to the target material. In both model states, strata refer to **original CLEAN camera support**; masked models did not receive those cameras.

{support_table}

The weak material predictions persist descriptively within the approximate covered support. KF-D’s covered-support accuracy slightly exceeds the soil reference, whereas BF-D’s does not; neither result identifies a mechanism. Stratum-specific semantic confusions are restricted to true terrain, so nonterrain truth support is zero there: they assess terrain classification/errors, not person detection. Full-map semantic metrics remain the source for rare-class claims. No hidden cells are removed from any official score.

## Rare classes and separate hazard-buffer prediction

The following intact rows are a readable clean/severe/rough summary. Counts pool the three fitted seeds and thus repeat the same labelled pixels three times; they are not independent physical observations. All seven conditions and both camera states remain in `class_metrics.csv`.

{rare_table}

KF-D’s zero person/obstacle predictions in these displayed conditions do not support validated recognition of those classes. BF-D also has low recall, particularly for obstacles. The separate hazard-buffer head nevertheless has nonzero recall:

{haz_table}

These hazard pixels are not the unsafe-per-attempt endpoint or proof of physical safety. The larger designed buffer target differs from the semantic person/obstacle footprint. No algebraic conversion from zero semantic predictions to zero hazard recall is valid.

## Numerical convention, integrity and outputs

Saved `mat_label` records pre-softmax logit argmax. Original official material confusions use `mat_prob.argmax` after FP32 softmax/storage. Four CLEAN cells across the 6,144 retained prediction files have different labels because the stored probabilities are **exactly tied at the maximum**; three of the four lie on true terrain. This diagnostic follows the original official probability-argmax convention and preserves every tie in `stored_label_probability_argmax_ties.csv`. Reconstructed complete CLEAN confusions exactly match all original seed/mask/architecture confusions. No historical output or selector changed. This is a numerical representation caveat, not an explanation for weak material learning; see the versioned convention receipt.

The first job stopped on a JSON key-name mismatch before outputs. A second stopped at an intentionally strict stored-label/probability-label equality assertion; its partial support outputs and first-case record remain in `material/`. The corrected diagnostic reader preserves the official convention, permits only explained maximum-probability ties, and writes the complete version to `material/complete_v1/`; every failure remains in the resource ledger. A non-tie mismatch would have blocked affected integration. No such mismatch was found in the scoped CLEAN replay.

Authoritative outputs:

- [Training prior](references/training_material_prior.json), [prior lock](references/prior_lock.json), and exact [training inventory](references/training_inventory.csv).
- [Full confusion matrices and support](material/complete_v1/full_confusions_and_support.json); [every class metric](material/complete_v1/class_metrics.csv); [summary metrics](material/complete_v1/summary_metrics.csv); [hazard counts/denominators](material/complete_v1/hazard_head_metrics.csv).
- [CLEAN per-frame coverage](material/complete_v1/camera_support_per_frame.csv), packed coverage-mask NPZ, and [saved-input identities](material/complete_v1/source_records.csv).
- [Saved material loss history](material/complete_v1/saved_material_loss_history.csv), [diagnostic receipt](material/complete_v1/diagnostic_receipt.json), [convention clarification](material/convention_clarification_v1.json), [source check](material/source_review.md), and [standalone reconstruction code](code/material_diagnostics.py).

**Claim-level decision:** integrate a concrete material/rare-class limitation and the training-selected reference. Retain the approximate coverage result with its explicit missing artifacts. Do not claim a cause, full semantic understanding, camera-material observability, a corrected experiment or physical validation. This diagnostic does not change either primary family.
'''
(S/'material_and_support_diagnostics.md').write_text(report)

paths = [
    'external_validation_run_EV1_AFAST/source/excascene_snapshot/kinafford/sim/scene.py',
    'external_validation_run_EV1_AFAST/source/excascene_snapshot/kinafford/sim/render.py',
    'external_validation_run_EV1_AFAST/source/excascene_snapshot/kinafford/sim/kinematics.py',
    'external_validation_run_EV1_AFAST/source/excascene_snapshot/kinafford/data/dataset.py',
    'external_validation_run_EV1_AFAST/source/excascene_snapshot/kinafford/models/blocks.py',
    'external_validation_run_EV1_AFAST/source/excascene_snapshot/kinafford/models/variants.py',
    'external_validation_run_EV1_AFAST/source/excascene_snapshot/kinafford/losses.py',
    'external_validation_run_EV1_AFAST/code/data.py',
    'external_validation_run_EV1_AFAST/code/models.py',
    'external_validation_run_EV1_AFAST/code/train_formal.py',
    'external_validation_run_EV1_AFAST/code/generate_holdout.py',
    'external_validation_run_EV1_AFAST/code/infer_holdout.py',
    'external_validation_run_EV1_AFAST/code/scoring.py']
hashes={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}
(S/'material/source_review_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
convention=dict(version=1,status='CLARIFIED_OFFICIAL_CONVENTION_NO_HISTORICAL_CORRECTION',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    plan_sha256=json.loads((S/'records/descriptive_plan_lock.json').read_text())['plan_sha256'],
    original_neutral_source='external_validation_run_EV1_AFAST/code/scoring.py:neutral',
    original_confusion_source='external_validation_run_EV1_AFAST/code/infer_holdout.py:labels/material',
    source_hashes={k:v for k,v in hashes.items() if k.endswith('/infer_holdout.py') or k.endswith('/scoring.py')},
    official_convention='argmax saved material softmax probabilities; lowest-index exact ties',
    alternate_saved_field='pre-softmax logit argmax mat_label',
    scoped_clean_files=6144,exact_maximum_probability_ties=4,ties_on_true_terrain=3,non_tie_discrepancies=0,
    all_original_clean_confusions_exactly_reconstructed=True,
    plan_mapping='saved full-map/prediction diagnostics retain original official class confusion semantics',
    historical_changes=0,new_model_forwards=0,scientific_correction_required=False,
    interpretation='numerical representation caveat only; not cause of weak material classification',
    complete_output='submission_strengthening/material/complete_v1',
    partial_outputs_preserved='submission_strengthening/material',
    failed_jobs=['material_support_diagnostics','material_support_diagnostics_schema_recovery'])
(S/'material/convention_clarification_v1.json').write_text(json.dumps(convention,indent=2)+'\n')
print('Material report and convention/source sidecars written')
