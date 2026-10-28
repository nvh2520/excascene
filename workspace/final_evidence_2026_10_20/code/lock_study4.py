"""Seal all scientific choices before any fresh holdout generation."""
from common import *
import datetime
def main():
    dest=ROOT/'protocol/study4_final_lock.json';assert not dest.exists() and not (ROOT/'study4/holdout').exists()
    assert json.loads((ROOT/'study4/validation/validation_audit.json').read_text())['status']=='PASS_INPUTS_ONLY'
    assert json.loads((ROOT/'study4/threshold_lock.json').read_text())['no_new_test_generated']
    controls=json.loads((ROOT/'protocol/prelock_controls.json').read_text());assert controls['status']=='PASS'
    for name,h in controls['code_sha256'].items():assert sha(ROOT/'code'/name)==h,('prelock fixture source changed',name)
    analytic=json.loads((ROOT/'protocol/analytic_pipeline_control.json').read_text());assert analytic['status']=='PASS'
    for name,h in analytic['code_sha256'].items():assert sha(ROOT/'code'/name)==h,('analytic fixture source changed',name)
    train=json.loads((ROOT/'protocol/training_lock.json').read_text())
    for rel,h in train['files'].items():assert sha(ROOT/rel)==h
    inventory=json.loads((ROOT/'protocol/checkpoint_inventory.json').read_text());assert len(inventory['checkpoints'])==12
    files={};external=dict(train['external_files'])
    paths=list((ROOT/'code').glob('*.py'))+list((ROOT/'protocol').glob('*.md'))
    paths+=[ROOT/'protocol'/n for n in ['training_lock.json','training_class_weights.json','checkpoint_inventory.json','holdout_manifest.json','release_fixture_plan.json','prelock_controls.json','analytic_pipeline_control.json','statistical_controls.json','study3_thresholds_unchanged.json']]
    paths+=[ROOT/'study4'/n for n in ['thresholds.json','threshold_lock.json','validation_threshold_grid.csv','validation_gate_scores.csv','gate/final.pt','validation/files.csv','validation/validation_audit.json']]
    for cp in inventory['checkpoints']:
        p=PROJECT/cp['path'];assert sha(p)==cp['sha256']
        if p.is_relative_to(ROOT):paths.append(p)
        else:external[str(p.relative_to(PROJECT))]=cp['sha256']
    for rel in ['submission_strengthening/references/training_prior_maps.npz','submission_strengthening/references/geometric_source.py','strengthening_2026_10_18/phase_a/classical/validation_lock.json']:
        external[rel]=sha(PROJECT/rel)
    collision_inputs=[AFAST/'manifests'/n for n in ['historical_source_geometry_hashes.json','historical_scene_pose_fingerprints.json','holdout_content_validity.csv','collision_audit_receipt.json']]
    collision_inputs+=[PROJECT/'strengthening_2026_10_18/study3/holdout/scene_pose_fingerprints.csv']
    collision_inputs+=list((PROJECT/'strengthening_2026_10_18/study3/holdout/generation').glob('*.json'))
    for p in collision_inputs:external[str(p.relative_to(PROJECT))]=sha(p)
    for p in paths:files[str(p.relative_to(ROOT))]=sha(p)
    for rel,h in external.items():assert sha(PROJECT/rel)==h
    result=dict(status='LOCKED_BEFORE_STUDY4_HOLDOUT_GENERATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=files,external_files=external,holdout_generated=False,test_predictions=0,test_scores=0,test_bootstrap_draws=0,
        scientific_seeds=[10,11,12],epochs=42,selected_checkpoint_epoch=42,primary_family_size=4,
        primary_marginal_coverage=.9875,bootstrap_draws=20000,empty_scene_rule='exclude whole without replacement before inference',
        no_outcome_based_extension=True)
    js(dest,result);js(ROOT/'protocol/study4_final_lock_digest.json',dict(sha256=sha(dest),utc=result['utc'],files=len(files),external_files=len(external)))
    print('LOCKED',sha(dest),flush=True)
if __name__=='__main__':main()
