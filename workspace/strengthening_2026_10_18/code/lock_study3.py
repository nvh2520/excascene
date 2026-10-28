"""One final prospective lock, after development and before new holdout generation."""
from common import *
import datetime
def main():
    destination=ROOT/'protocol/study3_final_lock.json'
    assert not destination.exists() and not (ROOT/'study3/holdout').exists()
    assert json.loads((ROOT/'protocol/prelock_controls.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'study3/threshold_lock.json').read_text())['no_new_test_generated']
    assert json.loads((ROOT/'study3/deployment/receipt.json').read_text())['status'].startswith('COMPLETE')
    for architecture,seeds in [('KF-D',[10,11,12]),('BF-D',[10,11,12]),('S5A',[0,1,2])]:
        for seed in seeds:
            path=ROOT/f'study3/validation_subsets/{architecture}_seed{seed}.json'
            assert json.loads(path.read_text())['status']=='COMPLETE_VALIDATION_ONLY'
    paths=list((ROOT/'code').glob('*.py'))
    paths += [ROOT/rel for rel in ['protocol/study3_protocol.md','protocol/study3_development_plan.md','protocol/holdout_manifest.json',
        'protocol/checkpoint_inventory.json','protocol/resource_amendment.md','protocol/prelock_controls.json','protocol/statistical_controls.json',
        'study3/power_resource_proposal.json','study3/gate/final.pt','study3/thresholds.json','study3/threshold_lock.json','study3/validation_threshold_grid.csv',
        'study3/deployment/receipt.json','phase_a/classical/validation_lock.json']]
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    external=[]
    for tree in [AFAST/'source/excascene_snapshot',AFAST/'source/bevfusion_adapted']:
        external.extend(p for p in tree.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml','.so','.cpp','.cu','.h'])
    external.extend((AFAST/'code').glob('*.py'))
    external += [AFAST/'source/selectors_v5_original.py',PROJECT/'submission_strengthening/references/geometric_source.py',
        PROJECT/'submission_strengthening/references/training_prior_maps.npz',AFAST/'manifests/historical_scene_pose_fingerprints.json',
        AFAST/'manifests/historical_source_geometry_hashes.json',AFAST/'manifests/holdout_content_validity.csv']
    for row in json.loads((ROOT/'protocol/checkpoint_inventory.json').read_text())['checkpoints']:
        p=PROJECT/row['path'];assert sha(p)==row['sha256'];external.append(p)
    external_files={str(p.relative_to(PROJECT)):sha(p) for p in sorted(set(external))}
    js(destination,dict(status='LOCKED_BEFORE_HOLDOUT_GENERATION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=files,external_files=external_files,new_holdout_exists=False,selection_inputs='original training and declared validation only; historical holdout solely authorized power/runtime planning',
        precision='actual FP32 for gate and every frozen-model forward;TF32/autocast off',independent_custodian=False,
        amendment='resource-limited frozen perception models; no six-fit perception retraining',no_postlock_choice_changes=True))
    js(ROOT/'protocol/study3_final_lock_digest.json',dict(path=str(destination.relative_to(ROOT)),sha256=sha(destination)))
    print('LOCKED',sha(destination),'new files',len(files),'historical dependencies',len(external_files))
if __name__=='__main__':main()
