"""Release requires six complete fixed-horizon fits and a matching input audit."""
import csv,json,pathlib,time
from common_afast import sha
ROOT=pathlib.Path(__file__).resolve().parents[1]
clock=json.loads((ROOT/'records/start_clock.json').read_text())
assert time.time()<=clock['t0_unix']+26*3600
assert json.loads((ROOT/'records/all_six_final_checkpoints.json').read_text())['status']=='ALL_SIX_FINALS_COMPLETE'
assert json.loads((ROOT/'records/sealed_holdout_audit.json').read_text())['status']=='SEALED_PHYSICAL_CONTENT_AUDIT_PASSED'
assert json.loads((ROOT/'tests/analysis_fixture_receipt.json').read_text())['status']=='PASSED'
lock=json.loads((ROOT/'manifests/execution_lock.json').read_text())
for rel,h in lock['locked_files'].items():assert sha(ROOT/rel)==h,('execution_lock_mismatch',rel)
analysis_lock=json.loads((ROOT/'manifests/analysis_lock.json').read_text())
for rel,h in analysis_lock['files'].items():assert sha(ROOT/rel)==h,('analysis_lock_mismatch',rel)
checkpoints={};matched=[]
for seed in [10,11,12]:
    tables=[]
    for arch in ['BF-D','KF-D']:
        directory=ROOT/'checkpoints'/arch/f'seed{seed}'
        completion=json.loads((directory/'completion.json').read_text())
        assert completion['status']=='FORMAL_FIT_COMPLETE' and completion['epochs']==lock['epochs'] and completion['seed']==seed
        assert completion['counts']==dict(optimizer_updates=750*lock['epochs'],training_frame_exposures=12000*lock['epochs'],epoch_validation_forwards=1200*lock['epochs'])
        path=directory/'final.pt';digest=sha(path);assert digest==completion['final_checkpoint_sha256']
        checkpoints[str(path.relative_to(ROOT))]=digest
        tables.append(list(csv.DictReader((directory/'minibatches.csv').open())))
    assert len(tables[0])==len(tables[1])==3000*lock['epochs']
    assert tables[0]==tables[1],('INVALID_UNMATCHED_TRAINING_INPUTS',seed)
    matched.append(dict(seed=seed,microbatches=len(tables[0]),identical_frame_epoch_order_neural_bytes_and_dropout=True,
        BF_input_manifest_sha256=sha(ROOT/'checkpoints/BF-D'/f'seed{seed}/minibatches.csv'),KF_input_manifest_sha256=sha(ROOT/'checkpoints/KF-D'/f'seed{seed}/minibatches.csv')))
receipt=dict(status='RELEASED_AFTER_SIX_FINALS',utc_unix=time.time(),elapsed_hours=(time.time()-clock['t0_unix'])/3600,
    checkpoint_files=checkpoints,analysis_files=analysis_lock['files'],matched_training=matched,
    independent_custody=False,selection='final checkpoints only; no holdout predictions existed before release')
(ROOT/'records/holdout_prediction_release.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
