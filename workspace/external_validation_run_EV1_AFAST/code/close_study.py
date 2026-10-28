"""Scientific closure and compact local evidence; no manuscript modification."""
import csv,hashlib,json,os,pathlib,resource,time,zipfile
import numpy as np
import pandas as pd
from common_afast import sha
ROOT=pathlib.Path(__file__).resolve().parents[1]
clock=json.loads((ROOT/'records/start_clock.json').read_text())
def elapsed():return max(time.time()-clock['t0_unix'],time.monotonic()-clock['t0_monotonic'])/3600
def js(p,d):p.write_text(json.dumps(d,indent=2,default=str,allow_nan=False))
def fmt(x):return 'undefined' if pd.isna(x) else f'{float(x):.6f}'
assert elapsed()<36
primary=json.loads((ROOT/'analysis/primary_results.json').read_text())
assert primary['status']=='COMPLETE_STATISTICAL_ESTIMATES' and primary['actual_logical_forwards']==39936
assert json.loads((ROOT/'records/scalar_reconstruction.json').read_text())['status']=='SCALAR_RECONSTRUCTION_MATCHED'
assert json.loads((ROOT/'records/holdout_prediction_release.json').read_text())['status']=='RELEASED_AFTER_SIX_FINALS'
assert not (ROOT/'records/formal_schedule_failure.json').exists()
preserved=[]
for row in json.loads((ROOT/'manifests/protected_before.json').read_text()):
    actual=sha(ROOT.parent/row['path']);assert actual==row['sha256'],('PROTECTED_EVIDENCE_CHANGED',row['path'])
    preserved.append(dict(path=row['path'],sha256=actual,unchanged=True))
js(ROOT/'manifests/protected_after.json',preserved)
development=list(csv.DictReader((ROOT/'manifests/train_validation_inventory.csv').open()))
for row in development:assert sha(ROOT.parent/row['path'])==row['sha256'],('INVALID_DEVELOPMENT_INPUT_CHANGED',row['path'])
js(ROOT/'records/development_inputs_preserved.json',dict(status='ALL_ORIGINAL_TRAIN_VALIDATION_HASHES_UNCHANGED',records=len(development),utc_unix=time.time()))
fits=json.loads((ROOT/'records/all_six_final_checkpoints.json').read_text())['fits']
assert len(fits)==6 and all(f['finish_unix']<=clock['t0_unix']+26*3600 for f in fits)
inference=[json.loads((ROOT/'records'/f'inference_{arch}_seed{seed}.json').read_text()) for arch in ['BF-D','KF-D'] for seed in [10,11,12]]
assert all(r['status']=='COMPLETE' and r['utc_unix']<=clock['t0_unix']+32*3600 for r in inference)
deployment=json.loads((ROOT/'reports/deployment_timing.json').read_text());assert deployment['logical_forwards']==176
events=[json.loads(x) for x in (ROOT/'resources/jobs.jsonl').read_text().splitlines()]
jobs=[e for e in events if e['event']=='finish'];failed=[e for e in jobs if e['returncode']!=0]
known_gpu=sum(e['occupied_gpu_device_hours'] for e in jobs);known_cpu=sum(e['cpu_core_hours'] for e in jobs)
first_launch=min(e['start_unix'] for e in events if e['event']=='launch')
initial_window=(first_launch-clock['t0_unix'])/3600
# Bounds are disclosed allowances, not invented exact measurements.
setup_cpu_bound=64*initial_window;other_unwrapped_cpu_allowance=8.;setup_gpu_bound=initial_window
charged_cpu=known_cpu+setup_cpu_bound+other_unwrapped_cpu_allowance
charged_gpu=known_gpu+setup_gpu_bound
pilot_paths=[ROOT/'reports/pilots/BF-D/pilot_failure.json',ROOT/'reports/pilots/BF-D/attempt_2/pilot_receipt.json',
    ROOT/'reports/pilots/KF-D/attempt_1/pilot_receipt.json',ROOT/'reports/pilots/KF-D/attempt_2/pilot_receipt.json']
pilot_counts=[json.loads(p.read_text())['counts'] for p in pilot_paths]
nontraining_pilot=sum(c['pilot_measurement_forwards']+c['epoch_validation_forwards']+c['export_forwards'] for c in pilot_counts)
engineering_charge=4+3*42+6+8+12+12
forward_charge=nontraining_pilot+engineering_charge+176+sum(r['attempted_logical_forwards'] for r in inference)
assert forward_charge<=60000 and charged_gpu<64 and charged_cpu<240
disk_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
outside_cache_disk_allowance=1000000000
assert disk_bytes+outside_cache_disk_allowance<120e9
cost=dict(status='METERED_JOBS_PLUS_DISCLOSED_CONSERVATIVE_ALLOWANCES',as_of_elapsed_hours=elapsed(),
    metered_occupied_gpu_device_hours=known_gpu,metered_launched_job_cpu_core_hours=known_cpu,
    initial_unmetered_window_hours=initial_window,initial_gpu_device_hour_upper_allowance=setup_gpu_bound,
    initial_cpu_core_hour_upper_allowance=setup_cpu_bound,initial_bound_basis='Entire pre-wrapper window charged at all 64 logical host CPU cores and one GPU; actual use was smaller and not metered.',
    other_unwrapped_scalar_shell_and_reporting_cpu_core_hour_allowance=other_unwrapped_cpu_allowance,
    charged_gpu_device_hours_before_closure=charged_gpu,charged_cpu_core_hours_before_closure=charged_cpu,
    closure_cost_location='Final final_scientific_report_and_evidence finish event in resources/jobs.jsonl; packaging elapsed/CPU in records/evidence_zip_receipt.json. Both remain outside the archive snapshot.',
    disk_bytes_before_zip=disk_bytes,outside_run_dependency_cache_disk_allowance_bytes=outside_cache_disk_allowance,
    charged_disk_bytes_before_zip=disk_bytes+outside_cache_disk_allowance,
    disk_measurement='Sum of logical file bytes in the isolated run, excluding symlinks; hardlinked bytes may be counted twice. Outside dependency-cache increments were not baseline-metered and receive a separate 1GB conservative allowance.',
    network_source_bytes=9238451,network_dependency_charge_bytes=1000000000,
    network_measurement='Source tar bytes are measured; dependency/metadata wire traffic was not packet-metered. A 1GB conservative allowance is charged; known attempted distribution sizes are far smaller. No later acquisition or weight retrieval.',
    non_training_non_epoch_validation_forward_charge=forward_charge,
    formal_training_frame_exposures=sum(f['counts']['training_frame_exposures'] for f in fits),
    formal_epoch_validation_forwards=sum(f['counts']['epoch_validation_forwards'] for f in fits),
    pilot_training_frame_exposures=sum(c['training_frame_exposures'] for c in pilot_counts),pilot_optimizer_updates=sum(c['optimizer_updates'] for c in pilot_counts),
    pilot_nontraining_forwards=nontraining_pilot,engineering_forward_charge=engineering_charge,
    deployment_forwards=176,holdout_forwards=39936,failed_launched_jobs=len(failed),cpu_affinity='Every instrumented job shares CPUs0..7; one GPU, one fit at a time',
    purchases=0,externally_billed_services=0,electricity_and_hardware_amortization_currency_cost=None,
    currency_cost_basis='No transaction or service purchased; local electricity and hardware amortization were not metered or priced.',
    caps=dict(gpu_device_hours=64,cpu_core_hours=240,disk_bytes=120000000000,network_bytes=20000000000,nontraining_forwards=60000))
js(ROOT/'resources/final_resource_snapshot.json',cost)
with (ROOT/'resources/failed_jobs.csv').open('w',newline='') as f:
    fields=['name','returncode','elapsed_seconds','occupied_gpu_device_hours','cpu_core_hours','log'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:e[k] for k in fields} for e in failed)

# Large outside-ZIP checkpoints carry individual hashes. Pred/data manifests
# already record every file; do not read historical test scores for this report.
large=[]
for p in sorted((ROOT/'checkpoints').rglob('*.pt')):large.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),artifact='checkpoint'))
for p in sorted((ROOT/'latents').glob('*.npz')):large.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),artifact='shared_fault_latent'))
with (ROOT/'manifests/large_artifacts.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256','artifact']);w.writeheader();w.writerows(large)

summary=pd.read_csv(ROOT/'analysis/condition_results.csv');pool=pd.read_csv(ROOT/'analysis/fixed_weight_pool_results.csv')
effects=pd.read_csv(ROOT/'analysis/camera_removal_costs.csv');architecture=pd.read_csv(ROOT/'analysis/architecture_comparisons.csv')
seedrows=pd.read_csv(ROOT/'analysis/primary_per_seed_leave_one_out.csv')
report=[]
report+=['# EV1-AFAST results','', '**Status: COMPLETE.** Six new fixed-horizon fits and the full planned holdout inference completed. This executes the six-fit AFAST amendment; the original nine-fit EV-1 draft was not executed. Statistical support is stated separately for each endpoint.','',
    f'The setup clock began {clock["t0_utc"]}. All final checkpoints were available by elapsed hour {max((f["finish_unix"]-clock["t0_unix"])/3600 for f in fits):.3f}; holdout inference finished by hour {max((r["utc_unix"]-clock["t0_unix"])/3600 for r in inference):.3f}. This report was generated at hour {elapsed():.3f}; archive completion and verification time are in `records/evidence_zip_receipt.json`. The target was24h, with an absolute36h deadline.','',
    '## Protocol and validity','',
    'Both architectures used E=14, seeds10/11/12, fresh random initialization, actual FP32 with TF32/autocast off, microbatch4 with accumulation4, the same supervised losses, structured dropout, AdamW and final checkpoints only. Each fit used10500 updates,168000 training-frame exposures and16800 original-validation forwards. Runtime-only forecasts excluded42 and28 epochs and selected14 before formal fitting; no outcome-based shortening or extra epoch was used. The six fitted trajectories are budget limited and are not claimed to have converged.','',
    'BF-D retains the pinned MIT HAN Lab Swin/FPN, learned LSS depth-probability lift, native CUDA BEV pooling, native sparse encoder, convolutional fuser and SECOND/FPN decoder. Minimal task/runtime adapters are enumerated in `adapter_record.md` and the source diff. The random-fit task adaptation is not an unmodified published nuScenes result. Empty/trailing-empty LiDAR cardinality, floor/boundaries, U/camera/radial geometry, pixel anchors, presence masking, all branches/heads and native forward/backward controls passed before launch. Early broken attempts were resolved and retained as engineering failures, not negative scientific results.','',
    'The holdout contains128 nominal and64 rough new scene identities with four accepted slots each:3328 records and39936 actual logical forwards. All slot bundles passed preprediction validity; none was exhausted or excluded. Shared poses/fault arrays preceded interventions; retained inputs were checked exactly. Collision screens checked historical identifiers, stored pose/oracle fingerprints and source-compatible geometry without accessing historical performance scores. These are simulator-internal validity checks, not independent physical ground truth. All six final checkpoints and exact matched training minibatches were verified before prediction release. The same agent administered custody; no independent experimenter is claimed.','',
    '## Two co-primary endpoints','',
    'The estimand is BF-D severe cam_s3, all-camera-masked minus intact, under the frozen strict direct selector. Metrics first average the three fitted seeds, then eligible slots within each scene, then equal scenes. Negative means lower loss. Raw regret retains low/zero oracle cases; abstention gets zero utility/full feasible-oracle regret. Unsafe uses attempts as its primary denominator.','',
    '| Endpoint | Estimate | Ordinary95% | Adjusted97.5% | Interpretation |','|---|---:|---|---|---|']
for e in primary['endpoints']:report.append(f'| {e["name"]} | {e["estimate"]:.9f} | [{e["ordinary95"][0]:.9f}, {e["ordinary95"][1]:.9f}] | [{e["adjusted97_5"][0]:.9f}, {e["adjusted97_5"][1]:.9f}] | {e["interpretation"]} |')
u=primary['endpoints'][1]
report+=['',f'Unsafe arithmetic: intact {u["unsafe_intact_count"]}/{u["attempts_per_state"]} attempts and {u["executions_intact"]} executions; masked {u["unsafe_masked_count"]}/{u["attempts_per_state"]} attempts and {u["executions_masked"]} executions. Exact effect {u["estimate_exact"]}; ordinary bounds {u["ordinary95_exact"]}; adjusted bounds {u["adjusted97_5_exact"]}. All abstentions remain in the attempt denominator. These pooled event counts reconstruct the scene estimand because every scene retained four slots; per-scene counts are also supplied.', '',
    ('Both adjusted upper limits are belowzero, supporting a conditional joint lower-loss finding.' if primary['joint_lower_loss'] else 'The prespecified joint lower-loss criterion is not satisfied. This does not establish equivalence, absence of an effect or physical safety.'),'',
    'Intervals use20000 shared-scene draws, default_rng2026101303, with adjusted quantiles1/80 and79/80. Continuous arithmetic is float64; unsafe percentile interpolation is exact rational arithmetic. Marginal percentile coverage is approximate, so the Bonferroni familywise95% interpretation is nominal, not an exact coverage guarantee. A degenerate interval reflects empirical resampling support and does not establish equivalence or bound unobserved event risk. Inference is conditional on these three fitted seeds and the declared synthetic population, not arbitrary retraining seeds. Saved draws and standalone scalar reconstruction match the reported quantities.','',
    '## Per-seed and sensitivity results','',
    '| Fitted seeds | Raw regret effect | Unsafe effect | Execution effect |','|---|---:|---:|---:|']
for _,r in seedrows.iterrows():report.append(f'| {r["summary"]} ({r["seeds"]}) | {fmt(r.raw_regret_effect)} | {fmt(r.unsafe_effect)} | {fmt(r.execution_effect)} |')
report+=['','## Required descriptive comparisons','',
    'All rows below are descriptive. Intact/masked absolute unsafe per attempt, execution and unsafe per execution must be interpreted together. Pooled unsafe-per-execution ratios are diagnostic denominators, separate from equal-scene primary inference. Full direct/soft and per-seed results, count denominators and undefined quantities are in `analysis/`.','',
    '| Architecture | Condition | Mask | Direct raw regret | Unsafe/attempt | Execution rate | Unsafe events / attempts | Unsafe/executions |','|---|---|---:|---:|---:|---:|---|---|']
for _,r in summary[summary.policy=='direct'].iterrows():report.append(f'| {r.architecture} | {r.condition} | {int(r["mask"])} | {fmt(r.regret)} | {fmt(r.unsafe)} | {fmt(r.executed)} | {int(r.unsafe_count)}/{int(r.attempted_count)} | {int(r.unsafe_count)}/{int(r.executed_count)} ({fmt(r.unsafe_per_execution_pooled_diagnostic)}) |')
report+=['','| Architecture | Condition | Direct removal raw regret cost | Unsafe change | Execution change |','|---|---|---:|---:|---:|']
for _,r in effects[effects.policy=='direct'].iterrows():report.append(f'| {r.architecture} | {r.condition} | {fmt(r.regret_effect)} | {fmt(r.unsafe_effect)} | {fmt(r.executed_effect)} |')
report+=['','| Context | Mask | Policy | BF−KF reachable-terrain height MAE | BF−KF raw regret | BF−KF unsafe/attempt | BF−KF execution |','|---|---:|---|---:|---:|---:|---:|']
for _,r in architecture.iterrows():report.append(f'| {r.condition} | {int(r["mask"])} | {r.policy} | {fmt(r.H_reachable_terrain)} | {fmt(r.regret)} | {fmt(r.unsafe)} | {fmt(r.executed)} |')
report+=['', 'The iid/all_s2/rough pool uses fixed one-third context weights. Direct receives no grade; soft composition receives the same oracle grade/reach privilege in both architectures. Undefined reachable-height support cannot exclude decision events. `pixel_diagnostics.json` supplies full per-seed/aggregate semantic confusion matrices and all six class IoUs, true-terrain material counts, hazard precision/recall and denominators, utility error and support. A ground-truth-absent class with false positives has IoU0; only zero union is undefined. Condition tables retain oracle-zero/low, empty reach/candidate, boundary and tie diagnostics.','',
    '## Optimization and computation','',
    '| Architecture | Seed | Trainable/total parameter note | Fit seconds | Peak allocated GPU GB | Final original-validation loss |','|---|---:|---|---:|---:|---:|']
for f in fits:
    history=json.loads((ROOT/'checkpoints'/f['architecture']/f'seed{f["seed"]}/history.json').read_text())
    params=next(x for x in deployment['rows'] if x['architecture']==f['architecture'])
    report.append(f'| {f["architecture"]} | {f["seed"]} | {params["parameters_trainable"]}/{params["parameters_total"]} | {f["elapsed_seconds"]:.2f} | {f["peak_allocated_bytes"]/1e9:.3f} | {history[-1]["validation_losses"]["total"]:.6f} |')
    js(ROOT/'reports'/f'optimization_{f["architecture"]}_seed{f["seed"]}.json',dict(first=history[0],last=history[-1],all_epochs=len(history),
        interpretation='Descriptive trajectory and coverage; no assertion of convergence or checkpoint selection. Declining final losses remain an optimization limitation.'))
report+=['', '| Architecture (fixed seed10) | Single-frame end-to-end mean s | Single-frame p95 s | Batch4 throughput frames/s |','|---|---:|---:|---:|']
for r in deployment['rows']:report.append(f'| {r["architecture"]} | {r["single_frame_latency_seconds_mean"]:.6f} | {r["single_frame_latency_seconds_p95"]:.6f} | {r["batch4_throughput_frames_per_second"]:.3f} |')
report+=['', 'Deployment timing uses only the fixed first32 original-validation inputs. It includes input deserialize/normalization/dynamic transforms, transfers, network and neutral FP32 in-memory serialization; it excludes checkpoint setup, disk writes and selector/oracle scoring. Batch throughput is not single-frame latency.','',
    f'One actual A100-PCIE-40GB was used, PyTorch2.8/CUDA12.9, MMCV1.4/MMDetection2.20, and CPUs0..7. Metered occupied GPU device-hours before closure: {known_gpu:.4f}; metered launched-job CPU core-hours: {known_cpu:.4f}. Including disclosed setup/scalar allowances gives {charged_gpu:.4f} GPU device-hours and {charged_cpu:.4f} CPU core-hours before archive closure. All pilots and failed jobs are included. Non-training/non-epoch-validation forward charge: {forward_charge}/60000. New disk before ZIP: {disk_bytes/1e9:.3f}GB. Measured source download9.238451MB plus a conservative1GB dependency/metadata allowance; wire traffic is not claimed exactly measured. Purchases and externally billed services:0; local electricity and hardware amortization were not priced. Final closure costs/time are recorded outside the archive snapshot as explained in `resources/final_resource_snapshot.json`.','',
    'Sparse/GPU reductions may be nondeterministic; no bit-identical retraining claim is made. Engineering numerical controls do not establish statistical equivalence. Matching data exposure and nominal recipe do not eliminate architecture-specific optimization difficulty or capacity differences. Full epoch trajectories are preserved; optimization-limited or inconclusive results do not imply published-method inferiority.','',
    '## Manuscript impact and preserved scope','',
    'This is a separately reported external-architecture synthetic validation with new matched G3 fits. It can strengthen evidence about whether the supplied severe-camera removal effect extends to an independently developed fusion trunk under this adapted task and budget. It cannot establish real-world transfer, physical kinematic accuracy, online sensor diagnosis, safe robotic execution, or general original BEVFusion performance. A-G is NOT_TESTED because no BF-static arm exists. S remains unresolved because no external noaff arm exists. Study B is NOT_TESTED. No follow-on experiment or extension was run.','',
    f'All {len(preserved)} protected historical/manuscript/design artifacts retain their original SHA256 hashes. The four v5.1 primary endpoint values and multiplicity family are unchanged; the new two-endpoint family remains separate. The nine-fit draft, manuscript and appendix were preserved. Large checkpoints/data/latents/predictions remain outside the compact ZIP with hashes. No pretrained-weight acquisition, public release, annotation, submission or manuscript rewriting occurred.','']
(ROOT/'AFAST_results.md').write_text('\n'.join(report))
impact=['# Manuscript impact note','', 'EV1-AFAST is a separate six-fit, E=14, random-initialization external-architecture study. The original nine-fit EV-1 draft was not executed, and the four completed v5.1 primary results are preserved. No manuscript edit is included in this run.','']
for e in primary['endpoints']:impact.append(f'{e["name"]}: {e["interpretation"]}; estimate {e["estimate"]:.9f}, adjusted97.5% interval [{e["adjusted97_5"][0]:.9f}, {e["adjusted97_5"][1]:.9f}].')
supported=[e['interpretation']=='supported lower loss' for e in primary['endpoints']]
if all(supported):
    impact+=['', 'The two-endpoint joint criterion is met. Supplied severe-camera removal lowers both raw regret and unsafe events per attempt for this adapted external trunk, conditional on the three fitted seeds and the declared synthetic population. Interpret the unsafe reduction alongside the reported absolute execution and abstention rates; it does not establish physical safety.']
elif supported[0]:
    impact+=['', 'The external study supports the raw-regret removal effect, while the unsafe endpoint does not meet the lower-loss criterion. The prespecified joint claim is not supported. Any manuscript addition should extend the raw-regret evidence only and retain the unsafe result and its event/execution denominators.']
elif supported[1]:
    impact+=['', 'The external study supports the unsafe-per-attempt removal effect, while raw regret does not meet the lower-loss criterion. The prespecified joint claim is not supported. Any manuscript addition must retain the utility result and interpret unsafe counts alongside execution and abstention.']
else:
    impact+=['', 'Neither external endpoint meets the adjusted lower-loss criterion, and the prespecified joint claim is not supported. The external study does not extend the joint supplied-camera removal conclusion to this adapted trunk at the locked training budget. Retain both intervals and per-seed directions; an inconclusive result does not establish absence of an effect.']
impact+=['', 'A supported effect would extend conditional supplied-camera removal evidence to this adapted external trunk and synthetic population. An inconclusive or adverse result narrows that claim and must be retained with per-seed effects, execution/abstention and residual unsafe rates. BF/KF differences are descriptive and budget dependent; they do not rank original published methods or establish convergence/equivalence.','',
    'A-G is NOT_TESTED; S remains unresolved; Study B and physical safety are NOT_TESTED. No claim of real articulated geometry, operational fault detection or independent experimental custody follows. If incorporated later, add a separate external-validation subsection/table and preserve the existing four-endpoint family. Future manuscript edits require a separate instruction.','']
(ROOT/'manuscript_impact_note.md').write_text('\n'.join(impact))

# Select compact evidence. Never include raw sensor/latent/prediction or weights.
selected=set()
for d in ['code','analysis','metrics','records','manifests','resources','logs']:
    selected.update(p for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.zip')
selected.update(p for p in ROOT.glob('*.md'))
selected.update(p for p in ROOT.glob('generation*.csv'))
selected.update(p for p in ROOT.glob('generation*.json'))
selected.update(p for p in (ROOT/'tests').glob('*.json'))
selected.update(p for p in (ROOT/'reports').glob('*.json'))
selected.update(p for p in (ROOT/'reports/pilots').rglob('*') if p.is_file() and p.suffix in ['.json','.csv'])
selected.update(p for p in (ROOT/'environment').glob('*.json'))
selected.update(p for p in (ROOT/'environment').glob('*.txt'))
selected.update(p for p in (ROOT/'source/excascene_snapshot').rglob('*.py'))
selected.add(ROOT/'source/bevfusion_original.tar.gz')
selected.add(ROOT/'source/LICENSE_BEVFusion.txt')
selected.update(p for p in (ROOT/'source/design_original').glob('*') if p.is_file())
selected.update(p for p in (ROOT/'source').glob('*.py'))
for arch in ['BF-D','KF-D']:
    for seed in [10,11,12]:selected.update(p for p in (ROOT/'checkpoints'/arch/f'seed{seed}').glob('*') if p.suffix in ['.json','.csv'])
selected={p for p in selected if p.name not in ['evidence_zip_receipt.json','post_closure_resource_totals.json']}
manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)]
js(ROOT/'manifests/evidence_package_files.json',manifest);selected.add(ROOT/'manifests/evidence_package_files.json')
zip_start=time.monotonic();cpu_start=resource.getrusage(resource.RUSAGE_SELF)
archive=ROOT/'EV1_AFAST_evidence.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(selected):z.write(p,str(p.relative_to(ROOT)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for row in manifest:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
zip_seconds=time.monotonic()-zip_start;cpu_end=resource.getrusage(resource.RUSAGE_SELF)
after_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
assert after_bytes+outside_cache_disk_allowance<120e9 and elapsed()<36
receipt=dict(status='COMPLETE',zip_path=str(archive.relative_to(ROOT)),zip_bytes=archive.stat().st_size,sha256=sha(archive),
    included_files=len(selected),all_included_files_verified=True,report_and_archive_elapsed_hours=elapsed(),
    target24h_met=elapsed()<=24,hard36h_met=elapsed()<=36,zip_build_and_verification_seconds=zip_seconds,
    zip_build_and_verification_cpu_core_hours=(cpu_end.ru_utime+cpu_end.ru_stime-cpu_start.ru_utime-cpu_start.ru_stime)/3600,
    incremental_disk_bytes_after_zip=after_bytes,
    charged_disk_bytes_after_zip=after_bytes+outside_cache_disk_allowance,
    outside_run_dependency_cache_disk_allowance_bytes=outside_cache_disk_allowance,
    large_artifacts_outside_zip=True,scalar_reconstruction='records/scalar_reconstruction.json',
    preservation_verified_files=len(preserved),utc_unix=time.time())
js(ROOT/'records/evidence_zip_receipt.json',receipt);js(ROOT/'final_status.json',dict(status='COMPLETE',elapsed_hours=elapsed(),utc_unix=time.time(),scope='EV1-AFAST six-fit amendment only',evidence_receipt=receipt))
print(json.dumps(receipt,indent=2))
