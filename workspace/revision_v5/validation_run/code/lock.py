from common import *
import pandas as pd,torch,subprocess
def main():
    assert not (RUN/'manifests/artifact_lock.json').exists()
    tests=(RUN/'logs/preflight_tests_final.log').read_text();assert 'Ran 9 tests' in tests and tests.rstrip().endswith('OK')
    assert len(pd.read_csv(RUN/'legacy/camera_valid_only.csv'))==288
    cp=pd.read_csv(RUN/'manifests/checkpoint_manifest.csv');epochs=cp[cp.model.isin(['S5A_G3_structured','S2E_G6_lidar'])][['model','seed','epoch_one_based']]
    # Bind corrected R02 numbers to the actual historical checkpoint versions, without evaluation.
    extra=[]
    for tag,folder in [('G6_lidar','runs'),('S2D_R2_oracle','runs_stage2')]:
      for seed in range(3):
        p=ROOT/f'{folder}/{tag}/seed{seed}/best.pt';z=torch.load(p,map_location='cpu',weights_only=False)
        extra.append(dict(model=tag,seed=seed,checkpoint='best',path=str(p.relative_to(ROOT)),sha256=sha(p),epoch_one_based=int(z['epoch'])+1,training_budget=int(z['cfg']['epochs']),endpoints='H_mae_vis,regret_direct',suite='cam_s3 (quoted); ext_s3/all_s3 separate rows'))
    csvwrite(RUN/'legacy/R02_checkpoint_bindings.csv',extra)
    protocol='''# Amended frozen-model validation protocol — execution lock, 2026-10-10

The user supplied `ExcaScene_v5_Post_Prompt1_Execution.md` authorizes this one package. Its full text is preserved in `original_prompt1/`; it controls over the original proposed protocol. No further generic approval gate applies to the authorized synthetic work. The original four reports remain unchanged. This amendment is recorded before any fresh predictions or scene generation.

## Fixed questions and checkpoint versions

G: final epoch-14 G3_dyn minus G2_static, reachable-terrain height MAE, weights 1/2 clean and 1/2 clean wider articulation. S: final epoch-14 R2_full/soft minus R2_noaff/soft raw feasible-oracle regret, weights 1/3 nominal iid, nominal all_s2 and rough iid. Both primary soft selectors have no extra hazard filter. C-regret and C-unsafe: original validation-selected S5A best, masked minus unmasked, cam_s3, same strict direct hazard-filtered selector. There are always four primary slots, even if blocked. S5A selected epochs are 42/42/39; S2E_G6 selected epochs are 38/36/40, within 42-epoch budgets. G/S final checkpoints are all epoch14. Seeds are 0/1/2. Best-checkpoint S and hazard-soft sensitivities are separately labeled.

These are frozen fitted-model comparisons. R2 full/noaff retain original loss, initialization and data-order implementations; equal final epoch does not manufacture a perfectly controlled training intervention. The checkpoint configurations are retained verbatim. Soft composition on a directly supervised backbone cannot establish that direct supervision is unnecessary.

## Paired cohort and streams

`generation_manifest.json` fixes nominal IDs100000–100127, rough IDs110000–110063, four slots, ten nominal conditions and rough_iid. The sorted 28-scene subset from default_rng(20261013) adds severe combined-present RGB+depth, 112 records, total5488. No replacement IDs. Conditions and 58608 fresh forwards are exhaustively specified by `run_matrix.csv`; no other frozen-model matrix is authorized.

Terrain, rig, intrinsic matrices, grade and proposal mixture use the original source. Terrain uses its original scene-seed formula. Other RNGs derive from SeedSequence([20261010,family,scene,slot,namespace,sensor,latent_identity,attempt]). Namespaces10/11 pose proposals;20 shared removed-camera identity;21 iid mixture;30 physical drifts;31 joint bias;40 camera latent arrays;41 LiDAR arrays;50 rendering noise. No condition code enters retained noise/pose streams. Per-mechanism arrays are generated before condition application and saved losslessly. The same ellipse sequence supplies severity prefixes; fixed camera suites retain all five mechanisms in original order, while iid retains its one/two-kind mixture rule. Dense indexed Gaussian/uniform fields replace conditional-consumption RNG calls to make interventions paired; this preserves marginal fault laws but intentionally changes coupling and exact historical realizations.

The 2x2 subset uses clean, appearance_only, depth_only and combined_present, with the identical component arrays. cam_s3 is that combined condition plus one fixed camera's absence. Appearance restores both original depth paths; depth restores original RGB; removal zeros RGBD, projection depth and presence/meta. Camera-only interventions retain LiDAR, joints, extrinsics, grade and oracle maps. Serialization-level tensor equality assertions precede inference. Oracle labels, true joints and reliability labels are excluded from the model input allowlist. Masking is supplied fault knowledge. GT visibility preserves historical rendered-hit semantics, not measured-depth availability.

Whole same-pose bundles are accepted only after original_pose_ok and >=1mm camera-origin clearance from terrain/all renderer solids, including drift. At most50 independent proposal attempts per slot. No condition-specific pose repair and no nominal substitution for failed wider poses. Retain every attempt/failure and original proposal component plus final swing-overwritten phase. Apply 10% invalid-slot ceilings separately to nominal/rough/wide and overall. G uses the preprediction common nominal/wide key intersection; S nominal conditions share accepted bundle keys; subset mechanisms use complete blocks. Exact content fingerprints (FP16 canonicalized original target maps plus FP32 true joints) and original/fresh scene IDs are checked before evaluation; near-duplicate absence is not claimed. No policy-dependent exclusions.

## Numeric and inference rules

Original sensor storage remains RGBuint8 and measured depth/rangeFP16, converted to FP32 tensors. Fresh H/utility/reach targets are computed FP32 by bev_ground_truth before storage. Shared FP16 utility sensitivity changes only utility scoring at the same picks/support. Targets used by composition: reach and scene grade. H/semantic/material/hazard targets score predictions or define oracle/support only. Composition is the implemented factorized product of marginal class probabilities, predicted nonhazard probability, deterministic reach, slope factor and clipped predicted height productivity; material probabilities trained with terrain-conditional loss are not a general joint posterior. No exact expected-oracle interpretation is asserted.

Strict finite reach>=.05 candidates; finite scores; direct/risk and secondary soft_hazard require finite predicted hazard<=.5. Empty finite eligibility means abstention, with zero executed utility and full feasible regret. No fallback relaxation. Oracle maximum is over the same reachable candidate set without predicted hazard filtering. Empty reach has optimum0; nonfinite authoritative oracle values fail the invariant. Positive-utility oracle picks must be safe. Hard score retains historical hazard<.5. Exact row-major ties and exact threshold boundary counts are logged. Raw regret is primary; normalized regret is undefined when optimum<=.05. Unsafe counts use attempted and executed denominators; zero executed makes the conditional rate undefined. Zero-event bootstrap[0,0] is degenerate.

All models eval(), gradients disabled, parameters frozen, and parameter/buffer digests checked before/after each run. No stochastic augmentation, optimizer, normalization update or calibration fitting. Original BF16-autocast forward is the fresh primary mode. Probabilities are computed in FP32 after head conversion, material probabilities captured before argmax. Immutable source modules are imported without modification.

`precision_keys.csv` fixes 128 historical keys:8 nominal IDs selected without replacement with default_rng(20261012), then8 rough IDs with that same RNG; sorted IDs, all four frames across iid/all_s2/cam_s3 or rough ood_terrain. The 16-key subset is frame0 of the first4 sorted IDs in each pool. R2/G3 best use128 keys, noaff/G2 final and S5A/G6 best use16. Both modes and3 seeds total1920, cap2048. Original contiguous batch16 replay is a separate bounded fidelity operation charged as cache_replay overhead, with original inputs/checkpoints/targets/selectors. Precision itself uses identical selected batch16 membership in both modes and logs any difference from original-batch replay. Additional-model historical probability caches are unavailable and not invented. Actual FP32 has FP32 parameters/inputs, autocast disabled, TF32 disabled for matmul/cuDNN and preconversion head dtype hooks. Genuine precision differences are results; no mode switch or subset expansion follows. A fidelity/dtype failure blocks the affected model interpretations only, with all four primary slots retained.

## Analysis and interpretation lock

Compute seed-specific frame metrics before averaging seeds. For each condition use mean across observed valid frames, then fixed condition weights in every replicate. Draw10000 scene clusters with default_rng(20261011): nominal128 first, rough64 next, then diagnostic subset28. Reuse nominal multiplicities across all related conditions; rough resampling is separate. Preserve partial scenes with actual frame denominators; record undefined replicates and do not discard them. Export multiplicities and primary paired effects. Environmental bootstrap intervals are conditional on these fitted seeds; report per-seed and leave-one-seed-out separately.

Report95% and98.75% marginal percentile intervals with quantiles.025/.975 and.00625/.99375, respectively. Bonferroni controls a fixed family of four under approximate bootstrap coverage, not an exact finite-sample guarantee. Integer unsafe differences and rational linear quantiles have no favorable epsilon. Continuous accumulation uses float64/stable order and no zero tolerance. Negative upper adjusted bounds support lower G/S loss. C joint improvement needs both adjusted upper bounds negative and explicit execution changes. Zero-spanning intervals do not establish equivalence, noninferiority, necessity or safety preservation.

Required secondary results: S best-checkpoint and hazard-filtered soft; existing-output known-invalid camera sensitivity; G3/R2 contextual selectors; masked/unmasked and masked/specialist C controls; all2x2 subset cell means and A-clean,D-clean,AD-A,AD-D,interaction,cam_s3-AD,absence-clean; reachable/visible/clean-fixed/intersection height supports and denominators; precision, utility-target quantization, ties/abstention/nonfinite/low-oracle strata. The completed Stage6 analyses remain historical evidence. No new dropout-necessity or specialist-equivalence claim.

## Conditional work and resources

Only official PP-LiteSeg ppliteseg_class_512 on the existing192 ALICE validation images is conditionally allowed, after official implementation/preprocessing, mapping/ignore rules, rights, byte digest and overlap status are locked. Full64-class per-image confusions must retain false-positive-only classes. The actual historical comparator is kinafford.real.goose_seg.SegNet(64), runs_stage2/real/goose_R2/seed{0,1,2}/segnet.pt, not an auxiliary/BEV head. No compatible output means no head-to-head claim. TASK-MATCHED EXTERNAL FUSION COMPARISON: NOT RUN.

Raw geometry is conditional on Ouster decoding metadata, camera_info/TF, joint units/zero/sign/link conventions and clock mapping for alice_scenario06_sequence02_sensors.bag. Before residuals, earliest eligible complete moving segment, <=60s, <=100 timestamps, with joint motion criterion cumulative change>=5deg in at least one non-swing joint over>=5s; no offset/calibration fitting. Missing prerequisites block only this branch. Shared-source TF/FK agreement is consistency, not independent accuracy. No sensor rig changes or real utility claims.

Hard caps:0training;24GPUhours on device0 only;120000 frame-model forwards including<=2048precision;96CPUcorehours and<=8 workers, raw<=8corehours;72active-wallhours/7elapsed days from lock;5500records;50GB additional storage;weights<=1GB network;ALICE<=14GB network/24GB extraction. Planned forwards58608fresh+1920precision+192conditional=60720 before bounded replay/recovery. Reserve at least4CPUcorehours/2activewallhours for analysis/reporting. Metered jobs enforce caps, preserve failures and incomplete rows. The initial wrapper dependency failure is retained; system Python supplies psutil and launches the unchanged analysis venv. Model construction disables pretrained downloads before strict full-state loading. No extra models/conditions are allowed with unused budget.

Storage preflight uses the existing0.43MB/cache record as a scale reference, not a guarantee: approximately25GB predictions,11GB shared latent arrays,4GB inputs/precision and<=5GB tables/logs/bundle reserve, below50GB; runtime cap remains binding. No full intermediate-activation dump. Precise bytes are logged.

Author questions remain unresolved until attributed dated human answers. No inferred approvals, serving identities or reconciliation. This package ends with results or exact scoped blocked reports, a legacy–v5 crosswalk and claim-by-claim impact memo. No manuscript reconstruction, release or submission.
'''
    (RUN/'validation_protocol_amended.md').write_text(protocol)
    (ROOT/'revision_v5/amendment_2026-10-10.md').write_text('Execution amendment, before fresh outcomes. The user-supplied post-Prompt1 instruction authorizes this bounded package. Original reports are preserved. See validation_run/validation_protocol_amended.md and manifests/artifact_lock.json. Corrections: U10 footprint5x5; C01 scenes versus draws; C03all_s2; R02cam_s3 attribution; all39 references/statuses; visibility semantics; fixed five-mechanism pairing and112-record2x2 completion. F6 remains13/24, F5net0 with234 event changes in each direction. No additional scientific question or training is introduced.\n')
    report='''# Mechanical preflight report

PASS before fresh outcomes. All required final14 checkpoints exist; full inventory and hashes are in manifests/checkpoint_manifest.csv. S5A best epochs42/42/39; G6 best38/36/40. No checkpoint replacement or training.

Nine executable CPU tests pass (logs/preflight_tests_final.log): finite candidates/ties/thresholds, safe/unsafe/oracle/abstention, original preprocessing and complete masking equality, no target leakage, RNG independence/fixed five mechanisms, serialized 2x2/absence identities on a historical pose, known invalid camera-origin detection, exact event arithmetic, fixed weights with unequal condition frame counts, matrix/precision/F6/historical counts (some grouped in one test). The historical fixture is not a fresh test scene and its latent artifact is labeled5000_0.

F6 exact arithmetic was independently rerun from B5_per_frame.csv using the original rng0,2000 scene draws and historical weights. All named corrected artifacts match by parsed value; Q3=13/24 (regret6/12,unsafe7/12). F5R2all_s3 footprint direct/soft each1464/2880; net0;234 soft-only and234 direct-only unsafe events. All49 historical tests copied with original family labels and original source files. The full bounded existing-output camera sensitivity finished before neural inference (288 summary rows plus paired frame differences).

All four primary slots remain fixed. Fresh generation, precision dtype/fidelity, resource and conditional external gates are runtime gates and are not claimed passed by this mechanical report. No fresh outcomes have been seen. Manifest integrity is custody, not scientific proof or external timestamped preregistration.

Sources are the actual repository at commit a38d607588a361360254a98a5c53209c3bdb8bd4. Correct manuscript sources are paper_v4/main_v4.tex and paper_v4/appendix_v4.tex, not main.tex/appendix.tex as stated in part of the original report. Original files remain preserved. Conditional branches have no authority beyond their stated artifact/rights gates.\n'''
    (RUN/'preflight_report.md').write_text(report)
    # Hash every source module potentially imported, source inputs used in the mechanical correction, and historical originals.
    paths=set()
    for folder in ['kinafford','stage2','stage3','stage4','stage5']:
        paths.update((ROOT/folder).rglob('*.py'))
    for rel in ['results_stage6/B/B5_per_frame.csv','results_stage6/B/B5_contrasts.csv','results_stage6/B/B5_reading.csv','results_stage6/B/B4_footprint_per_frame.csv','results_stage2/analysis/per_suite_endpoints.csv','revision_v5/analyses/existing_validity_flags.csv','audit/stage6/scripts/phase2/s6a.py','audit/stage6/scripts/phase2/s6b4.py']:
        paths.add(ROOT/rel)
    for folder in ['results','results_stage2','results_stage4','results_stage5','results_stage6/A']:
        paths.update((ROOT/folder).glob('*_s[012].csv'))
    rows=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths) if p.is_file()]
    csvwrite(RUN/'manifests/source_input_hashes.csv',rows)
    subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu','--format=csv'],stdout=(RUN/'manifests/device_at_lock.txt').open('w'),check=True)
    code=list((RUN/'code').glob('*.py'));files=code+list((RUN/'manifests').glob('*'))+[RUN/'validation_protocol_amended.md',RUN/'preflight_report.md',RUN/'claim_evidence_matrix_corrected.csv',RUN/'matrix_change_log.csv',RUN/'logs/preflight_tests_final.log']
    js(RUN/'manifests/artifact_lock.json',dict(utc=utc(),unix_time=time.time(),source_commit='a38d607588a361360254a98a5c53209c3bdb8bd4',fresh_outcomes_seen=False,files={str(p.relative_to(RUN)):sha(p) for p in sorted(files) if p.is_file()}))
    event('artifact_lock_created',fresh_outcomes_seen=False,files=len(files));print('LOCK CREATED',flush=True)
if __name__=='__main__':main()
