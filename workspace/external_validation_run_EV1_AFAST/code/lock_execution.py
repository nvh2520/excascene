"""Outcome-blind timing tier selection and pre-fit artifact lock."""
import hashlib,json,pathlib,time,math
ROOT=pathlib.Path(__file__).resolve().parents[1]
clock=json.loads((ROOT/'records/start_clock.json').read_text())
elapsed=(time.time()-clock['t0_unix'])/3600
assert elapsed<6,'Hour6 gate missed'
def read(rel):return json.loads((ROOT/rel).read_text())
for rel in ['tests/native_operator_receipt.json','tests/adapter_preflight_receipt.json',
            'tests/adapter_additional_controls.json','tests/coordinate_mask_controls.json',
            'tests/generator_selector_controls.json','manifests/data_inventory_receipt.json']:
    assert read(rel)['status']=='PASSED',(rel,'gate_not_passed')
assert read('manifests/collision_audit_receipt.json')['status']=='PASSED_NO_ID_COLLISION'
bf=read('reports/pilots/BF-D/attempt_2/pilot_receipt.json');kf=read('reports/pilots/KF-D/attempt_2/pilot_receipt.json')
assert bf['status']==kf['status']=='PASSED'
# Tier selection reads only measured timing/resources, never loss/metric values.
times=[bf['seconds_per_update_conservative'],kf['seconds_per_update_conservative']]
validation=[bf['validation_seconds'],kf['validation_seconds']]
io=[bf['checkpoint_io_seconds'],kf['checkpoint_io_seconds']]
generation=read('reports/generation_timing.json')['forecast_generation_seconds']/3600
eval_seconds=19968*(bf['export_seconds']/bf['export_frames']+kf['export_seconds']/kf['export_frames'])
choices=[]
for E in [42,28,14]:
    updates=3*750*E*sum(times)
    val=3*E*sum(validation)
    # Four resume saves/epoch, three tensor copies/model+optimizer, plus initial/final.
    checkpoints=3*(E*4*3+2)*sum(io)
    # Explicit fixed work allowance for actual-input hashes and orchestration;
    # it supplements rather than consumes the mandatory25% runtime margin.
    overhead=15*60
    formal_seconds=updates+val+checkpoints+overhead
    final_by=elapsed+1.25*formal_seconds/3600
    eval_by=max(final_by,elapsed+1.25*generation)+1.25*(eval_seconds+10*60)/3600
    report_by=eval_by+4
    occupied_forecast=1.25*(formal_seconds+eval_seconds+10*60)/3600+.5
    row=dict(epochs=E,seconds_per_update_BF=times[0],seconds_per_update_KF=times[1],
             update_seconds=updates,epoch_validation_seconds=val,checkpoint_io_seconds=checkpoints,
             fixed_hash_orchestration_seconds=overhead,margin=1.25,final_checkpoint_elapsed_hour=final_by,
             evaluation_elapsed_hour=eval_by,report_reserve_hours=4,report_elapsed_hour=report_by,
             occupied_gpu_device_hour_forecast=occupied_forecast,
             fits=(final_by<=26 and eval_by<=32 and report_by<=36 and occupied_forecast<=64))
    choices.append(row)
selected=next((r for r in choices if r['fits']),None)
status='LOCKED' if selected else 'BLOCKED_36H_FEASIBILITY'
forecast=dict(status=status,selection_basis='Measured runtime/resources only; no outcome values read',
              actual_prelaunch_elapsed_hours=elapsed,one_gpu=True,choices=choices,selected_epochs=selected['epochs'] if selected else None,
              measured_generation_forecast_hours=generation,measured_pilot_updates={'BF-D':288,'KF-D':288},
              planned_cpu_core_hour_bound=120,planned_incremental_disk_gb_bound=35,
              expected_elapsed_hours_from_means=elapsed+(3*750*(selected['epochs'] if selected else 14)*(bf['seconds_per_update_mean']+kf['seconds_per_update_mean'])+3*(selected['epochs'] if selected else 14)*sum(validation)+eval_seconds+15*60)/3600+1,
              target_hours=24,hard_hours=36,utc_unix=time.time())
(ROOT/'reports/timing_epoch_selection.json').write_text(json.dumps(forecast,indent=2))
assert selected,'BLOCKED_36H_FEASIBILITY: no14epoch six-fit design fits'
E=selected['epochs']
protocol=ROOT/'frozen_protocol.md'
protocol.write_text(protocol.read_text().replace('prospective protocol — gate pending',f'frozen protocol — {E} epochs').replace('E is not selected yet.',f'E={E} is selected from measured runtime only; all six fits use {E*750} updates.'))
paths=[p for p in (ROOT/'code').glob('*.py') if p.name not in ['lock_execution.py','run_job.py','collision_audit.py','support_audit.py']]
paths += [ROOT/'frozen_protocol.md',ROOT/'manifests/generation_manifest.json',ROOT/'manifests/train_validation_inventory.csv',ROOT/'source/selectors_v5_original.py']
paths += [p for p in (ROOT/'source/excascene_snapshot').rglob('*.py')]
paths += [p for p in (ROOT/'source/bevfusion_adapted').rglob('*') if p.is_file() and p.suffix in ['.py','.h','.cc','.cu','.so']]
lock=dict(status='AUTHORIZED_GATES_PASSED',epochs=E,seeds=[10,11,12],architectures=['BF-D','KF-D'],formal_fits=6,
          updates_per_fit=750*E,formal_update_total=6*750*E,frame_exposure_total=6*12000*E,
          primary_endpoints=['AFAST-C-R','AFAST-C-U'],primary_quantiles=['1/40','39/40','1/80','79/80'],
          epoch_selection='reports/timing_epoch_selection.json',holdout_access='forbidden until all six final checkpoints and analysis code hashed',
          locked_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},utc_unix=time.time())
(ROOT/'manifests/execution_lock.json').write_text(json.dumps(lock,indent=2))
(ROOT/'records/hour6_adapter_launch_gate.json').write_text(json.dumps(dict(status='PASSED',actual_elapsed_hours=elapsed,
        epochs=E,operator_fwd_bwd=True,geometry_and_modality_controls=True,both_pilots=True,
        timing_selection='reports/timing_epoch_selection.json',formal_fits_started=False,holdout_generated=False,utc_unix=time.time()),indent=2))
print(json.dumps(forecast,indent=2))
