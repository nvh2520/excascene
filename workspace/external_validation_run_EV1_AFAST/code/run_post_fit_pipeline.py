"""Authorized release/evaluation/closure after all six finals; no extra fits."""
import json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
clock=json.loads((ROOT/'records/start_clock.json').read_text())
def elapsed():return max(time.time()-clock['t0_unix'],time.monotonic()-clock['t0_monotonic'])/3600
def bounded(name,script,args=(),gpu=0,stage=36):
    assert elapsed()<stage,'Stage deadline reached before '+name
    events=[json.loads(x) for x in (ROOT/'resources/jobs.jsonl').read_text().splitlines()]
    complete=[e for e in events if e['event']=='finish']
    assert sum(e['occupied_gpu_device_hours'] for e in complete)+.1<64
    assert sum(e['cpu_core_hours'] for e in complete)+16<240
    print('POST_STAGE',name,'elapsed_h',round(elapsed(),4),flush=True)
    cmd=[sys.executable,str(ROOT/'code/run_job.py'),'--name',name,'--gpu',str(gpu),'--threads','4' if gpu else '1','--hours',str(stage-elapsed()),'--',
        str(ROOT/'environment/venv/bin/python'),str(ROOT/'code'/script),*args]
    subprocess.run(cmd,check=True)
try:
    while not (ROOT/'records/all_six_final_checkpoints.json').exists():
        assert elapsed()<26,'All-final checkpoint stage deadline reached'
        assert not (ROOT/'records/formal_schedule_failure.json').exists(),'Formal fit failed; no substitute seed or extra run'
        time.sleep(30)
    assert (ROOT/'records/sealed_holdout_audit.json').exists(),'Sealed content audit is missing'
    bounded('final_checkpoint_input_release','release_holdout.py',stage=26)
    for seed in [10,11,12]:
        for arch in ['BF-D','KF-D']:
            bounded(f'{arch}_seed{seed}_holdout_inference','infer_holdout.py',('--architecture',arch,'--seed',str(seed)),gpu=1,stage=32)
    bounded('fixed_scalar_analysis','analyze_afast.py',stage=36)
    bounded('independent_scalar_reconstruction','reconstruct_primary.py',stage=36)
    bounded('single_frame_and_batch4_deployment_timing','measure_deployment.py',gpu=1,stage=36)
    bounded('final_scientific_report_and_evidence','close_study.py',stage=36)
except Exception as e:
    receipt=dict(status='INCOMPLETE',phase='post_fit_pipeline',error=repr(e),elapsed_hours=elapsed(),utc_unix=time.time())
    for p in (ROOT/'records').glob('inference_*.json'):
        r=json.loads(p.read_text())
        if 'INVALID_NONFINITE_OR_PRECISION' in r.get('error',''):receipt['status']='INVALID'
    (ROOT/'records/post_fit_pipeline_failure.json').write_text(json.dumps(receipt,indent=2))
    (ROOT/'final_status.json').write_text(json.dumps(receipt,indent=2))
    report=ROOT/'AFAST_results.md'
    if report.exists():report.write_text(report.read_text().replace('**Status: COMPLETE.**',f'**Status: {receipt["status"]}.** Closure failed; use the failure receipt for the authoritative status.'))
    else:report.write_text(f'# EV1-AFAST results\n\n**Status: {receipt["status"]}.**\n\nThe authorized six-fit study did not complete valid closure. No negative scientific result is inferred from this failure.\n\nExact block: {e!r}\n\nSee `records/post_fit_pipeline_failure.json`, formal fit completion/failure receipts, and full `logs/`. The original nine-fit draft, manuscript and historical endpoints remain preserved. No automatic extension or substitute seed is authorized.\n')
    raise
