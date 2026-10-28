"""Sequential six-fit schedule on the single actual A100. Stops on any failure."""
import json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
clock=json.loads((ROOT/'records/start_clock.json').read_text())
plan=[(seed,arch) for seed in [10,11,12] for arch in ['BF-D','KF-D']]
for seed,arch in plan:
    assert time.time()<clock['t0_unix']+26*3600,'All-final checkpoint deadline reached'
    name=arch.replace('-','_')+f'_formal_seed{seed}'
    remaining=(clock['t0_unix']+26*3600-time.time())/3600
    cmd=[sys.executable,str(ROOT/'code/run_job.py'),'--name',name,'--gpu','1','--hours',str(remaining),'--',
         str(ROOT/'environment/venv/bin/python'),str(ROOT/'code/train_formal.py'),'--architecture',arch,'--seed',str(seed)]
    print('FORMAL_LAUNCH',arch,seed,flush=True)
    r=subprocess.run(cmd)
    if r.returncode:
        (ROOT/'records/formal_schedule_failure.json').write_text(json.dumps(dict(architecture=arch,seed=seed,returncode=r.returncode,utc_unix=time.time()),indent=2))
        raise SystemExit(r.returncode)
receipt=dict(status='ALL_SIX_FINALS_COMPLETE',utc_unix=time.time(),fits=[json.loads((ROOT/'checkpoints'/arch/f'seed{seed}/completion.json').read_text()) for seed,arch in plan])
(ROOT/'records/all_six_final_checkpoints.json').write_text(json.dumps(receipt,indent=2));print('ALL_SIX_FINALS_COMPLETE',flush=True)
