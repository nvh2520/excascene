"""Read-only progress monitor. No model/holdout access or study modification."""
import json,os,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.sched_setaffinity(0,[min(os.sched_getaffinity(0))])
clock=json.loads((ROOT/'records/start_clock.json').read_text())
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
while True:
    elapsed=max(time.time()-clock['t0_unix'],time.monotonic()-clock['t0_monotonic'])/3600
    final=read(ROOT/'final_status.json')
    if final:
        print('EV1-AFAST final status:',final['status'],'| elapsed',round(elapsed,3),'hours',flush=True);break
    fits=[];active=[]
    for seed in [10,11,12]:
        for arch in ['BF-D','KF-D']:
            directory=ROOT/'checkpoints'/arch/f'seed{seed}'
            complete=read(directory/'completion.json')
            if complete:fits.append((arch,seed))
            elif read(directory/'progress.json'):
                p=read(directory/'progress.json');active.append(f'{arch} seed{seed}: epoch{p.get("epoch","?")}/14, update{p.get("step","?")}/10500')
    pred=[read(ROOT/'records'/f'inference_{arch}_seed{seed}.json') for arch in ['BF-D','KF-D'] for seed in [10,11,12]]
    forwards=sum(p.get('completed_logical_forwards',0) for p in pred)
    progress='; '.join(active) if active else 'setup between fixed fits'
    if (ROOT/'records/holdout_prediction_release.json').exists():progress=f'holdout forwards {forwards}/39936'
    print(f'Elapsed {elapsed:.3f}h | final fits {len(fits)}/6 | {progress} | final/evaluation/report deadlines 26/32/36h',flush=True)
    time.sleep(45)
