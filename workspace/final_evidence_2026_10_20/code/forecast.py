"""Whole authorized round forecast from recorded jobs; no outcome access."""
from pathlib import Path
import json,datetime,argparse
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OLD=PROJECT/'strengthening_2026_10_18'
jobs={}
def old(name):
    p=OLD/f'resources/job_{name}.json';r=json.loads(p.read_text());assert r['returncode']==0
    jobs[name]=dict(path=str(p.relative_to(PROJECT)),wall_seconds=r['wall_seconds'])
    return r['wall_seconds']
rows=[json.loads(s) for s in (PROJECT/'external_validation_run_EV1_AFAST/resources/jobs.jsonl').read_text().splitlines()]
fits={a:max(r['elapsed_seconds'] for r in rows if r['event']=='finish' and r.get('returncode')==0 and r['name'].startswith(a.replace('-','_')+'_formal_seed')) for a in ['KF-D','BF-D']}
rates={}
for a in ['KF-D','BF-D']:
    values=[]
    for s in [10,11,12]:
        t=old(f'B_holdout_{a}_seed{s}');m=json.loads((OLD/f'study3/metrics/{a}_seed{s}.json').read_text())
        values.append(t/m['actual_neural_forwards'])
    rates[a]=max(values)
valrates={a:max(old(f'B_val_{a[:2]}_seed{s}')/6000 for s in [10,11,12]) for a in ['KF-D','BF-D']}
plan=[]
def add(name,seconds,basis):
    plan.append(dict(name=name,base_seconds=seconds,forecast_seconds=1.25*seconds,forecast_hours=1.25*seconds/3600,basis=basis))
for a in ['KF-D','BF-D']:
    for s in [10,11,12]:
        add(f'E_fit_{a}_seed{s}',3*fits[a]+30,'Maximum recorded14epoch fit of same graph, scaled42/14, plus30s initialization; all epochs include original1200frame validation')
        add(f'E_validation_{a}_seed{s}',9600*8*valrates[a]+30,'Full1200 nominal pose bank across seven known conditions plus1200rough replay; all8subsets; maximum recorded validation seconds/forward')
        add(f'E_holdout_new_{a}_seed{s}',17408*6*rates[a]+30,'All6logical arms costed as physically distinct, no duplicate credit; maximum recorded actual-forward cost including map IO')
        add(f'E_holdout_old_{a}_seed{s}',17408*2*rates[a]+30,'Both frozen14epoch arms costed distinct; maximum recorded actual-forward cost including map IO')
for a in ['KF-D','BF-D']:
    add(f'E_training_only_pilot_{a}',240,'Recorded same-graph fit startup and first updates; bounded compatibility/backward pilot, no hyperparameter selection')
add('E_gate_validation',old('B_gate_validation_scores')*9600/750+15,'Same frozen gate; measured750validation records scaled9600')
add('E_gate_holdout',old('B_holdout_gate_scores')*17408/17376+15,'Same frozen gate; measured17376records scaled17408')
fit=max(old('C_real_fit'),old('C_sim_fit'))
for d in ['real','sim']:add(f'F_fit_{d}',fit+30,'Same24epoch MiT-B0 graph, maximum recorded domain fit; pretrained load only changes initialization')
ev=max(old('C_real_evaluate'),old('C_sim_evaluate'))
for d in ['real','sim']:add(f'F_evaluate_{d}_parity',ev+15,'Same15conditions and256x160 arrays; maximum recorded full-domain evaluation')
add('F_evaluate_real_width512',25*ev+30,'Recorded maximum full-domain evaluation;512x384 versus256x160 area4.8x, conservative25x attention-work bound plus30s setup,25percent margin')
add('F_training_only_pilot',180,'Recorded SegFormer pilot; bounded preload/FP32 and resolution compatibility timing')
total=sum(r['forecast_hours'] for r in plan)
result=dict(status='FITS_CAP' if total<=75 else 'STOP_FORECAST_EXCEEDS_CAP',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_cap_hours=75,margin=1.25,total_forecast_gpu_hours=total,
    jobs=plan,sources=jobs,maximum_recorded14epoch_fit_seconds=fits,maximum_actual_forward_seconds=rates,maximum_validation_forward_seconds=valrates,
    planned_test_records=17408,planned_test_logical_forwards=835584,validation_records=9600,
    fixed_epoch_budgets=dict(perception=42,image_check=24),no_holdout_outcomes_read=True,
    assumptions='Unchanged graphs/FP32/microbatch and accumulation; training-label CE weights add no new head. Test every logical arm is costed separately; implementation may reuse identical inputs. CPU generation/statistics do not occupy GPU. Every job is checked against actual usage plus remaining reservations.')
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'resources/reconstructed_forecast_v2.json');args=parser.parse_args()
assert not args.output.exists(),'Preserve prior forecasts; choose a new output path'
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','total_forecast_gpu_hours','planned_test_records','planned_test_logical_forwards','validation_records']},indent=2))
assert total<=75,'Stop before GPU work; complete scope does not fit'
