"""Full prescribed matrix only, with final pre-generation lock and no auto-repair."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
PY=PROJECT/'external_validation_run_EV1_AFAST/environment/venv/bin/python'
def command(name,script,args=(),gpu=False,cores=4,forecast=None,basis=None):
    cmd=[sys.executable,'-B',str(ROOT/'code/run_job.py'),'--name',name,'--cores',str(cores)]
    if gpu:cmd.append('--gpu')
    if forecast is not None:cmd+=['--forecast-seconds',str(forecast),'--forecast-source',basis]
    return cmd+['--',str(PY),'-B',str(ROOT/'code'/script),*map(str,args)]
def run(*args,**kw):subprocess.run(command(*args,**kw),check=True)
def wait_for(path):
    while not path.exists():time.sleep(30)
    return json.loads(path.read_text())
def main():
    wait_for(ROOT/'records/training_and_F_completion.json')
    assert wait_for(ROOT/'resources/job_E_full_validation_input_audit.json')['returncode']==0
    assert not (ROOT/'study4/holdout').exists()
    run('E_checkpoint_inventory','checkpoint_inventory.py',cores=2,forecast=120,
        basis='Recorded checkpoint preload/verification pilots under30s per architecture; six new plus six unchanged files,120s bound includes25percent margin')
    run('E_gate_validation','select_gate_thresholds.py',['predict'],gpu=True)
    for a in ['KF-D','BF-D']:
        for s in [10,11,12]:
            run(f'E_validation_{a}_seed{s}','validation_subsets.py',['--architecture',a+'42','--seed',s],gpu=True)
    run('E_select_thresholds','select_gate_thresholds.py',['select'],cores=2,forecast=300,
        basis='Recorded25pair selection2.12s on750records, scaled9600/750 and extra input IO, conservative300s with25percent margin')
    run('E_final_artifact_lock','lock_study4.py',cores=2,forecast=120,
        basis='Recorded protocol seal3.18s plus hashing12checkpoint files and native sources, conservative120s including25percent margin')
    run('E_generate_locked_holdout','generate_study4.py',['holdout','--workers',4],cores=4,forecast=4000,
        basis='Maximum recorded record-rate bound: current9600record development replay1624.88s scaled17408/9600, plus60s setup, times1.25; conservative4000s CPU-only bound, no scientific scope change')
    run('E_holdout_validity','audit_study4.py',cores=4,forecast=600,
        basis='Recorded retained-cohort traversal243.70s plus153.39s authorized audit,17376to17408records, times1.25 plus draw creation')
    run('E_gate_holdout','infer_study4.py',['gate_scores'],gpu=True)
    for a in ['KF-D','BF-D']:
        for suffix,kind in [('42','new'),('14','old')]:
            for s in [10,11,12]:
                run(f'E_holdout_{kind}_{a}_seed{s}','infer_study4.py',['model','--architecture',a+suffix,'--seed',s],gpu=True)
    analysis=subprocess.Popen(command('E_primary_and_secondaries','study4_statistics.py',cores=2,forecast=1800,
        basis='Recorded full Study3 analysis70.95s, complete expanded contrasts up to eightfold with25percent margin; conservative1800s'))
    diagnostic=subprocess.Popen(command('E_all_diagnostics','study4_diagnostics.py',['--workers',6],cores=6,forecast=11000,
        basis='Recorded Study3 diagnostic4268.55s, at most doubled learned arms and same scene population, times1.25; no credit for removing utility grid or identical-map reuse'))
    rc1=analysis.wait();rc2=diagnostic.wait();assert rc1==0 and rc2==0,('affected analysis/diagnostic artifact blocked',rc1,rc2)
    run('E_map_metric_comparisons','study4_map_metrics.py',cores=4,forecast=1200,
        basis='Recorded full scene-bootstrap diagnostic tables477.62s, new nonlinear confusion bootstrap4modeltypes and paired maps bounded at2x plus25percent margin')
    run('E_gate_operating_inventory','gate_operating_tables.py',cores=2,forecast=120,
        basis='Recorded gate operating inventory14.50s doubled architectures and same records plus25percent margin and IO allowance')
    run('E_diagnostic_tables','study4_diagnostic_tables.py',cores=4,forecast=1800,
        basis='Recorded Study3 diagnostic reduction477.62s, doubled arms plus five-cutoff inventory, conservative3x times1.25')
    (ROOT/'study4/campaign_completion.json').write_text(json.dumps(dict(status='ALL_LOCKED_STUDY4_JOBS_COMPLETE',finish_unix=time.time(),automatic_follow_on=False),indent=2)+'\n')
if __name__=='__main__':main()
