"""Execute only the already locked prospective matrix; no automatic repairs."""
from pathlib import Path
import subprocess,json,sys,time
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
PYTHON=PROJECT/'external_validation_run_EV1_AFAST/environment/venv/bin/python'

def job(name,script,args=(),gpu=False,cores=4,forecast=60,source=''):
    cmd=[sys.executable,str(ROOT/'code/run_job.py'),'--name',name,'--phase','study3','--cores',str(cores),
        '--forecast-seconds',str(forecast),'--forecast-source',source]
    if gpu:cmd.append('--gpu')
    return cmd+['--',str(PYTHON),str(ROOT/'code'/script),*map(str,args)]

def run(*args,**kw):subprocess.run(job(*args,**kw),check=True)

def main():
    lock=json.loads((ROOT/'protocol/study3_final_lock.json').read_text());assert lock['status']=='LOCKED_BEFORE_HOLDOUT_GENERATION'
    # Use the measured750-record validation generation cost without assuming a
    # parallel speedup from eight workers. This already includes25%margin.
    measured=json.loads((ROOT/'resources/job_B_validation_bundles.json').read_text())['wall_seconds']
    run('B_generate_locked_holdout','generate_study3.py',['holdout','--workers',8],cores=8,
        forecast=1.25*measured*17408/750,source='Measured750-record validation generation wall time scaled by17408/750; times1.25; no assumed worker speedup')
    run('B_holdout_validity','audit_study3_generation.py',cores=2,forecast=900,
        source='750-record prelock hash/data control6.9s scaled to17408 plus external source/checkpoint hashing and margin')
    run('B_holdout_gate_scores','infer_study3.py',['gate_scores'],gpu=True,cores=4,forecast=300,
        source='Measured750-record gate validation scoring scaled to17408 plus startup; 25percent margin included')
    for architecture,seeds in [('KF-D',[10,11,12]),('BF-D',[10,11,12]),('S5A',[0,1,2]),('G6',[0,1,2])]:
        rate=.07429090716781363 if architecture=='BF-D' else .048018880884910686
        arms=4 if architecture in ['KF-D','BF-D'] else 1
        forecast=1.25*(17408*arms*rate+30)
        for seed in seeds:
            run(f'B_holdout_{architecture}_seed{seed}','infer_study3.py',['model','--architecture',architecture,'--seed',seed],gpu=True,cores=4,forecast=forecast,
                source=f'17408records x{arms}logicalarms x historical maximum rate{rate}s plus30s setup, times1.25; identical-input reuse not credited')
    # The two jobs read complete frozen outputs independently. Six+two workers
    # bound this stage to eight CPU workers; neither launches GPU operations.
    analysis=subprocess.Popen(job('B_primary_and_secondaries','study3_statistics.py',cores=2,forecast=3600,
        source='Historical full reporting50.54s scaled conservatively for larger scene matrices and full new contrast inventory; margin included'))
    diagnostics=subprocess.Popen(job('B_all_diagnostics','study3_diagnostics.py',['--workers',6],cores=6,
        forecast=1.25*(849.1094183921814+689.6507439613342)*17408/3328*2,
        source='Measured PhaseA attribution+sensitivity wall times scaled by record ratio and four versus two main arms, times1.25; no cache/worker speedup credited'))
    a=analysis.wait();b=diagnostics.wait();assert a==0 and b==0,('analysis_or_diagnostics_failed',a,b)
    run('B_gate_operating_inventory','gate_operating_tables.py',cores=2,forecast=300,
        source='Stored-array metadata reads at measured validation IO rate plus table export and25percent margin')
    run('B_diagnostic_tables','study3_diagnostic_tables.py',cores=4,forecast=7200,
        source='Historical50.54s full diagnostic reduction and20.73s sensitivity table runtime scaled for fivefold records, fourfold scene draws and complete new arms; conservative25percent margin included')
    (ROOT/'study3/campaign_completion.json').write_text(json.dumps(dict(status='ALL_LOCKED_JOBS_COMPLETE',finish_unix=time.time(),automatic_follow_on=False),indent=2)+'\n')
if __name__=='__main__':main()
