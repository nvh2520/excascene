"""Serial, bounded validation jobs; abort on any failure, no automatic recovery."""
from pathlib import Path
import subprocess,time,json,sys
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
first=ROOT/'resources/job_B_val_BF_seed10.json'
while not first.exists():time.sleep(5)
assert json.loads(first.read_text())['returncode']==0,'First validation job failed'
python=PROJECT/'external_validation_run_EV1_AFAST/environment/venv/bin/python'
for arch,seeds in [('BF-D',[11,12]),('KF-D',[10,11,12]),('S5A',[0,1,2])]:
    rate=.07429090716781363 if arch=='BF-D' else .048018880884910686
    forecast=1.25*(6000*rate+30)
    name=f'B_val_{arch.replace("-D","")}_seed{seeds[0]}'
    for seed in seeds:
        name=f'B_val_{arch.replace("-D","")}_seed{seed}'
        cmd=[sys.executable,str(ROOT/'code/run_job.py'),'--name',name,'--phase','gate','--gpu','--cores','4',
            '--forecast-seconds',str(forecast),'--forecast-source','6000 validation subset forwards at historical architecture maximum rate plus30s setup, times1.25',
            '--',str(python),str(ROOT/'code/validation_subsets.py'),'--architecture',arch,'--seed',str(seed)]
        subprocess.run(cmd,check=True)
subprocess.run([sys.executable,str(ROOT/'code/run_job.py'),'--name','B_select_thresholds','--phase','gate','--cores','2',
    '--forecast-seconds','120','--forecast-source','75 fixed threshold-grid lookups from complete validation-only subset exports; CPU forecast includes25%margin',
    '--',str(python),str(ROOT/'code/select_gate_thresholds.py'),'select'],check=True)
