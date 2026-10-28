"""Execute only the fixed six fits and bounded pretrained image check."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
PY=PROJECT/'external_validation_run_EV1_AFAST/environment/venv/bin/python'
def job(name,script,args=(),gpu=True,forecast=None,basis=None):
    cmd=[sys.executable,'-B',str(ROOT/'code/run_job.py'),'--name',name,'--cores','4']
    if gpu:cmd.append('--gpu')
    if forecast is not None:cmd+=['--forecast-seconds',str(forecast),'--forecast-source',basis]
    subprocess.run(cmd+['--',str(PY),'-B',str(script),*map(str,args)],check=True)
def main():
    first=ROOT/'resources/job_E_fit_KF-D_seed10.json'
    while not first.exists():time.sleep(30)
    assert json.loads(first.read_text())['returncode']==0,'First formal fit failed; no automatic repair'
    assert json.loads((ROOT/'image_check/protocol/ready_to_execute.json').read_text())['status']=='READY_FIXED24_EPOCHS'
    seg=ROOT/'image_check/code/segmentation.py';novel=ROOT/'code/faults.py'
    job('F_training_only_pilot',seg,['pilot','--domain','real','--resource-id','F_training_only_pilot'])
    for domain in ['real','sim']:
        job('F_fit_'+domain,seg,['fit','--domain',domain,'--epochs',24,'--resource-id','F_fit_'+domain])
    for domain,res in [('sim','parity'),('real','parity'),('real','width512')]:
        name=f'F_evaluate_{domain}_{res}'
        job(name,seg,['evaluate','--domain',domain,'--resolution',res,'--batch',1 if res=='width512' else 16,'--novel-module',novel,'--resource-id',name])
    # Image statistics proceed on CPU while the next locked perception fit runs.
    analysis=subprocess.Popen([sys.executable,'-B',str(ROOT/'code/run_job.py'),'--name','F_descriptive_statistics','--cores','2','--forecast-seconds','300','--forecast-source','Recorded earlier full sequence-bootstrap analysis plus three versus two domain/resolution tables and25percent margin','--',str(PY),'-B',str(ROOT/'image_check/code/analyze.py')])
    for architecture,seed in [('BF-D',10),('KF-D',11),('BF-D',11),('KF-D',12),('BF-D',12)]:
        job(f'E_fit_{architecture}_seed{seed}',ROOT/'code/train_final.py',['--architecture',architecture,'--seed',seed])
    assert analysis.wait()==0,'Image statistical implementation failed; preserve record'
    (ROOT/'records/training_and_F_completion.json').write_text(json.dumps(dict(status='ALL_SIX42EPOCH_FITS_AND_F_COMPLETE',automatic_follow_on=False),indent=2)+'\n')
if __name__=='__main__':main()
