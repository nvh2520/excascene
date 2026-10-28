"""Wait for development generation; perform the prescribed input-only audit."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
receipt=ROOT/'resources/job_E_full_validation_replay_metadata_recovery.json'
while not receipt.exists():time.sleep(30)
assert json.loads(receipt.read_text())['returncode']==0,'Development generation failed; no automatic repair'
subprocess.run([sys.executable,'-B',str(ROOT/'code/run_job.py'),'--name','E_full_validation_input_audit','--cores','4',
    '--forecast-seconds','600','--forecast-source','Recorded17376record validity traversal243.70s, scaled9600records, plus original1200target comparisons and25percent margin; conservative600s bound',
    '--',str(PROJECT/'external_validation_run_EV1_AFAST/environment/venv/bin/python'),'-B',str(ROOT/'code/audit_validation.py')],check=True)
