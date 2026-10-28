"""Verify all completed prescribed fits; choose epoch42 without outcome access."""
from common import *
import torch
def main():
    out=ROOT/'protocol/checkpoint_inventory.json';assert not out.exists()
    rows=[]
    for a in ['KF-D','BF-D']:
        for seed in [10,11,12]:
            d=ROOT/f'checkpoints/{a}/seed{seed}';receipt=json.loads((d/'completion.json').read_text())
            resource=json.loads((ROOT/f'resources/job_E_fit_{a}_seed{seed}.json').read_text());assert resource['returncode']==0
            cp=d/'epoch42.pt';state=torch.load(cp,map_location='cpu',weights_only=False)
            assert state['epochs']==42 and state['step']==31500 and state['precision']=='actual_fp32' and not state['tf32']
            assert state['counts']['training_frame_exposures']==504000
            loading=json.loads((d/'pretrained_loading.json').read_text());assert loading['status'].startswith('VERIFIED') or loading['status'].startswith('PASS')
            assert all((d/f'epoch{e:02d}.pt').exists() for e in [14,28,42])
            assert all(t.dtype==torch.float32 for t in state['model'].values() if t.is_floating_point())
            rows.append(dict(architecture=a+'42',seed=seed,path=str(cp.relative_to(PROJECT)),sha256=sha(cp),selection='prespecified epoch42 only',
                training_precision='actual FP32',epochs=42,training_frame_exposures=504000,pretrained_loading_receipt=str((d/'pretrained_loading.json').relative_to(ROOT)),
                pretrained_loading_receipt_sha256=sha(d/'pretrained_loading.json'),completion_receipt_sha256=sha(d/'completion.json')))
    old=json.loads((PROJECT/'strengthening_2026_10_18/protocol/checkpoint_inventory.json').read_text())
    for r in old['checkpoints']:
        if r['architecture'] not in ['KF-D','BF-D']:continue
        assert sha(PROJECT/r['path'])==r['sha256'];r=dict(r);r['architecture']+='14';rows.append(r)
    assert len(rows)==12
    js(out,dict(status='VERIFIED_SIX_EPOCH42_PLUS_SIX_UNCHANGED_EPOCH14',evaluation_precision='actual FP32',checkpoints=rows,selection_uses_validation_outcomes=False))
    print('PASS checkpoint inventory12',flush=True)
if __name__=='__main__':main()
