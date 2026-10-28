"""Apply fixed deployed thresholds to recorded frozen-gate FP32 probabilities.

No inference, threshold selection, recalibration or new bootstrap draws.
"""
from pathlib import Path
import csv, hashlib, json, time, datetime
import numpy as np

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
OLD = PROJECT / 'strengthening_2026_10_18/goose_ex'
FINAL = PROJECT / 'final_evidence_2026_10_20'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_csv(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    start = time.time()
    inputs = [OLD/'metrics/real_gate_probabilities.csv', OLD/'metrics/sequence_bootstrap_draws.npz',
              OLD/'resources/real_gate_transfer.json', FINAL/'study4/thresholds.json',
              FINAL/'study4/gate/final.pt', OLD/'metrics/real_gate_rates.csv']
    receipt = json.loads(inputs[2].read_text())
    assert receipt['status'] == 'COMPLETE'
    assert receipt['checkpoint_sha256'] == sha(inputs[4])
    thresholds = json.loads(inputs[3].read_text())
    assert thresholds['KF-D42']['tau'] == .75 and thresholds['BF-D42']['tau'] == .25
    rows = [r for r in csv.DictReader(inputs[0].open()) if r['split'] == 'val']
    groups = sorted({(r['mechanism'], int(r['severity'])) for r in rows})
    ids = {r['image_id'] for r in rows if r['mechanism'] == 'clean'}
    assert len(ids) == 192 and len(groups) == 15 and len(rows) == 2880
    with np.load(inputs[1], allow_pickle=False) as z:
        counts = z['real_counts'].astype(np.int64); sequences = z['real_sequences'].tolist()
    assert counts.shape == (20000, 10) and (counts.sum(1) == 10).all()
    old = list(csv.DictReader(inputs[5].open())); output = []; individual = []
    for model in ['KF-D42', 'BF-D42']:
        tau = thresholds[model]['tau']
        for mechanism, severity in groups:
            rr = [r for r in rows if r['mechanism'] == mechanism and int(r['severity']) == severity]
            assert len(rr) == 192 and {r['image_id'] for r in rr} == ids
            ns = np.zeros(10, np.int64); ks = np.zeros(10, np.int64)
            for r in rr:
                p = float(r['probability_severe']); assert 0 <= p <= 1
                event = int(p >= tau); j = sequences.index(r['sequence']); ns[j] += 1; ks[j] += event
                individual.append(dict(model=model, threshold=tau, **r, removed=event))
            den = counts @ ns; assert (den > 0).all()
            replicates = (counts @ ks) / den
            lo, hi = np.quantile(replicates, [.025, .975])
            result = dict(model=model, threshold=tau, mechanism=mechanism, severity=severity,
                          images=192, removed=int(ks.sum()), rate=float(ks.sum()/192),
                          percent=float(100*ks.sum()/192), lower95=float(lo), upper95=float(hi),
                          lower95_percent=float(100*lo), upper95_percent=float(100*hi),
                          interpretation='original-image removal' if mechanism=='clean' else 'detection under imposed corruption',
                          branch='RGB only; real depth unavailable', interval='descriptive saved-sequence percentile 95%')
            if model == 'BF-D42':
                prior = next(x for x in old if x['downstream_model']=='KF-D' and x['threshold_source']=='sim_tuned'
                             and x['split']=='val' and x['mechanism']==mechanism and int(x['severity'])==severity)
                assert result['removed'] == int(prior['removed'])
                assert np.allclose([lo,hi],[float(prior['ordinary_ci95_low']),float(prior['ordinary_ci95_high'])],rtol=0,atol=1e-15)
            output.append(result)
    write_csv(OUT/'C1/deployed_threshold_real_transfer.csv', output)
    write_csv(OUT/'C1/per_image_threshold_applications.csv', individual)
    lines = ['# Frozen gate at replication-selected RGB thresholds', '',
        'The unchanged FP32 gate probabilities for the original 192 validation images and all 14 corruptions are reused. Thresholds are applied without recalibration. Ordinary 95% descriptive intervals reuse the original 20,000 sequence multiplicities. Real depth is unavailable, so these are RGB-branch results.', '',
        '| Downstream model | Threshold | Mechanism | Severity | Removed / 192 | Removal or detection % [95%] |',
        '|---|---:|---|---:|---:|---:|']
    names={'KF-D42':'KinAfford (42 epochs)','BF-D42':'BEVFusion-Exca (42 epochs)'}
    for r in output:
        lines.append(f"| {names[r['model']]} | {r['threshold']:.2f} | {r['mechanism']} | {r['severity']} | {r['removed']} | {r['percent']:.2f} [{r['lower95_percent']:.2f}, {r['upper95_percent']:.2f}] |")
    (OUT/'C1/C1_table.md').write_text('\n'.join(lines)+'\n')
    result=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method='Exact threshold application to saved unchanged-network probabilities; no neural replay needed.',
        input_sha256={str(p.relative_to(PROJECT)):sha(p) for p in inputs},
        original_images=192,conditions=15,rows=30,per_image_applications=len(individual),
        unchanged_checkpoint_sha256=receipt['checkpoint_sha256'],saved_draws=20000,new_draws=0,
        new_neural_forwards=0,gpu_hours=0,recalibration=False,
        old_tau025_all_conditions_and_intervals_match=True,seconds=time.time()-start)
    (OUT/'C1/receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([r for r in output if r['mechanism']=='clean'],indent=2))

if __name__ == '__main__': main()
