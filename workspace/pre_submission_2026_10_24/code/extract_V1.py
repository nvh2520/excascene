"""Read existing descriptive contrasts and operating counts only; no new draws."""
from pathlib import Path
import csv,hashlib,json
OUT=Path(__file__).resolve().parents[1];PROJECT=OUT.parent
F=PROJECT/'final_evidence_2026_10_20';S=PROJECT/'strengthening_2026_10_18'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvw(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 sources=[F/'study4/analysis/sensor_impact_comparisons.csv',F/'study4/analysis/gate_operating_inventory.csv',F/'study4/analysis/gate_per_camera.csv',S/'study3/analysis/gate_per_camera.csv']
 cost=[r for r in csv.DictReader(sources[0].open()) if r['architecture'] in ['KF-D42','BF-D42'] and r['condition']=='novel' and r['policy']=='direct']
 assert len(cost)==4;csvw(OUT/'V1/novel_keep_all_costs.csv',cost)
 inventory=[r for r in csv.DictReader(sources[1].open()) if r['architecture'] in ['KF-D42','BF-D42'] and r['condition']=='novel' and r['novel']=='1']
 assert len(inventory)==2;csvw(OUT/'V1/novel_camera_detection_inventory.csv',inventory)
 proof=[]
 for path,arches in [(sources[2],['KF-D42','BF-D42']),(sources[3],['KF-D','BF-D'])]:
  rows=list(csv.DictReader(path.open()))
  for a in arches:
   rr=[r for r in rows if r['architecture']==a and r['condition']=='cam_s3'];assert rr
   mismatches=sum((int(r['removed']) or not int(r['present'])) != (int(r['supplied_oracle_drop']) or not int(r['present'])) for r in rr)
   assert mismatches==0;proof.append(dict(source=str(path.relative_to(PROJECT)),architecture=a,rows=len(rr),mask_mismatches=0))
 (OUT/'V1/severe_mask_equality.json').write_text(json.dumps(proof,indent=2)+'\n')
 lines=['# V1 — saved-output descriptive context','',
 'No model, selector, score or interval was rerun. These rows are copied from the existing replication sensor-impact contrasts. Intervals are the saved ordinary 95% paired scene-bootstrap intervals. Detection uses the existing operating-inventory removal fraction for the one present camera carrying a withheld fault.','',
 '| Model | Novel minus clean keep-all regret [95%] | Unsafe difference, percentage points [95%] | Novel-camera gate detections | Detection % |',
 '|---|---:|---:|---:|---:|']
 for a,name in [('KF-D42','KinAfford, 42 epochs'),('BF-D42','BEVFusion-Exca, 42 epochs')]:
  r=next(x for x in cost if x['architecture']==a and x['metric']=='regret');u=next(x for x in cost if x['architecture']==a and x['metric']=='unsafe');g=next(x for x in inventory if x['architecture']==a)
  text=lambda x,scale,d: f"{float(x['estimate'])*scale:+.{d}f} [{float(x['lower95'])*scale:+.{d}f}, {float(x['upper95'])*scale:+.{d}f}]"
  lines.append(f"| {name} | {text(r,1,5)} | {text(u,100,3)} | {g['removed']}/{g['present_camera_observations']} | {100*float(g['removal_fraction']):.2f} |")
 lines+=['','Both regret costs are small and positive; their saved intervals exclude zero. Unsafe differences are inconclusive. With little added decision loss and low novel-camera detection, these conditions cannot establish recovery from a consequential unseen fault. This does not identify novel faults as harmless in real use.','',
 'In both prospective studies, the supplied severe-condition mask equals the deployed gate mask on every saved camera observation for both main architectures, including inherited absence. The classifier was trained on these fault families. The 100% benefit recovery in that condition is an in-family detection result.','',
 'Sources:',*[f'- `{p.relative_to(PROJECT)}` — SHA-256 `{sha(p)}`.' for p in sources]]
 (OUT/'V1/V1_numbers_with_sources.md').write_text('\n'.join(lines)+'\n')
 (OUT/'V1/receipt.json').write_text(json.dumps(dict(status='COMPLETE',scope='copy saved contrasts and inventory; verify recorded masks',source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in sources},new_bootstrap_draws=0,new_neural_forwards=0,new_selector_calls=0),indent=2)+'\n')
 print('\n'.join(lines[:10]))
if __name__=='__main__':main()
