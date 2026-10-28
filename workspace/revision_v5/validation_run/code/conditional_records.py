"""Read-only conditional artifact gates; no downloads of weights or raw recordings, no forwards."""
from common import *
import urllib.request,re,torch
def main():
    pages=[]
    for name,url in [('setup','https://goose-dataset.de/docs/setup/'),('examples','https://goose-dataset.de/docs/examples/'),('alice','https://goose-dataset.de/docs/alice/')]:
        with urllib.request.urlopen(url,timeout=30) as r:
            b=r.read(2_000_001);assert len(b)<=2_000_000
        p=RUN/f'conditional/official_{name}.html';p.write_bytes(b);pages.append(dict(url=url,accessed_utc=utc(),bytes=len(b),sha256=sha(p)))
    csvwrite(RUN/'conditional/web_sources.csv',pages)
    manifest=ROOT/'data_stage2/real_A7/val_only.json';entries=json.loads(manifest.read_text());assert len(entries)==192
    images=[]
    for e in entries:
        assert e['split']=='val' and e['platform']=='excavator'
        a={k:ROOT/('data_stage2/'+e[k].split('/data_stage2/',1)[1]) for k in ['rgb','label']}
        assert all(p.exists() for p in a.values())
        # Original manifest groups scenario; finer filename sequence is retained explicitly.
        sequence=re.match(r'(alice_scenario\d+_sequence\d+)',a['rgb'].name).group(1)
        images.append(dict(image_id=a['rgb'].name,sequence=sequence,legacy_group=e['sequence'],split='val',rgb=str(a['rgb'].relative_to(ROOT)),rgb_sha256=sha(a['rgb']),label=str(a['label'].relative_to(ROOT)),label_sha256=sha(a['label'])))
    csvwrite(RUN/'conditional/alice192_manifest.csv',images)
    from kinafford.real.goose_seg import SegNet
    checkpoints=[]
    for seed in range(3):
        p=ROOT/f'runs_stage2/real/goose_R2/seed{seed}/segnet.pt';m=SegNet(64,pretrained=False)
        z=torch.load(p,map_location='cpu',weights_only=True);m.load_state_dict(z,strict=True)
        checkpoints.append(dict(seed=seed,path=str(p.relative_to(ROOT)),sha256=sha(p),architecture='kinafford.real.goose_seg.SegNet(64)',strict_load=True,neural_forwards=0))
    csvwrite(RUN/'conditional/historical_image_comparator.csv',checkpoints)
    facts=ROOT/'audit/stage6/facts_data/FACT-10c/goose_alice_bag_probe.json';probe=json.loads(facts.read_text())
    gate=dict(external_image_status='BLOCKED',image_artifacts_present=192,
      missing=['Checkpoint-specific official PP-LiteSeg architecture/config/version and preprocessing (official example is DDRNet39 at768, not proof of this PP-LiteSeg512 pipeline)',
        'Weight-specific usage conditions or an authoritative statement applying the dataset license to this checkpoint',
        'Checkpoint-specific output-index mapping/ignore convention and documented training/evaluation overlap status for these ALICE IDs'],
      model_url='https://goose-dataset.de/models/ppliteseg_class_512.pth',weight_digest=None,weight_bytes_acquired=0,forwards=0,
      raw_geometry_status='BLOCKED',raw_missing=['Ouster OS0-128 boom and house-reference sensor metadata JSON: firmware/packet profile, beam angles, timestamp mode and extrinsics',
        'Recording-specific URDF/kinematic configuration with verified joint units, zero/sign/axis conventions and link transforms for the18 named joints',
        'Joint-header/PTP/device-clock mapping and timing provenance, beyond timestamp proximity',
        'Verified complete recording-specific camera_info/TF chain and an eligible moving interval; the audited0.4778628s prefix is insufficient',
        'Provenance of TF versus FK and an independent reference for any physical accuracy claim'],
      raw_source=str(facts.relative_to(ROOT)),raw_source_sha256=sha(facts),raw_bytes_acquired=0,raw_decoding_cpu_seconds=0,
      historical_topic_inspection_complete=True,author_confirmations_received=False)
    js(RUN/'conditional/branch_gates.json',gate)
    (RUN/'conditional/external_image_reference_report.md').write_text('''# External image-semantic reference — BLOCKED

The192 historical ALICE validation image/label pairs exist and are hashed in alice192_manifest.csv. No new cohort or inference was used. The [official setup page](https://goose-dataset.de/docs/setup/) lists PP-LiteSeg ppliteseg_class_512,64 classes at512x512. It identifies a dataset license; that alone does not verify checkpoint-specific usage conditions. The [official example](https://goose-dataset.de/docs/examples/) uses a DDRNet39 example, square center cropping, nearest-neighbor resizing and ToTensor. It does not establish the prescribed PP-LiteSeg checkpoint's exact architecture/version, preprocessing and output mapping. No such facts are inferred from its filename.

Required artifacts are enumerated in branch_gates.json: checkpoint-specific implementation/config/preprocessing, applicable weight terms, class/ignore mapping and overlap records. The unavailable weight digest is explicitly null; no weights were acquired and0 external forwards ran. This branch stops at its artifact/rights/compatibility gates with no substitute model.

The actual historical comparator was CPU strict-loaded: kinafford.real.goose_seg.SegNet(64), runs_stage2/real/goose_R2/seed0,1,2/segnet.pt. It is a separate real-image segmentation network. Original RealSegDataset uses384x240 inputs; evaluate reports IoU only over ground-truth-present classes and does not export complete compatible per-image confusions. Neither a synthetic auxiliary head nor a BEV/utility head is a valid replacement. No head-to-head result is reported.

If the missing records are later supplied, declared full64-class confusion IoU uses union>0; false-positive-only classes remain in aggregation, and zero-union classes are undefined. Historical ground-truth-present-class aggregation would be separate. Finer filename sequence IDs are preserved for clustering. No scalar published mIoU is presented as this run's result.

TASK-MATCHED EXTERNAL FUSION COMPARISON: NOT RUN.
''')
    (RUN/'conditional/real_geometry_report.md').write_text('''# ALICE raw geometry — BLOCKED

The completed Stage6 audit is retained: FACT-10c/goose_alice_bag_probe.json parsed24 JointState messages and dynamic/static TF in an80MB compressed prefix spanning0.4778628s. It identified the boom LiDAR parent,18 joint names and approximate timestamps. This establishes topic presence, not verified physical joint conventions or synchronization.

Missing prerequisites for alice_scenario06_sequence02_sensors.bag are exactly listed in branch_gates.json: Ouster metadata for boom/reference sensors; recording-specific units/zero/sign/axis/link configuration; joint clock mapping; complete calibrated TF/camera_info and usable moving segment; independent-reference provenance. No additional raw archive was acquired, no geometric residuals inspected, and no calibration inferred. The [ALICE documentation](https://goose-dataset.de/docs/alice/) describes PTP and calibration records, while also noting calibration can differ between recordings; it does not resolve these local prerequisites.

The locked prospective segment rule remains earliest complete interval satisfying cumulative>=5deg movement in a nonswing joint over>=5s, at most60s/100 uniformly spread timestamps, selected before residuals. No eligible segment is claimed selected. Agreement between TF and FK sharing the generating joints/model would be implementation consistency. Dynamic/frozen registration requires a separate house-frame reference with coverage, object/ego motion and scan timing limitations. No real excavation utility, hazard, material, articulated-camera deployment or safety result is inferred.
''')
    event('conditional_branches_blocked',image_forwards=0,raw_acquisition_bytes=0,weight_acquisition_bytes=0,web_bytes=sum(p['bytes'] for p in pages))
    print('Conditional reports written: precise missing-artifact gates;192 pairs and3 comparator checkpoints verified',flush=True)
if __name__=='__main__':main()
