"""Identity/content metadata audit only; no historical predictions or scores used."""
import csv,hashlib,json,pathlib,re,subprocess,time,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
workspace=ROOT.parent;proposed=set(range(300000,300128))|set(range(310000,310064))
start=time.monotonic();paths=subprocess.check_output(['rg','--files','--hidden','--no-ignore','-g','*.npz','-g','!external_validation_run_EV1_AFAST/**','-g','!.venv/**'],cwd=workspace,text=True).splitlines()
rows=[];collisions=[];content={};skipped=0
for i,rel in enumerate(paths):
    p=workspace/rel
    match=re.fullmatch(r'(?:f_)?(\d+)_(\d+)(?:_.*)?',p.stem)
    if match and int(match[1]) in proposed:collisions.append(dict(path=rel,method='filename',scene_id=int(match[1])))
    with zipfile.ZipFile(p) as z:
        names=z.namelist()
        if 'scene_id.npy' not in names:skipped+=1;continue
    with np.load(p,allow_pickle=False) as z:
        sids=np.asarray(z['scene_id']).reshape(-1)
        frames=np.asarray(z['frame_id']).reshape(-1) if 'frame_id' in z.files else np.full(len(sids),-1)
        if len(frames)!=len(sids):frames=np.full(len(sids),-1)
        for sid in sids:
            if int(sid) in proposed:collisions.append(dict(path=rel,method='authoritative_scene_id',scene_id=int(sid)))
        fingerprint=None
        if len(sids)==1 and all(k in z.files for k in ['q_true','gt_H','gt_sem']):
            h=hashlib.sha256()
            for k,dtype in [('q_true','<f4'),('gt_H','<f2'),('gt_sem','u1')]:
                a=np.ascontiguousarray(z[k],dtype=dtype);h.update(k.encode());h.update(a.tobytes())
            fingerprint=h.hexdigest();content.setdefault(fingerprint,dict(path=rel,scene_id=int(sids[0]),frame_id=int(frames[0])))
    for sid,frame in zip(sids,frames):rows.append(dict(path=rel,scene_id=int(sid),frame_id=int(frame),scene_pose_fingerprint=fingerprint))
    if (i+1)%10000==0:print('metadata audited',i+1,flush=True)
with (ROOT/'manifests/historical_identity_content_inventory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','scene_id','frame_id','scene_pose_fingerprint']);w.writeheader();w.writerows(rows)
(ROOT/'manifests/historical_scene_pose_fingerprints.json').write_text(json.dumps(content,indent=2))
receipt=dict(status='PASSED_NO_ID_COLLISION' if not collisions else 'BLOCKED_COLLISION',npz_paths_inspected=len(paths),
             metadata_records=len(rows),without_identity_fields=skipped,unique_scene_pose_fingerprints=len(content),
             actual_scene_ids=sorted(set(x['scene_id'] for x in rows)),collisions=collisions,
             audit_fields=['scene_id','frame_id','q_true','gt_H','gt_sem'],historical_scores_read=False,
             content_definition='q_true float32 + H float16 canonical comparison + semantics uint8; fresh full precision targets are never rewritten',
             fresh_scene_pose_audit='required after generation and before prediction; compare against saved fingerprints',
             elapsed_s=time.monotonic()-start,utc_unix=time.time())
(ROOT/'manifests/collision_audit_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='actual_scene_ids'},indent=2))
assert not collisions,'Proposed IDs already present; no reassignment based on outcomes allowed'
