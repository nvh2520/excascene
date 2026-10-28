"""Read-only train/validation schema, identity and integrity inventory."""
import csv, hashlib, json, pathlib, sys, time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
required={'rgb':((3,160,256,3),np.uint8),'depth':((3,160,256),np.float16),
 'img_sem':((3,160,256),np.uint8),'cam_present':((3,),np.uint8),
 'lidar_range':((32,512),np.float16),'lidar_valid':((32,512),np.uint8),
 'q_rep':((4,),np.float32),'q_true':((4,),np.float32),
 'gt_H':((96,96),np.float16),'gt_sem':((96,96),np.uint8),
 'gt_mat':((96,96),np.uint8),'gt_haz':((96,96),np.uint8),
 'gt_aff':((96,96),np.float16),'gt_reach':((96,96),np.float16),
 'gt_vis':((96,96),np.uint8),'rel_gt':((4,),np.float32)}
rows=[];schema={};identities=set();start=time.monotonic()
for split,n,lo,hi in [('train',12000,0,3000),('val',1200,3000,3300)]:
    files=sorted((ROOT.parent/'data'/split).glob('*.npz'));assert len(files)==n,(split,len(files))
    for i,p in enumerate(files):
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p,allow_pickle=False) as z:
            scene=int(z['scene_id']);slot=int(z['frame_id']);assert lo<=scene<hi and 0<=slot<4
            assert (scene,slot) not in identities;identities.add((scene,slot))
            for k,(shape,dtype) in required.items():
                assert k in z.files and z[k].shape==shape,(str(p),k,z[k].shape)
                # Preserve measured schema; record observed dtype rather than silently convert files.
                schema.setdefault(k,set()).add(str(z[k].dtype))
                if np.issubdtype(z[k].dtype,np.floating):assert np.isfinite(z[k]).all(),(str(p),k,'nonfinite')
            assert np.isin(z['gt_sem'],np.arange(6)).all()
            assert np.isin(z['gt_mat'],np.arange(4)).all()
            assert np.isin(z['img_sem'],np.arange(7)).all()
            assert np.isin(z['cam_present'],[0,1]).all() and np.isin(z['lidar_valid'],[0,1]).all()
            for k in ['gt_aff','gt_reach','gt_vis','gt_haz']:assert ((z[k]>=0)&(z[k]<=1)).all(),k
        rows.append(dict(split=split,path=str(p.relative_to(ROOT.parent)),scene_id=scene,frame_id=slot,bytes=p.stat().st_size,sha256=h))
        if (i+1)%1000==0:print(split,i+1,flush=True)
with (ROOT/'manifests/train_validation_inventory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
receipt=dict(status='PASSED',records=len(rows),scenes=len({x[0] for x in identities}),
             schema={k:sorted(v) for k,v in schema.items()},elapsed_s=time.monotonic()-start,
             input_bytes=sum(x['bytes'] for x in rows),utc_unix=time.time())
(ROOT/'manifests/data_inventory_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
