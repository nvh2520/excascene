"""One command: frozen final42 checkpoint + gate replay on four saved records."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,sys,time,traceback

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    candidate=Path(__file__).resolve().parent;root=candidate/'workspace/final_evidence_2026_10_20'
    assert not a.output.exists();a.output.mkdir(parents=True)
    started=time.time();receipt=dict(status='FAILED',start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        candidate=str(candidate),output=str(a.output),new_training=False,new_generation=False,
        fresh_compilation=False,training_reproduction=False,precision='FP32; TF32/autocast disabled',
        atol=2e-5,rtol=2e-5,fixture_records=4)
    try:
        for r in csv.DictReader((candidate/'package_inventory.csv').open()):
            p=candidate/r['path'];assert p.resolve().is_relative_to(candidate),(p,'escaping link')
            assert digest(p)==r['sha256'],('package digest',r['path'])
        sys.path.insert(0,str(root/'code'))
        import numpy as np,torch
        from common import verify_protocol,load_npz,csvwrite
        from neural import backend,frozen,batch,neutral,decisions,ensure_fp32
        from gate import load_gate,severe_probability
        from infer_study4 import masks
        verify_protocol();backend()
        assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,0)
        receipt['device']=torch.cuda.get_device_name();receipt['torch']=torch.__version__
        fixture=json.loads((candidate/'fixture_manifest.json').read_text())
        data=[load_npz(root/r['data']) for r in fixture];assert len(data)==4
        expected={(r['architecture'],int(r['scene_id']),int(r['frame_id']),r['arm'],r['policy']):r
                  for r in csv.DictReader((candidate/'expected_decisions.csv').open())}
        expected_gate={(int(r['scene_id']),int(r['frame_id'])):r for r in csv.DictReader((candidate/'expected_gate.csv').open())}
        thresholds=json.loads((root/'study4/thresholds.json').read_text())
        gate=load_gate(root/'study4/gate/final.pt')
        assert all(p.dtype==torch.float32 for p in gate.parameters())
        pp=severe_probability(gate,np.concatenate([d['rgb'] for d in data])).reshape(4,3)
        saved_scores=[];got_scores=[];gate_checks=[]
        for r,d,p in zip(fixture,data,pp):
            want=expected_gate[(r['scene_id'],r['frame_id'])]
            invalid=np.mean((~np.isfinite(d['depth']))|(d['depth']<=.05),axis=(-2,-1))
            old={k:float(v) for k,v in want.items() if k.startswith(('p','invalid')) and k not in ['policy']}
            got={f'p{j}':float(p[j]) for j in range(3)};got.update({f'invalid{j}':float(invalid[j]) for j in range(3)})
            assert np.allclose(p,[old[f'p{j}'] for j in range(3)],atol=2e-5,rtol=2e-5)
            assert np.array_equal(invalid,[old[f'invalid{j}'] for j in range(3)])
            for arch in ['KF-D42','BF-D42']:
                expected_mask=masks(d,old,arch,thresholds)['gate'];actual_mask=masks(d,got,arch,thresholds)['gate']
                assert np.array_equal(actual_mask,expected_mask)
                gate_checks.append(dict(architecture=arch,scene_id=r['scene_id'],frame_id=r['frame_id'],
                    mask=actual_mask.astype(int).tolist(),exact=True,
                    maximum_probability_difference=float(np.max(np.abs(p-[old[f'p{j}'] for j in range(3)])))))
            saved_scores.append(old);got_scores.append(got)
        receipt['gate_camera_forwards']=12;del gate;torch.cuda.empty_cache()
        comparisons=[];decisions_out=[];checkpoints=[];actual=0
        for arch in ['KF-D42','BF-D42']:
            model,checkpoint=frozen(arch,10);checkpoints.append(checkpoint);todo=[];arm_masks=[]
            for i,(d,p) in enumerate(zip(data,got_scores)):
                arm={k:v for k,v in masks(d,p,arch,thresholds).items() if k in ['keep_all','gate']};arm_masks.append(arm)
                codes={sum(int(v)*(1<<j) for j,v in enumerate(drop)):drop for drop in arm.values()}
                todo.extend((i,code,drop) for code,drop in sorted(codes.items()))
            cache={}
            for offset in range(0,len(todo),4):
                sub=todo[offset:offset+4]
                with torch.inference_mode():out=model(batch([data[i] for i,_,_ in sub],[dr for _,_,dr in sub]))
                ensure_fp32(out);actual+=len(sub)
                for j,(i,code,drop) in enumerate(sub):
                    got=neutral(out,j);r=fixture[i]
                    e=next(e for key,e in expected.items() if key[0]==arch and key[1]==r['scene_id'] and key[2]==r['frame_id'] and int(e['subset'])==code)
                    want=load_npz(root/e['prediction_path'])
                    for head,x in got.items():
                        err=float(np.abs(x.astype(float)-want[head].astype(float)).max())
                        ok=np.array_equal(x,want[head]) if head.endswith('_label') else np.allclose(x,want[head],atol=2e-5,rtol=2e-5)
                        comparisons.append(dict(architecture=arch,scene_id=r['scene_id'],frame_id=r['frame_id'],subset=code,
                            head=head,maximum_absolute_difference=err,agreement=bool(ok)))
                        assert ok,(arch,r['frame_id'],head,err)
                    cache[(i,code)]=got
            for i,(r,d,arm) in enumerate(zip(fixture,data,arm_masks)):
                for arm_name,drop in arm.items():
                    code=sum(int(v)*(1<<j) for j,v in enumerate(drop))
                    for policy,met in decisions(cache[(i,code)],d).items():
                        e=expected[(arch,r['scene_id'],r['frame_id'],arm_name,policy)]
                        for k in ['pick_row','pick_col','unsafe','executed','abstained','tie_count','eligible_cells']:
                            assert met[k]==int(e[k]),(arch,r['frame_id'],arm_name,policy,k,met[k],e[k])
                        assert np.isclose(met['regret'],float(e['regret']),atol=2e-5,rtol=2e-5)
                        decisions_out.append(dict(architecture=arch,scene_id=r['scene_id'],frame_id=r['frame_id'],arm=arm_name,policy=policy,**met))
            del model;torch.cuda.empty_cache()
        torch.cuda.synchronize()
        scientific={}
        for name,m in sys.modules.copy().items():
            value=getattr(m,'__file__',None)
            if not value:continue
            p=Path(value).resolve()
            if name.split('.')[0] in ['common','neural','gate','infer_study4','models','inputs','upstream_loader','kinafford','mmdet','mmdet3d']:
                assert p.is_relative_to(candidate),(name,p,'noncandidate scientific source')
                scientific[name]=str(p.relative_to(candidate))
        csvwrite(a.output/'decisions.csv',decisions_out);csvwrite(a.output/'map_comparisons.csv',comparisons)
        receipt.update(status='PASS',actual_perception_forwards=actual,logical_perception_arm_forwards=16,
            exact_decision_rows=len(decisions_out),map_comparisons=len(comparisons),
            maximum_map_absolute_difference=max(r['maximum_absolute_difference'] for r in comparisons),
            maps_within_declared_tolerance=True,all_labels_and_discrete_events_exact=True,
            gate_checks=gate_checks,checkpoints=checkpoints,gate_sha256=digest(root/'study4/gate/final.pt'),
            package_inventory_sha256=digest(candidate/'package_inventory.csv'),
            lock_sha256=digest(root/'protocol/study4_final_lock.json'),
            source_imports=scientific,source_imports_all_within_candidate=True,
            runtime='Existing installed Torch/MMCV/etc.; packaged MMDetection source and existing sm80 binaries, no compilation',
            missing_base_receipt=json.loads((candidate/'audit_receipt_resolution.json').read_text()))
    except BaseException:
        receipt['error']=traceback.format_exc();raise
    finally:
        receipt['elapsed_seconds']=time.time()-started;receipt['finish_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_imports','gate_checks','missing_base_receipt','checkpoints']},indent=2))

if __name__=='__main__':main()
