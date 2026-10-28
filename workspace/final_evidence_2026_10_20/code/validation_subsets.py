"""Enumerate all eight camera subsets on development scenes; no holdout access."""
from common import *
import argparse,time,torch
from neural import backend,frozen,batch,neutral,decisions,ensure_fp32

def main():
    p=argparse.ArgumentParser();p.add_argument('--architecture',required=True,choices=['KF-D42','BF-D42']);p.add_argument('--seed',required=True,type=int);a=p.parse_args()
    backend();model,cp=frozen(a.architecture,a.seed)
    source=ROOT/'study4/validation';files=list(csv.DictReader((source/'files.csv').open()))
    out=ROOT/f'study4/validation_subsets/{a.architecture}_seed{a.seed}.csv';out.parent.mkdir(exist_ok=True);assert not out.exists()
    count=0;start=time.time();writer=None
    with out.open('w',newline='') as f:
        for code in range(8):
            for offset in range(0,len(files),4):
                group=files[offset:offset+4];records=[load_npz(source/r['path']) for r in group]
                drops=[np.array([bool(code&(1<<j)) for j in range(3)])|(d['cam_present']<=.5) for d in records]
                b=batch(records,drops)
                with torch.inference_mode():outputs=model(b)
                ensure_fp32(outputs)
                for j,(r,d) in enumerate(zip(group,records)):
                    pred=neutral(outputs,j)
                    for policy,met in decisions(pred,d).items():
                        row=dict(architecture=a.architecture,seed=a.seed,subset=code,condition=r['condition'],scene_id=int(r['scene_id']),frame_id=int(r['frame_id']),policy=policy,**met)
                        if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row)
                count+=len(group)
            f.flush();print(a.architecture,a.seed,'subset',code,'forwards',count,'seconds',time.time()-start,flush=True)
    js(out.with_suffix('.json'),dict(status='COMPLETE_VALIDATION_ONLY',architecture=a.architecture,seed=a.seed,forwards=count,seconds=time.time()-start,
        checkpoint=cp,output_sha256=sha(out),data_manifest_sha256=sha(source/'files.csv'),precision='actual FP32',no_test_access=True))
if __name__=='__main__':main()
