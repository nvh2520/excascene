from common import *
import pandas as pd
from inputs import load
FIELDS={'gt_H':np.float16,'gt_sem':np.uint8,'gt_mat':np.uint8,'gt_haz':np.uint8,'gt_aff':np.float16,'gt_reach':np.float16,'q_true':np.float32}
def fingerprint(path):
    with np.load(path) as z:
        h=hashlib.sha256()
        for k,t in FIELDS.items():h.update(array_sha(z[k].astype(t)).encode())
        return h.hexdigest(),int(z['scene_id'])
def main():
    locked();status=json.loads((RUN/'generation_status.json').read_text());slots=pd.read_csv(RUN/'manifests/generation_slots.csv');files=pd.read_csv(RUN/'manifests/generated_files.csv')
    blocked=[p['population'] for p in status['populations'] if p['invalid_fraction']>.1]
    if status['overall_invalid_fraction']>.1:blocked=sorted(set(blocked+['nominal','wide','rough']))
    legacy=[]
    for s in ['train','val','cal','test_iid','suite_clean','suite_ood_artic','suite_ood_terrain']:
        legacy+=sorted((ROOT/'data'/s).glob('*.npz'))
    legacy+=sorted((ROOT/'data_stage2').rglob('*.npz'))
    hashes={};historical_ids=set();index=[]
    for p in legacy:
        fp,s=fingerprint(p);hashes.setdefault(fp,str(p.relative_to(ROOT)));historical_ids.add(s)
        index.append(dict(path=str(p.relative_to(ROOT)),scene_id=s,content_sha256=fp))
    fresh_ids=set(slots.scene_id);assert not fresh_ids&historical_ids,'historical scene identity collision'
    duplicate=[];fresh_index=[]
    for _,r in files[files.condition.isin(['clean','wider_articulation','rough_iid'])].iterrows():
        fp,s=fingerprint(RUN/r.path);fresh_index.append(dict(path=r.path,scene_id=s,content_sha256=fp))
        if fp in hashes:duplicate.append(dict(fresh_path=r.path,historical_path=hashes[fp]));blocked.append('rough' if r.condition=='rough_iid' else 'wide' if r.condition=='wider_articulation' else 'nominal')
    csvwrite(RUN/'manifests/historical_content_index.csv',index);csvwrite(RUN/'manifests/fresh_content_index.csv',fresh_index)
    js(RUN/'manifests/content_duplicates.json',duplicate)
    nominal=set(map(tuple,slots[(slots.population=='nominal')&slots.accepted][['scene_id','frame_id']].values))
    wide=set(map(tuple,slots[(slots.population=='wide')&slots.accepted][['scene_id','frame_id']].values))
    gkeys=[list(map(int,k)) for k in sorted(nominal&wide)]
    js(RUN/'generation_gates.json',dict(blocked_populations=sorted(set(blocked)),G_keys=gkeys,identity_disjoint=True,historical_files=len(legacy),historical_unique_contents=len(hashes),content_duplicate_count=len(duplicate),complete=True))
    print('Generation gates',len(gkeys),'G paired keys; blocked',blocked,flush=True)
if __name__=='__main__':main()
