"""Paired recipe/map metrics, scene clustered, from saved per-frame confusions."""
from common import *
import pandas as pd
from statistical_core4 import CONDITIONS
ARCHES=['KF-D42','BF-D42','KF-D14','BF-D14'];SEEDS=[10,11,12]

def metrics(semantic,material):
    # Last two axes are true,predicted; arbitrary leading bootstrap axes.
    diag=np.diagonal(material,axis1=-2,axis2=-1)
    union=material.sum(-1)+material.sum(-2)-diag
    with np.errstate(divide='ignore',invalid='ignore'):
        acc=diag.sum(-1)/material.sum((-2,-1))
        iou=diag/union
        # Macro over the fixed four material classes. An absent union is an
        # undefined class, never an automatically perfect/zero class.
        macro=iou.mean(-1)
        person=semantic[...,3,3]/semantic[...,3,:].sum(-1)
        obstacle=semantic[...,4,4]/semantic[...,4,:].sum(-1)
    return dict(material_accuracy=acc,material_macro_iou=macro,person_cell_recall=person,obstacle_cell_recall=obstacle)

def bounds(reps):
    valid=np.isfinite(reps);q=np.quantile(reps,[.025,.975]) if valid.all() else [None,None]
    return dict(lower95=q[0],upper95=q[1],undefined_draws=int((~valid).sum()),
        interval_status='DEFINED' if valid.all() else 'UNDEFINED_SUPPORT_IN_SAVED_DRAWS')

def main():
    from infer_study4 import verify_lock
    verify_lock();out=ROOT/'study4/map_metrics';out.mkdir(exist_ok=False)
    draws=load_npz(ROOT/'study4/bootstrap_plan/bootstrap_multiplicities.npz')
    # Aggregates are normalized by accepted slots and the three fixed seeds
    # within each scene. This retains the primary analysis's equal-scene unit.
    aggregate={};perseed=[];absolute=[];bootstrap={};classrows=[]
    for a in ARCHES:
        tables=[];sem=[];mat=[];heights=[]
        for seed in SEEDS:
            prefix=ROOT/f'study4/metrics/{a}_seed{seed}'
            keys=pd.read_csv(str(prefix)+'_confusion_keys.csv');z=load_npz(str(prefix)+'_frame_confusions.npz')
            assert len(keys)==len(z['semantic'])==len(z['material'])
            take=keys.arm.eq('keep_all').to_numpy();k=keys[take].copy();k['seed']=seed
            tables.append(k);sem.append(z['semantic'][take]);mat.append(z['material'][take])
            h=pd.read_csv(str(prefix)+'.csv');h=h[(h.arm=='keep_all')&(h.policy=='direct')]
            heights.append(h[['condition','scene_id','frame_id','seed','height_reachable_terrain','reachable_terrain_cells']])
            for c,g in k.groupby('condition'):
                ix=np.flatnonzero(k.condition.to_numpy()==c)
                sm=z['semantic'][take][ix].sum(0);mt=z['material'][take][ix].sum(0)
                v=metrics(sm,mt)
                for metric,value in v.items():perseed.append(dict(architecture=a,seed=seed,condition=c,metric=metric,estimate=float(value) if np.isfinite(value) else None,estimand='fixed-seed pooled-cell diagnostic'))
                gh=h[h.condition==c]
                perseed.append(dict(architecture=a,seed=seed,condition=c,metric='height_frame_mae',estimate=float(gh.height_reachable_terrain.mean()),estimand='fixed-seed supported-frame mean; support exported separately'))
                for head,cm,names in [('semantic',sm,['terrain','rock','truck','person','obstacle','machine']),('material',mt,['sand','soil','gravel','rocky'])]:
                    for j,name in enumerate(names):
                        tp=int(cm[j,j]);truth=int(cm[j].sum());predicted=int(cm[:,j].sum());union=truth+predicted-tp
                        classrows.append(dict(architecture=a,seed=seed,condition=c,head=head,class_id=j,class_name=name,truth_cells=truth,predicted_cells=predicted,true_positive_cells=tp,
                            recall=None if truth==0 else tp/truth,iou=None if union==0 else tp/union,support_status='NO_TRUE_CLASS_SUPPORT' if truth==0 else 'SUPPORTED'))
        keys=pd.concat(tables,ignore_index=True);S=np.concatenate(sem);T=np.concatenate(mat)
        h=pd.concat(heights,ignore_index=True)
        assert not keys.duplicated(['condition','scene_id','frame_id','seed']).any()
        for c in CONDITIONS:
            pop='rough' if c=='rough_iid' else 'nominal';ids=draws[pop+'_scene_ids'];M=draws[pop].astype(float)
            sg=[];tg=[];hg=[];support=[]
            for sid in ids:
                ix=np.flatnonzero((keys.condition.to_numpy()==c)&(keys.scene_id.to_numpy()==sid));assert len(ix)>0
                sg.append(S[ix].mean(0));tg.append(T[ix].mean(0))
                g=h[(h.condition==c)&(h.scene_id==sid)]
                assert len(g)==len(ix)
                finite=np.isfinite(g.height_reachable_terrain.to_numpy())
                assert np.array_equal(finite,g.reachable_terrain_cells.to_numpy()>0)
                hg.append(float(g.height_reachable_terrain.mean()) if finite.any() else np.nan)
                support.append(dict(architecture=a,condition=c,scene_id=int(sid),attempts=len(g),undefined_height_frames=int((~finite).sum()),reachable_terrain_cells=int(g.reachable_terrain_cells.sum())))
            sg=np.stack(sg);tg=np.stack(tg);hg=np.asarray(hg)
            aggregate[(a,c)]=(sg,tg,hg);csvwrite(out/f'support_{a}_{c}.csv',support)
            bS=(M@sg.reshape(len(ids),-1)/len(ids)).reshape(-1,6,6)
            bT=(M@tg.reshape(len(ids),-1)/len(ids)).reshape(-1,4,4)
            point=metrics(sg.mean(0),tg.mean(0));reps=metrics(bS,bT)
            valid=np.isfinite(hg);den=M@valid.astype(float)
            with np.errstate(divide='ignore',invalid='ignore'):bh=(M@np.nan_to_num(hg))/den
            point['height_frame_mae']=float(hg[valid].mean()) if valid.any() else np.nan
            reps['height_frame_mae']=bh
            for metric,value in point.items():
                bootstrap[(a,c,metric)]=reps[metric]
                absolute.append(dict(architecture=a,condition=c,metric=metric,estimate=float(value) if np.isfinite(value) else None,
                    **bounds(reps[metric]),scenes=len(ids),estimand='equal-scene, fixed-seed mixture; class metrics from scene-normalized confusion counts; height over supported frames within scene'))
        print('map metrics',a,flush=True)
    comparisons=[]
    pairs=[('KF-D42','KF-D14',CONDITIONS,'bundled_recipe'),('BF-D42','BF-D14',CONDITIONS,'bundled_recipe'),
           ('BF-D42','KF-D42',['clean','cam_s3','rough_iid'],'architecture_under42epoch_recipe')]
    points={(r['architecture'],r['condition'],r['metric']):r['estimate'] for r in absolute}
    for left,right,conditions,scope in pairs:
        for c in conditions:
            for metric in ['height_frame_mae','material_accuracy','material_macro_iou','person_cell_recall','obstacle_cell_recall']:
                l=points[(left,c,metric)];r=points[(right,c,metric)]
                delta=bootstrap[(left,c,metric)]-bootstrap[(right,c,metric)]
                comparisons.append(dict(left=left,right=right,condition=c,metric=metric,estimate=None if l is None or r is None else l-r,
                    **bounds(delta),scope=scope+'; descriptive ordinary95; person/obstacle are BEV-cell recalls'))
    csvwrite(out/'absolute_metrics.csv',absolute);csvwrite(out/'paired_comparisons.csv',comparisons);csvwrite(out/'per_seed_metrics.csv',perseed);csvwrite(out/'full_class_diagnostics.csv',classrows)
    np.savez_compressed(out/'scene_confusions.npz',**{f'{a}_{c}_{k}':v for (a,c),arr in aggregate.items() for k,v in zip(['semantic','material','height'],arr)})
    js(out/'completion.json',dict(status='COMPLETE',bootstrap_sha256=sha(ROOT/'study4/bootstrap_plan/bootstrap_multiplicities.npz'),new_draws=0,
        perception_forwards=0,absolute_rows=len(absolute),paired_rows=len(comparisons),per_seed_rows=len(perseed),code_sha256=sha(__file__),
        undefined_intervals=sum(r['undefined_draws']>0 for r in absolute+comparisons),semantic_person_index=3,semantic_obstacle_index=4))
if __name__=='__main__':main()
