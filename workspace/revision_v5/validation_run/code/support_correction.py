"""Versioned correction of rough clean-fixed/intersection diagnostics only."""
from common import *
import pandas as pd
from inputs import load
def main():
    rows=[];audit=[]
    for path in sorted((RUN/'metrics').glob('*.csv')):
        m=pd.read_csv(path);old=m.copy()
        for (scene,frame),group in m[m.condition=='rough_iid'].groupby(['scene_id','frame_id']):
            ref=load(RUN/f'support_reference/rough_iid/{scene}_{frame}.npz')['gt_vis_clean']>.5
            d=load(RUN/f'data/rough_iid/f_{scene:06d}_{frame}.npz');vis=d['gt_vis']>.5
            p=load(RUN/f'predictions/{path.stem}/rough_iid/{scene}_{frame}.npz')
            err=np.abs(p['H'].astype(np.float64)-d['gt_H'].astype(np.float64));both=vis&ref
            changes=dict(H_clean_fixed=float(err[ref].mean()) if ref.any() else np.nan,H_intersection=float(err[both].mean()) if both.any() else np.nan,n_clean_fixed=int(ref.sum()),n_intersection=int(both.sum()),visibility_lost=int((ref&~vis).sum()),visibility_gained=int((vis&~ref).sum()))
            for key,v in changes.items():m.loc[group.index,key]=v
            audit.append(dict(model=group.iloc[0].model,checkpoint=group.iloc[0].checkpoint,seed=int(group.iloc[0].seed),scene_id=int(scene),frame_id=int(frame),old_H_clean_fixed=float(group.iloc[0].H_clean_fixed),new_H_clean_fixed=changes['H_clean_fixed'],old_n_clean_fixed=int(group.iloc[0].n_clean_fixed),new_n_clean_fixed=changes['n_clean_fixed']))
        # No primary endpoint, selector, candidate or decision changed.
        for c in ['H_reachable_terrain','regret','unsafe','executed','pick_row','pick_col','oracle','selected_utility']:
            assert np.array_equal(m[c].values,old[c].values,equal_nan=True),c
        target=RUN/'metrics_corrected'/path.name;target.parent.mkdir(exist_ok=True);m.to_csv(target,index=False);rows.append(m)
    csvwrite(RUN/'reports/rough_support_old_new.csv',audit)
    a=pd.concat(rows,ignore_index=True);keys=['model','checkpoint','seed','mask','condition','policy']
    summary=pd.read_csv(RUN/'decision_support_summary.csv').set_index(keys)
    replacement=a.groupby(keys).agg(H_clean_fixed=('H_clean_fixed','mean'),H_intersection=('H_intersection','mean'),n_clean_fixed=('n_clean_fixed','sum'),n_intersection=('n_intersection','sum'),visibility_lost=('visibility_lost','sum'),visibility_gained=('visibility_gained','sum'))
    summary.update(replacement);summary.to_csv(RUN/'decision_support_summary_corrected.csv')
    event('support_diagnostic_correction',rough_model_seed_frames=len(audit),primary_endpoints_unchanged=True)
    print('Support diagnostics corrected; primary values unchanged',len(audit),flush=True)
if __name__=='__main__':main()
