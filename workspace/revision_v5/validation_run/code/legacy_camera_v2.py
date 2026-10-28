"""Bounded existing-output sensitivity; no historical rows are deleted."""
from common import *
import pandas as pd
KEY=['seed','suite','scene_id','frame_id']
def main():
    flags=pd.read_csv(ROOT/'revision_v5/analyses/existing_validity_flags.csv')
    nominal=set(map(tuple,flags[flags.dir.eq('suite_clean')][['scene_id','frame_id']].values))
    testi=set(map(tuple,flags[flags.dir.eq('test_iid')][['scene_id','frame_id']].values))
    rows=[];effects=[]
    for tag,folder in [('G3_dyn','results'),('S2E_G3_dyn','results_stage2'),('S4A_G3_structured','results_stage4'),('S5A_G3_structured','results_stage5')]:
        arms={}
        for arm in ['masked','unmasked','specialist']:
            paths=[ROOT/(f'results_stage6/A/S6A_masked_{tag}_s{s}.csv' if arm=='masked' else f'{folder}/{tag}_s{s}.csv' if arm=='unmasked' else (f'results/G6_lidar_s{s}.csv' if tag in ['G3_dyn','S4A_G3_structured'] else f'results_stage2/S2E_G6_lidar_s{s}.csv')) for s in range(3)]
            arms[arm]=pd.concat([pd.read_csv(p,usecols=KEY+['H_mae_vis','regret_direct','unsafe_direct']) for p in paths]).set_index(KEY).sort_index()
        assert set(arms['masked'].index.get_level_values('suite'))=={'test_iid','clean','iid','cam_s3','all_s2','all_s3'}
        for comparator in ['unmasked','specialist']:
            a=arms['masked'];b=arms[comparator].reindex(a.index);assert not b.isna().any().any()
            z=(a-b).reset_index()
            z['flag']= [(s,f) in (testi if c=='test_iid' else nominal) for c,s,f in z[['suite','scene_id','frame_id']].itertuples(index=False,name=None)]
            z['model']=tag;z['comparator']=comparator;effects.append(z)
            for suite,one in z.groupby('suite',sort=True):
                ids=np.sort(one.scene_id.unique());draws=np.random.default_rng(20261011).choice(len(ids),size=(2000,len(ids)),replace=True)
                for valid in [False,True]:
                    t=one[~one.flag] if valid else one
                    for endpoint in ['H_mae_vis','regret_direct','unsafe_direct']:
                        frame=t.groupby(['scene_id','frame_id'])[endpoint].mean()
                        sc=frame.groupby(level='scene_id').agg(['sum','count']).reindex(ids).fillna(0)
                        ns=sc['sum'].to_numpy()[draws].sum(1);ds=sc['count'].to_numpy()[draws].sum(1)
                        assert (ds>0).all();rep=ns/ds
                        if endpoint=='unsafe_direct':
                            from fractions import Fraction
                            # Original seed event differences are integers. Avoid mean-of-thirds cancellation.
                            sums=t.groupby('scene_id')[endpoint].sum().reindex(ids).fillna(0).astype('int64').to_numpy()
                            fractions=sorted(Fraction(int(n),int(3*d)) for n,d in zip(sums[draws].sum(1),ds))
                            def fq(q):
                                p=Fraction(len(fractions)-1)*q;i=p.numerator//p.denominator
                                return fractions[i]*(1-(p-i))+fractions[min(i+1,len(fractions)-1)]*(p-i)
                            lo,hi=map(float,[fq(Fraction(1,40)),fq(Fraction(39,40))]);mean=float(Fraction(int(t[endpoint].sum()),len(t)))
                        else:lo,hi=np.quantile(rep,[.025,.975]);mean=frame.mean()
                        rows.append(dict(model=tag,contrast='masked - '+comparator,specialist_model='G6_lidar' if tag in ['G3_dyn','S4A_G3_structured'] else 'S2E_G6_lidar',suite=suite,endpoint=endpoint,population='known-valid-only' if valid else 'historical-all',mean_diff=mean,ci_lo=lo,ci_hi=hi,frame_pairs=len(frame),scene_clusters=frame.index.get_level_values('scene_id').nunique(),excluded_frames=int(one[one.flag][['scene_id','frame_id']].drop_duplicates().shape[0]) if valid else 0,bootstrap_draws=2000,rng=20261011,scope='known FACT-1/3 and S6C flags; common keys across arms/seeds; not exhaustive drifted-geometry validation'))
    csvwrite(RUN/'legacy/camera_valid_only_v2.csv',rows);pd.concat(effects).to_csv(RUN/'legacy/camera_paired_frame_effects_v2.csv',index=False)
    event('legacy_camera_sensitivity_complete',rows=len(rows));print('Camera sensitivity complete',len(rows))
if __name__=='__main__':main()
