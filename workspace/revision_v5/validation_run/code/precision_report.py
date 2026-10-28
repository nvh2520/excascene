"""Descriptive CPU analysis of the locked precision outputs; no additional forwards."""
from common import *
import pandas as pd
from inputs import load
from stage3.selectors import soft_composition
from inference import legacy_pick
from analysis import estimate,multiplicities
def main():
    keys=pd.read_csv(RUN/'manifests/precision_keys.csv');diff=pd.read_csv(RUN/'precision/differences.csv');met=pd.read_csv(RUN/'precision/metrics.csv');replay=pd.read_csv(RUN/'precision/legacy_replay.csv')
    r=np.random.default_rng(20261011);nom=sorted(keys[keys.suite=='iid'].scene_id.unique());rough=sorted(keys[keys.suite=='ood_terrain'].scene_id.unique())
    rows=[];scores=[];batch=[]
    for (tag,ckpt),group in met.groupby(['model','checkpoint']):
        nids=sorted(group[group.condition=='iid'].scene_id.unique());rids=sorted(group[group.condition=='ood_terrain'].scene_id.unique())
        draws={'nominal':(nids,multiplicities(len(nids),r)),'rough':(rids,multiplicities(len(rids),r))}
        for policy,one in group.groupby('policy'):
          for endpoint in ['regret','unsafe']:
            key=['seed','condition','scene_id','frame_id'];a=one[one['mode']=='fp32'].set_index(key)[endpoint];b=one[one['mode']=='bf16'].set_index(key)[endpoint]
            assert a.index.equals(b.index);z=(a-b).rename('difference').reset_index();z['condition']=z.condition.replace({'ood_terrain':'rough_iid'})
            result,_=estimate(z,endpoint,['iid','all_s2','cam_s3','rough_iid'],draws)
            zc=diff[(diff.model==tag)&(diff.policy==policy)]
            rows.append(dict(model=tag,checkpoint=ckpt,policy=policy,endpoint=endpoint,contrast='actual FP32 minus BF16; fixed1/4 condition weights',nominal_scenes=len(nids),rough_scenes=len(rids),keys_per_seed=len(one[one.seed==0])//2,decisions=len(zc),pick_changes=int(zc.pick_changed.sum()),bf16_unsafe_events=int(zc.unsafe_bf16.sum()),fp32_unsafe_events=int(zc.unsafe_fp32.sum()),**result))
        subset=keys[keys.additional_subset] if tag not in ['G3_dyn','R2_full'] else keys
        for seed in range(3):
          for _,k in subset.iterrows():
            root=RUN/f'precision/maps/{tag}_{ckpt}_s{seed}';name=f'{k.suite}_{k.scene_id}_{k.frame_id}.npz'
            a=load(root/'bf16'/name);b=load(root/'fp32'/name);d=load(ROOT/k.input_path)
            sa=soft_composition(a,d['gt_reach'].astype(np.float32),float(d['z_target']));sb=soft_composition(b,d['gt_reach'].astype(np.float32),float(d['z_target']))
            scores.append(dict(model=tag,checkpoint=ckpt,seed=seed,suite=k.suite,scene_id=int(k.scene_id),frame_id=int(k.frame_id),max_soft_score_delta=float(np.max(np.abs(sb-sa))),mean_soft_score_delta=float(np.mean(sb.astype(np.float64)-sa)),max_H_sigma_delta=float(np.max(np.abs(b['H_sigma']-a['H_sigma']))),max_material_probability_delta=float(np.max(np.abs(b['mat_prob']-a['mat_prob']))),max_semantic_probability_delta=float(np.max(np.abs(b['sem_prob']-a['sem_prob']))),max_direct_score_delta=float(np.max(np.abs(b['aff']-a['aff']))) if 'aff' in a else np.nan))
            if tag in ['G3_dyn','R2_full']:
                old=load(cache_path(tag,seed,k.suite,int(k.scene_id),int(k.frame_id)))
                batch.append(dict(model=tag,seed=seed,suite=k.suite,scene_id=int(k.scene_id),frame_id=int(k.frame_id),max_H_difference=float(np.max(np.abs(a['H']-old['H']))),all_legacy_picks_equal=all(legacy_pick(a,d,p)==legacy_pick(old,d,p) for p in ['direct','risk','hard','soft'])))
    csvwrite(RUN/'precision/summary.csv',rows);csvwrite(RUN/'precision/score_deltas.csv',scores);csvwrite(RUN/'precision/selected_batch_vs_original_cache.csv',batch)
    settings=[json.loads(l) for l in (RUN/'ledger.jsonl').read_text().splitlines() if json.loads(l)['kind']=='precision_batch']
    assert all(not x['backend']['autocast'] and not x['backend']['matmul_tf32'] and not x['backend']['cudnn_tf32'] and all(v=='torch.float32' for v in x['head_dtypes'].values()) for x in settings if x['mode']=='fp32')
    state=json.loads((RUN/'precision/gates.json').read_text());counter=json.loads((RUN/'forward_counter.json').read_text())
    lines=['# Actual precision and cached replay results','',f"Completed {state['precision_forwards']} precision forwards (cap2048), plus {counter['by_phase']['cache_replay']} separately counted original-batch replay forwards. No cache upcasting substitutes for neural inference. Original cache replay: {len(replay)} policy/key/seed comparisons, {int((~replay.pick_equal).sum())} pick mismatches; maximum H difference {replay.max_H_difference.max():.9g}.",'',
      'R2_full/G3_dyn use best checkpoints,128 keys each per seed; noaff/G2 final14 and S5A/G6 best use16 keys each per seed. Three seeds, both modes. This is not exhaustive over every fresh checkpoint: G3 best seed2 is epoch13 whereas its fresh primary is final14. S5A precision uses unmasked inputs only; masked S5A precision was not in the locked1920-forward matrix and is not claimed tested. Missing additional-model historical probability caches are disclosed. All actual FP32 output tensors were float32 before conversion; autocast and both TF32 paths were disabled. Models stayed eval/frozen and parameter/buffer digests were invariant. Full head outputs and FP32 probabilities, H/sigma are preserved under precision/maps.', '',
      'BF16 versus FP32 differences below use strict v5 selection on historical stored targets and inputs. Original legacy replay is separately verified with original conventions. Selected-batch versus original-batch comparisons are in selected_batch_vs_original_cache.csv. No mode or scene replacement followed the observed differences. Sensor storage, target storage and forward precision remain separate interventions.','',
      '| Model/version | Policy | Pick changes / decisions | FP32−BF16 regret [95% CI] | FP32−BF16 unsafe [95% CI] |','|---|---|---:|---|---|']
    table=pd.DataFrame(rows)
    for (tag,ckpt,pol),g in table.groupby(['model','checkpoint','policy']):
        a=g[g.endpoint=='regret'].iloc[0];b=g[g.endpoint=='unsafe'].iloc[0]
        lines.append(f'| {tag}/{ckpt} | {pol} | {a.pick_changes}/{a.decisions} | {a.mean_diff:.6g} [{a.ci95_lo:.6g}, {a.ci95_hi:.6g}] | {b.mean_diff:.6g} [{b.ci95_lo:.6g}, {b.ci95_hi:.6g}] |')
    lines+=['','Descriptive shared-scene intervals use fixed1/4 weights across the four conditions, with nominal and rough clusters resampled separately. Small subsets cannot establish whole-cohort sign stability; seed-specific sensitivity is in summary.csv. Degenerate zero-event intervals provide no proof of zero population probability. Threshold crossings, exact ties and score deltas are retained without favorable tolerances.','',f"Blocked model interpretations: {state['blocked_models'] or 'none from fidelity/dtype gates'}. The primary fresh mode remains BF16."]
    (RUN/'reports/precision_report.md').write_text('\n'.join(lines)+'\n');print('Precision report complete',flush=True)
if __name__=='__main__':main()
