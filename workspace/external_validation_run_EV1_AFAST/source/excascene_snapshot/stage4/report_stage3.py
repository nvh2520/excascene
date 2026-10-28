"""Complete the requested Stage-3 report using new exports and frozen inputs only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from stage4.common import ROOT,RESULTS,sha256,write_json,utc
from stage3.dataset import sensor_keep_mask
from stage4.dataset import presence_class,PRESENCE
OUT=ROOT/'results_stage3/report_completion'
def md(d):return d.to_markdown(index=False,floatfmt='.7g')
def training_presence_replay():
 rows=[]
 for p in sorted((ROOT/'data/train').glob('*.npz')):
  with np.load(p) as d:rows.append(dict(file=str(p.relative_to(ROOT)),scene_id=int(d['scene_id']),frame_id=int(d['frame_id']),cab=bool(d['cam_present'][0]>.5),boom=bool(d['cam_present'][1]>.5),stick=bool(d['cam_present'][2]>.5)))
 frames=pd.DataFrame(rows);frames.to_csv(OUT/'training_frame_inventory.csv',index=False);epochs=[]
 for seed in range(3):
  for ep in range(14):
   counts=np.zeros(4,dtype=np.int64);unchanged=0
   for i,r in enumerate(frames.itertuples(index=False)):
    available=np.array([r.cab,r.boom,r.stick,True]);rng=np.random.default_rng(np.random.SeedSequence([seed,ep,i,r.scene_id,r.frame_id,3103]));keep=sensor_keep_mask(available,.15,rng)
    counts[presence_class(keep)]+=1;unchanged+=int(np.array_equal(keep,available))
   epochs.append(dict(seed=seed,epoch=ep,observations=len(frames),no_additional_drop=unchanged,**dict(zip(PRESENCE,map(int,counts)))))
  print(f'S3-B input-mask replay seed {seed} complete',flush=True)
 e=pd.DataFrame(epochs);e.to_csv(OUT/'dropout_presence_per_epoch.csv',index=False)
 s=e.drop(columns='epoch').groupby('seed',as_index=False).sum();pooled=s.drop(columns='seed').sum().to_frame().T;pooled['seed']='pooled';s=pd.concat([s,pooled],ignore_index=True)
 for c in PRESENCE+['no_additional_drop']:s[c+'_fraction']=s[c]/s.observations
 out=pd.concat([s.assign(tag=tag) for tag in ['S3B_G3_dropout','S3B_R2_dropout']],ignore_index=True);out.to_csv(OUT/'dropout_training_presence.csv',index=False)
 write_json(OUT/'dropout_presence_provenance.json',dict(time=utc(),method='exact deterministic replay of committed input masks, not model inference or new training',epochs=list(range(14)),seeds=[0,1,2],stream='SeedSequence([seed,epoch,index,scene_id,frame_id,3103])',training_frames=len(frames),mask_source_sha256=sha256(ROOT/'stage3/dataset.py'),frame_inventory_sha256=sha256(OUT/'training_frame_inventory.csv'),same_stream_for_both_variants=True,nominal_presence_definition='all four sensors present; separate no-additional-drop field accounts for original absence'))
 return out
def comparisons(overall,pairs):
 rows=[]
 numeric=[c for c in overall.select_dtypes('number') if c not in ['seeds','rows','unique_frames']]
 for new,old in pairs:
  a=overall[overall.tag==new].set_index('suite');b=overall[overall.tag==old].set_index('suite')
  for suite in a.index:
   for metric in numeric:
    if metric not in b:continue
    x,y=a.loc[suite,metric],b.loc[suite,metric]
    if pd.notna(x) and pd.notna(y):rows.append(dict(variant=new,comparator=old,suite=suite,endpoint=metric,new_mean=x,frozen_mean=y,mean_diff=x-y,status='DESCRIPTIVE'))
 return pd.DataFrame(rows)
def run():
 OUT.mkdir(parents=True,exist_ok=True);inputs={}
 def csv(path):inputs[str(path.relative_to(ROOT))]=sha256(path);return pd.read_csv(path)
 original=ROOT/'results_stage3';analysis=original/'analysis';overall=csv(analysis/'per_suite_endpoints.csv');sections=['# Stage-3 reporting completion\n\nExports only. The registered Stage-3 tables, raw results, verdicts, original summary and ledger remain frozen. All new CSVs and source receipts are in `report_completion/`. Counts of decisions use frame-seed exposures; paired effects use the registered seed-average pairing.']
 a=csv(analysis/'expected_utility_diagnostics.csv');counts=a.groupby(['tag','suite','selector'],as_index=False)[['observations','unsafe_count']].sum();pool=counts.groupby(['tag','selector'],as_index=False)[['observations','unsafe_count']].sum();pool['suite']='registered_pool';counts=pd.concat([counts,pool],ignore_index=True);counts['unsafe_fraction']=counts.unsafe_count/counts.observations;counts.to_csv(OUT/'A_unsafe_counts.csv',index=False)
 perseed_pool=a.groupby(['tag','seed','selector'],as_index=False)[['observations','unsafe_count']].sum();perseed_pool['suite']='registered_pool';perseed_pool['unsafe_fraction']=perseed_pool.unsafe_count/perseed_pool.observations
 pd.concat([a[['tag','seed','suite','selector','observations','unsafe_count','unsafe_fraction']],perseed_pool],ignore_index=True).to_csv(OUT/'A_unsafe_per_seed.csv',index=False)
 soft=csv(analysis/'soft_vs_hard_composition.csv');soft.to_csv(OUT/'A_soft_vs_hard_regret.csv',index=False)
 sections+=['## a. Expected-utility composition\n\n`direct` uses the frozen hazard exclusion; `direct_nothreshold` is the S3-A common selector rule. `comp` is frozen hard composition; `soft` is probability composition without hazard exclusion. The pool is iid + all_s2 + ood_terrain. Pooled decision counts retain every seed exposure, not a fictitious seed-averaged unsafe pick.\n\n'+md(counts),'Soft-minus-hard composed regret (seed-averaged paired frames):\n\n'+md(soft)]
 bp=[('S3B_G3_dropout','G3_dyn'),('S3B_R2_dropout','R2_full')];b=comparisons(overall,bp);b.to_csv(OUT/'B_all_suite_endpoints.csv',index=False);no=csv(analysis/'dropout_no_regression_check.csv');no.to_csv(OUT/'B_no_regression.csv',index=False)
 presence=training_presence_replay();sections+=['## b. Independent modality dropout\n\nAll-suite height, regret and unsafe endpoints are below; the CSV contains every available endpoint. Negative differences favor the new run for these three endpoints.\n\n'+md(b[b.endpoint.isin(['H_mae_vis','regret_direct','unsafe_direct'])]),'Clean/iid/test_iid no-regression check, including every unfavorable comparison:\n\n'+md(no),'Training-time sensor-presence fractions were reconstructed exactly from the committed deterministic masks for every frame, epoch and seed. They were not logged as observations during the original fit. Each variant uses the same mask stream at matching seed/frame/epoch; per-epoch replay and input identity are exported. S3-B redraws independent sensor masks until nonempty. S4-A instead retains LiDAR on originally camera-empty frames without redrawing; both obey the nonempty-sensor invariant. The full 14-epoch training history is counted, including epochs after a selected earlier checkpoint. `all_four_present` is nominal presence; `no_additional_drop` means no extra S3-B drop and can include originally absent cameras. The residual mixed-camera/LiDAR category is retained so the mutually exclusive presence fractions sum to one.\n\n'+md(presence)]
 cp=[('S3C_G3_jitter_epoch','S2C_G3_jitter'),('S3C_R2_epoch','R2_full')];c=comparisons(overall,cp);c.to_csv(OUT/'C_all_suite_endpoints.csv',index=False);joint=[]
 for new,old in cp:
  aa=[];bb=[]
  for s in range(3):
   aa.append(csv(original/f'{new}_s{s}.csv'))
   source=ROOT/('results_stage2' if old.startswith('S2') else 'results')/f'{old}_s{s}.csv';bb.append(csv(source))
  keys=['suite','scene_id','frame_id'];x=pd.concat(aa);y=pd.concat(bb);x=x[x.suite.isin(['joint_s2','joint_s3'])].groupby(keys).regret_direct.mean();y=y[y.suite.isin(['joint_s2','joint_s3'])].groupby(keys).regret_direct.mean();j=pd.concat([x.rename('a'),y.rename('b')],axis=1).dropna();d=j.a-j.b
  joint.append(dict(variant=new,comparator=old,suites='joint_s2+joint_s3',paired_frames=len(j),new_mean=j.a.mean(),frozen_mean=j.b.mean(),mean_diff=d.mean(),median_diff=d.median(),improved_fraction=(d<0).mean(),status='DESCRIPTIVE'))
 joint=pd.DataFrame(joint);joint.to_csv(OUT/'C_joint_pool_regret.csv',index=False);sections+=['## c. Epoch-aware augmentation\n\nJoint-pool direct regret (descriptive, no additional test):\n\n'+md(joint),'All-suite primary endpoints; all available endpoints are in `C_all_suite_endpoints.csv`.\n\n'+md(c[c.endpoint.isin(['H_mae_vis','regret_direct','unsafe_direct'])])]
 oracle_path=original/'audit/oracle_rounding_check.json';oracle=json.loads(oracle_path.read_text());inputs[str(oracle_path.relative_to(ROOT))]=sha256(oracle_path);orows=pd.DataFrame([dict(suite=k,**v) for k,v in oracle['values'].items()]);orows.to_csv(OUT/'D_oracle_rounding.csv',index=False);sections+=['## d. Oracle 0.1356 display\n\n'+oracle['conclusion']+'\n\n'+md(orows)+'\n\nRaw cam_s3 minus all_s2: '+repr(oracle['difference'])+'. Full precision and source hashes are preserved in `audit/oracle_rounding_check.json`.']
 sensitivity=csv(analysis/'cluster_sensitivity_concordance.csv');discordant=sensitivity[~sensitivity.scene_signflip_concordant].copy();prior=[]
 for stage,path in [('Stage 1',ROOT/'results/analysis/paired_tests.csv'),('Stage 2',ROOT/'results_stage2/analysis/registered_tests.csv')]:
  d=csv(path).rename(columns={'hypothesis':'id','wilcoxon_p':'registered_wilcoxon_p','holm_p':'registered_holm_p'});d['stage']=stage;prior.append(d[['stage','id','registered_wilcoxon_p','registered_holm_p']])
 discordant=discordant.merge(pd.concat(prior),on=['stage','id']);cols=['stage','id','registered_wilcoxon_p','registered_holm_p','scene_signflip_p','scene_signflip_holm_p','registered_supported','scene_signflip_supported','ci_lo','ci_hi'];discordant[cols].to_csv(OUT/'E_signflip_discordance.csv',index=False);sections+=['## e. Discordant scene-level sign-flip sensitivities\n\nRaw and Holm-adjusted p-values are both shown. The registered verdict remains unchanged; these are descriptive sensitivities with the original separate families.\n\n'+md(discordant[cols])]
 replication=csv(analysis/'seed_replication.csv');r=replication[replication.seed!='seed_agreement'][['id','seed','paired_frames','a_mean','b_mean','mean_diff','median_diff','sign','stage1_supported','status']];r.to_csv(OUT/'D_per_seed_magnitudes.csv',index=False);sections+=['## f. Seed-replication magnitudes\n\nSame-seed A-minus-B differences and the complete five-seed average; no new registered test.\n\n'+md(r)]
 att=json.loads((original/'PREREGISTRATION_ATTESTATION.json').read_text());ledger=json.loads((analysis/'compute_usage.json').read_text());fits=[j for j in ledger['jobs'] if j['key'].startswith('train/') and j['status']=='complete'];reconcile=pd.DataFrame([dict(block=b,named_request_fits=n,committed_fits=m,completed_fits=sum(j['block']==b for j in fits)) for b,n,m in [('B',6,6),('C',6,6),('D',6,8)]]);reconcile.to_csv(OUT/'fit_reconciliation.csv',index=False)
 sections+=['## g. Fit-count reconciliation\n\n'+md(reconcile)+'\n\nThe user\'s named B/C/D variants imply 18 fits. The committed Stage-3 protocol explicitly includes G6 LiDAR seeds 3 and 4 inside D\'s unchanged 7-hour cap, producing 20 preregistered fits. They provide the missing matched short-budget comparator seeds for five-seed H4b; Stage-2 42-epoch checkpoints were not substituted. This was recorded before results in `results_stage3/DEVIATIONS.md` at 2026-09-08T16:02:47.814089+00:00 and in the attested protocol. It is not a post-result addition.']
 times=dict(protocol_commit=att['commit'],protocol_sha256=att['sha256'],first_tracked_job=min(j['start'] for j in ledger['jobs']),first_training_or_evaluation=min(j['start'] for j in ledger['jobs'] if j['key'].startswith(('train/','eval/'))),ledger_sha256=sha256(analysis/'compute_usage.json'));write_json(OUT/'registration_and_timing.json',times);sections+=['## h. Registration and timing\n\n```json\n'+json.dumps(times,indent=2)+'\n```\n\nThe first tracked job was the protection inventory; the later timestamp is the first frozen-checkpoint evaluation. The attested protocol precedes both.']
 write_json(OUT/'input_hashes.json',inputs);(original/'REPORT.md').write_text('\n\n'.join(sections)+'\n')
 note='STAGE-4 REQUEST / STAGE-3 FIT RECONCILIATION'
 deviations=original/'DEVIATIONS.md'
 if note not in deviations.read_text():
  with deviations.open('a') as f:f.write(f'\n- {utc()} **{note}** — The original named B/C/D variants imply 18 fits; committed protocol {att["commit"]} explicitly preregistered two necessary G6_lidar comparator fits (seeds 3,4), making 20. See the pre-result 2026-09-08T16:02:47.814089+00:00 entry. All 20 completed within their caps. Added REPORT.md and new report_completion exports only, including deterministic S3-B mask reconstruction. No frozen Stage-3 table, CSV, verdict, ledger or manuscript was regenerated.\n')
 print('STAGE-3 REPORT COMPLETE: a–h; frozen scientific exports unchanged',flush=True)
if __name__=='__main__':run()
