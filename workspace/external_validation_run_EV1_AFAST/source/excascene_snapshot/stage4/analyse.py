"""Four registered tests, all-suite descriptives and observed no-redraw exposure counts."""
import argparse,json
import numpy as np
import pandas as pd
from stage4.common import ROOT,RESULTS,ANALYSIS,SUITES,sha256,write_json
from stage4.stats import registered_tests,paired_values
from stage4.registry import REGISTRY
from stage4.dataset import BRANCHES,PRESENCE
from stage2.analyse import registry_table
from stage2.reporting import endpoint_points
from kinafford.analysis.aggregate import suite_level_metrics,MAIN_METRICS,CAL_METRICS
TAG='S4A_G3_structured'
def markdown(d):return d.astype(object).where(d.notna(),None).to_markdown(index=False,floatfmt='.7g',missingval='NA') if len(d) else 'No observations.'
def load_frames():
 frames=[];hashes={}
 paths=[ROOT/'results'/f'{tag}_s{s}.csv' for tag in ['G3_dyn','G6_lidar'] for s in range(3)]
 paths += [ROOT/'results_stage3'/f'S3B_G3_dropout_s{s}.csv' for s in range(3)]
 paths += list(RESULTS.glob(TAG+'_s[0-2].csv'))
 for p in paths:
  d=pd.read_csv(p);d['stage']='Stage 4' if p.parent==RESULTS else 'Stage 3' if p.parent.name=='results_stage3' else 'Stage 1';frames.append(d);hashes[str(p.relative_to(ROOT))]=sha256(p)
 frame=pd.concat(frames,ignore_index=True);assert not frame.duplicated(['tag','seed','suite','scene_id','frame_id']).any()
 write_json(ANALYSIS/'input_hashes.json',hashes);return frame

def summaries(frame):
 excluded=['seed','scene_id','frame_id'];metrics=[c for c in frame.select_dtypes('number') if c not in excluded and not c.startswith('cm_')]
 seeds,overall=endpoint_points(frame,metrics);special=suite_level_metrics(frame);cols=[c for c in special if c not in ['tag','seed','suite']]
 seeds=seeds.drop(columns=cols,errors='ignore').merge(special,on=['tag','seed','suite'],how='left')
 overall=overall.drop(columns=cols,errors='ignore').merge(special.groupby(['tag','suite'])[cols].mean().reset_index(),on=['tag','suite'],how='left')
 seeds=seeds.merge(frame.groupby(['tag','seed','suite']).size().rename('rows').reset_index(),on=['tag','seed','suite'])
 counts=frame.groupby(['tag','suite']).agg(seeds=('seed','nunique'),frame_seed_observations=('frame_id','size')).reset_index();overall=overall.merge(counts,on=['tag','suite'])
 seeds.to_csv(ANALYSIS/'per_seed_suite_endpoints.csv',index=False);overall.to_csv(ANALYSIS/'per_suite_endpoints.csv',index=False)
 trade=overall[overall.tag.isin(['G3_dyn','S3B_G3_dropout',TAG])&overall.suite.isin(['clean','iid','test_iid','cam_s3','all_s3'])][['tag','suite','H_mae_vis','regret_direct','seeds','frame_seed_observations']]
 trade.to_csv(ANALYSIS/'three_point_tradeoff.csv',index=False);return overall,trade

def selections_and_exposures():
 import torch
 selected=[];curves=[];exposures=[]
 for p in sorted((ROOT/'runs_stage4'/TAG).glob('seed*/done.json')):
  run=p.parent;ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=False);cfg=ckpt['cfg'];values=[]
  for line in (run/'log.jsonl').read_text().splitlines():
   x=json.loads(line)
   if 'val' in x:
    v=x['val'];criterion=v['H_mae_vis']+v.get('aff_mae',0);values.append((v['epoch'],criterion));curves.append(dict(tag=TAG,seed=cfg['seed'],epoch=v['epoch'],epoch_one_based=v['epoch']+1,H_mae_vis=v['H_mae_vis'],aff_mae=v.get('aff_mae'),criterion=criterion))
  assert ckpt['epoch']==min(values,key=lambda x:x[1])[0]
  selected.append(dict(tag=TAG,seed=cfg['seed'],profile='full',epochs=cfg['epochs'],selected_epoch_zero_based=ckpt['epoch'],selected_epoch_one_based=ckpt['epoch']+1,is_final_epoch=ckpt['epoch']+1==cfg['epochs'],checkpoint_sha256=sha256(run/'best.pt'),config_sha256=sha256(run/'config.json'),training_log_sha256=sha256(run/'log.jsonl'),steps=json.loads(p.read_text())['steps']))
  d=json.loads((run/'dropout_exposures.json').read_text())
  for e in d['epochs']:
   r={k:v for k,v in e.items() if not isinstance(v,dict)};r.update(seed=cfg['seed'])
   for group in ['accepted_branches','proposed_branches','presence']:
    r.update({group+'_'+k:v for k,v in e[group].items()})
   exposures.append(r)
 s=pd.DataFrame(selected);s.to_csv(ANALYSIS/'selected_epochs.csv',index=False);pd.DataFrame(curves).to_csv(ANALYSIS/'validation_curves.csv',index=False)
 e=pd.DataFrame(exposures);e.to_csv(ANALYSIS/'dropout_exposures_per_epoch.csv',index=False)
 if len(e):
  perseed=e.drop(columns=['epoch','epoch_one_based']).groupby('seed',as_index=False).sum();pooled=perseed.drop(columns='seed').sum().to_frame().T;pooled['seed']='pooled';perseed=pd.concat([perseed,pooled],ignore_index=True);rows=[]
  for r in perseed.to_dict('records'):
   assert r['observations']==sum(r['accepted_branches_'+b] for b in BRANCHES)==sum(r['proposed_branches_'+b] for b in BRANCHES) and r['redraws']==0
   for branch,label,nominal in zip(BRANCHES,['LiDAR-only','camera-only','one-camera-dropped','nominal'],[.30,.15,.20,.35]):
    rows.append(dict(seed=r['seed'],branch=label,nominal_frequency=nominal,proposed_count=r['proposed_branches_'+branch],proposed_frequency=r['proposed_branches_'+branch]/r['observations'],executed_count=r['accepted_branches_'+branch],realized_frequency=r['accepted_branches_'+branch]/r['observations'],observations=r['observations'],original_no_camera_frames=r['original_no_cameras'],precedence_changed_branch=r['precedence_changed_branch'],redraws=r['redraws']))
   for k in PRESENCE:r['presence_'+k+'_fraction']=r['presence_'+k]/r['observations']
  # Physical presence fractions are distinct from branch frequencies.
  for k in PRESENCE:perseed['presence_'+k+'_fraction']=perseed['presence_'+k]/perseed.observations
  perseed.to_csv(ANALYSIS/'dropout_exposures_per_seed.csv',index=False);pd.DataFrame(rows).to_csv(ANALYSIS/'dropout_branch_frequencies.csv',index=False)
 return s

def histograms(frame):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 out=ANALYSIS/'paired_difference_histograms';out.mkdir(exist_ok=True);manifest=[];values=[];bins=[]
 with plt.rc_context({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}),PdfPages(out/'stage4_registered_tests.pdf',metadata={'CreationDate':None,'ModDate':None}) as pdf:
  for row in REGISTRY:
   j=paired_values(frame,row);d=j.a-j.b
   if not len(d):continue
   counts,edges=np.histogram(d,bins=50);fig,ax=plt.subplots(figsize=(7.5,4.8));ax.stairs(counts,edges,fill=True,color='#0072B2',alpha=.8);ax.axvline(0,color='black');ax.axvline(d.mean(),color='#D55E00',ls='--',label='Mean');ax.axvline(d.median(),color='#009E73',ls=':',label='Median');ax.set(xlabel='Seed-averaged paired difference A − B',ylabel='Paired frames',title=row[0]);ax.legend();fig.text(.1,.02,f'{row[1]} vs {row[2]}\n{row[3]} / {row[4]}; suites: {", ".join(row[5])}',fontsize=8);fig.subplots_adjust(bottom=.21);path=out/(row[0]+'.pdf');fig.savefig(path);pdf.savefig(fig);fig.savefig(path.with_suffix('.png'),dpi=300);plt.close(fig)
   values.append(j.reset_index().assign(id=row[0],difference=d.to_numpy()));bins.append(pd.DataFrame(dict(id=row[0],left=edges[:-1],right=edges[1:],count=counts)));manifest.append(dict(id=row[0],pairs=len(j),mean_diff=d.mean(),median_diff=d.median(),pdf=path.name,sha256=sha256(path)))
 pd.DataFrame(manifest).to_csv(out/'manifest.csv',index=False)
 if values:pd.concat(values).to_csv(out/'paired_differences.csv',index=False);pd.concat(bins).to_csv(out/'histogram_bins.csv',index=False)

def report():
 ANALYSIS.mkdir(exist_ok=True);frame=load_frames();tests=registered_tests(frame);tests.to_csv(ANALYSIS/'registered_tests.csv',index=False);table=registry_table(tests);table.to_csv(ANALYSIS/'registered_4_row_table.csv',index=False);(ANALYSIS/'registered_tests.md').write_text(markdown(table)+'\n');overall,trade=summaries(frame);selected=selections_and_exposures();histograms(frame)
 branches=ANALYSIS/'dropout_branch_frequencies.csv';physical=ANALYSIS/'dropout_exposures_per_seed.csv';ledger=json.loads((ANALYSIS/'compute_usage.json').read_text())
 text='# S4-A\n\nSeparate four-slot Holm family; support requires Holm p < 0.05 and scene-bootstrap upper CI < 0. All tests use seeds 0–2, full profile, 14 epochs; all descriptive suites are retained.\n\n'+markdown(table)+'\n\nAll-suite endpoints\n\n'+markdown(overall[overall.tag==TAG])+'\n\nThree-point trade-off\n\n'+markdown(trade)+'\n\nCheckpoint selection\n\n'+markdown(selected)+'\n\nObserved dropout branches beside nominal probabilities\n\n'+(markdown(pd.read_csv(branches)) if branches.exists() else 'Not run.')+'\n\nPhysical sensor-presence counts and fractions\n\n'+(markdown(pd.read_csv(physical)) if physical.exists() else 'Not run.')+'\n\nS3-B redraws independent sensor masks until nonempty. S4-A draws once; originally camera-empty frames retain LiDAR and execute the LiDAR-only branch regardless of proposal. The counters above include all training epochs. Choosing an already absent camera in the one-camera branch causes no additional drop; nominal means no additional drop. Actual physical presence is reported separately.\n\nValidation curves: validation_curves.csv. Job-hours at report: '+str(ledger['job_hours_by_block']['A'])+'. Deviations: ../DEVIATIONS.md.\n'
 (ANALYSIS/'S4-A.md').write_text(text)
 grid=ANALYSIS/'selector_grid.csv'
 if grid.exists():
  replay=RESULTS/'audit/replay_column_differences.csv';r=pd.read_csv(replay);core=r[r.column.isin(['sem_acc_vis','mat_acc','H_mae_vis','H_mae_all','H_mae_reach','regret_direct'])].groupby(['original','suite','column']).max_abs_deviation.max().reset_index()
  (ANALYSIS/'S4-B.md').write_text('# S4-B\n\nDescriptive only, no registered tests. Endpoints average three seed means; counts retain frame-seed exposures.\n\n'+markdown(pd.read_csv(grid))+'\n\nMissing-suite replay: maximum absolute deviations across three seeds\n\n'+markdown(core)+'\n\nAll stored numeric columns and confusion counts: ../audit/replay_column_differences.csv. Frozen CSVs contain class-derived metrics and height errors but no cellwise argmax or height maps: map identity is unverifiable, and is not inferred from aggregate agreement. Canonical matching keys and input/output hashes: ../audit/replay_hashes.json.\n\nExactly hazard 0.5 is retained by the selector exclusion; the frozen hard composer itself uses hazard < 0.5. Reach < 0.05 is excluded, with original-score reachable fallback and row-major ties. No-reachable cases are separate from finite-max ties. Job-hours at report: '+str(ledger['job_hours_by_block']['B'])+'. Deviations: ../DEVIATIONS.md.\n')
 print(table.to_string(index=False),flush=True)
if __name__=='__main__':report()
