"""Stage-3 registered statistics and complete descriptive exports; writes Stage 3 only."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from stage3.common import ROOT,RESULTS,ANALYSIS,SUITES,write_json,sha256,utc
from stage3.stats import registered_tests,paired_values
from stage3.registry import REGISTRY,BLOCK_TAGS
from stage2.reporting import endpoint_points
from stage2.analyse import registry_table
from kinafford.analysis.aggregate import suite_level_metrics,MAIN_METRICS,CAL_METRICS

def markdown(d):return d.astype(object).where(d.notna(),None).to_markdown(index=False,floatfmt='.6g',missingval='NA') if len(d) else 'No observations.'

def load_frames():
 frames=[];inputs={}
 paths=[p for tag in ['G3_dyn','R2_full','G6_lidar'] for p in (ROOT/'results').glob(tag+'_s[0-9].csv')]
 paths+=list((ROOT/'results_stage2').glob('S2C_G3_jitter_s[0-9].csv'))
 paths+=list(RESULTS.glob('*_s[0-9].csv'))+list((RESULTS/'A').glob('*_s[0-9].csv'))
 for p in sorted(paths):
  d=pd.read_csv(p);d['stage']='Stage 3' if p.is_relative_to(RESULTS) else 'Stage 2' if p.is_relative_to(ROOT/'results_stage2') else 'Stage 1';frames.append(d);inputs[str(p.relative_to(ROOT))]=sha256(p)
 if not frames:raise RuntimeError('No frame records')
 frame=pd.concat(frames,ignore_index=True)
 if frame.duplicated(['tag','seed','suite','scene_id','frame_id']).any():raise RuntimeError('Duplicate run/frame key')
 write_json(ANALYSIS/'input_hashes.json',inputs);return frame

def summaries(frame):
 metrics=list(dict.fromkeys(MAIN_METRICS+CAL_METRICS+['regret_direct_norm','regret_comp_norm','regret_risk_norm','unsafe_comp','unsafe_risk','top5_regret_direct','top5_regret_comp','H_mae_all','H_mae_reach','mat_acc','haz_miss_rate','haz_brier','affc_mae','aff_spearman','H_cov90_all','H_sigma_mean_vis','H_sigma_mean_unvis','cp_aff_width','lb_pick_true_aff']+[c for c in frame if c.startswith(('regret_soft','regret_direct_nothreshold','unsafe_soft','unsafe_direct_nothreshold','fallback_','no_reachable_'))]))
 metrics=[m for m in metrics if m in frame];seeds,overall=endpoint_points(frame,metrics)
 special=suite_level_metrics(frame[frame.cm_0.notna()]);cols=[c for c in special if c not in ['tag','seed','suite']]
 seeds=seeds.drop(columns=cols,errors='ignore').merge(special,on=['tag','seed','suite'],how='left')
 overall=overall.drop(columns=cols,errors='ignore').merge(special.groupby(['tag','suite'])[cols].mean().reset_index(),on=['tag','suite'],how='left')
 counts=frame.groupby(['stage','tag','suite']).agg(seeds=('seed','nunique'),rows=('frame_id','size')).reset_index();unique=frame.drop_duplicates(['stage','tag','suite','scene_id','frame_id']).groupby(['stage','tag','suite']).size().rename('unique_frames').reset_index()
 overall=overall.merge(counts,on=['stage','tag','suite']).merge(unique,on=['stage','tag','suite']);seeds=seeds.merge(frame.groupby(['tag','seed','suite']).size().rename('rows').reset_index(),on=['tag','seed','suite'])
 seeds.to_csv(ANALYSIS/'per_seed_suite_endpoints.csv',index=False);overall.to_csv(ANALYSIS/'per_suite_endpoints.csv',index=False)
 rows=[]
 for tag,old in [('S3B_G3_dropout','G3_dyn'),('S3B_R2_dropout','R2_full')]:
  for suite in ['clean','iid','test_iid']:
   for metric in ['H_mae_vis','regret_direct']:
    a=overall[(overall.tag==tag)&(overall.suite==suite)];b=overall[(overall.tag==old)&(overall.suite==suite)];av=a.iloc[0][metric] if len(a) else np.nan;bv=b.iloc[0][metric]
    rows.append(dict(variant=tag,comparator=old,suite=suite,endpoint=metric,new_mean=av,frozen_mean=bv,mean_diff=av-bv,regression=bool(av>bv) if np.isfinite(av) else None,status='DESCRIPTIVE' if len(a) else 'NOT RUN'))
 pd.DataFrame(rows).to_csv(ANALYSIS/'dropout_no_regression_check.csv',index=False)
 return overall

def selections():
 import torch
 rows=[];curves=[]
 for p in sorted((ROOT/'runs_stage3').glob('*/seed*/done.json')):
  run=p.parent;state=torch.load(run/'best.pt',map_location='cpu',weights_only=False);cfg=state['cfg'];vals=[]
  for line in (run/'log.jsonl').read_text().splitlines():
   r=json.loads(line)
   if 'val' in r:
    v=r['val'];score=v['H_mae_vis']+v.get('aff_mae',0.);vals.append((v['epoch'],score));curves.append(dict(tag=cfg['variant'],seed=cfg['seed'],epoch=v['epoch'],epoch_one_based=v['epoch']+1,H_mae_vis=v['H_mae_vis'],aff_mae=v.get('aff_mae'),criterion=score))
  assert state['epoch']==min(vals,key=lambda x:x[1])[0]
  rows.append(dict(tag=cfg['variant'],seed=cfg['seed'],profile='full',epochs=cfg['epochs'],selected_epoch_zero_based=state['epoch'],selected_epoch_one_based=state['epoch']+1,is_final_epoch=state['epoch']+1==cfg['epochs'],checkpoint_sha256=sha256(run/'best.pt'),config_sha256=sha256(run/'config.json'),training_log_sha256=sha256(run/'log.jsonl'),epoch_rng=cfg['model'].get('epoch_rng',False),modality_dropout=cfg['model'].get('modality_dropout',0)))
 pd.DataFrame(rows).to_csv(ANALYSIS/'selected_epochs.csv',index=False);pd.DataFrame(curves).to_csv(ANALYSIS/'validation_curves.csv',index=False)
 return pd.DataFrame(rows)

def a_descriptions(frame):
 rows=[]
 for (tag,seed,suite),g in frame[frame.tag.str.startswith('S3A_')].groupby(['tag','seed','suite']):
  for sel in ['direct_nothreshold','soft','comp','direct']:
   c='unsafe_'+sel;r='regret_'+sel
   rows.append(dict(tag=tag,seed=seed,suite=suite,selector=sel,observations=len(g),unsafe_count=int(g[c].sum()),unsafe_fraction=g[c].mean(),mean_regret=g[r].mean(),fallback_count=int(g['fallback_'+sel].sum()) if 'fallback_'+sel in g else np.nan,no_reachable_count=int(g['no_reachable_'+sel].sum()) if 'no_reachable_'+sel in g else np.nan))
 pd.DataFrame(rows).to_csv(ANALYSIS/'expected_utility_diagnostics.csv',index=False)
 rows=[]
 for tag,g in frame[frame.tag.str.startswith('S3A_')].groupby('tag'):
  for suite in ['iid','all_s2','ood_terrain','registered_pool']:
   x=g if suite=='registered_pool' else g[g.suite==suite];j=x.groupby(['suite','scene_id','frame_id'])[['regret_soft','regret_comp']].mean();d=j.regret_soft-j.regret_comp
   rows.append(dict(tag=tag,suite=suite,paired_frames=len(j),soft_mean=j.regret_soft.mean(),hard_mean=j.regret_comp.mean(),mean_diff=d.mean(),median_diff=d.median(),improved_fraction=(d<0).mean(),status='DESCRIPTIVE'))
 pd.DataFrame(rows).to_csv(ANALYSIS/'soft_vs_hard_composition.csv',index=False)

def histograms(frame):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 out=ANALYSIS/'paired_difference_histograms';out.mkdir(exist_ok=True);manifest=[];values=[];bins=[]
 with plt.rc_context({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}),PdfPages(out/'stage3_registered_tests.pdf',metadata={'CreationDate':None,'ModDate':None}) as pdf:
  for row in REGISTRY:
   j=paired_values(frame,row);d=j.a-j.b
   if not len(d):continue
   counts,edges=np.histogram(d,bins=50);fig,ax=plt.subplots(figsize=(7.5,4.8));ax.stairs(counts,edges,fill=True,color='#0072B2',alpha=.8);ax.axvline(0,color='black');ax.axvline(d.mean(),color='#D55E00',ls='--',label='Mean');ax.axvline(d.median(),color='#009E73',ls=':',label='Median');ax.set(xlabel='Seed-averaged paired difference A − B',ylabel='Paired frames',title=row[0]);ax.legend();fig.text(.1,.02,f'{row[1]} vs {row[2]}\n{row[3]} / {row[4]}; suites: {", ".join(row[5])}',fontsize=8);fig.subplots_adjust(bottom=.21);path=out/(row[0]+'.pdf');fig.savefig(path);pdf.savefig(fig);fig.savefig(path.with_suffix('.png'),dpi=300);plt.close(fig)
   values.append(j.reset_index().assign(id=row[0],difference=d.to_numpy()));bins.append(pd.DataFrame(dict(id=row[0],left=edges[:-1],right=edges[1:],count=counts)));manifest.append(dict(id=row[0],pairs=len(j),mean_diff=d.mean(),median_diff=d.median(),pdf=path.name,sha256=sha256(path)))
 pd.DataFrame(manifest).to_csv(out/'manifest.csv',index=False)
 if values:pd.concat(values).to_csv(out/'paired_differences.csv',index=False);pd.concat(bins).to_csv(out/'histogram_bins.csv',index=False)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--block');ap.add_argument('--final',action='store_true');args=ap.parse_args();ANALYSIS.mkdir(exist_ok=True)
 frame=load_frames();tests=registered_tests(frame);tests.to_csv(ANALYSIS/'registered_tests.csv',index=False);table=registry_table(tests);table.to_csv(ANALYSIS/'registered_8_row_table.csv',index=False);(ANALYSIS/'registered_tests.md').write_text(markdown(table)+'\n');overall=summaries(frame);selected=selections();a_descriptions(frame)
 if args.final:histograms(frame)
 for block in (list('ABCDEF') if args.final else [args.block] if args.block else []):
  subset=tests[tests.hypothesis.str.startswith('S3-'+block+'-')];tags=BLOCK_TAGS.get(block,['S3A_R2','S3A_G3'] if block=='A' else []);part=overall[overall.tag.isin(tags)];ledger=json.loads((ANALYSIS/'compute_usage.json').read_text())
  text='# S3-'+block+'\n\n'+('Final global eight-slot Holm results.\n\n' if args.final else 'Provisional global eight-slot Holm results; missing slots retain p=1 bookkeeping.\n\n')+markdown(registry_table(subset))+'\n\n'+markdown(part)+'\n\nSelected epochs\n\n'+markdown(selected[selected.tag.isin(tags)] if len(selected) else selected)+'\n\nJob-hours at this report: '+str(ledger['job_hours_by_block'].get(block))+'\n'
  for name in ({'A':['expected_utility_diagnostics.csv','soft_vs_hard_composition.csv'],'B':['dropout_no_regression_check.csv'],'D':['seed_replication.csv'],'E':['cluster_sensitivity_concordance.csv']}.get(block,[])):
   p=ANALYSIS/name
   if p.exists():text+='\n'+name+'\n\n'+markdown(pd.read_csv(p))+'\n'
  if block=='F':text+='\nInventory and proposed checks: ../../audit/stage3_realdata_feasibility.md. No real experiment.\n'
  text+='\nDeviations: ../DEVIATIONS.md. Per-run seeds and complete endpoints: per_seed_suite_endpoints.csv; source/checkpoint records: ../audit/.\n'
  (ANALYSIS/f'S3-{block}.md').write_text(text)
 print(table.to_string(index=False),flush=True)
if __name__=='__main__':main()
