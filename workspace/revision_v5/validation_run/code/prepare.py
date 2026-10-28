"""Mechanical amendments and locked manifests; no neural inference or fresh scenes."""
from common import *
import pandas as pd,torch,shutil,subprocess,ast
def main():
    assert not (RUN/'manifests/artifact_lock.json').exists()
    protected={}
    for folder in ['paper_v4','kinafford','stage2','stage3','stage4','stage5','data','data_stage2','runs','runs_stage2','runs_stage3','runs_stage4','runs_stage5','results','results_stage2','results_stage3','results_stage4','results_stage5','results_stage6','audit','logs','logs_stage2','logs_stage3','logs_stage4','logs_stage5','review','protocols']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                st=p.stat();protected[str(p.relative_to(ROOT))]=[st.st_size,st.st_mtime_ns]
    js(RUN/'manifests/protected_before.json',protected)
    originals=['claim_evidence_matrix.csv','correctness_report.md','author_questions.md','validation_protocol.md','ExcaScene_v5_Post_Prompt1_Execution.md']
    for name in originals:shutil.copyfile(ROOT/'revision_v5'/name,RUN/'original_prompt1'/name)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    m=pd.read_csv(ROOT/'revision_v5/claim_evidence_matrix.csv',keep_default_na=False);assert len(m)==39
    locations={
    'G01':'main §6.1/Table 5','G02':'main §§6.1–6.2/Tables 5–6','G03':'main §7; Appendix H (S3-D/E)',
    'U01':'main §6.1/Table 5','U02':'main §6.1/Table 5','U03':'main §7.1/Tables 8–9; Appendix F.2',
    'U04':'main §4.5; Appendix F','U05':'Appendix F.1–F.2; selector diagnostics','U06':'main §3.2; Appendix B.3','U07':'Appendix F.1; source edge-case audit',
    'U08':'main §7.1; Appendix F.4','U09':'main §7.1/Table 9; Appendix F.2','U10':'Appendix F.5','U11':'Appendix F.6','U12':'Appendix F.6',
    'R01':'main §6.2/Table 6; Appendix E.2–E.4','R02':'main §6.2; Appendix E.6–E.8',
    'D01':'main §7; Appendix G.1','D02':'main §7.2; Appendix G.1','D03':'main §7.2/Table 10; Appendix G.1/G.3',
    'C01':'main §7.2; Appendix G.2','C02':'main §7.2; Appendix G.2','C03':'main §7.2; Appendix G.2','C04':'Appendix G.2; masking source',
    'N01':'main §§6.1–6.2; Appendix E','N02':'main §6.2/Figure 5; Appendix E.12','N03':'Appendix F.2/G.2/H; statistics source','N04':'Appendix I.6 and resampling source','N05':'Appendix A.3–A.4; evaluator source','N06':'main §5; Appendix B.3; renderer source',
    'L01':'main §3; Appendix B.7','L02':'main §8.3; Appendix B.7/I.6','L03':'v5 existing-output sensitivity (no v4 result claimed)','L04':'main §3.3; Appendix B.5','L05':'main §§3.4/8.3; Appendix I.6, Tables I.6–I.8','L06':'main §§2/8.3; Appendix E.14',
    'R03':'Appendix A.1–A.2/J','R04':'main §8.3; Appendix J','R05':'main §5.1; Appendix I.1–I.5/J'}
    changes=[]
    def setv(cid,col,new,why):
        i=m.index[m.claim_id.eq(cid)][0];old=m.at[i,col]
        if old!=new:changes.append(dict(claim_id=cid,field=col,old=old,new=new,reason=why));m.at[i,col]=new
    for cid,loc in locations.items():setv(cid,'manuscript_location',loc,'Resolved against paper_v4/main_v4.tex, appendix_v4.tex and their supplied PDF/aux numbering; obsolete table/section references removed.')
    setv('U10','scene_population','6 suites and pool; 5x5 footprint (1.25 m side at 0.25 m resolution) versus 1x1 cell','Executed audit/stage6/scripts/phase2/s6b4.py W=5; Appendix F.5 agrees. CSV-only error.')
    setv('C01','statistic','paired seed-averaged frame H/regret/unsafe; 2,000 bootstrap draws; 240 scene clusters / 960 frames per ordinary suite; test_iid 500 / 2,000','Actual exports and Appendix G.2; scenes and draws are distinct.')
    setv('C03','scene_population','clean/iid/test_iid/cam_s3/all_s2/all_s3; no historical cam_s1/cam_s2 masked export','Verified unique suite values in all twelve masked exports; ood_terrain absent.')
    setv('R02','scene_population','cam_s3 for quoted H/regret values; ext_s3/all_s3 are separate severity-gate diagnostics','Mixed-suite attribution corrected; see legacy/R02_source_rows.csv.')
    setv('R02','reported_evidence','cam_s3: S2D_R2_oracle H_mae_vis≈0.1356, regret_direct≈0.2025; G6_lidar ≈0.0819/0.1850 (14-epoch budget best checkpoints, seeds 0/1/2). all_s3 gates all zero is a separate result.','Exact source rows exported separately; no number assigned to ext_s3/all_s3 by implication.')
    setv('R02','source_artifact',m.loc[m.claim_id.eq('R02'),'source_artifact'].iloc[0]+'; results_stage2/analysis/per_suite_endpoints.csv; results/G6_lidar_s{0,1,2}.csv; results_stage2/S2D_R2_oracle_s{0,1,2}.csv','Bind numbers to endpoint/suite/model; checkpoint inventory supplement records historical originals.')
    setv('N06','reported_evidence','gt_vis uses rendered terrain hits before measured-depth/photo corruption, restricted by camera presence and affected by physical drift; it is not the post-corruption measured-depth support.','Trace kinafford/sim/generate.py and render.py; correct prior description.')
    setv('L04','reported_evidence','Fixed camera suites apply occlusion, dust, blur, noise and depth faults simultaneously in that order; cam_s3 also drops one camera. iid separately samples one/two kinds when its mixture trigger fires.','Trace sample_spec and CAM_PHOTO_KINDS; preserve all five mechanisms.')
    setv('U05','claim','REJECTED: selector fallback explains observed gaps','Observed fallback count is zero; no evidence for this explanation.')
    m['claim_status']=['REJECTED' if c=='U05' else 'HUMAN_CONFIRMATION_PENDING' if c=='R05' else 'LIMITED_AS_STATED' for c in m.claim_id]
    m['v5_execution_evidence']='Pending locked execution; historical evidence is not fresh validation.'
    m.to_csv(RUN/'claim_evidence_matrix_corrected.csv',index=False);csvwrite(RUN/'matrix_change_log.csv',changes)
    e=pd.read_csv(ROOT/'results_stage2/analysis/per_suite_endpoints.csv')
    e[e.astype(str).apply(lambda c:c.str.contains('S2D_R2_oracle|G6_lidar')).any(axis=1)&e.astype(str).apply(lambda c:c.str.fullmatch('cam_s3|all_s3|ext_s3')).any(axis=1)].to_csv(RUN/'legacy/R02_source_rows.csv',index=False)
    # Re-run the exact Prompt-1 F.6 arithmetic, redirecting only output/copy destinations.
    tree=ast.parse((ROOT/'revision_v5/checks/analyze_existing.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef)) and not (isinstance(n,ast.FunctionDef) and n.name not in ['read','pos','fraction_percentile','arithmetic'])]
    fn=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='arithmetic')
    fn.body=[n for n in fn.body if not (isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='name')]
    ns={'ROOT':ROOT,'OUT':RUN/'legacy','KEY':['suite','scene_id','frame_id']};exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_arithmetic_redirected','exec'),ns);ns['arithmetic']()
    verification=[]
    for name in ['F6_contrasts_corrected.csv','F6_reading_corrected.csv','F6_stability_counts.csv','F6_interval_class_changes.csv']:
        a=pd.read_csv(ROOT/'revision_v5/analyses'/name);b=pd.read_csv(RUN/'legacy'/name)
        pd.testing.assert_frame_equal(a,b);verification.append(dict(artifact=name,original_sha256=sha(ROOT/'revision_v5/analyses'/name),new_sha256=sha(RUN/'legacy'/name),values_equal=True))
    csvwrite(RUN/'legacy/F6_verification.csv',verification)
    hist=[]
    for stage,p in enumerate(['results/analysis/paired_tests.csv']+[f'results_stage{s}/analysis/registered_tests.csv' for s in [2,3,4,5]],1):
        t=pd.read_csv(ROOT/p);t.insert(0,'historical_family',stage);t.insert(1,'source_artifact',p);hist.append(t)
        shutil.copyfile(ROOT/p,RUN/f'legacy/historical_stage{stage}_original.csv')
    h=pd.concat(hist,ignore_index=True);assert len(h)==49;h.to_csv(RUN/'legacy/historical_tests_49.csv',index=False)
    b=pd.read_csv(ROOT/'results_stage6/B/B4_footprint_per_frame.csv');b=b[(b.tag=='R2_full')&(b.suite=='all_s3')]
    p=b.pivot(index=['seed','suite','scene_id','frame_id'],columns='policy',values='unsafe_fp')
    js(RUN/'legacy/F5_exact_events.json',dict(n=len(p),direct_events=int(p.direct.sum()),soft_events=int(p.soft_composed.sum()),net=int((p.soft_composed-p.direct).sum()),soft_only=int(((p.soft_composed==1)&(p.direct==0)).sum()),direct_only=int(((p.soft_composed==0)&(p.direct==1)).sum())))
    # Actual frozen inventory, including best versions required by precision and sensitivity.
    inventory=[]
    for tag in ['G2_static','G3_dyn','R2_full','R2_noaff','S5A_G3_structured','S2E_G6_lidar']:
      for ckpt in (['best'] if tag.startswith(('S5','S2')) else ['last','best']):
       for seed in [0,1,2]:
        path=cp_path(tag,seed,ckpt);assert path.exists(),path
        z=torch.load(path,map_location='cpu',weights_only=False);epoch=int(z['epoch'])+1
        if ckpt=='last':assert epoch==14,(path,epoch)
        digest=hashlib.sha256()
        for k,v in sorted(z['model'].items()):digest.update(k.encode());digest.update(array_sha(v.detach().numpy()).encode())
        inventory.append(dict(model=tag,seed=seed,checkpoint=ckpt,path=str(path.relative_to(ROOT)),file_sha256=sha(path),weight_sha256=digest.hexdigest(),epoch_one_based=epoch,configuration=json.dumps(z['cfg'],sort_keys=True),head_inventory=json.dumps([k for k in z['model'] if 'head' in k]),validation_record=json.dumps(z.get('val'),default=str)))
    csvwrite(RUN/'manifests/checkpoint_manifest.csv',inventory)
    nominal=list(range(100000,100128));rough=list(range(110000,110064));subset=sorted(map(int,np.random.default_rng(20261013).choice(nominal,28,replace=False)))
    from generator import NOMINAL
    js(RUN/'manifests/generation_manifest.json',dict(source_commit=source,nominal_scene_ids=nominal,rough_scene_ids=rough,slots=[0,1,2,3],conditions=NOMINAL,rough_conditions=['rough_iid'],mechanism_subset=subset,subset_rng='default_rng(20261013).choice(sorted nominal IDs,28,replace=False); sorted',added_condition='combined_present',planned_records=5488,root_seed=20261010,stream_key='SeedSequence([root,family,scene,slot,namespace,sensor,identity,attempt]); terrain uses original deterministic scene formula',namespaces={'base_pose':10,'wider_pose':11,'removed_camera':20,'iid_spec':21,'drifts':30,'joint':31,'camera_fault_latents':40,'lidar_latents':41,'render_shading':50},acceptance='whole same-pose bundle, original_pose_ok + all present drifted camera origins outside terrain/solids by >=1 mm; 50 attempts; no failed-slot substitution; ceilings 10% separately and overall',content_duplicate_check='Before prediction compare SHA256 of canonical float16 H, uint8 sem/material/hazard, float16 utility/reach and float32 q_true against every unique historical base-frame population; exact duplicates block affected population. Near-duplicates are not claimed absent.',sensor_storage='Original uint8 RGB/presence and FP16 measured depth/range; FP32 q/drift/meta; canonical oracles computed FP32 before storage.',source_files={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'kinafford/sim/scene.py',ROOT/'kinafford/sim/kinematics.py',ROOT/'kinafford/sim/render.py',ROOT/'kinafford/sim/degrade.py',RUN/'code/generator.py']}))
    G=['clean','wider_articulation'];S=['iid','all_s2','rough_iid'];C=['clean','cam_s1','cam_s2','cam_s3','appearance_only','depth_only','absence_only','combined_present']
    matrix=[]
    for tag,ckpt,conditions,mask in [('G2_static','last',G,False),('G3_dyn','last',G+S,False),('R2_full','last',S,False),('R2_noaff','last',S,False),('R2_full','best',S,False),('R2_noaff','best',S,False),('S5A_G3_structured','best',C,False),('S5A_G3_structured','best',C,True),('S2E_G6_lidar','best',C,False)]:
      for seed in range(3):
       for condition in conditions:
        n=256 if condition=='rough_iid' else 112 if condition=='combined_present' else 512
        matrix.append(dict(model=tag,checkpoint=ckpt,seed=seed,condition=condition,mask=mask,precision='bf16_autocast',planned_frames=n,planned_forwards=n,artifact_requirement='locked checkpoint + common-valid paired inputs + passed precision/custody gate'))
    assert sum(r['planned_forwards'] for r in matrix)==58608;csvwrite(RUN/'manifests/run_matrix.csv',matrix)
    r=np.random.default_rng(20261012);pn=sorted(map(int,r.choice(np.arange(5000,5240),8,replace=False)));pr=sorted(map(int,r.choice(np.arange(6000,6240),8,replace=False)))
    precision=[]
    for suite,ids in [('iid',pn),('all_s2',pn),('cam_s3',pn),('ood_terrain',pr)]:
      for scene in ids:
       for frame in range(4):
        path=ROOT/f'data/suite_{suite}/f_{scene:06d}_{frame}.npz';assert path.exists(),path
        precision.append(dict(suite=suite,scene_id=scene,frame_id=frame,input_path=str(path.relative_to(ROOT)),input_sha256=sha(path),additional_subset=frame==0 and scene in ids[:4]))
    assert len(precision)==128 and sum(x['additional_subset'] for x in precision)==16
    csvwrite(RUN/'manifests/precision_keys.csv',precision)
    cache=[]
    for tag in ['R2_full','G3_dyn']:
     for seed in range(3):
      for k in precision:
       path=cache_path(tag,seed,k['suite'],k['scene_id'],k['frame_id']);assert path.exists(),path
       cache.append(dict(model=tag,seed=seed,suite=k['suite'],scene_id=k['scene_id'],frame_id=k['frame_id'],path=str(path.relative_to(ROOT)),sha256=sha(path)))
    csvwrite(RUN/'manifests/precision_cache_manifest.csv',cache)
    js(RUN/'manifests/analysis_lock.json',dict(primary_slots=['G','S','C-regret','C-unsafe'],G={'difference':'G3_dyn last - G2_static last','endpoint':'H_reachable_terrain','conditions':G,'weights':[.5,.5]},S={'difference':'R2_full last/soft - R2_noaff last/soft','endpoint':'regret','conditions':S,'weights':[1/3]*3},C={'difference':'S5A best mask - unmask','policy':'direct','condition':'cam_s3','endpoints':['regret','unsafe']},seeds=[0,1,2],bootstrap_draws=10000,bootstrap_rng=20261011,quantiles=[.025,.975,.00625,.99375],bootstrap='shared original nominal scene multiplicities across conditions; separate rough draw; mean across valid frames within condition, then fixed weights; compute each seed metric before seed averaging; record undefined replicates',unsafe_arithmetic='integer numerator, exact Fraction means and linear percentile interpolation; no tolerance',continuous_arithmetic='float64 stable order; no zero tolerance',candidate='finite reach >=0.05, finite score, direct/hazard-soft finite hazard <=0.5; abstain if empty; never relax',oracle='maximum FP32 utility over same finite reachable set, without predicted hazard filter; abstention utility0; empty reachable optimum0',low_utility='zero=0; low=(0,.05]; normalized regret undefined for optimum<=.05',ties='exact row-major first maximum; report threshold boundaries and tie counts',exclusions='preprediction bundle failures only; G common nominal/wide keys, S shared nominal keys; mechanisms complete common 28-scene subset; no model-dependent deletion',height_support='finite oracle reach>=.05 and sem terrain0; no sensor visibility in primary; visible/clean/intersection secondary',interpretation='negative upper adjusted bound supports lower loss; C joint improvement only both negative, with execution changes; no equivalence; degenerate zero-event intervals not zero population probability',precision='1920 forwards, cap2048; selected batch16 in both modes; separate original contiguous batch16 cached replay counted as replay overhead; FP32 params/input, autocast off and TF32 off; compare cache original picks before strict evaluator',caps=CAPS,conditional_caps={'weight_network_bytes':10**9,'alice_network_bytes':14*10**9,'alice_extraction_bytes':24*10**9,'raw_cpu_seconds':8*3600},planned_fresh_forwards=58608,planned_precision_forwards=1920,conditional_external_forwards=192,planned_total_forwards=60720))
    inputs=[]
    for p in [ROOT/'revision_v5'/n for n in originals]+list((ROOT/'paper_v4').glob('*v4.*'))+list((RUN/'legacy').glob('*original.csv')):
        inputs.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    csvwrite(RUN/'manifests/initial_inputs.csv',inputs)
    event('preparation_complete',source_commit=source,protected_files=len(protected),fresh_outcomes_seen=False)
    print('Preparation complete; not locked yet. Checkpoints',len(inventory),'precision keys',len(precision),flush=True)
if __name__=='__main__':main()
