"""Tables and manuscript integration from the completed K1--K5 outputs."""
from pathlib import Path
import csv,json,re,hashlib,shutil
import pandas as pd
O=Path(__file__).resolve().parents[1];P=O.parent;M=O/'manuscript'
REG=[]
def src(rel,lookup,field):
    d=pd.read_csv(O/rel);q=d
    for k,v in lookup.items():q=q[q[k]==v]
    assert len(q)==1,(rel,lookup,len(q))
    val=q.iloc[0][field]
    REG.append(dict(source=rel,lookup=json.dumps(lookup),field=field,value=str(val)))
    return val
def write(name,s): (M/name).write_text(s.strip()+'\n')
def replace(name,a,b):
    p=M/name;s=p.read_text();assert a in s,(name,a[:80]);p.write_text(s.replace(a,b))
def longtable(caption,label,columns,header,rows):
    return '\n'.join([r'\begingroup\small\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{'+columns+'}',r'\caption{'+caption+r'}\label{tab:'+label+r'}\\\toprule',header+r'\\\midrule\endfirsthead',r'\toprule '+header+r'\\\midrule\endhead',*rows,r'\bottomrule\end{longtable}\endgroup'])
def pct(x,n=2):return f'{100*float(x):.{n}f}'
def model(s):return {'KF-D42':'KinAfford dynamic (42)','BF-D42':'BEVFusion-Exca (42)','KF-D14':'KinAfford dynamic (14)','BF-D14':'BEVFusion-Exca (14)','R0':'Uniform reachable','R1':'Training prior','classical_ground':'Classical'}.get(s,s)
ARMS={'keep_all':'Keep all','gate':'Gate','always_mask':'Always mask','oracle_mask':'Fault-label mask','rgb_only':'RGB only','depth_only':'Depth only','uniform_reachable':'Expected','training_prior':'Fixed','classical_ground':'Fixed'}
def ci(r,scale=1,digits=4):return f"${scale*r['estimate']:+.{digits}f}$ [$ {scale*r['lower95']:+.{digits}f}, {scale*r['upper95']:+.{digits}f} $]"

def main():
    # K1 retains the previous table position and reporting convention.
    d=pd.read_csv(O/'K1/compact_sensitivity_table.csv');rows=[]
    def sig(s):return r'$-$' if s=='negative' else r'$+$' if s=='positive' else '0'
    prefs={'nominal':'Nominal','flat':'Flat','compressed':'Compressed','steeper':'Steeper','soil_first':'Soil-first'}
    for i,r in enumerate(d.to_dict('records')):
        vals=['Nominal' if i==0 else f'S{i:02d}',prefs[r['material_variant']],f"{r['productivity']:g}",f"{r['slope_low']:g}--{r['slope_high']:g}",f"{r['buffer_scale']:g}"]
        for a in ['KF-D42','BF-D42']:
            for metric in ['regret','unsafe']:
                s,c=r[a+'_'+metric].split('; ');vals.append(sig(s)+{'LOWER':'L','INCONCLUSIVE':'I','HIGHER':'H'}[c])
        for c in ['clean','cam_s3','rough_iid']:vals.append('('+', '.join(sig(r[f'BF42_minus_KF42_{c}_{x}']) for x in ['regret','unsafe'])+')')
        rows.append(' & '.join(vals)+r'\\')
    cap=(r'Relative-preference and buffer sensitivity of fixed replication choices. Preference vectors use sand, soil, gravel and rocky order: nominal $(1,0.80,0.65,0.35)$, flat $(1,1,1,1)$, compressed $(1,0.900,0.825,0.675)$, steeper $(1,0.640,0.4225,0.1225)$ and soil-first $(0.80,1,0.65,0.35)$. $p$ is the productivity clip (m), $s$ the slope band (degrees), and $b$ the truth-buffer multiplier. Primary cells give sign and adjusted 98.75\% marginal class: L lower, I inconclusive, H higher. Architecture cells give direct keep-all BEVFusion-Exca minus KinAfford dynamic signs (regret, unsafe). Both KinAfford gate primaries remain lower; its Severe camera gate effects and both models\' clean-regret advantages over the classical policy stay negative throughout. The four relative vectors replace a common multiplier that only rescales regret. All choices are fixed and multiplicities are reused; the locked primary findings are unchanged.')
    s=longtable(cap,'replicationsensitivity','llrrrlllllll',r'Setting & Preference & $p$ & $s$ & $b$ & \multicolumn{2}{c}{KinAfford dynamic} & \multicolumn{2}{c}{BEVFusion-Exca} & Clean & Severe camera & Rough terrain\\ & & & & & Regret & Unsafe & Regret & Unsafe & \multicolumn{3}{c}{Architecture signs}',rows).replace(r'\small',r'\scriptsize')
    write('tables/replication_sensitivity.tex',s)
    write('sections/sensitivity_replication.tex',r'''
\section{Replication utility and buffer sensitivity}
This retrospective analysis changes truth utility, its oracle and unsafe labels while holding every direct, soft, classical and fixed-reference choice fixed. Uniform selection retains analytical expected outcomes on its original support. The common material multiplier was replaced because it only rescales utility and regret. The factorial instead uses compressed and steeper relative preferences, and the material-only rows also include flat and soil-first preferences. All other settings and the interval construction remain unchanged, giving 27 settings including nominal. The original replication scene multiplicities provide ordinary 95\% and adjusted 98.75\% marginal percentile intervals with approximate coverage. These descriptive sensitivity classifications do not replace the locked primary family or imply policy reselection under a new objective.
\input{tables/replication_sensitivity.tex}
The two lower BEVFusion-Exca regret classifications use compressed preferences, the longer productivity clip and smaller buffers, with either slope band. Its unsafe endpoint remains inconclusive in every setting, so its joint criterion is never met. Its regret point estimate becomes positive in four settings. Direct clean architecture regret and unsafe signs change in 11 and nine settings, respectively; direct severe and rough signs persist. Soft-composition regret signs vary in four severe and nine rough comparisons. Full estimates, intervals and signs for both selectors accompany the table. Aggregate predicted hazard maps remain fixed rather than being decomposed into hypothetical object buffers.
''')
    p=M/'tables/decision_grounding.tex';s=p.read_text();s=s.replace('Common multiplier 0.75, 1, 1.25','Flat, compressed, steeper and soil-first vectors (Table~\\ref{S-tab:replicationsensitivity})',1);p.write_text(s)

    # K2: the empty complementary strata are explicitly undefined, not zeros.
    k2=pd.read_csv(O/'K2/stratified_safety.csv');rows=[]
    for (a,arm),g in k2.groupby(['architecture','arm'],sort=True):
        vals=[model(a),ARMS[arm]]
        for pool in ['clean','cam_s3','equal9']:
            q=g[g.pool==pool].iloc[0];vals.extend([pct(q.unsafe_rate),pct(q.person_rate)])
        rows.append(' & '.join(vals)+r'\\')
    cap=(r'Replication direct-policy and reference safety stratification. Every evaluated pose has at least one reachable safe terrain cell and strictly positive oracle utility: safe-available share is 100\% and zero-oracle share is 0\% in Clean, Severe camera and the equal-condition pool. Thus the percentages below are simultaneously the safe-available and positive-oracle stratum rates; the no-safe and zero-oracle rates are undefined (no attempts), not zero. All unsafe selections occur where a safe cell exists (100\% for every row). Unsafe and person-buffer rates use attempted decisions; pooled columns give each of nine conditions equal weight. Uniform rows are expected probabilities. All neural arms, including the earlier recipe, are retained. Counts and denominators are in the accompanying table data.')
    write('tables/always_act_strata.tex',longtable(cap,'alwaysact','llrrrrrr',r'Policy (epochs) & Arm & \multicolumn{2}{c}{Clean (\%)} & \multicolumn{2}{c}{Severe camera (\%)} & \multicolumn{2}{c}{Equal pool (\%)}\\ & & Unsafe & Person & Unsafe & Person & Unsafe & Person',rows).replace(r'\small',r'\scriptsize'))

    # K3 gives ordinary descriptive intervals, and labels cross-unit normalization.
    k3=pd.read_csv(O/'K3/prevalence_mixtures.csv');rows=[]
    for arch in ['KF-D42','BF-D42']:
        rows.append(r'\multicolumn{4}{l}{\textbf{'+model(arch)+r'}}\\')
        for p in [1,5,10,25,50]:
            for metric in ['regret','unsafe']:
                r=k3[(k3.architecture==arch)&(k3.severe_prevalence_percent==p)&(k3.metric==metric)].iloc[0]
                raw=ci(r,1 if metric=='regret' else 100,5 if metric=='regret' else 3)
                rel=f"${r.percent_of_clean_regret:+.2f}$ [$ {r.percent_clean_lower95:+.2f}, {r.percent_clean_upper95:+.2f} $]"
                rows.append(f"{p} & "+('Regret' if metric=='regret' else 'Unsafe (pp)')+' & '+raw+' & '+rel+r'\\')
    write('tables/prevalence.tex',longtable(r'Clean/Severe camera mixtures for replication direct selection. Each row is gate minus keep-all at the stated severe-fault prevalence; remaining weight is Clean. Ordinary 95\% descriptive intervals reuse the paired nominal-scene multiplicities. The last column is $100\Delta/\overline L_{\rm clean,keep}$ with a paired-bootstrap denominator. For regret this is a relative loss percentage; for unsafe probability it is a cross-unit normalization, not a percentage change in unsafe frequency. No prevalence is estimated from construction operations.','prevalence','rlll',r'Severe (\%) & Endpoint & Contrast [95\%] & / clean regret (\%) [95\%]',rows))

    # K4: candidate and oracle support, plus all model/arm/selector rows.
    k4=pd.read_csv(O/'K4/support_summary.csv');k4=k4[k4.seed_scope=='all_seeds'];rows=[]
    for cond in ['clean','cam_s3']:
        for arch,arm,name in [('input','keep_all','All cameras'),('KF-D42','gate','KinAfford dynamic gate'),('BF-D42','gate','BEVFusion-Exca gate'),('input','always_mask','All cameras masked')]:
            if cond=='cam_s3' and arm=='always_mask':continue
            g=k4[(k4.condition==cond)&(k4.architecture==arch)&(k4.arm==arm)&(k4.input_state=='actual')]
            vals=[('Clean' if cond=='clean' else 'Severe camera'),name]
            for scope in ['reachable_candidates','oracle_cell']:
                r=g[g.scope==scope].iloc[0];vals += [pct(r[m+'_share']) for m in ['camera','lidar','either']]
            rows.append(' & '.join(vals)+r'\\')
    write('tables/region_support.tex',longtable(r'Replication nominal-scene projected-return support (\%). Reachable candidate coverage pools supported cells over all candidate cells. Oracle support counts the first row-major truth-utility maximizer per pose. Projection uses measured depth/range, camera presence, reported joints and nominal calibration; cells are not filled or interpolated. All-camera masking leaves the same LiDAR support under Clean and Severe camera. Gate rows use the deployed camera subset. Return occupancy does not certify correct surface correspondence or visible material.','regionsupport','llrrrrrr',r'Condition & Input & \multicolumn{3}{c}{Reachable candidates} & \multicolumn{3}{c}{Oracle cell}\\ & & Camera & LiDAR & Either & Camera & LiDAR & Either',rows).replace(r'\small',r'\scriptsize'))
    rows=[]
    for policy in ['direct','soft']:
        rows.append(r'\multicolumn{6}{l}{\textbf{'+policy.capitalize()+r' selector}}\\')
        for cond in ['clean','cam_s3']:
            for arch in ['KF-D42','BF-D42']:
                for arm in ['keep_all','gate','always_mask']:
                    r=k4[(k4.condition==cond)&(k4.architecture==arch)&(k4.arm==arm)&(k4.policy==policy)&(k4.scope=='selected_cell')&(k4.input_state=='actual')].iloc[0]
                    rows.append(' & '.join([('Clean' if cond=='clean' else 'Severe camera'),model(arch),ARMS[arm],*[pct(r[m+'_share']) for m in ['camera','lidar','either']]])+r'\\')
    write('tables/selected_support.tex',longtable(r'Projected-return support at replication choices over all three seeds (\%). Columns use the camera subset actually supplied to the named arm. Removing all cameras while holding a keep-all or gate choice fixed sets its camera support to zero and its either-sensor support to the LiDAR column; these support-only counts and all per-seed rows are in the table data. Always-mask rows instead use that arm\'s own original choices. No inference or reselection is performed by this diagnostic.','selectedsupport','lllrrr',r'Condition & Model (epochs) & Arm & Camera & LiDAR & Either',rows).replace(r'\small',r'\scriptsize'))

    # K5 matrix retains every original validation pair.
    k5=pd.read_csv(O/'K5/validation_grid.csv');buffers=sorted(k5.buffer.unique());rows=[]
    for tau,g in k5.groupby('tau'):
        vals=[f'{tau:g}']
        for b in buffers:
            x=float(g[g.buffer==b].regret.iloc[0]);v=f'{x:.5f}';vals.append(r'\textbf{'+v+'}' if tau==1 and b==4 else v)
        rows.append(' & '.join(vals)+r'\\')
    write('tables/classical_validation.tex',longtable(r'Original validation regret for all 30 classical-policy pairs. Rows are height-above-ground thresholds $\tau$ (m); columns are dilation buffers $b$ (m). Bold is the previously frozen pair, not a new choice. Regret decreases at every tested buffer step for all five thresholds, including the last step to 4.0 m. The boundary therefore does not establish a validation optimum beyond the tested grid.','classicalvalidation','r'+'r'*len(buffers),r'$\tau\backslash b$ & '+' & '.join(f'{b:g}' for b in buffers),rows))

    # Insert compact numerical prose, with exact source lookups for every new value.
    getcov=lambda c,m:float(src('K4/support_summary.csv',dict(condition=c,scope='reachable_candidates',architecture='input',arm='keep_all',policy='truth',input_state='actual',seed_scope='all_seeds'),m+'_share'))
    ca,li,eu=[getcov('clean',x) for x in ['camera','lidar','either']];sa,se=[getcov('cam_s3',x) for x in ['camera','either']]
    write('generated/coverage.tex',f"On the replication's nominal scenes, measured camera-depth returns occupy {pct(ca)}\\% of reachable candidate cells, LiDAR returns occupy {pct(li)}\\%, and either sensor occupies {pct(eu)}\\%. Under severe camera faults, camera and combined support are {pct(sa)}\\% and {pct(se)}\\%; all-camera masking leaves {pct(li)}\\% LiDAR support. These fractions count projected returns without interpolation (Table~\\ref{{S-tab:regionsupport}}).")
    b=M/'sections/benchmark.tex';s=b.read_text();s=s.replace("The earlier clean support diagnostic covers approximately 19.67\\% of all terrain cells, not a reachable-only subset or this single frame.",f"Across clean replication nominal poses, measured camera-depth support covers {pct(ca)}\\% of reachable candidate cells; this is a population fraction, not a measurement for the illustrated frame.")
    b.write_text(s)
    diag=M/'sections/diagnostics.tex';s=diag.read_text()
    paragraph=(r'Every replication pose has at least one reachable safe terrain cell and positive oracle utility: the safe-available share is 100\% and the zero-oracle share is 0\% in clean inputs, severe camera faults and the equal-condition pool. All unsafe and person-buffer selections therefore occur despite safe alternatives. Rates for the no-safe and zero-oracle strata are undefined because neither stratum contains an attempt; Table~\ref{S-tab:alwaysact} gives the rates for the populated strata, including the classical and fixed-reference policies.')
    s=s.replace(r'\input{tables/severe_person_safety.tex}',r'\input{tables/severe_person_safety.tex}'+'\n\n'+paragraph)
    s=s.replace('Approximate measured-depth projection covered 19.67\\% of clean terrain cells.',f"In the replication, camera-depth support covers {pct(ca)}\\% of reachable candidate cells under clean inputs, falling to {pct(sa)}\\% under severe camera faults (Tables~\\ref{{S-tab:regionsupport}} and~\\ref{{S-tab:selectedsupport}}). This limited projected support complements the material-accuracy results, but does not by itself establish which material errors arise from occlusion.")
    s=s[:s.index('Across all 25 retrospective')]+r'''Across all 27 relative-preference and buffer settings, KinAfford dynamic retains both lower-loss primary classifications and its severe-camera benefit, and both models keep lower clean regret than the classical policy. BEVFusion-Exca's regret interval becomes lower in two settings and its point estimate changes sign in four, but its unsafe endpoint remains inconclusive throughout. The clean architecture regret and unsafe signs change in 11 and nine settings, while the direct severe and rough signs persist (Table~\ref{S-tab:replicationsensitivity}).
'''
    diag.write_text(s)
    for pool in ['clean','cam_s3','equal9']:
        for flag in ['safe_available','zero_oracle']:src('K2/population_support.csv',dict(pool=pool,flag=flag),'attempt_share')
    for field in ['safe_present_attempt_share','oracle_zero_attempt_share','unsafe_share_with_safe_available']:
        for a in ['KF-D42','BF-D42']:src('K2/stratified_safety.csv',dict(architecture=a,arm='gate',pool='equal9'),field)
    primary=pd.read_csv(O/'K1/sign_class_changes.csv');jsum={}
    for key,g in primary.groupby(['kind','identity','field']):jsum['|'.join(key)]=len(g)
    (O/'K1/display_change_counts.json').write_text(json.dumps(jsum,indent=2)+'\n')
    def prev(p,metric,field):return float(src('K3/prevalence_mixtures.csv',dict(architecture='KF-D42',severe_prevalence_percent=p,metric=metric),field))
    gtext=f"In a clean/severe-camera mixture, KinAfford's expected regret reduction grows from {-prev(1,'regret','percent_of_clean_regret'):.2f}\\% of clean regret [95\\%: ${-prev(1,'regret','percent_clean_upper95'):.2f},{-prev(1,'regret','percent_clean_lower95'):.2f}$] at 1\\% severe-fault prevalence to {-prev(50,'regret','percent_of_clean_regret'):.2f}\\% [${-prev(50,'regret','percent_clean_upper95'):.2f},{-prev(50,'regret','percent_clean_lower95'):.2f}$] at 50\\%, while BEVFusion-Exca's intervals span zero throughout (Table~\\ref{{S-tab:prevalence}})."
    p=M/'sections/gate_results.tex';p.write_text(p.read_text()+'\n'+gtext+'\n')
    disc=M/'sections/discussion.tex';s=disc.read_text()
    prevalence=f"For KinAfford, the expected unsafe reduction is {-100*prev(1,'unsafe','estimate'):.3f} percentage points [95\\%: ${-100*prev(1,'unsafe','upper95'):.3f},{-100*prev(1,'unsafe','lower95'):.3f}$] at 1\\% severe-fault prevalence and {-100*prev(50,'unsafe','estimate'):.3f} points [${-100*prev(50,'unsafe','upper95'):.3f},{-100*prev(50,'unsafe','lower95'):.3f}$] at 50\\%, so the operational value depends on the fault mixture."
    f4=pd.read_csv(P/'final_pre_submission_2026_10_26/F4/decision_change_counts.csv');r=f4[(f4.architecture=='KF-D42')&(f4.policy=='direct')&(f4.scope=='all_three_seeds')].iloc[0]
    improved=100*r.changed_share*r.lower_share_changed;worse=100*r.changed_share*r.higher_share_changed
    (O/'report/F4_all_attempt_percentages.json').write_text(json.dumps(dict(source='final_pre_submission_2026_10_26/F4/decision_change_counts.csv',formula='100 * changed_share * conditional lower/higher share',improved_percent=improved,worsened_percent=worse),indent=2)+'\n')
    s=s.replace('The real-image transfer rows make the deployed RGB threshold\'s original-image removal cost visible.',f"Under severe faults, fallback improves about {improved:.0f}\\% and worsens about {worse:.0f}\\% of all KinAfford decisions (Table~\\ref{{tab:decisionchanges}}). "+prevalence)
    disc.write_text(s)
    replace('sections/limitations.tex','The common material multiplier tests loss scale, not relative material preference; grade and reach were not varied.','The common material multiplier was replaced because it only rescales regret; the new relative-preference sensitivity still evaluates fixed choices rather than adaptation to a changed objective. Grade and reach were not varied.')
    replace('sections/limitations.tex','A zero-utility oracle can give zero regret to an unsafe selected point, which is why unsafe-event reporting remains separate; the fixed selector is a ranking task and provides no operational stop action.','Although a zero-utility oracle could give zero regret to an unsafe point, every replication pose has positive oracle utility and a reachable safe alternative. The fixed selector remains a ranking task without an operational stop action.')
    replace('sections/limitations.tex','The classical baseline has one fixed ground-estimation procedure and a bounded validation grid whose selected buffer is on the boundary.','The classical baseline has one fixed ground-estimation procedure and a bounded validation grid; validation regret was still decreasing at the largest tested buffer, 4.0 m (Table~\\ref{S-tab:classicalvalidation}).')
    replace('sections/design.tex','The selected values are 1.0 m and 4.0 m.','The selected values are 1.0 m and 4.0 m; regret was still falling at that buffer boundary for every tested height threshold (Table~\\ref{S-tab:classicalvalidation}).')

    # Author-selected B wording. These are manuscript drafts, not uploads or grants.
    access='Reviewers can access the code, configurations, split identities, checkpoints, rendered observations, evaluator and numerical outputs at [LINK]. Public release on acceptance will be provided at [REPOSITORY/DOI] under Apache-2.0 for code and CC BY 4.0 for data and checkpoints, with existing third-party notices retained. GOOSE-Ex images are not redistributed.'
    write('sections/using_excascene.tex',r'\paragraph{Using ExcaScene} '+access+' The evaluation entry point accepts an included checkpoint or a new model adapter on the same split, then applies the fixed direct and composed selectors. Installation and reproduction capabilities are distinguished in the supplement.')
    write('sections/artifact_locators.tex',r'\subsection{Versioned code and data access}\label{sec:artifactlocators}'+'\n'+access+'\nThe source package includes generator and figure code, camera-health monitors and baselines. The data package includes latent fields, checksums and complete study tables. Original and corrected numerical exports have distinct identifiers. Primary estimates, descriptive comparisons, unsafe-cause counts and undefined oracle-benefit ratios are separately identified.')
    p=M/'sections/limitations.tex';s=p.read_text();s=s[:s.index('The executable record reuses')]+('The tested evaluator reuses installed dependencies and native binaries; fresh compilation and reproduction of training remain untested. '+access+'\n');p.write_text(s)
    methods=(r'Claude Code (Anthropic) and Codex (OpenAI) implemented code to author-specified protocols, executed training and evaluation, wrote statistical-analysis and figure code, and assisted text drafting. The authors set the study questions, approved protocols and locks, and authorized deviations. [Authors: confirm the outputs reviewed and the responsibility undertaken.] Tool-version identifiers where documented, executable code, configurations and run logs accompany the reviewer package. Every figure and the graphical abstract was generated programmatically from data or diagram code, not by generative image tools.')
    declaration=(r'Claude Code (Anthropic) and Codex (OpenAI) assisted implementation, research execution, statistical and figure code, drafting and revision as described in Methods. Every figure and the graphical abstract was generated programmatically from data or diagram code, not by generative image tools. [Authors: confirm that the content was reviewed and edited, and state responsibility for the final article.]')
    p=M/'sections/design.tex';s=p.read_text();s=s[:s.index('The versioned record preserves')]+methods+'\n';p.write_text(s)
    write('sections/declarations.tex',r'''\section*{Declarations}
\textbf{Author contributions, funding and funder role, competing interests.}
[Authors: complete contributions, funding and funder role, and competing-interest declarations.]

\textbf{Data and code availability.}
'''+access+'\n\n'+r'\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}'+'\n'+declaration)
    (O/'AI_methods_and_declaration.md').write_text('# AI use: Methods and declaration drafts\n\n## Methods\n\n'+methods+'\n\n## Declaration\n\n'+declaration+'\n\nThe author-attributed earlier tool identifiers are preserved in sources/ai_tool_provenance.json; no unlogged model version is inferred.\n')
    # Main-text internal workflow vocabulary; figure captions use the same reader-facing replacements.
    subs={'saved choices':'original choices','saved selections':'original selections','saved scene multiplicities':'paired scene multiplicities','Saved multiplicities':'Paired multiplicities','saved sequence multiplicities':'paired sequence multiplicities','saved regret export':'per-frame regret values','saved ':'','recorded recipes':'specified recipes','recorded real images':'real images','recorded selected':'selected','recorded ':'','numerical record':'numerical outputs','versioned record':'versioned implementation','Analysis and computational record':'Analysis and computation','the supplement records':'the supplement gives','parameters and latent fields are saved':'parameters and latent fields are specified','clean record':'clean observation','one saved scene':'one scene','One saved scene':'One scene','unavailable':'not provided','author-completed':'author-specified'}
    main_sections=['introduction','related_work','benchmark','design','maps_results','removal_results','gate_results','diagnostics','image_check','discussion','limitations']
    names=['Clean','Independent faults','Mild camera','Moderate camera','Severe camera','One-camera severe','One-camera novel','Moderate all-sensor','Rough terrain']
    for name in main_sections:
        p=M/f'sections/{name}.tex';s=p.read_text()
        for a,b in subs.items():s=s.replace(a,b)
        lines=[]
        for line in s.splitlines():
            if not line.startswith('\\fig{'):
                for n in names:line=line.replace(n,n.lower())
            lines.append(line)
        p.write_text('\n'.join(lines)+'\n')
    for p in (M/'tables').glob('*.tex'):
        s=p.read_text().replace('saved regret export','per-frame regret values').replace('saved cutoff curve','cutoff curve').replace('saved-scene','scene').replace('saved ','').replace("\\'", "'")
        p.write_text(s)
    # Keep the requested supplement section numbering; new tables are within §9.
    p=M/'sections/supp_run_details.tex';p.write_text(p.read_text()+r'''
\subsection{Additional descriptive analyses}
The relative-preference analysis in Table~\ref{tab:replicationsensitivity}, the safety strata, the prevalence mixtures and the projected-support counts are descriptive additions on the inspected replication population. No model, choice, threshold or bootstrap draw is added. All original primary results remain unchanged.
\input{tables/always_act_strata.tex}
\input{tables/prevalence.tex}
\input{tables/region_support.tex}
\input{tables/selected_support.tex}
\input{tables/classical_validation.tex}
''')
    p=M/'refs.bib';s=p.read_text();s=s.replace(' url={https://wheelercat.com/wp-content/uploads/2025/03/Cat-Performance-Handbook-Edition-50-June-2022.pdf},\n','');s=re.sub(r'(title=\{Caterpillar Performance Handbook\},\n)',r'\1 publisher={{Caterpillar}},\n',s);s=s.replace('published document hosted by its dealer; accessed 26 October 2026','Caterpillar publication');p.write_text(s)
    letter=(P/'final_pre_submission_2026_10_26/variants/B/cover_letter_draft.md').read_text()
    old=letter[letter.index('The conditional reviewer-access'):letter.index('\n\n[Authors:')]
    letter=letter.replace(old,access).replace('confirm the selected access arrangement; ','')
    (O/'cover_letter_draft.md').write_text(letter)
    tex=(M/'cover_letter.tex').read_text();a=tex.index('The conditional reviewer-access');b=tex.index('\n\n',a);tex=tex[:a]+access+tex[b:];tex=tex.replace('confirm the selected access arrangement; ','');(M/'cover_letter.tex').write_text(tex)
    with (O/'report/new_number_sources.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(REG[0]));w.writeheader();w.writerows(REG)
    (O/'sources/ai_tool_provenance.json').write_text(json.dumps(dict(source='revision_v5/readiness_v5_1/author_response_form.md',line=42,source_sha256=hashlib.sha256((P/'revision_v5/readiness_v5_1/author_response_form.md').read_bytes()).hexdigest(),author_reported_identifiers={'Claude Code':['Opus 5','Fable 5'],'Codex':['GPT-5.6-Sol','GPT-5.6-Terra']},qualification='Author-attributed earlier tool identifiers, not independently inferred identifiers for every execution in every phase; no unsupported model version is added.'),indent=2)+'\n')
    print('Presentation integrated; source lookups',len(REG))
if __name__=='__main__':main()
