"""Source-derived parameter specification. Numeric content is extracted, never keyed in."""
from pathlib import Path
import ast,csv,json,hashlib,re,sys
import numpy as np
OUT=Path(__file__).resolve().parents[1];P=OUT.parent;S=P/'external_validation_run_EV1_AFAST/source/excascene_snapshot'
sys.path.insert(0,str(S))
from kinafford.sim import kinematics as K,scene as SC,degrade as DG
ROWS=[];CACHE={}
def source(rel):
    p=P/rel
    if rel not in CACHE:CACHE[rel]=(p.read_text(),ast.parse(p.read_text()),hashlib.sha256(p.read_bytes()).hexdigest())
    return CACHE[rel]
def node(rel,name):
    text,tree,_=source(rel);parts=name.split('.');found=tree
    for part in parts:
        found=next(n for n in found.body if getattr(n,'name',None)==part or (isinstance(n,(ast.Assign,ast.AnnAssign)) and any(isinstance(c,ast.Name) and c.id==part for t in (n.targets if isinstance(n,ast.Assign) else [n.target]) for c in ast.walk(t))))
    return found
def value(rel,name):
    n=node(rel,name);expr=n.value
    return eval(compile(ast.Expression(expr),str(P/rel),'eval'),{'np':np,'dict':dict,'range':range,'list':list,'IMG_W':K.IMG_W,'IMG_H':K.IMG_H,'EXCAVATOR':K.EXCAVATOR,'CAMERAS':K.CAMERAS})
def add(section,label,rel,n,display=None,method='verbatim source expression'):
    text,_,digest=source(rel);excerpt=ast.get_source_segment(text,n)
    expr=ast.unparse(n) if display is None else display
    row=dict(id=f'SP{len(ROWS)+1:04d}',section=section,parameter=label,value=str(expr),source_file=rel,line_start=n.lineno,line_end=n.end_lineno,source_sha256=digest,source_excerpt=excerpt,extraction=method)
    ROWS.append(row);return row
def esc(s):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in str(s))
def code(s):return r'{\ttfamily\footnotesize '+esc(s).replace(', ',r',\allowbreak ').replace(' = ',r' =\allowbreak ').replace(' * ',r' *\allowbreak ').replace(' + ',r' +\allowbreak ')+'}'
def table(caption,label,rows):
    out=[r'\begingroup\small',r'\begin{longtable}{p{.27\linewidth}p{.66\linewidth}}',r'\caption{'+caption+r'}\label{tab:'+label+r'}\\\toprule Parameter & Extracted specification\\\midrule\endfirsthead',r'\toprule Parameter & Table \thetable\ continued\\\midrule\endhead']
    for r in rows:out.append(esc(r['parameter'].replace('.__init__','')).replace('.',r'.\allowbreak ').replace(r'\_',r'\_\allowbreak{}')+' & '+code(r['value'])+r'\\')
    out += [r'\bottomrule\end{longtable}\endgroup'];return '\n'.join(out)
def sampling(rel,func,section):
    root=node(rel,func);parent={c:n for n in ast.walk(root) for c in ast.iter_child_nodes(n)};out=[]
    for n in ast.walk(root):
        if not isinstance(n,ast.Call) or not isinstance(n.func,ast.Attribute) or not isinstance(n.func.value,ast.Name) or n.func.value.id!='rng' or n.func.attr not in ['uniform','normal','integers','choice']:continue
        a=n
        while a in parent and not isinstance(a,(ast.Assign,ast.AnnAssign,ast.For,ast.If,ast.Return,ast.AugAssign)):a=parent[a]
        label=func+': '+(ast.unparse(a.targets[0]) if isinstance(a,ast.Assign) else ast.unparse(a.target) if isinstance(a,(ast.For,ast.AugAssign,ast.AnnAssign)) else a.__class__.__name__)
        tests=[];b=a
        while b in parent:
            child=b;b=parent[b]
            if isinstance(b,ast.If):tests.append(('not ('+ast.unparse(b.test)+')') if child in b.orelse else ast.unparse(b.test))
        label+=' ('+'; '.join(reversed(tests))+')' if tests else ''
        # Source comments identify the sampled object; retain them as readable context.
        if func=='generate_scene':
            lines=source(rel)[0].splitlines()
            comments=[l.strip()[2:] for l in lines[root.lineno:n.lineno] if l.strip().startswith('# ')]
            if comments:label=comments[-1].split(':')[0]+': '+label.split(': ',1)[-1]
        display_node=n
        if isinstance(a,ast.If):
            label=label.replace(': If',': condition draw');display_node=a.test
        elif isinstance(a,ast.Assign) and any(isinstance(c,ast.IfExp) for c in ast.walk(a.value)):
            display_node=a.value
        out.append(add(section,label,rel,display_node,ast.unparse(display_node)))
    return sorted(out,key=lambda r:r['line_start'])
def statements(rel,func,section,include=None):
    root=node(rel,func);out=[]
    for n in ast.walk(root):
        if not isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.If,ast.For,ast.Return)):continue
        expr=n.test if isinstance(n,ast.If) else n.iter if isinstance(n,ast.For) else n.value
        if not any(isinstance(c,ast.Constant) and isinstance(c.value,(int,float)) for c in ast.walk(expr)):continue
        target=ast.unparse(n.targets[0]) if isinstance(n,ast.Assign) else ast.unparse(n.target) if isinstance(n,(ast.AnnAssign,ast.AugAssign)) else n.__class__.__name__
        if include and not include(n,target):continue
        display=ast.unparse(expr) if isinstance(n,(ast.If,ast.For)) else ast.unparse(n)
        if isinstance(n,ast.If):target='condition: '+ast.unparse(n.test)
        out.append(add(section,func+': '+target,rel,n,display))
    return sorted(out,key=lambda r:r['line_start'])
def jsonrow(section,label,rel,key,display=None):
    p=P/rel;text=p.read_text();v=json.loads(text)
    for k in key:v=v[k]
    decoder=json.JSONDecoder()
    def ws(i):
        while text[i].isspace():i+=1
        return i
    def locate(i,path):
        i=ws(i)
        if not path:return i,decoder.raw_decode(text,i)[1]
        if text[i]=='{':
            i=ws(i+1)
            while text[i]!='}':
                name,j=decoder.raw_decode(text,i);j=ws(j);assert text[j]==':'
                start=ws(j+1);_,end=decoder.raw_decode(text,start)
                if name==path[0]:return locate(start,path[1:])
                i=ws(end)
                if text[i]==',':i=ws(i+1)
        raise KeyError(path)
    start,end=locate(0,key);lineno=text.count('\n',0,start)+1;lastline=text.count('\n',0,end)+1
    lines=text.splitlines()
    r=dict(id=f'SP{len(ROWS)+1:04d}',section=section,parameter=label,value=json.dumps(v) if display is None else str(display(v)),source_file=rel,line_start=lineno,line_end=lastline,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_excerpt='\n'.join(lines[lineno-1:lastline]),extraction='JSON key '+'.'.join(map(str,key)))
    ROWS.append(r);return r
def main():
    sr=str((S/'kinafford/sim/scene.py').relative_to(P));kr=str((S/'kinafford/sim/kinematics.py').relative_to(P));dr=str((S/'kinafford/sim/degrade.py').relative_to(P));rr=str((S/'kinafford/sim/render.py').relative_to(P))
    sections=[];rows=[]
    for name in ['TERRAIN_EXTENT','TERRAIN_N','BEV_RES','BEV_N','BEV_X0','MAT_CLASSES','N_BEV_SEM','MAT_DIGGABILITY','MAT_ALBEDO','MAT_TEXTURE_AMP','PERSON_EXCL','REACH_MIN']:
        rows.append(add('geometry',name,sr,node(sr,name)))
    # The Gaussian terrain bands and trench edge equations are not sampler calls.
    root=node(sr,'generate_scene')
    for n in ast.walk(root):
        if isinstance(n,ast.For) and ast.unparse(n.target)=='(wl_m, amp)':rows.append(add('geometry','Terrain wavelength/amplitude bands',sr,n.iter))
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prof' for t in n.targets):rows.append(add('geometry','Trench edge profile',sr,n))
    base=next(n for n in root.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='base_level')
    ret=next(n for n in root.body if isinstance(n,ast.Return));grade=next(k.value for k in ret.value.keywords if k.arg=='z_target')
    rows += [add('geometry','Base terrain level before piles and trenches',sr,base),add('geometry','Scene target grade',sr,grade)]
    sections.append(table("Scene geometry and material constants. Coordinates are metres in the base frame unless noted; the bird's-eye grid is in the upper-structure frame.",'spec_geometry',rows))
    rows=sampling(sr,'generate_scene','scene sampling')
    sections.append("Continuous draws use uniform or normal distributions as written; integer upper bounds are exclusive. Pile, pit, trench, material-patch, rock, truck and object parameters below follow the source order. The normal draw arguments are mean and standard deviation. A trench is optional; persons and obstacles use bounded rejection around the truck.\n"+table('Scene sampling distributions. Object labels follow the source comments; the complete source excerpt and branch context are in the sidecar.','spec_sampling',rows))
    # Explicit branch probabilities, deterministic object geometry, and seed mapping.
    extra=statements(sr,'generate_scene','scene branches',lambda n,t:isinstance(n,ast.If) or t in ['rng','truck','rock','z_target'])
    placement=next(n for n in root.body if isinstance(n,ast.For) and ast.unparse(n.target)=='(lst, n, rmin)')
    extra.append(add('scene branches','Person and obstacle counts and minimum radius',sr,placement.iter))
    rows=extra+[add('joints',n,kr,node(kr,n)) for n in ['EXCAVATOR','PHASES']]
    rows+=sampling(kr,'sample_joint_config','joints')+sampling(sr,'sample_frame','joints')
    rows+=statements(kr,'sample_joint_config','joint bounds',lambda n,t:t in ['lo2','hi2','lo','hi','q'])
    sections.append(table('Object-presence branches, machine dimensions and joint sampling. Joint order is swing, boom, arm and bucket; angles are radians except explicit degree conversions. Arm and bucket angles are relative.','spec_joints',rows))
    sensors=[]
    for name,spec in K.SENSORS.items():
        mount=K.mount_transform(spec['pos'],spec['yaw'],spec['pitch'],spec['kind'])
        n=node(kr,'SENSORS');val={k:v for k,v in spec.items()};val['mount_parent_from_sensor']=np.round(mount,9).tolist()
        if spec['kind']=='camera':val['intrinsics']=np.round(K.camera_intrinsics(spec),9).tolist()
        sensors.append(add('sensors',name,kr,n,json.dumps(val),method='Evaluate frozen constant dictionary and pure mount_transform/camera_intrinsics equations; no sensor rendering'))
    sensors += [add('sensors','Nominal depth formation and range',rr,node(rr,'Renderer.render_camera')),add('sensors','Nominal LiDAR range/validity',rr,node(rr,'Renderer.render_lidar'))]
    # Compact the two renderer rows to parameter-bearing statements; retain full snippets in CSV.
    sensors[-2]['value']='; '.join(ast.unparse(n) for n in node(rr,'Renderer.render_camera').body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in ['hit','depth'])
    sensors[-1]['value']='; '.join(ast.unparse(n) for n in node(rr,'Renderer.render_lidar').body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) in ['hit','rng_img','valid'])
    sections.append(table('Sensor specification. Mount matrices map the sensor into its named parent link. Camera axes are right, down, forward; LiDAR axes are forward, left, up. Camera depth is optical-axis depth, not ray range. Nominal depth and LiDAR range have no added measurement noise; fault-dependent noise is specified separately.','spec_sensors',sensors))
    sev=[]
    for key,v in DG.SEV.items():sev.append(add('fault severities',key,dr,node(dr,'SEV'),json.dumps(v),method='Extract severity dictionary entry '+key))
    sections.append(table('Severity parameters in mild, moderate and severe order. Gaussian blur uses a standard deviation, not a hard-coded kernel width. Ring stride retains each indicated ring before dropout and spurious-return insertion.','spec_severity',sev))
    detail=[]
    for fn in ['_ellipse_mask','degrade_camera','degrade_lidar','drift_transform','joint_bias','sample_spec']:
        detail+=statements(dr,fn,'fault mechanisms',lambda n,t: (isinstance(n,(ast.If,ast.For)) or (t not in ['H, W','H','W','img','spec','photo','rel'] and ('rng.' in ast.unparse(n) or 'SEV[' in ast.unparse(n) or t in ['target','mask','depth','b','m','haze','blob']))))
    nr='final_evidence_2026_10_20/code/faults.py'
    detail+=statements(nr,'novel_latents','withheld faults')+statements(nr,'apply_novel','withheld faults',lambda n,t:t not in ['h, w','yy, xx','img'] or 'latent' in ast.unparse(n))
    sections.append(table('Fault implementation details and independent-mixture probabilities. Occlusion adds ellipses until the target union fraction is reached; the count is therefore not fixed. The fixed camera suites compose all listed camera mechanisms, whereas independent faults sample their subset.','spec_fault_detail',detail))
    # Complete numeric source catalogue also covers all branches excluded from compact tables.
    for rel,fn in [(sr,'value_noise'),(sr,'generate_scene'),(sr,'sample_frame'),(sr,'_pose_ok'),(sr,'bev_ground_truth'),(kr,'sample_joint_config'),(rr,'Renderer.shade'),(dr,'sample_spec'),(nr,'novel_latents'),(nr,'apply_novel')]:statements(rel,fn,'complete numeric source catalogue')
    br=str((S/'kinafford/models/blocks.py').relative_to(P));vr=str((S/'kinafford/models/variants.py').relative_to(P));ar='external_validation_run_EV1_AFAST/code/models.py';lr=str((S/'kinafford/losses.py').relative_to(P))
    models=[add('models','Task model configuration',ar,node(ar,'CFG'))]
    for rel,fn in [(br,'ImageEncoder.__init__'),(br,'LidarPillars.__init__'),(vr,'GeometricFusionNet.__init__'),(br,'BEVUNet.__init__'),(br,'Heads.__init__')]:
        for n in node(rel,fn).body:
            if isinstance(n,ast.Assign):models.append(add('KinAfford dynamic architecture',fn+': '+ast.unparse(n.targets[0]),rel,n))
            if fn=='Heads.__init__' and isinstance(n,ast.If):
                for child in n.body:
                    if isinstance(child,ast.Assign):models.append(add('KinAfford dynamic architecture',fn+': '+ast.unparse(child.targets[0]),rel,child))
    for name in ['STRIDE','FEAT_H','N_IMG_SEM']:models.append(add('KinAfford dynamic architecture',name,br,node(br,name)))
    models.append(add('KinAfford dynamic architecture','Convolution block',br,node(br,'conv_block')))
    models+=statements(br,'CameraSplat._forward','KinAfford dynamic architecture',lambda n,t:t in ['u','v','valid'])
    sections.append(table('KinAfford dynamic architecture, extracted constructor and head definitions. Replication encoder weights are loaded separately as specified in the recipe crosswalk; constructor defaults are not loading receipts. The shared backbone retains ResNet stages through its third residual stage. Lifting uses measured depth and reported joints; sensor maps include count and height statistics. Joints condition bottleneck scale and shift.','spec_kinafford',models))
    models=statements(ar,'BEVFusionExca.__init__','BEVFusion-Exca architecture',lambda n,t:t.startswith('self.') and not t.startswith('self.register'))
    sections.append('BEVFusion-Exca retains the upstream Swin image encoder, learned depth lifting, sparse encoder, convolutional fuser and SECOND decoder. Task adaptations are the RGB-D input, excavator coordinates and bounds, presence filtering before lifting/pooling, explicit logical batch cardinality for empty LiDAR samples, axis-order correction, joint conditioning, image auxiliary head and shared dense task heads. No artificial points are added to empty batches.\n'+table('BEVFusion-Exca task configuration. The full adapter source, upstream notice and patch are identified in the release inventory.','spec_bevfusion',models))
    losses=[add('losses','Head coefficients',lr,node(lr,'DEFAULT_WEIGHTS'))]
    losses+=statements(lr,'gaussian_nll','loss equations')
    # Keep every loss and its intermediate definition, including assignments
    # with no numeric literal (hazard BCE, image auxiliary CE and total).
    for n in ast.walk(node(lr,'compute_losses')):
        if isinstance(n,ast.Assign):
            label=ast.unparse(n.targets[0])
            if label!='losses':losses.append(add('loss equations','Original loss: '+label,lr,n))
        elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='seg_losses.append':
            losses.append(add('loss equations','Image auxiliary camera-presence weighting',lr,n))
    mr='final_evidence_2026_10_20/code/models_final.py'
    for n in ast.walk(node(mr,'compute_losses')):
        if isinstance(n,ast.Assign):losses.append(add('balanced losses','Replication loss: '+ast.unparse(n.targets[0]),mr,n))
    for k in ['semantics','material']:losses.append(jsonrow('balanced losses',k+' class weights','final_evidence_2026_10_20/protocol/training_class_weights.json',[k,'inverse_sqrt_mean_one_weights']))
    dp=str((S/'stage4/dataset.py').relative_to(P))
    sections.append(table('Training losses and structured dropout. The original visibility weights and task supports are retained; the strengthened recipe changes only the semantic and material cross-entropy class weights in addition to its stated horizon and initialization. The internal reliability-gate loss exists in the library but is inactive for these dynamic models; the separate camera-health classifier has its own frozen fit.','spec_losses',losses))
    mask=node(dp,'structured_keep_mask');branch=next(n for n in mask.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='branch')
    bounds=sorted(n.comparators[0].value for n in ast.walk(branch) if isinstance(n,ast.Compare));prob=np.diff([0,*bounds,1])
    names=ast.literal_eval(node(dp,'BRANCHES').value);drops=[]
    for name,p in zip(names,prob):
        action={'lidar_only':'keep LiDAR; remove all cameras','drop_lidar':'remove LiDAR; keep available cameras','drop_one_camera':'choose a camera uniformly; remove it if available','nominal':'keep all originally available sensors'}[name]
        drops.append(add('structured dropout',name,dp,branch,f'{100*p:g}%: {action}',method='Probability differences from source branch thresholds; action from source mask branches'))
    drops.append(add('structured dropout','No available cameras takes precedence',dp,next(n for n in mask.body if isinstance(n,ast.If) and 'available[:3].any' in ast.unparse(n.test))))
    sections.append(table('Structured-dropout probabilities and actions. Masks begin with the originally available sensors. A proposed branch is never redrawn; the no-camera case preserves LiDAR.','spec_dropout',drops))
    tr='final_evidence_2026_10_20/code/select_gate_thresholds.py';grid=next(n for n in ast.walk(node(tr,'select')) if isinstance(n,ast.Call) and ast.unparse(n.func)=='itertools.product')
    thresholds=[add('gate thresholds','Probability and invalid-depth threshold grid',tr,grid),add('gate thresholds','Validation selection and tie order',tr,next(n for n in ast.walk(node(tr,'select')) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='best' for t in n.targets)))]
    sections.append(table('Camera-health threshold search. The Cartesian grid is shared by the two probability/fraction thresholds; a threshold greater than unity disables its branch. Selection minimizes equal-condition validation regret, with larger thresholds preferred on exact ties.','spec_gategrid',thresholds))
    splits=[]
    for k in json.loads((P/'data/generation_config.json').read_text())['spec']:splits.append(jsonrow('splits','Original '+k,'data/generation_config.json',['spec',k]))
    for label,rel in [('Replication','final_evidence_2026_10_20/protocol/holdout_manifest.json'),('First prospective gate','strengthening_2026_10_18/protocol/holdout_manifest.json'),('External architecture','external_validation_run_EV1_AFAST/manifests/generation_manifest.json'),('Reference study','release_candidate/versions/v5_reference_study/manifests/generation_manifest.json')]:
        if not (P/rel).exists():continue
        obj=json.loads((P/rel).read_text())
        for k in ['root_seed','bootstrap_seed','nominal_scene_ids','rough_scene_ids']:
            if k in obj:splits.append(jsonrow('splits',label+' '+k,rel,[k],lambda v:json.dumps(dict(first=v[0],last=v[-1],count=len(v))) if isinstance(v,list) else v))
    # Original deterministic scene and slot seed equations.
    for fn in ['generate_scene','sample_frame']:
        n=next(n for n in node(sr,fn).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rng' for t in n.targets));splits.append(add('splits',fn+' random seed',sr,n))
    for rel,fn in [('final_evidence_2026_10_20/code/generate_study4.py','phase_rng'),('external_validation_run_EV1_AFAST/code/data.py','rng')]:splits.append(add('splits',fn,rel,node(rel,fn)))
    sections.append(table('Split identities and seed definitions. Original scene ranges are half-open. Later planned ID spans are inclusive; the documented first prospective whole-scene exclusion remains in the study crosswalk and deviation record.','spec_splits',splits))
    receipt=json.loads((P/'submission_strengthening/material/complete_v1/diagnostic_receipt.json').read_text());cov=receipt['coverage'];percent=100*cov['projected_depth_covered']/cov['terrain_cells']
    coverage={'saved_camera_coverage_percent':percent,'denominator':'all truth terrain cells in original external-study clean records, not restricted to reachable terrain','saved_reachable_camera_coverage':'NOT_AVAILABLE_IN_SAVED_SUPPORT_EXPORT','saved_separate_lidar_coverage':'NOT_AVAILABLE_IN_SAVED_SUPPORT_EXPORT','source':'submission_strengthening/material/complete_v1/diagnostic_receipt.json','source_sha256':hashlib.sha256((P/'submission_strengthening/material/complete_v1/diagnostic_receipt.json').read_bytes()).hexdigest(),'reanalysis_performed':False}
    (OUT/'specification/coverage_scope.json').write_text(json.dumps(coverage,indent=2)+'\n')
    (OUT/'manuscript/generated/coverage.tex').write_text(f'The saved clean-image support diagnostic covers approximately {percent:.2f}\\% of all terrain cells in the external-study population, using projected measured-depth hits; this is not a reach-restricted or exact material-observation fraction. Separate saved reachable-terrain and LiDAR coverage fractions are unavailable.\n')
    out='\\section{Self-contained implementation specification}\\label{sec:specification}\nThe following values are extracted programmatically from frozen source and configuration files. The sidecar \\code{specification/source_parameters.csv} identifies the source file, line interval, excerpt and digest of every entry. Expressions use the original variable names; all distances are metres unless otherwise stated.\\par\n'+ '\n\n'.join(sections)
    (OUT/'manuscript/sections/specification.tex').write_text(out)
    with (OUT/'specification/source_parameters.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(ROWS)
    (OUT/'specification/source_parameters.json').write_text(json.dumps(ROWS,indent=2)+'\n')
    (OUT/'specification/extraction_receipt.json').write_text(json.dumps(dict(status='EXTRACTED_FROM_FROZEN_SOURCES',rows=len(ROWS),source_files=len({r['source_file'] for r in ROWS}),numeric_values_hand_transcribed=False,new_neural_forwards=0,gpu_hours=0,coverage=coverage),indent=2)+'\n')
    print('Source entries',len(ROWS),'coverage',percent)
if __name__=='__main__':main()
