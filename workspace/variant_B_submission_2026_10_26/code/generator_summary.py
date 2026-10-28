"""Plain-language parameter summary derived from frozen AST values, without replay."""
from pathlib import Path
import ast,json,hashlib,sys,csv
import numpy as np
O=Path(__file__).resolve().parents[1];P=O.parent
S=P/'external_validation_run_EV1_AFAST/source/excascene_snapshot';sys.path.insert(0,str(S))
from kinafford.sim.kinematics import EXCAVATOR,PHASES
from kinafford.sim.scene import TERRAIN_EXTENT,TERRAIN_N
path=S/'kinafford/sim/scene.py';text=path.read_text();tree=ast.parse(text);root=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='generate_scene')
ROWS=[]
def obtain(anchor,index=0):
    line=next(i for i,l in enumerate(text.splitlines(),1) if anchor in l)
    calls=sorted([n for n in ast.walk(root) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='rng' and n.lineno==line],key=lambda n:n.col_offset)
    n=calls[index];args=[eval(compile(ast.Expression(a),'<constant>','eval'),{'np':np,'E':TERRAIN_EXTENT}) for a in n.args]
    ROWS.append(dict(anchor=anchor,index=index,source_file=str(path.relative_to(P)),line=line,expression=ast.get_source_segment(text,n),values=json.dumps(args),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return args
def ranges(v,unit='m'):return (f'{v[0]:g}--{v[1]:g} {unit}').rstrip()
def discrete(v):return f'{int(v[0])}--{int(v[1])-1}'
def vals(a):return ', '.join(f'{x:g}' for x in a)
def main():
    count=obtain('n_piles =');r=obtain('r = rng.uniform(4.0, 9.0)');amp=obtain('hp, sg = rng.uniform(1.0, 3.5)');width=obtain('hp, sg = rng.uniform(1.0, 3.5)',1)
    ramp=obtain('hp, sg = rng.uniform(3.0, 5.0)');rwidth=obtain('hp, sg = rng.uniform(3.0, 5.0)',1);scale=obtain('sgx, sgy =');angle=np.rad2deg(obtain('a = theta_work + rng.uniform(-np.deg2rad(60)'))
    pitn=obtain('for _ in range(rng.integers(0, 3))');pitr=obtain('r = rng.uniform(3.0, 11.0)');pita=obtain('h -= rng.uniform',0);pitw=obtain('h -= rng.uniform',1)
    # Conditional probability is extracted as a literal from the branch test.
    tr=next(n for n in root.body if isinstance(n,ast.If) and ast.unparse(n.test)=='rng.uniform() < 0.5')
    trenchp=tr.test.comparators[0].value
    trr=obtain('r = rng.uniform(4.0, 8.5)');trl=obtain('L, W, D =',0);trw=obtain('L, W, D =',1);trd=obtain('L, W, D =',2);tra=np.rad2deg(obtain('a = theta_work + rng.uniform(-np.deg2rad(50)'))
    matn=obtain('n_seed =');priors=[]
    for n in ast.walk(root):
        if isinstance(n,ast.Call) and ast.unparse(n.func)=='rng.choice':
            ps=[x.value for x in n.keywords if x.arg=='p']
            if ps:priors.append(ast.literal_eval(ps[0]))
    rockn=obtain('n_rock =',1);rocknr=obtain('n_rock =',0);rockr=obtain('r = rng.uniform(2.5, 13.0)');rockh=obtain('hr, sr =',0);rockw=obtain('hr, sr =',1)
    truck=next(n for n in root.body if isinstance(n,ast.If) and ast.unparse(n.test)=='rng.uniform() < 0.9');truckp=truck.test.comparators[0].value;truckr=obtain('r = rng.uniform(7.5, 9.5)')
    work_direction=obtain('theta_work = rng.uniform');truck_direction=np.rad2deg(obtain('theta_truck = theta_work + side *'))
    truck_yaw=obtain('yaw = theta_truck + np.pi / 2')
    truck_call=next(n for n in ast.walk(truck) if isinstance(n,ast.Call) and ast.unparse(n.func)=='dict' and any(k.arg=='center' for k in n.keywords))
    truck_half=ast.literal_eval(next(k.value for k in truck_call.keywords if k.arg=='half'))
    cab=next(k.value for k in truck_call.keywords if k.arg=='cab')
    cab_values={k.arg:ast.literal_eval(k.value) for k in cab.keywords}
    ROWS.append(dict(anchor='truck box and cab dimensions',index=0,source_file=str(path.relative_to(P)),line=truck_call.lineno,expression=ast.get_source_segment(text,truck_call),values=json.dumps(dict(body_half=truck_half,**cab_values)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    placed=next(n for n in root.body if isinstance(n,ast.For) and ast.unparse(n.target)=='(lst, n, rmin)');prmin=placed.iter.elts[0].elts[2].value;ormin=placed.iter.elts[1].elts[2].value
    on=obtain('for lst, n, rmin in',0);radial=next(n for n in ast.walk(placed) if isinstance(n,ast.Call) and ast.unparse(n.func)=='rng.uniform' and ast.unparse(n.args[0])=='rmin').args[1].value
    oa=np.rad2deg(obtain('a = theta_work + rng.uniform(-np.deg2rad(150)'));light=np.rad2deg(obtain('el = rng.uniform'));tint=obtain('tint = rng.uniform')
    fmt=lambda a,b:f'$[{a:g},{b:g}]$'
    phase_rows=[]
    for name,p,rs in PHASES:
        rs=rs if rs is not None else list(zip(EXCAVATOR['q_min'][1:],EXCAVATOR['q_max'][1:]))
        phase_rows.append(f'{name.capitalize()} & {100*p:g}\\% & '+' & '.join(fmt(a,b) for a,b in rs)+r'\\')
    rough=obtain('rough =');roughscale=next(n for n in ast.walk(root) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rough' for t in n.targets)).value.right.body.value
    kt=S/'kinafford/sim/kinematics.py';ktree=ast.parse(kt.read_text());joint=next(n for n in ktree.body if isinstance(n,ast.FunctionDef) and n.name=='sample_joint_config')
    jitter=next(n for n in ast.walk(joint) if isinstance(n,ast.Call) and ast.unparse(n.func)=='rng.normal').args[1].value
    pose=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample_frame')
    branch=next(n for n in ast.walk(pose) if isinstance(n,ast.If) and ast.unparse(n.test)=='rng.uniform() < 0.7')
    workp=branch.test.comparators[0].value
    spreads=[eval(compile(ast.Expression(n.args[1]),'<angle>','eval'),{'np':np}) for n in ast.walk(branch) if isinstance(n,ast.Call) and ast.unparse(n.func)=='rng.normal']
    s=rf'''
\subsection*{{Generator in plain language}}
\begingroup\small
Continuous ranges below are uniform draws; count ranges are discrete uniform, including both endpoints. Distances are in metres. A {2*TERRAIN_EXTENT:g} m square terrain is sampled on a {TERRAIN_N}$\times${TERRAIN_N} height grid. Smooth noise uses the wavelength and amplitude bands in the next table, scaled by a roughness draw in {ranges(rough, '')}; rough scenes multiply this draw by {roughscale:g}. Grade is defined before piles and trenches. The work azimuth is uniform over the full circle.

\textbf{{Piles and pits.}} Each scene has {discrete(count)} Gaussian piles, at radii {ranges(r)} and angles {ranges(angle,'degrees')} about the work direction. Nominal pile amplitude is {ranges(amp)} and base width {ranges(width)}; rough-scene ranges are {ranges(ramp)} and {ranges(rwidth)}, respectively. Independent horizontal width factors lie in {ranges(scale,'')}. There are {discrete(pitn)} negative Gaussian pits at radii {ranges(pitr)}, with full-circle azimuth, depth {ranges(pita)} and horizontal widths {ranges(pitw)}. A trench occurs with probability {trenchp:g}: radius {ranges(trr)}, direction offset {ranges(tra,'degrees')}, length {ranges(trl)}, width {ranges(trw)} and depth {ranges(trd)}. Its long axis follows that direction or its perpendicular, with equal probability; edges are smooth.

\textbf{{Materials and objects.}} Materials form Voronoi regions from {discrete(matn)} uniformly placed centres. Sand, soil, gravel and rocky probabilities are ({vals(priors[1])}) nominally and ({vals(priors[0])}) for rough terrain; a single-class draw is changed to contain another class. Boulder counts are {discrete(rockn)} nominally and {discrete(rocknr)} on rough terrain, at radii {ranges(rockr)}, with bump heights {ranges(rockh)} and widths {ranges(rockw)}. The truck is present with probability {truckp:g}, at radius {ranges(truckr)} and azimuth {ranges(truck_direction,'degrees')} to either side of the work direction, equally likely. Its tangential yaw has zero-mean Gaussian noise with standard deviation {truck_yaw[1]:g} rad. Body half-dimensions are ({vals(truck_half)}) m; cab half-dimensions are ({vals(cab_values['half'])}) m with offset {cab_values['offset']:g} m.

\textbf{{People, obstacles and lighting.}} Each object category requests {discrete(on)} placements. People use radii {prmin:g}--{radial:g} m and obstacles {ormin:g}--{radial:g} m, at offsets {ranges(oa,'degrees')} about the work direction. Truck-clearance rejection can reduce the accepted count. Sun elevation is {ranges(light,'degrees')}, azimuth spans the full circle, and each RGB tint factor is drawn in {ranges(tint,'')}.

\textbf{{Joint phases.}} Boom, arm and bucket angles are relative, in radians. Phase selection uses the probabilities and independent uniform ranges below. Approach, penetrate and lift add zero-mean Gaussian jitter with standard deviation {jitter:g} rad before clipping to the machine limits. Swing centres on the work direction with probability {workp:g}, otherwise on the truck; Gaussian angular standard deviations are {np.rad2deg(spreads[0]):g} and {np.rad2deg(spreads[1]):g} degrees, respectively. Common pose and sensor-origin validity checks follow the proposals.
\begin{{center}}\begin{{tabular}}{{lrrrr}}\toprule
Phase & Probability & Boom & Arm & Bucket\\\midrule
{chr(10).join(phase_rows)}
\bottomrule\end{{tabular}}\end{{center}}
\endgroup
\clearpage
'''
    (O/'manuscript/sections/generator_plain.tex').write_text(s)
    p=O/'manuscript/sections/specification.tex';s=p.read_text()
    if '\\input{sections/generator_plain.tex}' not in s:
        s='\\clearpage\n'+s;s=s.replace('\\par\n','\\par\n\\input{sections/generator_plain.tex}\n',1);p.write_text(s)
    # Copy the complete provenance catalogue, with additional summary anchors.
    (O/'specification').mkdir(exist_ok=True)
    import shutil
    for f in (P/'final_pre_submission_2026_10_26/specification').glob('*'):
        if f.is_file():shutil.copy2(f,O/'specification'/f.name)
    with (O/'specification/plain_summary_sources.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ROWS[0]));w.writeheader();w.writerows(ROWS)
    (O/'specification/plain_summary_constants.json').write_text(json.dumps(dict(scene_source=str(path.relative_to(P)),scene_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trench_probability=trenchp,truck_probability=truckp,material_probabilities_nominal=priors[1],material_probabilities_rough=priors[0],phases=PHASES,source_of_phases='external_validation_run_EV1_AFAST/source/excascene_snapshot/kinafford/sim/kinematics.py:94',object_min_radii=[prmin,ormin],max_radius=radial),indent=2)+'\n')
    print('Plain summary generated from',len(ROWS),'anchored draws and frozen constants')
if __name__=='__main__':main()
