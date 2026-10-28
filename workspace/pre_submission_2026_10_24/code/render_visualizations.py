"""Visualization only, using a locked saved-frame rule and existing maps.
No inference, generator sensor replay, scoring or new reported measurement.
"""
from pathlib import Path
import ast,csv,hashlib,json,sys,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle,Rectangle
OUT=Path(__file__).resolve().parents[1];PROJECT=OUT.parent;F=PROJECT/'final_evidence_2026_10_20';P=OUT/'manuscript'
sys.path.insert(0,str(F/'code'))
from common import generate_scene,safety_masks,load_npz,sha
from faults import novel_camera
from kinafford.sim.degrade import degrade_camera
from kinafford.sim.kinematics import link_transforms,sensor_pose_U,EXCAVATOR
plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
SOURCES={}
def use(path):SOURCES[str(path.relative_to(PROJECT))]=sha(path);return path
def save(fig,name):
 fig.savefig(P/'figures'/f'{name}.pdf',bbox_inches='tight');fig.savefig(P/'figures'/f'{name}.svg',bbox_inches='tight');fig.savefig(P/'figures'/f'{name}.png',dpi=240,bbox_inches='tight');plt.close(fig)
def framework(lock):
 fig=plt.figure(figsize=(8,5.65));ax=fig.add_axes([.015,.335,.97,.65]);ax.set_xlim(-.2,7.8);ax.set_ylim(-.15,3.65);ax.axis('off')
 def box(x,y,w,h,label,face='#E8F1F7'):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',facecolor=face,edgecolor='#263c4c',lw=1));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9.7)
 def arr(a,b,label=None):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=1,color='#263c4c'))
  if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',va='bottom',fontsize=8)
 xs=[0,1.95,3.9,5.9]
 box(0,2.12,1.5,1.0,'Scene generator\nTerrain and objects\nArticulated pose')
 box(1.95,2.12,1.5,1.0,'Three articulated\nRGB-D cameras\nand roof LiDAR')
 box(3.9,2.12,1.5,1.0,'Paired sensor\nfault interventions\nShared latent fields')
 box(5.9,2.12,1.5,1.0,'Frozen perception\nmodel\nCamera–LiDAR fusion')
 arr((1.55,2.62),(1.9,2.62));arr((3.5,2.62),(3.85,2.62))
 # Camera branch switch, with an explicitly separate LiDAR path.
 arr((5.43,2.84),(5.85,2.84));ax.add_patch(Circle((5.64,2.84),.075,color='#D55E00',zorder=4))
 ax.text(5.64,3.3,'Per-camera\nhealth switch',ha='center',va='bottom',fontsize=8,color='#9b4100')
 ax.plot([5.64,5.64],[3.27,2.94],color='#9b4100',lw=.8)
 arr((5.43,2.35),(5.85,2.35));ax.text(5.65,2.12,'LiDAR',ha='center',va='top',fontsize=7.5)
 box(5.9,.2,1.5,1.15,'BEV maps\nHeight, material\nHazard, terrain\nDirect utility')
 box(2.75,.2,2.4,1.15,'Predicted utility\nDirect head or composition\nStrict point selector',face='#F1F4E7')
 box(0,.2,2,1.15,'Selected point\nRegret and unsafe\nTruth-based assessment',face='#F7EBE3')
 arr((6.65,2.06),(6.65,1.4));arr((5.85,.78),(5.2,.78));arr((2.7,.78),(2.05,.78))
 ax.text(3.95,1.75,'Site grade and kinematic reach',ha='center',fontsize=8);arr((3.95,1.62),(3.95,1.4))
 ax.text(0,3.5,'(a) Paired decision evaluation',fontsize=10,weight='bold')
 # Side view reconstructed only from the same saved fixture pose; no new sensor rendering.
 d=load_npz(use(PROJECT/lock['F2_source']));q=d['q_rep'].astype(float);T=link_transforms(q)
 rig=fig.add_axes([.12,.075,.76,.225]);rig.set_aspect('equal');rig.set_xlim(-3.6,10.5);rig.set_ylim(-2,4.9)
 rig.add_patch(Rectangle((-2.3,0),4.6,1.1,facecolor='#c5c9cb',edgecolor='#596168'))
 rig.add_patch(Rectangle((-3.1,1.1),5,1.2,facecolor='#e5ba57',edgecolor='#8a6b24'))
 points=[T['boom'][:3,3],T['arm'][:3,3],T['bucket'][:3,3]]
 tip=T['bucket']@np.array([EXCAVATOR['L_bucket'],0,0,1]);points.append(tip[:3]);points=np.array(points)
 rig.plot(points[:,0],points[:,2],'-o',color='#8a6b24',lw=4,ms=3)
 for name,offset in [('cab',(-.9,.3)),('boom',(.45,.15)),('stick',(.45,.15)),('lidar',(-1.35,.45))]:
  pos=sensor_pose_U(name,q)[:3,3];color='#0072B2' if name!='lidar' else '#D55E00'
  rig.plot(pos[0],pos[2],'^' if name!='lidar' else 's',color=color,ms=5,zorder=5)
  rig.annotate('Roof LiDAR' if name=='lidar' else name.title()+' camera',(pos[0],pos[2]),xytext=(pos[0]+offset[0],pos[2]+offset[1]),fontsize=8,color=color,
   arrowprops=dict(arrowstyle='-',lw=.5,color=color))
 rig.axhline(0,color='gray',lw=.6);rig.set_xlabel('Upper-structure x (m); +x forward',fontsize=8);rig.set_ylabel('z (m)',fontsize=8)
 rig.tick_params(labelsize=7);rig.set_title('(b) Sensor arrangement at the saved fixture pose (side view)',fontsize=9,loc='left')
 save(fig,'framework')
 # The official guide encourages a separate graphical abstract, PDF preferred.
 import shutil
 shutil.copy2(P/'figures/framework.pdf',P/'graphical_abstract.pdf')
 shutil.copy2(P/'figures/framework.svg',P/'graphical_abstract.svg')
 shutil.copy2(P/'figures/framework.png',P/'graphical_abstract.png')
def inputs(lock):
 d=load_npz(use(PROJECT/lock['F2_source']));assert sha(PROJECT/lock['F2_source'])==lock['F2_source_sha256']
 code=PROJECT/'strengthening_2026_10_18/goose_ex/code/segmentation.py';tree=ast.parse(use(code).read_text())
 f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_condition')
 glob=dict(np=np,hashlib=hashlib,degrade_camera=degrade_camera)
 exec(compile(ast.Module(body=[f],type_ignores=[]),str(code),'exec'),glob)
 condition=glob['apply_condition'];conds=[('clean',0),('occlusion',3),('dust',3),('blur',3),('noise',3),('mud_water',3),('sun_glare',3)]
 labels=['Clean','Occlusion','Haze','Blur and\nexposure','Noise','Mud and\ndroplets','Glare and\nsaturation']
 fig,axes=plt.subplots(7,4,figsize=(8.1,8.55))
 for i,(c,label) in enumerate(zip(conds,labels)):
  for j,cam in enumerate(['Cab','Boom','Stick']):
   im=condition(d['rgb'][j].copy(),f'visualization_1200000_0_{cam}',c,novel_camera)
   axes[i,j].imshow(im);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
   for spine in axes[i,j].spines.values():spine.set_visible(False)
   if i==0:axes[i,j].set_title(cam,fontsize=10)
   if j==0:axes[i,j].set_ylabel(label,fontsize=9,labelpad=8,rotation=0,ha='right',va='center')
  # Representative depth coupling for the cab, using the identical latent key.
  kind,level=c;rgb=d['rgb'][0].copy();depth=d['depth'][0].copy()
  if level:
   key=f'C_fault_20261018:visualization_1200000_0_Cab:{kind}:{level}'
   rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
   image,depth=(novel_camera(rgb,depth,kind,rng) if kind in ('mud_water','sun_glare') else degrade_camera(rgb,depth,kind,level,rng))
   assert np.array_equal(image,condition(rgb,'visualization_1200000_0_Cab',c,novel_camera))
  valid=np.isfinite(depth)&(depth>.05);axes[i,3].imshow(valid,cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[i,3].set_axis_off()
  if i==0:axes[i,3].set_title('Cab depth valid',fontsize=10)
 fig.subplots_adjust(left=.17,right=.995,top=.96,bottom=.01,wspace=.015,hspace=.05);save(fig,'example_inputs')
def decisions(lock):
 fig,axes=plt.subplots(3,3,figsize=(7.6,7.6));extent=(-8,16,-12,12)
 keys={(r['scene_id'],r['frame_id']) for r in lock['F3_frames']}
 source=use(PROJECT/lock['F3_source']);saved={}
 for r in csv.DictReader(source.open()):
  k=(int(r['scene_id']),int(r['frame_id']))
  if k in keys and r['condition']=='cam_s3' and r['policy']=='direct' and r['arm'] in ['keep_all','gate']:saved.setdefault(k,{})[r['arm']]=r
 effects=[]
 for i,r in enumerate(lock['F3_frames']):
  d=load_npz(use(PROJECT/r['data']));assert sha(PROJECT/r['data'])==r['data_sha256']
  rec=json.loads(use(F/f"study4/holdout/generation/{r['scene_id']}.json").read_text())
  scene=generate_scene(rec['geometry_seed'],ood_terrain=rec['rough'])
  pose=next(a['q'] for a in rec['attempts'] if a['accepted'] and int(a['frame_id'])==r['frame_id'])
  masks=safety_masks(scene,np.asarray(pose),d['gt_sem'])
  # Geometry replay is only for display contours; it generates no sensor frame or score.
  pred=[load_npz(use(F/r[a]['prediction_path'])) for a in ['keep_all','gate']]
  pair=saved[(r['scene_id'],r['frame_id'])];delta=float(pair['gate']['regret'])-float(pair['keep_all']['regret'])
  effects.append(dict(scene_id=r['scene_id'],frame_id=r['frame_id'],saved_keep_regret=pair['keep_all']['regret'],saved_gate_regret=pair['gate']['regret'],gate_minus_keep_regret=delta,source=lock['F3_source']))
  for j,m in enumerate([d['gt_aff'],pred[0]['aff'],pred[1]['aff']]):
   ax=axes[i,j];im=ax.imshow(m,origin='lower',vmin=0,vmax=1,cmap='viridis',extent=extent,interpolation='nearest')
   xx=np.linspace(-8+.125,16-.125,m.shape[1]);yy=np.linspace(-12+.125,12-.125,m.shape[0])
   ax.contour(xx,yy,masks[:4].any(0).astype(int),levels=[.5],colors=['#ffcf66'],linewidths=.6)
   if masks[0].any():ax.contour(xx,yy,masks[0].astype(int),levels=[.5],colors=['#E64B35'],linewidths=1.1)
   ax.contour(xx,yy,d['gt_reach'],levels=[.05],colors=['white'],linewidths=.65,linestyles='dashed')
   for arm in (['gate','keep_all'] if j==0 else [['keep_all','gate'][j-1]]):
    e=r[arm];row,col=int(e['pick_row']),int(e['pick_col'])
    if row>=0:
     if arm=='gate':ax.plot(-8+(col+.5)*.25,-12+(row+.5)*.25,'o',ms=8,mew=1.5,mfc='none',color='#ffb14e')
     else:ax.plot(-8+(col+.5)*.25,-12+(row+.5)*.25,'x',ms=7,mew=1.5,color='white')
   ax.plot(0,0,'^',color='black',markeredgecolor='white',ms=5)
   ax.set_xticks([-8,0,8,16]);ax.set_yticks([-12,0,12]);ax.tick_params(labelsize=6,pad=1)
   if i<2:ax.set_xticklabels([])
   else:ax.set_xlabel('x (m), forward →',fontsize=7,labelpad=2)
   if j>0:ax.set_yticklabels([])
   if i==0:ax.set_title(['Truth utility','Keep-all predicted utility','Gated predicted utility'][j],fontsize=9)
   if j==0:ax.set_ylabel(f"{r['percentile_rank']}th percentile; Δregret {delta:+.3f}\nScene {r['scene_id']}, slot {r['frame_id']}; y (m)",fontsize=7.5)
 fig.subplots_adjust(left=.17,right=.9,top=.93,bottom=.105,wspace=.08,hspace=.12)
 cax=fig.add_axes([.92,.18,.015,.64]);fig.colorbar(im,cax=cax,label='Utility (shared scale)')
 fig.text(.54,.035,'White ×: keep-all   Orange ○: gate   Black △: swing origin\nRed: person buffer   Gold: hazard union   White dashed: reach boundary',ha='center',fontsize=8)
 with (OUT/'V1/qualitative_pair_effects.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)
 save(fig,'qualitative_decisions')
def main():
 start=time.time();lock=json.loads((OUT/'visualizations/selection_lock.json').read_text());assert lock['status']=='LOCKED_BEFORE_RENDERING'
 use(F/'code/faults.py');use(PROJECT/'external_validation_run_EV1_AFAST/source/excascene_snapshot/kinafford/sim/degrade.py')
 framework(lock);inputs(lock);decisions(lock)
 (OUT/'visualizations/render_receipt.json').write_text(json.dumps(dict(status='COMPLETE',seconds=time.time()-start,source_sha256=SOURCES,
  new_metrics=False,new_scores=False,new_neural_forwards=0,gpu_hours=0,existing_prediction_maps_used=True,
  geometry_replay='Three saved scene geometries for buffer contours only; no sensor or cohort generation.',
  deterministic_image_degradation='Frozen image-check apply_condition and shared novel fault functions on one saved clean frame.',
  real_images_included=False),indent=2)+'\n');print('Four vector assets rendered: framework, inputs, qualitative decisions and graphical abstract; zero GPU.')
if __name__=='__main__':main()
